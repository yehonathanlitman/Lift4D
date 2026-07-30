# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

from tqdm import tqdm
import torch
from loguru import logger
from functools import wraps
from torch.utils._pytree import tree_map_only


def set_attention_backend():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    logger.info(f"GPU name is {gpu_name}")
    if "A100" in gpu_name or "H100" in gpu_name or "H200" in gpu_name:
        try:
            import flash_attn  # noqa: F401
            os.environ["ATTN_BACKEND"] = "flash_attn"
            os.environ["SPARSE_ATTN_BACKEND"] = "flash_attn"
        except ImportError:
            logger.warning("flash_attn not installed; falling back to sdpa")
            os.environ.setdefault("ATTN_BACKEND", "sdpa")
            os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")

set_attention_backend()

from typing import List, Union, Optional
from hydra.utils import instantiate
from omegaconf import OmegaConf
import numpy as np

from PIL import Image
from sam3d_objects.pipeline import preprocess_utils
from sam3d_objects.data.dataset.tdfy.img_and_mask_transforms import (
    get_mask,
)
from sam3d_objects.pipeline.inference_utils import (
    get_pose_decoder,
    SLAT_MEAN,
    SLAT_STD,
    downsample_sparse_structure,
    prune_sparse_structure,
)

from sam3d_objects.model.io import (
    load_model_from_checkpoint,
    filter_and_remove_prefix_state_dict_fn,
)

from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
from sam3d_objects.model.backbone.tdfy_dit.utils import postprocessing_utils
from safetensors.torch import load_file


class InferencePipeline:
    def __init__(
        self,
        ss_generator_config_path,
        ss_generator_ckpt_path,
        slat_generator_config_path,
        slat_generator_ckpt_path,
        ss_decoder_config_path,
        ss_decoder_ckpt_path,
        slat_decoder_gs_config_path,
        slat_decoder_gs_ckpt_path,
        slat_decoder_mesh_config_path,
        slat_decoder_mesh_ckpt_path,
        slat_decoder_gs_4_config_path=None,
        slat_decoder_gs_4_ckpt_path=None,
        ss_encoder_config_path=None,
        ss_encoder_ckpt_path=None,
        decode_formats=["gaussian", "mesh"],
        dtype="bfloat16",
        pad_size=1.0,
        version="v0",
        device="cuda",
        ss_preprocessor=preprocess_utils.get_default_preprocessor(),
        slat_preprocessor=preprocess_utils.get_default_preprocessor(),
        ss_condition_input_mapping=["image"],
        slat_condition_input_mapping=["image"],
        pose_decoder_name="default",
        workspace_dir="",
        downsample_ss_dist=0,  # the distance we use to downsample
        ss_inference_steps=25,
        ss_rescale_t=3,
        ss_cfg_strength=7,
        ss_cfg_interval=[0, 500],
        ss_cfg_strength_pm=0.0,
        slat_inference_steps=25,
        slat_rescale_t=3,
        slat_cfg_strength=5,
        slat_cfg_interval=[0, 500],
        rendering_engine: str = "nvdiffrast",  # nvdiffrast OR pytorch3d,
        shape_model_dtype=None,
        compile_model=False,
        slat_mean=SLAT_MEAN,
        slat_std=SLAT_STD,
    ):
        self.rendering_engine = rendering_engine
        self.device = torch.device(device)
        self.compile_model = compile_model
        logger.info(f"self.device: {self.device}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
        logger.info(f"Actually using GPU: {torch.cuda.current_device()}")
        with self.device:
            self.decode_formats = decode_formats
            self.pad_size = pad_size
            self.version = version
            self.ss_condition_input_mapping = ss_condition_input_mapping
            self.slat_condition_input_mapping = slat_condition_input_mapping
            self.workspace_dir = workspace_dir
            self.downsample_ss_dist = downsample_ss_dist
            self.ss_inference_steps = ss_inference_steps
            self.ss_rescale_t = ss_rescale_t
            self.ss_cfg_strength = ss_cfg_strength
            self.ss_cfg_interval = ss_cfg_interval
            self.ss_cfg_strength_pm = ss_cfg_strength_pm
            self.slat_inference_steps = slat_inference_steps
            self.slat_rescale_t = slat_rescale_t
            self.slat_cfg_strength = slat_cfg_strength
            self.slat_cfg_interval = slat_cfg_interval

            self.dtype = self._get_dtype(dtype)
            if shape_model_dtype is None:
                self.shape_model_dtype = self.dtype
            else:
                self.shape_model_dtype = self._get_dtype(shape_model_dtype) 


            # Setup preprocessors
            self.pose_decoder = self.init_pose_decoder(ss_generator_config_path, pose_decoder_name)
            self.ss_preprocessor = self.init_ss_preprocessor(ss_preprocessor, ss_generator_config_path)
            self.slat_preprocessor = slat_preprocessor
    
            logger.info("Loading model weights...")

            ss_generator = self.init_ss_generator(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_generator = self.init_slat_generator(
                slat_generator_config_path, slat_generator_ckpt_path
            )
            ss_decoder = self.init_ss_decoder(
                ss_decoder_config_path, ss_decoder_ckpt_path
            )
            ss_encoder = self.init_ss_encoder(
                ss_encoder_config_path, ss_encoder_ckpt_path
            )
            slat_decoder_gs = self.init_slat_decoder_gs(
                slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
            )
            slat_decoder_gs_4 = self.init_slat_decoder_gs(
                slat_decoder_gs_4_config_path, slat_decoder_gs_4_ckpt_path
            )
            slat_decoder_mesh = self.init_slat_decoder_mesh(
                slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
            )

            # Load conditioner embedder so that we only load it once
            ss_condition_embedder = self.init_ss_condition_embedder(
                ss_generator_config_path, ss_generator_ckpt_path
            )
            slat_condition_embedder = self.init_slat_condition_embedder(
                slat_generator_config_path, slat_generator_ckpt_path
            )

            self.condition_embedders = {
                "ss_condition_embedder": ss_condition_embedder,
                "slat_condition_embedder": slat_condition_embedder,
            }

            # override generator and condition embedder setting
            self.override_ss_generator_cfg_config(
                ss_generator,
                cfg_strength=ss_cfg_strength,
                inference_steps=ss_inference_steps,
                rescale_t=ss_rescale_t,
                cfg_interval=ss_cfg_interval,
                cfg_strength_pm=ss_cfg_strength_pm,
            )
            self.override_slat_generator_cfg_config(
                slat_generator,
                cfg_strength=slat_cfg_strength,
                inference_steps=slat_inference_steps,
                rescale_t=slat_rescale_t,
                cfg_interval=slat_cfg_interval,
            )

            self.models = torch.nn.ModuleDict(
                {
                    "ss_generator": ss_generator,
                    "slat_generator": slat_generator,
                    "ss_encoder": ss_encoder,
                    "ss_decoder": ss_decoder,
                    "slat_decoder_gs": slat_decoder_gs,
                    "slat_decoder_gs_4": slat_decoder_gs_4,
                    "slat_decoder_mesh": slat_decoder_mesh,
                }
            )
            logger.info("Loading model weights completed!")

            if self.compile_model:
                logger.info("Compiling model...")
                self._compile()
                logger.info("Model compilation completed!")
            self.slat_mean = torch.tensor(slat_mean)
            self.slat_std = torch.tensor(slat_std)

    def _compile(self):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        compile_mode = "max-autotune"
        logger.info(f"Compile mode {compile_mode}")

        def clone_output_wrapper(f):
            @wraps(f)
            def wrapped(*args, **kwargs):
                outputs = f(*args, **kwargs)
                return tree_map_only(
                    torch.Tensor, lambda t: t.clone() if t.is_cuda else t, outputs
                )

            return wrapped

        self.embed_condition = clone_output_wrapper(
            torch.compile(
                self.embed_condition,
                mode=compile_mode,
                fullgraph=True,  # _preprocess_input in dino is not compatible with fullgraph
            )
        )
        self.models["ss_generator"].reverse_fn.inner_forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_generator"].reverse_fn.inner_forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self.models["ss_decoder"].forward = clone_output_wrapper(
            torch.compile(
                self.models["ss_decoder"].forward,
                mode=compile_mode,
                fullgraph=True,
            )
        )

        self._warmup()

    def _warmup(self, num_warmup_iters=3):
        test_image = np.ones((512, 512, 4), dtype=np.uint8) * 255
        test_image[:, :, :3] = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(test_image)
        mask = None
        image = self.merge_image_and_mask(image, mask)

        for _ in tqdm(range(num_warmup_iters)):
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            ss_return_dict = self.sample_sparse_structure(ss_input_dict)
            coords = ss_return_dict["coords"]
            slat = self.sample_slat(slat_input_dict, coords)

    def instantiate_and_load_from_pretrained(
        self,
        config,
        ckpt_path,
        state_dict_fn=None,
        state_dict_key="state_dict",
        device="cuda", 
    ):
        model = instantiate(config)

        if ckpt_path.endswith(".safetensors"):
            state_dict = load_file(ckpt_path, device="cuda")
            if state_dict_fn is not None:
                state_dict = state_dict_fn(state_dict)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
        else:
            model = load_model_from_checkpoint(
                model,
                ckpt_path,
                strict=True,
                device="cpu",
                freeze=True,
                eval=True,
                state_dict_key=state_dict_key,
                state_dict_fn=state_dict_fn,
            )
        model = model.to(device)

        return model

    def init_pose_decoder(self, ss_generator_config_path, pose_decoder_name):
        if pose_decoder_name is None:
            pose_decoder_name = OmegaConf.load(os.path.join(self.workspace_dir, ss_generator_config_path))["module"]["pose_target_convention"]
        logger.info(f"Using pose decoder: {pose_decoder_name}")
        return get_pose_decoder(pose_decoder_name)

    def init_ss_preprocessor(self, ss_preprocessor, ss_generator_config_path):
        if ss_preprocessor is not None:
            return ss_preprocessor
        config = OmegaConf.load(os.path.join(self.workspace_dir, ss_generator_config_path))["tdfy"]["val_preprocessor"]
        return instantiate(config)

    def init_ss_generator(self, ss_generator_config_path, ss_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )["module"]["generator"]["backbone"]

        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )

        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.device,
        )

    def init_slat_generator(self, slat_generator_config_path, slat_generator_ckpt_path):
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, slat_generator_config_path)
        )["module"]["generator"]["backbone"]
        state_dict_prefix_func = filter_and_remove_prefix_state_dict_fn(
            "_base_models.generator."
        )
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, slat_generator_ckpt_path),
            state_dict_fn=state_dict_prefix_func,
            device=self.device,
        )

    def init_ss_encoder(self, ss_encoder_config_path, ss_encoder_ckpt_path):
        if ss_encoder_ckpt_path is not None:
            # override to avoid problem loading
            config = OmegaConf.load(
                os.path.join(self.workspace_dir, ss_encoder_config_path)
            )
            if "pretrained_ckpt_path" in config:
                del config["pretrained_ckpt_path"]
            return self.instantiate_and_load_from_pretrained(
                config,
                os.path.join(self.workspace_dir, ss_encoder_ckpt_path),
                device=self.device,
                state_dict_key=None,
            )
        else:
            return None

    def init_ss_decoder(self, ss_decoder_config_path, ss_decoder_ckpt_path):
        # override to avoid problem loading
        config = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_decoder_config_path)
        )
        if "pretrained_ckpt_path" in config:
            del config["pretrained_ckpt_path"]
        return self.instantiate_and_load_from_pretrained(
            config,
            os.path.join(self.workspace_dir, ss_decoder_ckpt_path),
            device=self.device,
            state_dict_key=None,
        )

    def init_slat_decoder_gs(
        self, slat_decoder_gs_config_path, slat_decoder_gs_ckpt_path
    ):
        if slat_decoder_gs_config_path is None:
            return None
        else:
            return self.instantiate_and_load_from_pretrained(
                OmegaConf.load(
                    os.path.join(self.workspace_dir, slat_decoder_gs_config_path)
                ),
                os.path.join(self.workspace_dir, slat_decoder_gs_ckpt_path),
                device=self.device,
                state_dict_key=None,
            )

    def init_slat_decoder_mesh(
        self, slat_decoder_mesh_config_path, slat_decoder_mesh_ckpt_path
    ):
        return self.instantiate_and_load_from_pretrained(
            OmegaConf.load(
                os.path.join(self.workspace_dir, slat_decoder_mesh_config_path)
            ),
            os.path.join(self.workspace_dir, slat_decoder_mesh_ckpt_path),
            device=self.device,
            state_dict_key=None,
        )

    def init_ss_condition_embedder(
        self, ss_generator_config_path, ss_generator_ckpt_path
    ):
        conf = OmegaConf.load(
            os.path.join(self.workspace_dir, ss_generator_config_path)
        )
        if "condition_embedder" in conf["module"]:
            return self.instantiate_and_load_from_pretrained(
                conf["module"]["condition_embedder"]["backbone"],
                os.path.join(self.workspace_dir, ss_generator_ckpt_path),
                state_dict_fn=filter_and_remove_prefix_state_dict_fn(
                    "_base_models.condition_embedder."
                ),
                device=self.device,
            )
        else:
            return None

    def init_slat_condition_embedder(
        self, slat_generator_config_path, slat_generator_ckpt_path
    ):
        return self.init_ss_condition_embedder(
            slat_generator_config_path, slat_generator_ckpt_path
        )


    def override_ss_generator_cfg_config(
        self,
        ss_generator,
        cfg_strength=7,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
        cfg_strength_pm=0.0,
    ):
        # override generator setting
        ss_generator.inference_steps = inference_steps
        ss_generator.reverse_fn.strength = cfg_strength
        ss_generator.reverse_fn.interval = cfg_interval
        ss_generator.rescale_t = rescale_t
        ss_generator.reverse_fn.backbone.condition_embedder.normalize_images = True
        ss_generator.reverse_fn.unconditional_handling = "add_flag"
        ss_generator.reverse_fn.strength_pm = cfg_strength_pm

        logger.info(
            "ss_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}, cfg_strength_pm={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
            cfg_strength_pm,
        )

    def override_slat_generator_cfg_config(
        self,
        slat_generator,
        cfg_strength=5,
        inference_steps=25,
        rescale_t=3,
        cfg_interval=[0, 500],
    ):
        slat_generator.inference_steps = inference_steps
        slat_generator.reverse_fn.strength = cfg_strength
        slat_generator.reverse_fn.interval = cfg_interval
        slat_generator.rescale_t = rescale_t

        logger.info(
            "slat_generator parameters: inference_steps={}, cfg_strength={}, cfg_interval={}, rescale_t={}",
            inference_steps,
            cfg_strength,
            cfg_interval,
            rescale_t,
        )


    def run(
        self,
        image: Union[None, Image.Image, np.ndarray],
        mask: Union[None, Image.Image, np.ndarray] = None,
        seed=42,
        stage1_only=False,
        with_mesh_postprocess=True,
        with_texture_baking=True,
        use_vertex_color=False,
        stage1_inference_steps=None,
        stage2_inference_steps=None,
        use_stage1_distillation=False,
        use_stage2_distillation=False,
        decode_formats=None,
    ) -> dict:
        """
        Parameters:
        - image (Image): The input image to be processed.
        - seed (int, optional): The random seed for reproducibility. Default is 42.
        - stage1_only (bool, optional): If True, only the sparse structure is sampled and returned. Default is False.
        - with_mesh_postprocess (bool, optional): If True, performs mesh post-processing. Default is True.
        - with_texture_baking (bool, optional): If True, applies texture baking to the 3D model. Default is True.
        Returns:
        - dict: A dictionary containing the GLB file and additional data from the sparse structure sampling.
        """
        # This should only happen if called from demo
        image = self.merge_image_and_mask(image, mask)
        with self.device:
            ss_input_dict = self.preprocess_image(image, self.ss_preprocessor)
            slat_input_dict = self.preprocess_image(image, self.slat_preprocessor)
            torch.manual_seed(seed)
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                use_distillation=use_stage1_distillation,
            )

            ss_return_dict.update(self.pose_decoder(ss_return_dict))

            if "scale" in ss_return_dict:
                logger.info(f"Rescaling scale by {ss_return_dict['downsample_factor']}")
                ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]
            if stage1_only:
                logger.info("Finished!")
                ss_return_dict["voxel"] = ss_return_dict["coords"][:, 1:] / 64 - 0.5
                return ss_return_dict

            coords = ss_return_dict["coords"]
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
                use_distillation=use_stage2_distillation,
            )
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )
            logger.info("Finished!")

            return {
                **ss_return_dict,
                **outputs,
            }

    def postprocess_slat_output(
        self, outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
    ):
        # GLB files can be extracted from the outputs
        logger.info(
            f"Postprocessing mesh with option with_mesh_postprocess {with_mesh_postprocess}, with_texture_baking {with_texture_baking}..."
        )
        if "mesh" in outputs:
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                # Optional parameters
                simplify=0.95,  # Ratio of triangles to remove in the simplification process
                texture_size=1024,  # Size of the texture used for the GLB
                verbose=False,
                with_mesh_postprocess=with_mesh_postprocess,
                with_texture_baking=with_texture_baking,
                use_vertex_color=use_vertex_color,
                rendering_engine=self.rendering_engine,
            )

        # glb.export("sample.glb")
        else:
            glb = None

        outputs["glb"] = glb

        if "gaussian" in outputs:
            outputs["gs"] = outputs["gaussian"][0]

        if "gaussian_4" in outputs:
            outputs["gs_4"] = outputs["gaussian_4"][0]

        return outputs

    def merge_image_and_mask(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Union[None, np.ndarray, Image.Image],
    ):
        if mask is not None:
            if isinstance(image, Image.Image):
                image = np.array(image)

            mask = np.array(mask)
            if mask.ndim == 2:
                mask = mask[..., None]

            logger.info(f"Replacing alpha channel with the provided mask")
            assert mask.shape[:2] == image.shape[:2]
            image = np.concatenate([image[..., :3], mask], axis=-1)

        image = np.array(image)
        return image

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        logger.info("Decoding sparse latent...")
        ret = {}
        with torch.no_grad():
            if "mesh" in formats:
                ret["mesh"] = self.models["slat_decoder_mesh"](slat)
            if "gaussian" in formats:
                ret["gaussian"] = self.models["slat_decoder_gs"](slat)
            if "gaussian_4" in formats:
                ret["gaussian_4"] = self.models["slat_decoder_gs_4"](slat)
        # if "radiance_field" in formats:
        #     ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def is_mm_dit(self, model_name="ss_generator"):
        return hasattr(self.models[model_name].reverse_fn.backbone, "latent_mapping")

    def embed_condition(self, condition_embedder, *args, **kwargs):
        if condition_embedder is not None:
            tokens = condition_embedder(*args, **kwargs)
            return tokens, None, None
        return None, args, kwargs

    def get_condition_input(self, condition_embedder, input_dict, input_mapping):
        condition_args = self.map_input_keys(input_dict, input_mapping)
        condition_kwargs = {
            k: v for k, v in input_dict.items() if k not in input_mapping
        }
        logger.info("Running condition embedder ...")
        embedded_cond, condition_args, condition_kwargs = self.embed_condition(
            condition_embedder, *condition_args, **condition_kwargs
        )
        logger.info("Condition embedder finishes!")
        if embedded_cond is not None:
            condition_args = (embedded_cond,)
            condition_kwargs = {}

        return condition_args, condition_kwargs

    def sample_sparse_structure(
        self,
        ss_input_dict: dict,
        inference_steps=None,
        use_distillation=False,
        x_start=None,
        t_start=0.0,
    ):
        ss_generator = self.models["ss_generator"]
        ss_decoder = self.models["ss_decoder"]
        if use_distillation:
            ss_generator.no_shortcut = False
            ss_generator.reverse_fn.strength = 0
            ss_generator.reverse_fn.strength_pm = 0
        else:
            ss_generator.no_shortcut = True
            ss_generator.reverse_fn.strength = self.ss_cfg_strength
            ss_generator.reverse_fn.strength_pm = self.ss_cfg_strength_pm

        prev_inference_steps = ss_generator.inference_steps
        if inference_steps:
            ss_generator.inference_steps = inference_steps

        image = ss_input_dict["image"]
        bs = image.shape[0]
        logger.info(
            "Sampling sparse structure: inference_steps={}, strength={}, interval={}, rescale_t={}, cfg_strength_pm={}",
            ss_generator.inference_steps,
            ss_generator.reverse_fn.strength,
            ss_generator.reverse_fn.interval,
            ss_generator.rescale_t,
            ss_generator.reverse_fn.strength_pm,
        )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=self.shape_model_dtype):
                if self.is_mm_dit():
                    latent_shape_dict = {
                        k: (bs,) + (v.pos_emb.shape[0], v.input_layer.in_features)
                        for k, v in ss_generator.reverse_fn.backbone.latent_mapping.items()
                    }
                else:
                    latent_shape_dict = (bs,) + (4096, 8)

                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["ss_condition_embedder"],
                    ss_input_dict,
                    self.ss_condition_input_mapping,
                )
                # Add warm-start args if provided
                if x_start is not None:
                    # Fix for MM-DiT models which expect dict inputs
                    if self.is_mm_dit() and torch.is_tensor(x_start):
                         x_start = {"shape": x_start}
                    
                    condition_kwargs["x_start"] = x_start
                    condition_kwargs["t_start"] = t_start
                
                return_dict = ss_generator(
                    latent_shape_dict,
                    image.device,
                    *condition_args,
                    **condition_kwargs,
                )
                if not self.is_mm_dit():
                    return_dict = {"shape": return_dict}

                shape_latent = return_dict["shape"]
                ss = ss_decoder(
                    shape_latent.permute(0, 2, 1)
                    .contiguous()
                    .view(shape_latent.shape[0], 8, 16, 16, 16)
                )
                coords = torch.argwhere(ss > 0)[:, [0, 2, 3, 4]].int()

                # downsample output
                return_dict["coords_original"] = coords
                original_shape = coords.shape
                if self.downsample_ss_dist > 0:
                    coords = prune_sparse_structure(
                        coords,
                        max_neighbor_axes_dist=self.downsample_ss_dist,
                    )
                coords, downsample_factor = downsample_sparse_structure(coords)
                logger.info(
                    f"Downsampled coords from {original_shape[0]} to {coords.shape[0]}"
                )
                return_dict["coords"] = coords
                return_dict["downsample_factor"] = downsample_factor

        ss_generator.inference_steps = prev_inference_steps
        return return_dict

    def sample_slat(
        self,
        slat_input: dict,
        coords: torch.Tensor,
        inference_steps=25,
        use_distillation=False,
        x_start=None,
        t_start=0.0,
    ) -> sp.SparseTensor:
        image = slat_input["image"]
        DEVICE = image.device
        slat_generator = self.models["slat_generator"]
        latent_shape = (image.shape[0],) + (coords.shape[0], 8)
        prev_inference_steps = slat_generator.inference_steps
        if inference_steps:
            slat_generator.inference_steps = inference_steps
        if use_distillation:
            slat_generator.no_shortcut = False
            slat_generator.reverse_fn.strength = 0
        else:
            slat_generator.no_shortcut = True
            slat_generator.reverse_fn.strength = self.slat_cfg_strength

        logger.info(
            "Sampling sparse latent: inference_steps={}, strength={}, interval={}, rescale_t={}",
            slat_generator.inference_steps,
            slat_generator.reverse_fn.strength,
            slat_generator.reverse_fn.interval,
            slat_generator.rescale_t,
        )

        with torch.autocast(device_type="cuda", dtype=self.dtype):
            with torch.no_grad():
                condition_args, condition_kwargs = self.get_condition_input(
                    self.condition_embedders["slat_condition_embedder"],
                    slat_input,
                    self.slat_condition_input_mapping,
                )
                condition_args += (coords.cpu().numpy(),)

                # Add warm-start args if provided
                if x_start is not None:
                    condition_kwargs["x_start"] = x_start
                    condition_kwargs["t_start"] = t_start

                slat = slat_generator(
                    latent_shape, DEVICE, *condition_args, **condition_kwargs
                )
                slat = sp.SparseTensor(
                    coords=coords,
                    feats=slat[0],
                ).to(DEVICE)
                slat = slat * self.slat_std.to(DEVICE) + self.slat_mean.to(DEVICE)

        slat_generator.inference_steps = prev_inference_steps
        return slat

    def _apply_transform(self, input: torch.Tensor, transform):
        if input is not None:
            input = transform(input)

        return input

    def _preprocess_image_and_mask(
        self, rgb_image, mask_image, img_mask_joint_transform
    ):
        for trans in img_mask_joint_transform:
            rgb_image, mask_image = trans(rgb_image, mask_image)
        return rgb_image, mask_image

    def map_input_keys(self, item, condition_input_mapping):
        output = [item[k] for k in condition_input_mapping]

        return output

    def image_to_float(self, image):
        image = np.array(image)
        image = image / 255
        image = image.astype(np.float32)
        return image

    def preprocess_image(
        self, image: Union[Image.Image, np.ndarray], preprocessor
    ) -> torch.Tensor:
        # canonical type is numpy
        if not isinstance(input, np.ndarray):
            image = np.array(image)

        assert image.ndim == 3  # no batch dimension as of now
        assert image.shape[-1] == 4  # rgba format
        assert image.dtype == np.uint8  # [0,255] range

        rgba_image = torch.from_numpy(self.image_to_float(image))
        rgba_image = rgba_image.permute(2, 0, 1).contiguous()
        rgb_image = rgba_image[:3]
        rgb_image_mask = (get_mask(rgba_image, None, "ALPHA_CHANNEL") > 0).float()
        processed_rgb_image, processed_mask = self._preprocess_image_and_mask(
            rgb_image, rgb_image_mask, preprocessor.img_mask_joint_transform
        )

        # transform tensor to model input
        processed_rgb_image = self._apply_transform(
            processed_rgb_image, preprocessor.img_transform
        )
        processed_mask = self._apply_transform(
            processed_mask, preprocessor.mask_transform
        )

        # full image, with only processing from the image
        rgb_image = self._apply_transform(rgb_image, preprocessor.img_transform)
        rgb_image_mask = self._apply_transform(
            rgb_image_mask, preprocessor.mask_transform
        )
        item = {
            "mask": processed_mask[None].to(self.device),
            "image": processed_rgb_image[None].to(self.device),
            "rgb_image": rgb_image[None].to(self.device),
            "rgb_image_mask": rgb_image_mask[None].to(self.device),
        }

        return item

    @staticmethod
    def _get_dtype(dtype):
        if dtype == "bfloat16":
            return torch.bfloat16
        elif dtype == "float16":
            return torch.float16
        elif dtype == "float32":
            return torch.float32
        else:
            raise NotImplementedError

    @torch.no_grad()
    def run_video_inference(
        self,
        frame_images: List[Union[np.ndarray, Image.Image]],
        frame_masks: List[Optional[Union[None, np.ndarray, Image.Image]]],
        seed: Optional[int] = None,
        stage1_inference_steps: Optional[int] = None,
        stage2_inference_steps: Optional[int] = None,
        decode_formats: Optional[List[str]] = None,
        with_mesh_postprocess: bool = True,
        with_texture_baking: bool = True,
        use_vertex_color: bool = False,
        initial_frame_index: int = 0,
        consistency_strength: float = 0.3,
    ) -> List[dict]:
        """
        Run video inference on a sequence of frames.

        Frames are processed starting from initial_frame_index and
        propagating bi-directionally. Each subsequent frame is warm-started
        (SDEdit) from its already-processed neighbor's Stage 1 latents,
        blended with fresh noise at consistency_strength.

        Args:
            frame_images: List of frame images (one per video frame)
            frame_masks: List of masks corresponding to each frame
            seed: Random seed (applied at the initial frame)
            stage1_inference_steps: Stage 1 (SS) inference steps
            stage2_inference_steps: Stage 2 (SLAT) inference steps
            decode_formats: Output formats (e.g., ["gaussian", "mesh"])
            with_mesh_postprocess: Whether to postprocess mesh
            with_texture_baking: Whether to bake textures
            use_vertex_color: Whether to use vertex colors
            initial_frame_index: Index of the frame to start processing from.
            consistency_strength: Strength of temporal consistency [0.0, 1.0].
                Higher preserves more structure from the neighbor frame,
                lower allows more deformation.

        Returns:
            List of result dicts, one per frame.
        """
        num_frames = len(frame_images)
        if frame_masks is None:
            frame_masks = [None] * num_frames
        assert len(frame_masks) == num_frames, "Number of masks must match number of frames"

        if initial_frame_index < 0 or initial_frame_index >= num_frames:
            logger.warning(f"Initial frame index {initial_frame_index} out of bounds 0-{num_frames-1}, clipping.")
            initial_frame_index = max(0, min(initial_frame_index, num_frames - 1))

        logger.info(f"Running video inference on {num_frames} frames with SDEdit consistency={consistency_strength}")
        logger.info(f"Starting from frame {initial_frame_index}, propagating bi-directionally")

        # Initialize results list (will be filled out of order)
        results = [None] * num_frames

        # Define processing order: [initial, initial+1...end, initial-1...0]
        process_order = [initial_frame_index]
        process_order.extend(range(initial_frame_index + 1, num_frames))
        process_order.extend(range(initial_frame_index - 1, -1, -1))

        # Keep track of previous results for conditioning
        processed_results = {}

        for frame_idx in process_order:
            logger.info(f"Processing frame {frame_idx + 1}/{num_frames}")

            if seed is not None and frame_idx == initial_frame_index:
                torch.manual_seed(seed)

            image = frame_images[frame_idx]
            mask = frame_masks[frame_idx]
            rgba_image = self._prepare_frame_rgba(image, mask)

            # Preprocessing (with self.device sets default device for tensor
            # creation, matching how run() calls compute_pointmap)
            frame_intrinsics = None
            if hasattr(self, 'compute_pointmap'):
                with self.device:
                    pointmap_dict = self.compute_pointmap(rgba_image)
                pointmap = pointmap_dict["pointmap"]
                frame_intrinsics = pointmap_dict.get("intrinsics", None)
                ss_input_dict = self.preprocess_image(rgba_image, self.ss_preprocessor, pointmap=pointmap)
                slat_input_dict = self.preprocess_image(rgba_image, self.slat_preprocessor)
            else:
                ss_input_dict = self.preprocess_image(rgba_image, self.ss_preprocessor)
                slat_input_dict = self.preprocess_image(rgba_image, self.slat_preprocessor)

            # SDEdit warm-start from the already-processed neighbor frame
            x_start = None
            t_start = 0.0
            if frame_idx != initial_frame_index:
                neighbor_idx = frame_idx - 1 if frame_idx > initial_frame_index else frame_idx + 1
                if neighbor_idx in processed_results:
                    t_start = consistency_strength
                    neighbor = processed_results[neighbor_idx]
                    if self.is_mm_dit():
                        # MM-DiT expects a dict keyed by the generator's latent names
                        latent_keys = list(self.models["ss_generator"].reverse_fn.backbone.latent_mapping.keys())
                        x_start = {}
                        for key in latent_keys:
                            if key in neighbor:
                                noise = torch.randn_like(neighbor[key])
                                x_start[key] = (1 - t_start) * noise + t_start * neighbor[key]
                            else:
                                logger.warning(f"Latent key '{key}' not found in neighbor results, skipping warm-start for this key")
                    elif "shape" in neighbor:
                        noise = torch.randn_like(neighbor["shape"])
                        x_start = (1 - t_start) * noise + t_start * neighbor["shape"]

                if x_start is not None:
                    logger.info(f"Warm-starting frame {frame_idx} from frame {neighbor_idx} with strength {t_start}")

            # Stage 1: Sample SS with optional warm-start
            ss_return_dict = self.sample_sparse_structure(
                ss_input_dict,
                inference_steps=stage1_inference_steps,
                x_start=x_start,
                t_start=t_start,
            )

            # Update pose decoder
            pointmap_scale = ss_input_dict.get("pointmap_scale", None)
            pointmap_shift = ss_input_dict.get("pointmap_shift", None)
            ss_return_dict.update(
                self.pose_decoder(
                    ss_return_dict,
                    scene_scale=pointmap_scale,
                    scene_shift=pointmap_shift,
                )
            )

            if "scale" in ss_return_dict:
                ss_return_dict["scale"] = ss_return_dict["scale"] * ss_return_dict["downsample_factor"]

            coords = ss_return_dict["coords"]

            # Stage 2: SLAT
            slat = self.sample_slat(
                slat_input_dict,
                coords,
                inference_steps=stage2_inference_steps,
            )

            # Decode
            outputs = self.decode_slat(
                slat, self.decode_formats if decode_formats is None else decode_formats
            )
            outputs = self.postprocess_slat_output(
                outputs, with_mesh_postprocess, with_texture_baking, use_vertex_color
            )

            result = {
                **ss_return_dict,
                **outputs,
                "frame_idx": frame_idx,
                "intrinsics": frame_intrinsics,
            }
            results[frame_idx] = result
            processed_results[frame_idx] = result

            logger.info(f"Frame {frame_idx + 1}/{num_frames} completed")

        logger.info(f"Video inference completed: {num_frames} frames processed")
        return results

    def _prepare_frame_rgba(
        self,
        image: Union[np.ndarray, Image.Image],
        mask: Optional[Union[np.ndarray, Image.Image]],
    ) -> np.ndarray:
        """Prepare frame image with mask merged into alpha channel."""
        # Convert to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        else:
            image = np.array(image)
        
        if mask is not None:
            mask = np.array(mask)
            
            # Convert mask to uint8 [0, 255]
            if mask.dtype == bool:
                mask = mask.astype(np.uint8) * 255
            elif mask.dtype != np.uint8:
                if mask.max() <= 1.0:
                    mask = (mask * 255).astype(np.uint8)
                else:
                    mask = mask.astype(np.uint8)
            
            if mask.ndim == 2:
                mask = mask[..., None]
            
            # Merge to RGBA
            if image.shape[-1] == 3:
                rgba_image = np.concatenate([image, mask], axis=-1).astype(np.uint8)
            elif image.shape[-1] == 4:
                rgba_image = np.concatenate([image[..., :3], mask], axis=-1).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
        else:
            # Assume image is already RGBA or add full alpha
            if image.shape[-1] == 3:
                alpha = np.ones((*image.shape[:2], 1), dtype=np.uint8) * 255
                rgba_image = np.concatenate([image, alpha], axis=-1)
            else:
                rgba_image = image
        
        return rgba_image

