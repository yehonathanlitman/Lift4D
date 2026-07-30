import os
import torch
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr as psnr_func
from utils.image_utils import ssim as ssim_metric
from utils.image_utils import lpips as lpips_metric
import sys
from scene import GaussianModel, DeformModel
from utils.general_utils import safe_state, get_expon_lr_func
import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, OptimizationParams
import math
import numpy as np
import imageio
from PIL import Image
import glob
import json
import torch.nn as nn
import torch.nn.functional as F

# Make the sibling stage-1 package (sam3d/) and the shared dataset registry
# importable, regardless of where the repo lives. Override with $SAM3D_PATH.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAM3D_PATH = os.environ.get("SAM3D_PATH", os.path.join(_REPO_ROOT, "sam3d"))
if SAM3D_PATH not in sys.path:
    sys.path.insert(0, SAM3D_PATH)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.environ['LIDRA_SKIP_INIT'] = '1'
import lift4d_datasets as ds_registry

from pytorch3d.transforms import quaternion_multiply, quaternion_invert, matrix_to_quaternion
from pytorch3d.ops import knn_points
from gsplat import rasterization



# =============================================================================
# SAM3D Data Loading
# =============================================================================

SH_C0 = 0.28209479177387814
R_PYTORCH3D_TO_CAM = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)

def get_dataset_config(dataset: str, object_name: str) -> dict:
    """Build the loader config for a consistent4d or davis object (see lift4d_datasets)."""
    if dataset is None or object_name is None:
        raise ValueError("get_dataset_config requires dataset and object_name")

    spec = ds_registry.resolve(dataset, object_name)

    if dataset == "davis":
        num_frames = len(glob.glob(os.path.join(spec.frames_dir, "*.jpg")))
        return {
            "is_c4d": True,
            "is_davis": True,
            "image_dir_abs": spec.frames_dir,
            "mask_dir_abs": spec.masks_dir,
            "num_infer_frames": num_frames,
        }

    if dataset == "consistent4d":
        num_frames = len(glob.glob(os.path.join(spec.frames_dir, "*.png")))
        return {
            "is_c4d": True,
            "is_davis": False,
            "image_dir_abs": spec.frames_dir,
            "num_infer_frames": num_frames,
        }

    raise ValueError(
        f"get_dataset_config supports 'consistent4d' and 'davis'; got '{dataset}'. "
        "Custom videos use --dataset custom (routed through generic mode)."
    )


class SAM3DDataLoader:
    """
    Data loader for SAM3D Gaussian Splat outputs.

    Loads per-frame Gaussian PLY files, SE(3) transforms, and GT images.
    """

    def __init__(
        self,
        selector: str = None,
        canonical_frame_idx: int = 0,
        device: str = "cuda",
        dataset: str = None,
        object_name: str = None,
        input_dir: str = None,
        max_frames: int = 0,
        # Generic path mode
        frames_dir: str = None,
        masks_dir: str = None,
        mask_name: str = None,
        da3_npz: str = None,
    ):
        """
        Args:
            selector: Label for logging (object/sequence name)
            canonical_frame_idx: Which frame to use as canonical
            device: Device to load tensors on
            dataset: Consistent4D / DAVIS dataset name
            object_name: Object name within the C4D dataset
            frames_dir: Generic mode — directory with video frames
            masks_dir: Generic mode — directory with SAM3 masks
            mask_name: Generic mode — mask suffix (e.g., "box_1")
            da3_npz: Generic mode — path to DA3 da3_output.npz
        """
        self.canonical_frame_idx = canonical_frame_idx
        self.device = device

        # Generic path mode: frames_dir + masks_dir + mask_name
        self._generic_mode = frames_dir is not None
        self._mask_name = mask_name
        self._generic_masks_dir = masks_dir
        self._generic_da3_npz = da3_npz

        if self._generic_mode:
            # Generic mode — bypass dataset config entirely
            self.config = {"is_c4d": False, "num_infer_frames": 9999}
            self.is_c4d = False
            self.vis_dir = input_dir  # SAM3D output dir (frame_0000/result.ply)
            self.image_dir = frames_dir
            print(f"SAM3DDataLoader: Generic mode")
            print(f"  Frames: {frames_dir}")
            print(f"  Masks: {masks_dir}")
            print(f"  SAM3D: {input_dir}")
            print(f"  Mask name: {mask_name}")
        else:
            # Get config
            self.config = get_dataset_config(dataset, object_name)
            self.is_c4d = self.config.get("is_c4d", False)

            # Setup paths
            if input_dir is None:
                raise ValueError("input_dir is required (stage-1 SAM3D output dir with frame_*/result.ply)")
            self.vis_dir = input_dir
            self.image_dir = self.config["image_dir_abs"]

        # Discover frames. Sort NUMERICALLY by the frame index in the dir name
        # (e.g. "0","1",...,"10") — a lexical sort would order 0,1,10,11,...,2,20
        # and scramble the temporal frame order (esp. for unpadded consistent4d dirs).
        def _dir_num_key(d):
            digits = ''.join(c for c in os.path.basename(d.rstrip('/')) if c.isdigit())
            return (0, int(digits)) if digits else (1, os.path.basename(d))

        frame_dirs = sorted(glob.glob(os.path.join(self.vis_dir, "frame_*")), key=_dir_num_key)
        if not frame_dirs:
            frame_dirs = sorted(
                (d for d in glob.glob(os.path.join(self.vis_dir, "[0-9]*"))
                 if os.path.isdir(d) and os.path.basename(d).isdigit()),
                key=_dir_num_key,
            )

        # Filter to valid frames (have both PLY and transform)
        self.frame_data = []
        for frame_dir in frame_dirs:
            ply_path = os.path.join(frame_dir, "result.ply")
            transform_path = os.path.join(frame_dir, "obj_transform.json")

            if not os.path.exists(ply_path) or not os.path.exists(transform_path):
                continue

            frame_idx = int(os.path.basename(frame_dir).split("_")[-1])

            self.frame_data.append({
                "frame_idx": frame_idx,
                "frame_dir": frame_dir,
                "ply_path": ply_path,
                "transform_path": transform_path,
            })

        # Limit frames if max_frames > 0: include canonical + frames after it,
        # backfill from before canonical if not enough frames after
        if max_frames > 0 and len(self.frame_data) > max_frames:
            all_indices = [fd["frame_idx"] for fd in self.frame_data]
            if canonical_frame_idx in all_indices:
                canon_pos = all_indices.index(canonical_frame_idx)
            else:
                canon_pos = 0
            # Take canonical + frames after it
            end = min(canon_pos + max_frames, len(self.frame_data))
            start = end - max_frames
            start = max(start, 0)
            end = start + max_frames
            self.frame_data = self.frame_data[start:end]
            kept = [fd["frame_idx"] for fd in self.frame_data]
            print(f"SAM3DDataLoader: Limited to {max_frames} frames: {kept[0]}..{kept[-1]} (canonical={canonical_frame_idx})")

        self.num_frames = len(self.frame_data)
        self._valid_frame_indices = set(fd["frame_idx"] for fd in self.frame_data)
        self._frame_data_by_idx = {fd["frame_idx"]: fd for fd in self.frame_data}
        print(f"SAM3DDataLoader: Found {self.num_frames} valid frames for selector '{selector}'")

        # --- Camera focal length (pixels, for the max(H,W) square render) ---
        # Priority: (1) SAM3D's own per-frame focal estimate saved in
        # obj_transform.json ("focal"); (2) an explicit DA3 npz; (3) a
        # resolution-based fallback. SAM3D's estimate is the pixel-per-angle
        # scale its pose was optimized against, so it is the correct default.
        self.da3_depths = {}
        self._da3_process_res = None
        self.focal_length = None

        # (1) SAM3D per-frame focal — use the median across frames (one camera).
        sam3d_focals = []
        for fd in self.frame_data:
            try:
                with open(fd["transform_path"]) as f:
                    tf = json.load(f)
                if tf.get("focal"):
                    sam3d_focals.append(float(tf["focal"]))
            except Exception:
                pass
        if sam3d_focals:
            self.focal_length = float(np.median(sam3d_focals))
            print(f"  Focal length (SAM3D estimate, median of {len(sam3d_focals)}): {self.focal_length:.2f}")

        # (2) Explicit DA3 npz overrides (pixel intrinsics at DA3 resolution).
        da3_npz_path = self._generic_da3_npz if self._generic_da3_npz is not None else \
            os.path.join(self.vis_dir.rstrip('/').removesuffix('_video_streaming'), "da3", "da3_output.npz")
        if os.path.exists(da3_npz_path):
            camera_data = np.load(da3_npz_path, allow_pickle=True)
            self.focal_length = float(camera_data['intrinsics'][0, 1, 1])
            if 'depth' in camera_data:
                depth_data = camera_data['depth']  # (N_frames, H, W)
                self._da3_process_res = max(depth_data.shape[1], depth_data.shape[2])
                for i in range(min(depth_data.shape[0], len(self.frame_data))):
                    self.da3_depths[i] = torch.tensor(depth_data[i], dtype=torch.float32, device=self.device)
                print(f"  Loaded DA3 depth maps: {len(self.da3_depths)} frames, shape={depth_data.shape[1:]}")
            print(f"  Focal length (DA3 raw): {self.focal_length:.2f}")

        # Load GT images
        self._load_gt_images()

        # (3) Fallback: resolution-relative focal if nothing else provided one.
        if self.focal_length is None and len(self.gt_images) > 0:
            first_gt = next(iter(self.gt_images.values()))
            self.focal_length = max(first_gt.shape[0], first_gt.shape[1]) * 1.2
            print(f"  Focal length (fallback max_dim*1.2): {self.focal_length:.2f}")

        # Rescale ONLY the DA3-pixel focal from its processing resolution to GT.
        if self._da3_process_res is not None and len(self.gt_images) > 0:
            first_gt = next(iter(self.gt_images.values()))
            gt_max_dim = max(first_gt.shape[0], first_gt.shape[1])
            scale_factor = gt_max_dim / self._da3_process_res
            self.focal_length *= scale_factor
            print(f"  Focal length (rescaled to {gt_max_dim}px): {self.focal_length:.2f}")

        # Load masks
        self._load_masks()

        # Cache for loaded Gaussian data
        self._gaussian_cache = {}

        # Validate canonical frame
        frame_indices = [fd["frame_idx"] for fd in self.frame_data]
        if canonical_frame_idx not in frame_indices:
            print(f"Warning: canonical_frame_idx {canonical_frame_idx} not in available frames {frame_indices}")
            self.canonical_frame_idx = frame_indices[0]
            print(f"  Using frame {self.canonical_frame_idx} as canonical instead")

    def _load_gt_images(self):
        """Load ground truth video frames."""
        if self.is_c4d:
            # C4D/DAVIS: numeric filenames (0.png, 1.png, ...) or sequential (00000.jpg, ...)
            all_imgs = glob.glob(os.path.join(self.image_dir, "*.png")) + \
                       glob.glob(os.path.join(self.image_dir, "*.jpg"))

            def _num_key(p):
                try:
                    return (0, int(os.path.splitext(os.path.basename(p))[0]))
                except ValueError:
                    return (1, os.path.basename(p))

            video_frame_paths = sorted(all_imgs, key=_num_key)
        else:
            frame_paths_jpg = sorted(glob.glob(os.path.join(self.image_dir, "frame_*.jpg")))
            frame_paths_png = sorted(glob.glob(os.path.join(self.image_dir, "frame_*.png")))
            video_frame_paths = frame_paths_jpg if len(frame_paths_jpg) > 0 else frame_paths_png

        video_frame_paths = video_frame_paths[:self.config["num_infer_frames"]]

        self.gt_images = {}
        self.gt_image_sizes = {}
        for i, p in enumerate(video_frame_paths):
            if i not in self._valid_frame_indices:
                continue
            img = imageio.imread(p)
            if img.ndim == 3 and img.shape[2] == 4:
                # RGBA: composite on white background
                alpha = img[:, :, 3:4].astype(np.float32) / 255.0
                rgb = img[:, :, :3].astype(np.float32)
                img = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
            self.gt_images[i] = torch.tensor(img, dtype=torch.float32, device=self.device)
            self.gt_image_sizes[i] = (img.shape[0], img.shape[1])  # (H, W)

        print(f"  Loaded {len(self.gt_images)} GT images (of {len(video_frame_paths)} total)")

    def _load_masks(self):
        """Load object masks for each frame."""
        self.masks = {}

        if self.is_c4d:
            is_davis = self.config.get("is_davis", False)

            if is_davis:
                # DAVIS: separate mask PNGs in mask_dir (0 = background, >0 = foreground)
                mask_dir = self.config["mask_dir_abs"]
                mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
                mask_paths = mask_paths[:self.config["num_infer_frames"]]
                for i, p in enumerate(mask_paths):
                    if i not in self._valid_frame_indices:
                        continue
                    mask_img = imageio.imread(p)
                    if mask_img.ndim == 3:
                        mask_img = mask_img[:, :, 0]
                    mask = (mask_img > 0).astype(np.float32)
                    self.masks[i] = torch.tensor(mask, dtype=torch.float32, device=self.device)
            else:
                # C4D: extract mask from alpha channel of RGBA images
                all_pngs = glob.glob(os.path.join(self.image_dir, "*.png"))

                def _num_key(p):
                    try:
                        return (0, int(os.path.splitext(os.path.basename(p))[0]))
                    except ValueError:
                        return (1, os.path.basename(p))

                frame_paths = sorted(all_pngs, key=_num_key)
                frame_paths = frame_paths[:self.config["num_infer_frames"]]

                for i, p in enumerate(frame_paths):
                    if i not in self._valid_frame_indices:
                        continue
                    img = imageio.imread(p)
                    if img.ndim == 3 and img.shape[2] >= 4:
                        mask = (img[:, :, 3] > 0).astype(np.float32)
                    else:
                        mask = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
                    self.masks[i] = torch.tensor(mask, dtype=torch.float32, device=self.device)
        elif self._generic_mode:
            # Generic mode: masks_dir contains <frame_stem>_<mask_name>.png
            mask_dir = self._generic_masks_dir
            mask_name = self._mask_name

            frame_paths_jpg = sorted(glob.glob(os.path.join(self.image_dir, "frame_*.jpg")))
            frame_paths_png = sorted(glob.glob(os.path.join(self.image_dir, "frame_*.png")))
            video_frame_paths = frame_paths_jpg if len(frame_paths_jpg) > 0 else frame_paths_png
            video_frame_paths = video_frame_paths[:self.config["num_infer_frames"]]

            for i, p in enumerate(video_frame_paths):
                if i not in self._valid_frame_indices:
                    continue
                base = os.path.splitext(os.path.basename(p))[0]
                mask_path = os.path.join(mask_dir, f"{base}_{mask_name}.png")
                if not os.path.exists(mask_path):
                    # Try without mask_name suffix (just frame stem)
                    alt_masks = sorted(glob.glob(os.path.join(mask_dir, f"{base}_*.png")))
                    if alt_masks:
                        mask_path = alt_masks[0]
                    else:
                        continue
                mask = imageio.imread(mask_path)
                if len(mask.shape) == 3:
                    mask = mask[:, :, 0]
                mask = (mask > 127).astype(np.float32)
                self.masks[i] = torch.tensor(mask, dtype=torch.float32, device=self.device)
        print(f"  Loaded {len(self.masks)} masks (of {self.num_frames} SAM3D frames)")

    def get_mask(self, frame_idx: int) -> torch.Tensor:
        """Get object mask for a frame. Returns (H, W) binary tensor."""
        if frame_idx in self.masks:
            return self.masks[frame_idx]
        return None

    def get_da3_depth(self, frame_idx: int) -> torch.Tensor:
        """Get DA3 depth map for a frame. Returns (H, W) float tensor."""
        if frame_idx in self.da3_depths:
            return self.da3_depths[frame_idx]
        return None

    def load_frame(self, frame_idx: int, use_cache: bool = True) -> dict:
        """
        Load Gaussian parameters from a frame PLY file.

        Args:
            frame_idx: Frame index
            use_cache: Whether to use cached data

        Returns:
            dict with keys: xyz, rotation, scaling, opacity, features_dc
        """
        if use_cache and frame_idx in self._gaussian_cache:
            return self._gaussian_cache[frame_idx]

        # Find frame data
        fd = self._frame_data_by_idx.get(frame_idx)
        if fd is None:
            raise ValueError(f"Frame {frame_idx} not found")

        # Load using SAM3D's Gaussian class
        from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian

        gaussian = Gaussian(
            aabb=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            sh_degree=0,
            mininum_kernel_size=0.0,
            scaling_bias=0.01,
            opacity_bias=0.1,
            scaling_activation="exp",
            device=self.device,
        )
        gaussian.load_ply(fd["ply_path"])

        data = {
            'xyz': gaussian.get_xyz.detach().clone(),
            'rotation': gaussian.get_rotation.detach().clone(),
            'scaling': gaussian.get_scaling.detach().clone(),
            'opacity': gaussian.get_opacity.detach().clone(),
            'features_dc': gaussian._features_dc.detach().clone(),
        }

        if use_cache:
            self._gaussian_cache[frame_idx] = data

        return data

    def get_transform(self, frame_idx: int) -> dict:
        """Load SE(3) transform from obj_transform.json (cached after first load)."""
        if not hasattr(self, '_transform_cache'):
            self._transform_cache = {}
        if frame_idx in self._transform_cache:
            return self._transform_cache[frame_idx]

        fd = self._frame_data_by_idx.get(frame_idx)
        if fd is None:
            raise ValueError(f"Frame {frame_idx} not found")

        with open(fd["transform_path"], 'r') as f:
            data = json.load(f)

        # Keep original shape (don't squeeze) for compatibility with SceneVisualizer
        transform = {
            "scale": torch.tensor(data["scale"], dtype=torch.float32, device=self.device),
            "translation": torch.tensor(data["translation"], dtype=torch.float32, device=self.device),
            "rotation": torch.tensor(data["rotation"], dtype=torch.float32, device=self.device),
        }
        self._transform_cache[frame_idx] = transform
        return transform

    def get_gt_image(self, frame_idx: int) -> torch.Tensor:
        """Get ground truth image for a frame."""
        if frame_idx in self.gt_images:
            return self.gt_images[frame_idx]
        return None

    def get_frame_indices(self) -> list:
        """Get list of all available frame indices."""
        return [fd["frame_idx"] for fd in self.frame_data]

    def get_non_canonical_indices(self) -> list:
        """Get list of frame indices excluding canonical."""
        return [idx for idx in self.get_frame_indices() if idx != self.canonical_frame_idx]

    def apply_transform_to_gaussian(
        self,
        xyz: torch.Tensor,
        rotation: torch.Tensor,
        scaling: torch.Tensor,
        transform: dict,
    ) -> tuple:
        """
        Apply SE(3) transform to Gaussian attributes (object -> camera space).

        Args:
            xyz: (N, 3) positions
            rotation: (N, 4) quaternions
            scaling: (N, 3) scales
            transform: dict with scale, rotation, translation

        Returns:
            Transformed (xyz, rotation, scaling)
        """
        # Import SceneVisualizer for proper transform application
        from sam3d_objects.utils.visualization import SceneVisualizer

        # C4D objects: clamp scale >= 1.0 to unit scale (artifact workaround).
        # DAVIS/other datasets: use the true SAM3D scale.
        raw_scale = transform["scale"]
        scale_val = raw_scale.flatten()[0].item()
        clamp_scale = not self.config.get("is_davis", False)
        if clamp_scale and scale_val >= 1.0:
            scale_t = torch.ones_like(raw_scale)
        else:
            scale_t = raw_scale

        PC = SceneVisualizer.object_pointcloud(
            points_local=xyz.unsqueeze(0),
            quat_l2c=transform["rotation"],
            trans_l2c=transform["translation"],
            scale_l2c=scale_t,
        )
        transformed_xyz = PC.points_list()[0]

        # Apply PyTorch3D to camera space conversion
        R_p3d_to_cam = R_PYTORCH3D_TO_CAM.to(transformed_xyz.device, transformed_xyz.dtype)
        transformed_xyz = transformed_xyz @ R_p3d_to_cam.T

        # Transform rotations (squeeze for quaternion ops)
        rot_quat = transform["rotation"].squeeze()
        transformed_rotation = quaternion_multiply(
            quaternion_invert(rot_quat.unsqueeze(0).expand(rotation.shape[0], -1)),
            rotation,
        )

        # Transform scales
        scale_vec = scale_t.squeeze()
        transformed_scaling = scaling * scale_vec

        return transformed_xyz, transformed_rotation, transformed_scaling

    def render_gaussian(
        self,
        xyz: torch.Tensor,
        rotation: torch.Tensor,
        scaling: torch.Tensor,
        opacity: torch.Tensor,
        features_dc: torch.Tensor,
        width: int,
        height: int,
        bg_color: tuple = (0.0, 0.0, 0.0),
        return_depth: bool = False,
        focal_length: float = None,
        return_meta: bool = False,
    ) -> torch.Tensor:
        """
        Render Gaussian attributes using gsplat.

        Args:
            xyz: (N, 3) positions in camera space
            rotation: (N, 4) quaternions
            scaling: (N, 3) scales
            opacity: (N, 1) opacities
            features_dc: (N, 1, 3) or (N, 3) SH coefficients
            width, height: Image dimensions
            bg_color: Background color
            return_depth: If True, also render depth (z-values) as 4th channel
            focal_length: Override focal length (default: self.focal_length)

        Returns:
            rendered: (H, W, 3) in [0, 255]
            alpha: (H, W, 1)
            depth (optional): (H, W, 1) z-values (only if return_depth=True)
        """
        max_dim = max(width, height)

        # Camera intrinsics
        fx = focal_length if focal_length is not None else self.focal_length
        fy = fx
        cx = max_dim / 2.0
        cy = max_dim / 2.0

        Ks = torch.tensor([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=torch.float32, device=xyz.device).unsqueeze(0)

        viewmat = torch.eye(4, dtype=torch.float32, device=xyz.device).unsqueeze(0)

        # Convert SH to RGB — handle (N, 1, 3), (N, 3, 1), or (N, 3)
        dc = features_dc.reshape(-1, 3) if features_dc.dim() == 3 else features_dc
        rgb_colors = torch.clamp(dc * SH_C0 + 0.5, 0.0, 1.0)

        if return_depth:
            z_vals = xyz[:, 2:3]  # (N, 1)
            colors = torch.cat([rgb_colors, z_vals], dim=-1)  # (N, 4)
            bg = torch.zeros(4, dtype=torch.float32, device=xyz.device)
        else:
            colors = rgb_colors
            bg = torch.tensor(bg_color, dtype=torch.float32, device=xyz.device)

        render_colors, render_alphas, meta = rasterization(
            means=xyz,
            quats=rotation,
            scales=scaling,
            opacities=opacity.squeeze(-1),
            colors=colors,
            sh_degree=None,
            viewmats=viewmat,
            Ks=Ks,
            width=max_dim,
            height=max_dim,
            backgrounds=bg,
        )

        output = render_colors.squeeze(0)
        alpha = render_alphas.squeeze(0)  # (H, W, 1)

        if return_depth:
            rendered = output[:, :, :3] * 255.0
            depth = output[:, :, 3:4]
        else:
            rendered = output * 255.0
            depth = None

        # Crop to target resolution
        if width > height:
            crop_start = (max_dim - height) // 2
            rendered = rendered[crop_start:crop_start + height, :width]
            alpha = alpha[crop_start:crop_start + height, :width]
            if depth is not None:
                depth = depth[crop_start:crop_start + height, :width]
        else:
            crop_start = (max_dim - width) // 2
            rendered = rendered[:height, crop_start:crop_start + width]
            alpha = alpha[:height, crop_start:crop_start + width]
            if depth is not None:
                depth = depth[:height, crop_start:crop_start + width]

        if return_meta:
            if return_depth:
                return rendered, alpha, depth, meta
            return rendered, alpha, meta
        if return_depth:
            return rendered, alpha, depth
        return rendered, alpha

class ChamferLoss(nn.Module):
    """
    Bidirectional Chamfer distance loss between point clouds.
    """

    def __init__(self, subsample: int = 4096, use_sqrt: bool = True):
        """
        Args:
            subsample: Max points to use (for memory efficiency)
            use_sqrt: If True, use sqrt of squared distances (L2)
        """
        super().__init__()
        self.subsample = subsample
        self.use_sqrt = use_sqrt

    def forward(self, xyz_src: torch.Tensor, xyz_tgt: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional Chamfer distance.

        Args:
            xyz_src: Source point cloud (N, 3)
            xyz_tgt: Target point cloud (M, 3)

        Returns:
            Chamfer loss (scalar)
        """
        # Subsample if needed
        if self.subsample > 0:
            if xyz_src.shape[0] > self.subsample:
                idx = torch.randperm(xyz_src.shape[0], device=xyz_src.device)[:self.subsample]
                xyz_src = xyz_src[idx]
            if xyz_tgt.shape[0] > self.subsample:
                idx = torch.randperm(xyz_tgt.shape[0], device=xyz_tgt.device)[:self.subsample]
                xyz_tgt = xyz_tgt[idx]

        # Forward: src -> tgt
        dist_src_to_tgt, _, _ = knn_points(
            xyz_src.unsqueeze(0), xyz_tgt.unsqueeze(0), K=1
        )
        dist_src_to_tgt = dist_src_to_tgt.squeeze()

        # Backward: tgt -> src
        dist_tgt_to_src, _, _ = knn_points(
            xyz_tgt.unsqueeze(0), xyz_src.unsqueeze(0), K=1
        )
        dist_tgt_to_src = dist_tgt_to_src.squeeze()

        if self.use_sqrt:
            dist_src_to_tgt = dist_src_to_tgt.clamp(min=1e-8).sqrt()
            dist_tgt_to_src = dist_tgt_to_src.clamp(min=1e-8).sqrt()

        chamfer = dist_src_to_tgt.mean() + dist_tgt_to_src.mean()

        return chamfer


class GUI:
    def __init__(self, args, dataset, opt) -> None:
        self.args = args
        self.opt = opt

        # Lift4D settings (sam3d_data holds the per-frame SAM3D reconstructions)
        self.sam3d_data = getattr(args, 'sam3d_data', None)
        self.disable_rendering_loss_iter = getattr(args, 'disable_rendering_loss_iter', 0)
        self.lambda_render = getattr(args, 'lambda_render', 1.0)
        self.lambda_chamfer = getattr(args, 'lambda_chamfer', 0)
        self.lambda_temporal_tv = getattr(args, 'lambda_temporal_tv', 1.0)
        self.lambda_rigid = getattr(args, 'lambda_rigid', 1.0)
        self.knn_batch_size = getattr(args, 'knn_batch_size', 4096)
        self.vis_interval = getattr(args, 'vis_interval', 2000)
        self.optimize_gs_xyz = getattr(args, 'optimize_gs_xyz', False)
        self.optimize_canonical_color = getattr(args, 'optimize_canonical_color', False)
        self.enable_densify_and_prune = getattr(args, 'densify_and_prune', False)
        self.enable_densify_and_prune_rc = getattr(args, 'densify_and_prune_rc', False)
        self._position_lr_iter_offset = 0
        self._compose_lr_iter_offset = 0
        self.enable_reset_opacity_rc = getattr(args, 'reset_opacity_rc', False)
        self.optimize_per_frame_compose_transforms_app = getattr(args, 'optimize_per_frame_compose_transforms_app', False)

        self.is_node_delta = False
        self.save_checkpoints = getattr(args, 'save_checkpoints', False)
        self.occlusion_compositing = getattr(args, 'occlusion_compositing', False)
        self.lambda_rc = getattr(args, 'lambda_rc', 0.0)
        self.lambda_sds_rgb = getattr(args, 'lambda_sds_rgb', 0.0)
        self.enable_lpips = getattr(args, 'enable_lpips', False)

        if self.sam3d_data is None:
            raise ValueError("Lift4D requires a data mode: pass --dataset or --frames_dir")
        self._init_lift4d(args, dataset, opt)

    def _recompute_sds_focal_length(self):
        import math
        fov_deg = 20.0
        self.sds_focal_length = 256.0 / (2.0 * math.tan(math.radians(fov_deg / 2.0)))

        # Compute normalization scale from canonical Gaussians
        with torch.no_grad():
            canonical_idx = self.sam3d_data.canonical_frame_idx
            canonical_transform = self.sam3d_data.get_transform(canonical_idx)
            xyz_cam, _, _ = self.sam3d_data.apply_transform_to_gaussian(
                self.gaussians.get_xyz, self.gaussians.get_rotation, self.gaussians.get_scaling,
                canonical_transform,
            )
            centroid = xyz_cam.mean(dim=0)
            xyz_centered = xyz_cam - centroid
            half_extent = xyz_centered.abs().max().item()  # max abs across all xyz
            self.sds_norm_scale = 0.5 / max(half_extent, 1e-6)  # maps to [-0.5, 0.5]
        print(f"[SDS] SDS focal length: {self.sds_focal_length:.2f} (FOV={fov_deg}°), "
              f"norm scale: {self.sds_norm_scale:.4f} (object half-extent: {half_extent:.4f})")

    def _init_lift4d(self, args, dataset, opt):
        print("\n=== Initializing Lift4D ===")

        # Record the run configuration next to the checkpoints
        os.makedirs(self.args.model_path, exist_ok=True)
        with open(os.path.join(self.args.model_path, "cfg_args"), 'w') as cfg_f:
            cfg_f.write(str(Namespace(**vars(dataset))))

        # Load canonical frame data
        canonical_idx = self.sam3d_data.canonical_frame_idx
        canonical_data = self.sam3d_data.load_frame(canonical_idx)
        print(f"Loaded canonical frame {canonical_idx} with {canonical_data['xyz'].shape[0]} Gaussians")

        # Initialize GaussianModel from canonical frame
        gs_fea_dim = dataset.hyper_dim if hasattr(dataset, 'hyper_dim') else 4
        self.gaussians = GaussianModel(sh_degree=0, fea_dim=gs_fea_dim, with_motion_mask=False)

        # Create from SAM3D data
        from utils.graphics_utils import BasicPointCloud
        pcd = BasicPointCloud(
            points=canonical_data['xyz'].cpu().numpy(),
            colors=np.zeros_like(canonical_data['xyz'].cpu().numpy()),
            normals=np.zeros_like(canonical_data['xyz'].cpu().numpy())
        )
        self.gaussians.create_from_pcd(pcd, spatial_lr_scale=5.0)

        # Override with SAM3D values (clone to avoid sharing storage with cache)
        from utils.general_utils import inverse_sigmoid
        self.gaussians._rotation.data = canonical_data['rotation'].clone()
        self.gaussians._scaling.data = torch.log(canonical_data['scaling'].clone() + 1e-8)
        self.gaussians._opacity.data = inverse_sigmoid(torch.clamp(canonical_data['opacity'].clone(), 1e-4, 1-1e-4))
        # features_dc shape: (N, 1, 3) -> need to reshape for GaussianModel format (N, 3, 1)
        if canonical_data['features_dc'].dim() == 3 and canonical_data['features_dc'].shape[1] == 1:
            self.gaussians._features_dc.data = canonical_data['features_dc'].clone().permute(0, 2, 1)
        else:
            self.gaussians._features_dc.data = canonical_data['features_dc'].clone().unsqueeze(-1)

        # Subsample Gaussians at init to speed up early training
        subsample_ratio = getattr(args, 'init_subsample_ratio', 1.0)
        if subsample_ratio < 1.0:
            N = self.gaussians._xyz.shape[0]
            keep = max(1, int(N * subsample_ratio))
            step = N / keep
            idx = torch.arange(keep, device='cuda').float().mul_(step).long()
            self.gaussians._xyz = torch.nn.Parameter(self.gaussians._xyz.data[idx])
            self.gaussians._features_dc = torch.nn.Parameter(self.gaussians._features_dc.data[idx])
            self.gaussians._features_rest = torch.nn.Parameter(self.gaussians._features_rest.data[idx])
            self.gaussians._scaling = torch.nn.Parameter(self.gaussians._scaling.data[idx])
            self.gaussians._rotation = torch.nn.Parameter(self.gaussians._rotation.data[idx])
            self.gaussians._opacity = torch.nn.Parameter(self.gaussians._opacity.data[idx])
            self.gaussians.max_radii2D = self.gaussians.max_radii2D[idx]
            if self.gaussians.fea_dim > 0 and self.gaussians.feature.shape[0] == N:
                self.gaussians.feature = torch.nn.Parameter(self.gaussians.feature.data[idx])
            print(f"  Subsampled Gaussians: {N} -> {keep} ({subsample_ratio:.0%})")

        self.gaussians.training_setup(opt)
        print(f"  Gaussians initialized: {self.gaussians.get_xyz.shape[0]} points")

        # Initialize DeformModel
        deform_type = getattr(args, 'deform_type', 'node')
        self.is_node_delta = (deform_type == 'node_delta')
        actual_deform_type = 'delta' if self.is_node_delta else deform_type
        node_num = getattr(dataset, 'node_num', 2048)
        hyper_dim = getattr(dataset, 'hyper_dim', 4)

        deform_kwargs = dict(
            K=getattr(dataset, 'K', 4),
            deform_type=actual_deform_type,
            is_blender=True,
            d_rot_as_res=True,
            node_num=node_num,
            with_arap_loss=True,
        )
        if actual_deform_type == 'delta':
            deform_kwargs['num_frames'] = self.sam3d_data.num_frames
            deform_kwargs['opt_deform_rot'] = getattr(args, 'opt_deform_rot', False)
        else:
            deform_kwargs.update(
                skinning=getattr(args, 'skinning', False),
                hyper_dim=hyper_dim,
                pred_opacity=False,
                pred_color=False,
                use_hash=getattr(dataset, 'use_hash', False),
                hash_time=getattr(dataset, 'hash_time', False),
                local_frame=getattr(dataset, 'local_frame', False),
                progressive_brand_time=getattr(dataset, 'progressive_brand_time', False),
                max_d_scale=-1,
                enable_densify_prune=getattr(opt, 'node_enable_densify_prune', False),
                is_scene_static=False,
            )
        self.deform = DeformModel(**deform_kwargs)

        # Initialize deformation nodes from canonical positions BEFORE creating optimizer,
        # since init() reassigns control_point_deltas to a new nn.Parameter
        if self.deform.name == 'node':
            print("  Initializing control nodes from canonical Gaussians...")
            self.deform.deform.init(
                init_pcl=self.gaussians.get_xyz.detach(),
                force_init=True,
                opt=opt,
                as_gs_force_with_motion_mask=False,
                force_gs_keep_all=True
            )

        self.deform.train_setting(opt)
        # Store node kwargs for node_delta base model creation during checkpoint loading
        if self.is_node_delta:
            self._node_base_kwargs = dict(
                K=getattr(dataset, 'K', 4),
                deform_type='node',
                is_blender=True,
                d_rot_as_res=True,
                node_num=node_num,
                with_arap_loss=True,
                skinning=getattr(args, 'skinning', False),
                hyper_dim=hyper_dim,
                pred_opacity=False,
                pred_color=False,
                use_hash=getattr(dataset, 'use_hash', False),
                hash_time=getattr(dataset, 'hash_time', False),
                local_frame=getattr(dataset, 'local_frame', False),
                progressive_brand_time=getattr(dataset, 'progressive_brand_time', False),
                max_d_scale=-1,
                enable_densify_prune=False,
                is_scene_static=False,
            )
        suffix = '(node_delta)' if self.is_node_delta else ''
        print(f"  DeformModel initialized: type={actual_deform_type}{suffix}, node_num={node_num}")

        # Initialize Chamfer loss
        self.chamfer_loss = ChamferLoss(subsample=4096)

        # Background color

        # Training state
        self.iter_start = torch.cuda.Event(enable_timing=True)
        self.iter_end = torch.cuda.Event(enable_timing=True)
        self.iteration = 1
        self.ema_loss_for_log = 0.0
        self.progress_bar = tqdm.tqdm(range(opt.iterations), desc="Lift4D Training")

        # Store frame info
        self.num_frames = self.sam3d_data.num_frames
        self.frame_indices = self.sam3d_data.get_frame_indices()
        self.non_canonical_indices = self.sam3d_data.get_non_canonical_indices()

        st_win = int(getattr(self.args, 'smooth_transforms_window', 0) or 0)
        if st_win > 1:
            frames_sorted = sorted(self.frame_indices)
            raw = [self.sam3d_data.get_transform(fi) for fi in frames_sorted]
            tf_shapes = {k: raw[0][k].shape for k in ('scale', 'translation', 'rotation')}
            tf_dev = raw[0]['scale'].device

            def _ma(arr, w):  # centered moving average, arr (F, D)
                # Antisymmetric (linear-extrapolation) padding: ext value at -k is 2*a0 - a_k.
                # Preserves linear trends at the boundaries, so an object with net motion keeps
                # its true start/end pose (plain reflect padding would bias endpoints inward by
                # ~(w//2) frames of motion). Identical to reflect for a static pose.
                pad = w // 2
                head = 2.0 * arr[0] - arr[1:pad + 1][::-1]
                tail = 2.0 * arr[-1] - arr[-pad - 1:-1][::-1]
                ext = np.concatenate([head, arr, tail], axis=0)
                ker = np.ones(w, dtype=np.float64) / w
                return np.stack([np.convolve(ext[:, d], ker, mode='valid') for d in range(arr.shape[1])], axis=1)

            S = np.stack([t['scale'].detach().cpu().numpy().reshape(-1) for t in raw]).astype(np.float64)
            T = np.stack([t['translation'].detach().cpu().numpy().reshape(-1) for t in raw]).astype(np.float64)
            Q = np.stack([t['rotation'].detach().cpu().numpy().reshape(-1) for t in raw]).astype(np.float64)
            for i in range(1, len(Q)):  # quaternion hemisphere alignment before averaging
                if (Q[i] * Q[i - 1]).sum() < 0:
                    Q[i] = -Q[i]
            S_s = np.exp(_ma(np.log(S), st_win))  # positive-safe (log-space)
            T_s = _ma(T, st_win)
            Q_s = _ma(Q, st_win)
            Q_s /= np.linalg.norm(Q_s, axis=1, keepdims=True) + 1e-12
            for i, fi in enumerate(frames_sorted):
                self.sam3d_data._transform_cache[fi] = {
                    'scale': torch.tensor(S_s[i], dtype=torch.float32, device=tf_dev).reshape(tf_shapes['scale']),
                    'translation': torch.tensor(T_s[i], dtype=torch.float32, device=tf_dev).reshape(tf_shapes['translation']),
                    'rotation': torch.tensor(Q_s[i], dtype=torch.float32, device=tf_dev).reshape(tf_shapes['rotation']),
                }
            print(f"[SMOOTH-TF] Low-passed SAM3D transform trajectory (window {st_win}) for {len(frames_sorted)} frames")

        # Precompute per-step constants to avoid repeated work in train_lift4d_step
        self._sorted_frame_indices = sorted(self.frame_indices)
        self._frame_to_sorted_pos = {fidx: i for i, fidx in enumerate(self._sorted_frame_indices)}
        self._max_frame = max(1, max(self.frame_indices))
        self._camera_center = torch.zeros(3, device='cuda')

        # Set frame mapping for delta deform (needs frame_indices)
        if actual_deform_type == 'delta':
            self.deform.deform.set_frame_mapping(self.frame_indices)
        print(f"  Training frames: {len(self.non_canonical_indices)} (excluding canonical)")

        # Precompute occlusion masks and SAM3D renders for all frames
        self.precomputed_occ = {}
        self.precomputed_occ_weight = {}  # soft float weight with distance falloff at borders
        self.precomputed_sam3d_render = {}
        if self.occlusion_compositing:
            print("  Precomputing occlusion masks and SAM3D renders...")
            with torch.no_grad():
                for frame_idx in self.frame_indices:
                    gt_image = self.sam3d_data.get_gt_image(frame_idx)
                    if gt_image is None:
                        continue
                    h, w = gt_image.shape[:2]
                    transform = self.sam3d_data.get_transform(frame_idx)
                    sam3d_frame = self.sam3d_data.load_frame(frame_idx)

                    # Transform SAM3D GS to camera space and render with depth
                    xyz_cam, rot_cam, scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                        sam3d_frame['xyz'], sam3d_frame['rotation'], sam3d_frame['scaling'], transform
                    )
                    sam3d_rendered, sam3d_alpha, sam3d_depth = self.sam3d_data.render_gaussian(
                        xyz=xyz_cam, rotation=rot_cam, scaling=scale_cam,
                        opacity=sam3d_frame['opacity'], features_dc=sam3d_frame['features_dc'],
                        width=w, height=h, return_depth=True,
                    )
                    sam3d_norm = (sam3d_rendered / 255.0).detach()

                    # Compute occlusion mask
                    da3_depth = self.sam3d_data.get_da3_depth(frame_idx)
                    sam_mask = self.sam3d_data.get_mask(frame_idx)
                    if da3_depth is not None and sam_mask is not None:
                        from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt
                        m_rend = (sam3d_alpha.squeeze(-1) > 0.5)
                        m_sam_bool = (sam_mask > 0.5)
                        sam3d_d = sam3d_depth.squeeze(-1)

                        # Depth-scale alignment: rescale DA3 scene depth so its median matches the
                        # SAM3D render depth over the visible overlap, so the "scene in front of the
                        # object" test is done on a common scale.
                        vis = m_sam_bool & m_rend & (sam3d_d > 1e-3)
                        da3_aligned = da3_depth
                        if int(vis.sum()) > 50:
                            s = torch.median(sam3d_d[vis]) / torch.median(da3_depth[vis]).clamp_min(1e-6)
                            da3_aligned = da3_depth * s

                        # Occlusion mask + refinement: scene in front of the object AND (render XOR
                        # SAM mask), restricted to the render and kept just outside the visible
                        # silhouette, then morphologically opened to drop speckle.
                        m_occ_bool = (da3_aligned < sam3d_d) & (m_rend ^ m_sam_bool) & m_rend
                        m_sam_np = m_sam_bool.cpu().numpy()
                        m_occ_np = m_occ_bool.cpu().numpy() & ~binary_dilation(m_sam_np, iterations=2)
                        m_occ_np = binary_dilation(binary_erosion(m_occ_np, iterations=1), iterations=2)
                        m_occ = torch.from_numpy(m_occ_np).to('cuda')

                        # Color transfer: per-channel histogram matching using m_sam stats, applied to entire render
                        if m_occ.any() and m_sam_bool.any():
                            gt_norm_f = gt_image / 255.0
                            # Source reference: SAM3D render in visible SAM mask
                            src_ref = sam3d_norm[m_sam_bool]  # (P_vis, 3)
                            # Destination reference: GT in visible SAM mask
                            dst_ref = gt_norm_f[m_sam_bool]  # (P_vis, 3)
                            # Apply histogram matching to all rendered pixels (non-zero alpha)
                            rendered_mask = sam3d_norm.sum(dim=-1) > 0  # (H, W)
                            transferred = sam3d_norm.clone()
                            all_pixels = transferred[rendered_mask].clone()  # (P_all, 3)
                            for c in range(3):
                                src_sorted, _ = src_ref[:, c].sort()
                                dst_sorted, _ = dst_ref[:, c].sort()
                                ranks = torch.searchsorted(src_sorted, all_pixels[:, c].clamp(src_sorted[0], src_sorted[-1]))
                                ranks = ranks.clamp(0, dst_sorted.shape[0] - 1)
                                all_pixels[:, c] = dst_sorted[ranks]
                            transferred[rendered_mask] = all_pixels.clamp(0, 1)
                            self.precomputed_sam3d_render[frame_idx] = transferred
                        else:
                            self.precomputed_sam3d_render[frame_idx] = sam3d_norm

                        # Feathering: solid weight inside the (refined) occlusion mask, with a linear
                        # distance falloff over a 20px band expanding into the render.
                        m_rend_np = m_rend.cpu().numpy()
                        expand_radius = 20  # pixels
                        dist_outside = distance_transform_edt(~m_occ_np)
                        occ_weight = np.zeros(m_occ_np.shape, dtype=np.float32)
                        occ_weight[m_occ_np] = 1.0
                        band = (dist_outside <= expand_radius) & m_rend_np & ~m_occ_np
                        occ_weight[band] = (1.0 - dist_outside[band] / expand_radius).clip(0, 1)
                        self.precomputed_occ[frame_idx] = m_occ
                        self.precomputed_occ_weight[frame_idx] = torch.from_numpy(occ_weight).to('cuda')
                    else:
                        self.precomputed_occ[frame_idx] = None
                        self.precomputed_occ_weight[frame_idx] = None
                        self.precomputed_sam3d_render[frame_idx] = sam3d_norm
            print(f"  Precomputed {sum(v is not None for v in self.precomputed_occ.values())} occlusion masks")

            # Save compact debug grids for the occlusion pipeline:
            # input | occlusion mask | occlusion weight | color-matched SAM3D render
            if getattr(self.args, 'save_occ_debug', False):
                import cv2
                debug_dir = os.path.join(self.args.model_path, "debug_occ")
                os.makedirs(debug_dir, exist_ok=True)

                def _to_u8(x, h, w, ch3=True):
                    if x is None:
                        a = np.zeros((h, w, 3), dtype=np.float32)
                    else:
                        a = np.asarray(x, dtype=np.float32)
                        if a.ndim == 2:
                            a = np.repeat(a[..., None], 3, axis=2)
                    return (a * 255).clip(0, 255).astype(np.uint8)

                for frame_idx in self.frame_indices:
                    gt_image = self.sam3d_data.get_gt_image(frame_idx)
                    if gt_image is None:
                        continue
                    gt_f = gt_image.cpu().numpy().astype(np.float32)
                    if gt_f.max() > 1.0:
                        gt_f = gt_f / 255.0
                    h, w = gt_f.shape[:2]
                    m_occ = self.precomputed_occ.get(frame_idx)
                    occ_w = self.precomputed_occ_weight.get(frame_idx)
                    sam3d_cc = self.precomputed_sam3d_render.get(frame_idx)
                    grid = np.concatenate([
                        _to_u8(gt_f, h, w),
                        _to_u8(m_occ.float().cpu().numpy() if m_occ is not None else None, h, w),
                        _to_u8(occ_w.cpu().numpy() if occ_w is not None else None, h, w),
                        _to_u8(sam3d_cc.cpu().numpy() if sam3d_cc is not None else None, h, w),
                    ], axis=1)
                    cv2.imwrite(os.path.join(debug_dir, f"occ_{frame_idx:04d}.png"),
                                cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
                print(f"  Saved occlusion debug grids to {debug_dir}")

        else:
            print("  Occlusion compositing disabled")


        # Per-frame compose transforms T_t = (scale, translation) applied to targets
        # --optimize_per_frame_compose_transforms_app: separate compose (scale, translation)
        # for the rendering/SDS/eval paths, independent of the geometry compose transforms.
        if self.optimize_per_frame_compose_transforms_app:
            num_compose = len(self.frame_indices)
            self._compose_app_idx = {idx: i for i, idx in enumerate(self.frame_indices)}
            self.compose_app_scale = torch.nn.Parameter(torch.ones(num_compose, device='cuda'))
            self.compose_app_rotation = torch.nn.Parameter(torch.zeros(num_compose, 3, device='cuda'))
            self.compose_app_translation = torch.nn.Parameter(torch.zeros(num_compose, 3, device='cuda'))
            self.compose_app_optimizer = torch.optim.Adam(
                [self.compose_app_scale, self.compose_app_rotation, self.compose_app_translation], lr=1e-3,
            )
            self.compose_app_lr_func = get_expon_lr_func(
                lr_init=1e-3, lr_final=1e-7, max_steps=self.opt.iterations,
            )
            print(f"  Per-frame compose transforms (app): {num_compose} frames")

        # Initialize SDS rendering params (needed for both SDS and sds_rgb)
        self.zero123_guidance = None
        if self.lambda_sds_rgb > 0:
            canonical_idx = self.sam3d_data.canonical_frame_idx
            self.sds_cond_distance = 3.8
            self._recompute_sds_focal_length()
            print(f"[SDS] SDS orbit/cond distance: {self.sds_cond_distance:.3f}")

        # Load Zero123 model when SDS or SDS-RGB loss is active
        if self.lambda_sds_rgb > 0:
            from utils.stable_zero123_guidance import StableZero123Guidance
            zero123_dir = getattr(args, 'zero123_dir', 'extern/ldm_zero123')
            zero123_ckpt = os.path.join(zero123_dir, 'stable_zero123.ckpt')
            zero123_config = os.path.join(zero123_dir, 'sd-objaverse-finetune-c_concat-256.yaml')
            self.zero123_guidance = StableZero123Guidance(
                device='cuda',
                ckpt_path=zero123_ckpt,
                config_path=zero123_config,
                guidance_scale=3.0,
                min_step_ratio=getattr(args, 'sds_min_step', 0.02),
                max_step_ratio=getattr(args, 'sds_max_step', 0.98),
                cond_elevation_deg=0.0,
                cond_azimuth_deg=0.0,
                cond_distance=self.sds_cond_distance,
                prediction_type='epsilon',
                save_debug=getattr(args, 'save_sds_debug', False),
                keep_decoder=(self.lambda_sds_rgb > 0),  # DDIM decode needed for sds_rgb
                ddim_steps=getattr(args, 'ddim_steps', 5),
            )
            # Precompute reference embeddings for all frames (GT on white background)
            import cv2 as _cv2
            ref_debug_dir = os.path.join(args.model_path, "sds_debug_ref")
            os.makedirs(ref_debug_dir, exist_ok=True)
            frame_images = {}
            for frame_idx in self.frame_indices:
                gt_image = self.sam3d_data.get_gt_image(frame_idx)
                if gt_image is None:
                    continue
                mask = self.sam3d_data.get_mask(frame_idx)
                ref_img = gt_image / 255.0  # [0, 1]
                if mask is not None:
                    mask_3d = mask.unsqueeze(-1)
                    ref_img = ref_img * mask_3d + 1.0 * (1.0 - mask_3d)  # white bg

                    # Blend in occluded region from color-matched SAM3D render
                    occ_w = self.precomputed_occ_weight.get(frame_idx)
                    sam3d_cc = self.precomputed_sam3d_render.get(frame_idx)
                    if occ_w is not None and sam3d_cc is not None:
                        occ_w_3d = occ_w.unsqueeze(-1)  # (H, W, 1)
                        ref_img = sam3d_cc * occ_w_3d + ref_img * (1.0 - occ_w_3d)

                    # Crop around mask bbox for non-C4D datasets to fill the frame
                    if not self.sam3d_data.is_c4d:
                        ys, xs = torch.where(mask > 0.5)
                        if ys.numel() > 0:
                            y0, y1 = ys.min().item(), ys.max().item()
                            x0, x1 = xs.min().item(), xs.max().item()
                            # Pad to square with 10% margin, minimum 256 to avoid upscaling
                            h_box, w_box = y1 - y0 + 1, x1 - x0 + 1
                            side = max(h_box, w_box)
                            margin = int(side * 0.1)
                            side = max(side + 2 * margin, 256)
                            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
                            img_h, img_w = ref_img.shape[0], ref_img.shape[1]
                            # Clamp side to fit within image
                            side = min(side, img_h, img_w)
                            crop_y0 = max(cy - side // 2, 0)
                            crop_x0 = max(cx - side // 2, 0)
                            crop_y1 = min(crop_y0 + side, img_h)
                            crop_x1 = min(crop_x0 + side, img_w)
                            # Re-center if clamped
                            crop_y0 = max(crop_y1 - side, 0)
                            crop_x0 = max(crop_x1 - side, 0)
                            ref_img = ref_img[crop_y0:crop_y1, crop_x0:crop_x1]

                ref_img = ref_img.permute(2, 0, 1).unsqueeze(0)
                ref_img = F.interpolate(ref_img, size=(256, 256), mode='bilinear', align_corners=False)
                frame_images[frame_idx] = ref_img
                # Save for debugging
                ref_np = (ref_img[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                _cv2.imwrite(os.path.join(ref_debug_dir, f"reference_frame_{frame_idx:04d}.png"),
                             _cv2.cvtColor(ref_np, _cv2.COLOR_RGB2BGR))
            self.zero123_guidance.precompute_reference_embeddings(frame_images)
            self._sds_reference_images = frame_images  # keep for debug visualization
            print(f"[SDS] Saved {len(frame_images)} reference images to {ref_debug_dir}/")


        # Create scene attribute for compatibility (minimal)
        self.scene = type('obj', (object,), {
            'loaded_iter': None,
            'cameras_extent': 1.0,
            'getTrainCameras': lambda: [],
        })()

        print("=== Lift4D Initialization Complete ===\n")

    def train(self, iters=5000):
        """Train the Lift4D deformation model against the per-frame SAM3D reconstructions."""
        print(f"\nStarting Lift4D training for {iters} iterations...")
        print(f"  Lambda chamfer: {self.lambda_chamfer}")
        print(f"  Lambda temporal TV: {self.lambda_temporal_tv}")
        print(f"  Optimize GS xyz: {self.optimize_gs_xyz}")
        print(f"  Optimize canonical color: {self.optimize_canonical_color}")
        print(f"  Random camera loss: lambda_rc={self.lambda_rc}")
        print(f"  Densify and prune: {self.enable_densify_and_prune}")
        print(f"  Reset opacity (RC): {self.enable_reset_opacity_rc}")
        print(f"  Save checkpoints: {self.save_checkpoints}")
        print(f"  Occlusion compositing: {self.occlusion_compositing}")
        print(f"  Optimize xyz: {self.optimize_gs_xyz}")
        print(f"  Node-delta mode: {self.is_node_delta}")
        print(f"  Visualization interval: {self.vis_interval}")

        # Load checkpoint if requested
        load_iter = getattr(self.args, 'load_checkpoint', -1)
        if self.is_node_delta and load_iter < 0:
            print("[ERROR] --deform_type node_delta requires --load_checkpoint with a node deformation checkpoint.")
            sys.exit(1)
        if load_iter >= 0:
            if not self.load_lift4d_checkpoint(iteration=load_iter):
                print(f"[ERROR] Failed to load checkpoint at iteration {load_iter}")
                sys.exit(1)
            # Recompute SDS focal length with checkpoint Gaussians
            if self.lambda_sds_rgb > 0:
                self._recompute_sds_focal_length()
            # Rebuild position LR schedule to start fresh from checkpoint
            position_lr_init_app = getattr(self.args, 'position_lr_init_app', None)
            if position_lr_init_app is not None:
                remaining_iters = max(1, iters - self.iteration + 1)
                self._position_lr_iter_offset = self.iteration
                self.gaussians.xyz_scheduler_args = get_expon_lr_func(
                    lr_init=position_lr_init_app * self.gaussians.spatial_lr_scale,
                    lr_final=self.opt.position_lr_final * self.gaussians.spatial_lr_scale,
                    lr_delay_mult=self.opt.position_lr_delay_mult,
                    max_steps=remaining_iters,
                )
                print(f"  Position LR reset: init={position_lr_init_app}, final={self.opt.position_lr_final}, "
                      f"max_steps={remaining_iters} (offset={self._position_lr_iter_offset})")

            # Reset compose LR schedules to span remaining iters from checkpoint
            if getattr(self.args, 'compose_lr_reset', False):
                remaining_iters = max(1, iters - self.iteration + 1)
                self._compose_lr_iter_offset = self.iteration
                if self.optimize_per_frame_compose_transforms_app:
                    self.compose_app_lr_func = get_expon_lr_func(
                        lr_init=1e-3, lr_final=1e-7, max_steps=remaining_iters,
                    )
                    print(f"  Compose app LR reset: 1e-3 -> 1e-7 over {remaining_iters} steps (offset={self._compose_lr_iter_offset})")


        remaining = max(0, iters - self.iteration + 1)
        for i in tqdm.trange(remaining, desc="Lift4D Training"):
            self.train_lift4d_step()

            # Evaluate and save comparison video at intervals
            if self.iteration % self.vis_interval == 0:
                self.eval_metrics()
                self.save_comparison_video()
                self.save_orbit_video()
                if self.save_checkpoints:
                    self.save_lift4d_checkpoint()

        # Final evaluation and visualization
        self.eval_metrics()
        self.save_comparison_video()
        self.save_orbit_video()
        if self.save_checkpoints:
            self.save_lift4d_checkpoint()

        print(f"\nLift4D training complete. Final iteration: {self.iteration}")

    def load_lift4d_checkpoint(self, iteration=-1):
        """Load checkpoint from deform_gs/ directory.

        Args:
            iteration: Specific iteration to load. 0 = latest, -1 = none.
        """
        # node_delta: prefer own checkpoint dir (has saved base weights),
        # fall back to the stage-1 (_node) model path for first-time init.
        if self.is_node_delta:
            own_gs_dir = os.path.join(self.args.model_path, "deform_gs")
            base_model_path = self.args.model_path.replace('_node_delta', '_node')
            base_gs_dir = os.path.join(base_model_path, "deform_gs")
            if iteration > 0 and os.path.exists(os.path.join(own_gs_dir, f"iteration_{iteration}")):
                deform_gs_dir = own_gs_dir
            elif iteration == 0 and os.path.exists(own_gs_dir):
                deform_gs_dir = own_gs_dir
            else:
                deform_gs_dir = base_gs_dir
        else:
            deform_gs_dir = os.path.join(self.args.model_path, "deform_gs")
        if not os.path.exists(deform_gs_dir):
            print(f"[WARN] No checkpoints found at {deform_gs_dir}")
            return False

        # Find iteration to load
        if iteration == 0:
            # Find latest
            iter_dirs = [d for d in os.listdir(deform_gs_dir) if d.startswith("iteration_")]
            if not iter_dirs:
                print(f"[WARN] No iteration dirs found in {deform_gs_dir}")
                return False
            iteration = max(int(d.split("_")[1]) for d in iter_dirs)

        ckpt_dir = os.path.join(deform_gs_dir, f"iteration_{iteration}")
        if not os.path.exists(ckpt_dir):
            print(f"[WARN] Checkpoint dir not found: {ckpt_dir}")
            return False

        print(f"  Loading checkpoint from {ckpt_dir}")

        # Load canonical Gaussians
        ply_path = os.path.join(ckpt_dir, "gaussians.ply")
        if os.path.exists(ply_path):
            self.gaussians.load_ply(ply_path)
            # Re-setup optimizer: load_ply creates new nn.Parameter objects,
            # so the old optimizer references stale parameters.
            self.gaussians.training_setup(self.opt)
            print(f"    Loaded gaussians.ply ({self.gaussians.get_xyz.shape[0]} points)")

        # Load deform weights (skip for node_delta — handled separately below)
        deform_path = os.path.join(ckpt_dir, "deform.pth")
        if os.path.exists(deform_path):
            state_dict = torch.load(deform_path)
            if not self.is_node_delta:
                # Handle frame count mismatch (e.g. --max_frames limits loaded frames)
                cpd_key = 'control_point_deltas'
                if cpd_key in state_dict and cpd_key in dict(self.deform.deform.named_parameters()) and \
                        state_dict[cpd_key].shape[0] != self.deform.deform.control_point_deltas.shape[0]:
                    ckpt_deltas = state_dict[cpd_key]  # (F_ckpt, M, 3)
                    # Extract only the frames we're using (by original frame index)
                    frame_indices_to_load = []
                    for frame_idx in sorted(self.frame_indices):
                        if frame_idx < ckpt_deltas.shape[0]:
                            frame_indices_to_load.append(frame_idx)
                    new_deltas = torch.zeros_like(self.deform.deform.control_point_deltas)
                    for new_i, orig_idx in enumerate(frame_indices_to_load):
                        if new_i < new_deltas.shape[0]:
                            new_deltas[new_i] = ckpt_deltas[orig_idx]
                    state_dict[cpd_key] = new_deltas
                    print(f"    Remapped control_point_deltas: {ckpt_deltas.shape[0]} -> {new_deltas.shape[0]} frames")
                result = self.deform.deform.load_state_dict(state_dict, strict=False)
                if result is not None:
                    missing, unexpected = result
                    if missing:
                        print(f"    [WARN] Missing keys in deform checkpoint (initialized to default): {missing}")
                    if unexpected:
                        print(f"    [WARN] Unexpected keys in deform checkpoint (ignored): {unexpected}")
                print(f"    Loaded deform.pth")

        # node_delta: create frozen node base from checkpoint, initialize fresh delta layer
        if self.is_node_delta:
            # Create the frozen node base model
            self.deform_node_base = DeformModel(**self._node_base_kwargs)
            self.deform_node_base.deform.init(
                init_pcl=self.gaussians.get_xyz.detach(),
                force_init=True,
                opt=self.opt,
                as_gs_force_with_motion_mask=False,
                force_gs_keep_all=True,
            )
            # Load checkpoint weights into node base
            node_base_path = os.path.join(ckpt_dir, "deform_node_base.pth")
            if os.path.exists(node_base_path):
                # Loading a node_delta checkpoint — load the saved node base
                self.deform_node_base.deform.load_state_dict(torch.load(node_base_path))
                print(f"    node_delta: loaded node base from deform_node_base.pth")
            else:
                # First time: loading a node checkpoint as the base
                self.deform_node_base.deform.load_state_dict(state_dict)
                print(f"    node_delta: loaded node base from deform.pth (first init)")
            # Freeze node base
            self.deform_node_base.deform.eval()
            for p in self.deform_node_base.deform.parameters():
                p.requires_grad_(False)
            # Initialize delta model from base model's nodes (matching positions, radii, weights)
            self.deform.deform.init_from_base(
                opt=self.opt,
                base_deform=self.deform_node_base.deform,
            )
            self.deform.deform.set_frame_mapping(self.frame_indices)
            self.deform.deform.control_point_deltas.data.zero_()
            if hasattr(self.deform.deform, 'opt_deform_rot') and self.deform.deform.opt_deform_rot:
                self.deform.deform.control_point_rotations.data.zero_()
            self.deform.train_setting(self.opt)
            print(f"    node_delta: initialized delta layer ({self.deform.deform.control_point_deltas.shape})")


        # Load transform corrections

        # Load compose transforms

        ct_app_path = os.path.join(ckpt_dir, "compose_transforms_app.pt")
        if os.path.exists(ct_app_path) and self.optimize_per_frame_compose_transforms_app:
            ct = torch.load(ct_app_path)
            self.compose_app_scale.data = ct['compose_app_scale']
            if 'compose_app_rotation' in ct:
                self.compose_app_rotation.data = ct['compose_app_rotation']
            self.compose_app_translation.data = ct['compose_app_translation']
            print(f"    Loaded compose_transforms_app.pt")

        # Load per-frame weights

        self.iteration = iteration + 1
        print(f"  Resuming from iteration {self.iteration}")
        return True

    def save_lift4d_checkpoint(self):
        """Save Gaussian PLY, deform weights, and transform corrections."""
        ckpt_dir = os.path.join(self.args.model_path, "deform_gs", f"iteration_{self.iteration}")
        os.makedirs(ckpt_dir, exist_ok=True)

        # Save canonical Gaussians
        self.gaussians.save_ply(os.path.join(ckpt_dir, "gaussians.ply"))

        # Save deform weights (in same checkpoint dir)
        torch.save(self.deform.deform.state_dict(), os.path.join(ckpt_dir, "deform.pth"))

        # Save node_delta base model
        if self.is_node_delta:
            torch.save(self.deform_node_base.deform.state_dict(), os.path.join(ckpt_dir, "deform_node_base.pth"))

        # Save transform corrections if enabled

        # Save compose transforms if enabled

        if self.optimize_per_frame_compose_transforms_app:
            torch.save({
                'compose_app_scale': self.compose_app_scale.data,
                'compose_app_rotation': self.compose_app_rotation.data,
                'compose_app_translation': self.compose_app_translation.data,
                'compose_app_idx': self._compose_app_idx,
            }, os.path.join(ckpt_dir, "compose_transforms_app.pt"))


        print(f"[ITER {self.iteration}] Checkpoint saved to {ckpt_dir}")

    @staticmethod
    def _axis_angle_to_matrix(aa):
        """Convert axis-angle (3,) to rotation matrix (3, 3) via Rodrigues."""
        angle = aa.norm() + 1e-8
        axis = aa / angle
        K = torch.zeros(3, 3, device=aa.device, dtype=aa.dtype)
        K[0, 1] = -axis[2]; K[0, 2] = axis[1]
        K[1, 0] = axis[2];  K[1, 2] = -axis[0]
        K[2, 0] = -axis[1]; K[2, 1] = axis[0]
        R = torch.eye(3, device=aa.device) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        return R

    def apply_compose_transform_app(self, xyz, frame_idx, scaling=None):
        """Apply app-path compose: R_t @ (scale_t * x) + translation_t (object space)."""
        ci = self._compose_app_idx[frame_idx]
        s = self.compose_app_scale[ci]
        R = self._axis_angle_to_matrix(self.compose_app_rotation[ci])
        t = self.compose_app_translation[ci]
        xyz_out = (s * xyz) @ R.T + t
        if scaling is not None:
            return xyz_out, scaling * s
        return xyz_out

    @staticmethod
    def _quat_conjugate(q):
        """Conjugate of quaternion (w, x, y, z) -> (w, -x, -y, -z)."""
        return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

    @staticmethod
    def _quat_multiply(q1, q2):
        """Hamilton product of quaternions (w, x, y, z)."""
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)

    @staticmethod
    def _quat_rotate(q, v):
        """Rotate vector v by quaternion q. q: (..., 4), v: (..., 3)."""
        q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
        q_conj = torch.cat([q[..., :1], -q[..., 1:]], dim=-1)
        # q * q_v * q_conj
        w1, x1, y1, z1 = q.unbind(-1)
        w2, x2, y2, z2 = q_v.unbind(-1)
        t = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)
        w1, x1, y1, z1 = t.unbind(-1)
        w2, x2, y2, z2 = q_conj.unbind(-1)
        result = torch.stack([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ], dim=-1)
        return result[..., 1:]  # xyz components

    def _compute_node_delta_base(self, canonical_xyz, fid):
        """Run frozen node base model to get base deformation for node_delta mode."""
        with torch.no_grad():
            time_input = self.deform_node_base.deform.expand_time(fid)
            d_vals = self.deform_node_base.step(
                canonical_xyz.detach(), time_input,
                iteration=self.iteration,
                feature=self.gaussians.feature,
                motion_mask=torch.ones_like(canonical_xyz[:, :1]),
                camera_center=torch.zeros(3, device='cuda'),
                time_interval=1.0 / self.num_frames,
            )
        return d_vals['d_xyz'].detach()

    @torch.no_grad()
    def _deform_at_frame(self, frame_idx):
        """Evaluate the full deformation (trainable delta + frozen node_delta base) for one
        frame on the current canonical Gaussians. Shared by all no-grad render/eval paths.

        Returns:
            deformed_xyz: (N, 3) canonical + deformation
        """
        canonical_xyz = self.gaussians.get_xyz
        fid = torch.tensor(frame_idx / max(1, max(self.frame_indices))).cuda().float()
        if self.deform.name == 'node':
            time_input = self.deform.deform.expand_time(fid)
            d_values = self.deform.step(
                canonical_xyz.detach(), time_input,
                iteration=self.iteration, feature=self.gaussians.feature,
                motion_mask=torch.ones_like(canonical_xyz[:, :1]),
                camera_center=torch.zeros(3, device='cuda'),
                time_interval=1.0 / self.num_frames,
            )
        else:
            time_input = fid.unsqueeze(0).expand(canonical_xyz.shape[0], -1)
            d_values = self.deform.step(
                canonical_xyz.detach(), time_input,
                iteration=self.iteration, feature=self.gaussians.feature,
                camera_center=torch.zeros(3, device='cuda'),
            )
        d_xyz = d_values['d_xyz']
        if self.is_node_delta:
            d_xyz = d_xyz + self._compute_node_delta_base(canonical_xyz, fid)
        return canonical_xyz + d_xyz

    def _ensure_knn_precomputed(self):
        """Lazily precompute KNN indices, weights, and canonical distances for physical priors.
        Recomputes if Gaussian count changed (e.g. after densify_and_prune)."""
        N = self.gaussians.get_xyz.shape[0]
        if hasattr(self, '_knn_idx') and self._knn_idx is not None and self._knn_idx.shape[0] == N:
            return
        k = 20
        with torch.no_grad():
            xyz = self.gaussians.get_xyz.detach()  # (N, 3)
            N = xyz.shape[0]
            # Batched KNN to avoid N×N distance matrix OOM
            batch_size = min(N, self.knn_batch_size)
            all_idx = []
            all_dists = []
            for i in range(0, N, batch_size):
                end = min(i + batch_size, N)
                chunk = xyz[i:end]  # (B, 3)
                d = torch.cdist(chunk, xyz)  # (B, N)
                _, topk_idx = d.topk(k + 1, dim=1, largest=False)  # (B, k+1)
                topk_dists = d.gather(1, topk_idx)  # (B, k+1)
                all_idx.append(topk_idx[:, 1:])  # exclude self
                all_dists.append(topk_dists[:, 1:])
            self._knn_idx = torch.cat(all_idx, dim=0).contiguous()  # (N, k)
            knn_dists = torch.cat(all_dists, dim=0)  # (N, k)
            # Weights: w_{i,j} = exp(-lambda_w * ||mu_j - mu_i||^2), lambda_w=2000
            self._knn_weights = torch.exp(-2000.0 * knn_dists ** 2)  # (N, k)

    def train_lift4d_step(self):
        """Single Lift4D training step: anchoring + rendering + SDS losses."""
        self.iter_start.record()

        # Pick a random target frame (including canonical for pose correction)
        target_idx = np.random.choice(self.frame_indices)

        # Normalize time to [0, 1]
        fid = torch.tensor(target_idx / self._max_frame).cuda().float()

        # Load target frame Gaussian data (in object space)
        target_data = self.sam3d_data.load_frame(target_idx)
        target_xyz = target_data['xyz']

        # Get canonical Gaussian positions
        canonical_xyz = self.gaussians.get_xyz

        # Time interval for deformation
        time_interval = 1.0 / self.num_frames

        # Compute deformation
        # Lazily cache motion_mask (all ones); invalidate when Gaussian count changes
        N_gs = canonical_xyz.shape[0]
        if not hasattr(self, '_motion_mask') or self._motion_mask is None or self._motion_mask.shape[0] != N_gs:
            self._motion_mask = torch.ones(N_gs, 1, device='cuda')
        if self.deform.name == 'node':
            time_input = self.deform.deform.expand_time(fid)
            N = time_input.shape[0]
            d_values = self.deform.step(
                canonical_xyz.detach(),
                time_input,
                iteration=self.iteration,
                feature=self.gaussians.feature,
                motion_mask=self._motion_mask,
                camera_center=self._camera_center,
                time_interval=time_interval,
            )
        else:  # MLP
            N = canonical_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            d_values = self.deform.step(
                canonical_xyz.detach(),
                time_input,
                iteration=self.iteration,
                feature=self.gaussians.feature,
                camera_center=self._camera_center,
            )

        d_xyz = d_values['d_xyz']
        d_rotation = d_values['d_rotation']
        d_scaling = d_values['d_scaling']

        if self.is_node_delta:
            d_xyz = d_xyz + self._compute_node_delta_base(canonical_xyz, fid)

        # Apply deformation (in object space)
        gs_xyz_live = self.optimize_gs_xyz
        deformed_xyz = canonical_xyz + d_xyz if gs_xyz_live else canonical_xyz.detach() + d_xyz

        deformed_xyz_geo = deformed_xyz

        # === GEOMETRY LOSSES (compose-transformed space) ===

        # Per-frame weight (sigmoid-bounded [0,1])

        # Chamfer loss — apply per-frame compose transform to SAM3D target if enabled
        chamfer_enabled = self.lambda_chamfer > 0
        if chamfer_enabled:
            loss_chamfer = self.chamfer_loss(deformed_xyz_geo, target_xyz)
        else:
            loss_chamfer = torch.tensor(0.0, device='cuda')

        # ARAP loss (built-in from ControlNodeWarp)
        loss_arap = self.deform.reg_loss
        if isinstance(loss_arap, (int, float)):
            loss_arap = torch.tensor(loss_arap, device='cuda')

        # Temporal TV loss: penalize differences in control point deltas between consecutive frames
        loss_temporal_tv = torch.tensor(0.0, device='cuda')
        if self.lambda_temporal_tv > 0 and hasattr(self.deform, 'deform') and hasattr(self.deform.deform, 'control_point_deltas'):
            deltas = self.deform.deform.control_point_deltas  # (F, M, 3)
            temporal_diff = deltas[1:] - deltas[:-1]  # (F-1, M, 3)
            loss_temporal_tv = (temporal_diff ** 2).sum(dim=-1).mean()

        # === PHYSICALLY-BASED PRIOR LOSSES ===
        # Neighbor-frame deformation for the local-rigidity loss
        loss_rigid = torch.tensor(0.0, device='cuda')

        need_neighbors = (self.lambda_rigid > 0)
        if need_neighbors:
            sorted_frames = self._sorted_frame_indices
            pos_in_sorted = self._frame_to_sorted_pos[target_idx]
            prev_pos = max(0, pos_in_sorted - 1)
            next_pos = min(len(sorted_frames) - 1, pos_in_sorted + 1)
            prev_idx = sorted_frames[prev_pos]
            next_idx = sorted_frames[next_pos]

            canon_xyz_det = canonical_xyz.detach()

            def _deform_neighbor(fidx):
                """Deform canonical GS for a neighbor frame, returning (positions, d_rotation)."""
                fid_nb = torch.tensor(fidx / self._max_frame).cuda().float()
                if self.deform.name == 'node':
                    time_nb = self.deform.deform.expand_time(fid_nb)
                    d_nb = self.deform.step(canon_xyz_det, time_nb, iteration=self.iteration,
                        feature=self.gaussians.feature, motion_mask=self._motion_mask,
                        camera_center=self._camera_center, time_interval=time_interval)
                else:
                    time_nb = fid_nb.unsqueeze(0).expand(canon_xyz_det.shape[0], -1)
                    d_nb = self.deform.step(canon_xyz_det, time_nb, iteration=self.iteration,
                        feature=self.gaussians.feature, camera_center=self._camera_center)
                d_xyz_nb = d_nb['d_xyz']
                d_rot_nb = d_nb['d_rotation']
                if self.is_node_delta:
                    d_xyz_nb = d_xyz_nb + self._compute_node_delta_base(canon_xyz_det, fid_nb)
                return canon_xyz_det + d_xyz_nb, d_rot_nb

            # Deform for prev frame (needed by tracking, rigid, rot)
            # When target is the first frame, prev_idx == target_idx; still deform so
            # loss_rigid gets a grad-carrying zero rather than staying at tensor(0.0).
            xyz_prev_nb, d_rot_prev_nb = _deform_neighbor(prev_idx)

            # Restore current frame state (expand_time modifies _current_frame_idx)
            if self.deform.name == 'node':
                self.deform.deform.expand_time(fid)

            xyz_curr = deformed_xyz_geo.detach() if not gs_xyz_live else deformed_xyz_geo


            # --- Local rigidity loss: neighbors should follow rigid-body transform of i ---
            if self.lambda_rigid > 0 and xyz_prev_nb is not None:
                self._ensure_knn_precomputed()
                knn_idx = self._knn_idx  # (N, k)
                knn_w = self._knn_weights  # (N, k)
                canon_rot_det = self.gaussians.get_rotation.detach()  # (N, 4)

                # Per-frame rotations: canonical + delta, normalized
                rot_t = F.normalize(canon_rot_det + d_rotation, dim=-1)  # (N, 4)
                rot_prev = F.normalize(canon_rot_det + d_rot_prev_nb, dim=-1)  # (N, 4)

                # R_{i,t-1} @ R_{i,t}^{-1}: rotation from frame t coords back to frame t-1
                rot_t_inv = self._quat_conjugate(rot_t)  # (N, 4)
                rot_change = self._quat_multiply(rot_prev, rot_t_inv)  # (N, 4)

                # Relative positions at t: mu_j - mu_i for KNN pairs
                rel_t = xyz_curr[knn_idx] - xyz_curr.unsqueeze(1)  # (N, k, 3)

                # Rotate rel_t from frame t to frame t-1 using i's rotation change
                rot_change_k = rot_change.unsqueeze(1).expand(-1, knn_idx.shape[1], -1)  # (N, k, 4)
                rel_t_rotated = self._quat_rotate(rot_change_k, rel_t)  # (N, k, 3)

                # Relative positions at t-1
                rel_prev = xyz_prev_nb[knn_idx] - xyz_prev_nb.unsqueeze(1)  # (N, k, 3)

                # Weighted L2 norm per pair
                rigid_diff = (rel_prev - rel_t_rotated).norm(dim=-1)  # (N, k)
                loss_rigid = (knn_w * rigid_diff).mean()


        # Normal consistency loss is computed in the rendering section (needs depth map)

        # Total geometry loss
        loss = self.lambda_chamfer * loss_chamfer + self.lambda_temporal_tv * loss_temporal_tv + self.lambda_rigid * loss_rigid + loss_arap

        # === RENDERING LOSSES (camera space) ===
        loss_render = torch.tensor(0.0, device='cuda')
        render_meta = None
        rendering_loss_active = self.disable_rendering_loss_iter <= 0 or self.iteration < self.disable_rendering_loss_iter
        # Get SE(3) transform for projection
        transform = self.sam3d_data.get_transform(target_idx)

        # Get deformed Gaussian properties (rotation/scaling stay canonical and detached
        # in the render path; only colors/opacity/xyz receive rendering gradients via flags)
        canonical_rotation = self.gaussians.get_rotation.detach()
        canonical_scaling = self.gaussians.get_scaling.detach()
        opacity = self.gaussians.get_opacity.detach()
        features_dc = self.gaussians._features_dc if self.optimize_canonical_color else self.gaussians._features_dc.detach()

        deformed_rotation = canonical_rotation
        deformed_scaling = canonical_scaling

        # When gs_xyz_live=False, canonical_xyz is already detached in deformed_xyz,
        # so rendering gradients still flow to d_xyz (including deltas).
        xyz_for_render = deformed_xyz

        # Apply per-frame compose transforms to rendering path (only when _app flag is set)
        if self.optimize_per_frame_compose_transforms_app:
            xyz_for_render, deformed_scaling = self.apply_compose_transform_app(xyz_for_render, target_idx, deformed_scaling)

        # Transform to camera space
        xyz_cam, rot_cam, scale_cam = self.sam3d_data.apply_transform_to_gaussian(
            xyz_for_render, deformed_rotation, deformed_scaling, transform
        )

        # Apply per-frame SE(3) correction

        # Get image dimensions from GT
        gt_image = self.sam3d_data.get_gt_image(target_idx)
        if gt_image is not None:
            h, w = gt_image.shape[:2]
            mask_hw = self.sam3d_data.get_mask(target_idx)

            # Render SAM3D target depth for visibility test
            with torch.no_grad():
                sam3d_xyz_cam, sam3d_rot_cam, sam3d_scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                    target_data['xyz'], target_data['rotation'], target_data['scaling'], transform
                )
                _, _, sam3d_depth = self.sam3d_data.render_gaussian(
                    xyz=sam3d_xyz_cam, rotation=sam3d_rot_cam, scaling=sam3d_scale_cam,
                    opacity=target_data['opacity'], features_dc=target_data['features_dc'],
                    width=w, height=h, return_depth=True,
                )


            features_dc_render = features_dc.permute(0, 2, 1)  # (N, 3, 1) -> (N, 1, 3)

            # Render
            need_densify_meta = self.enable_densify_and_prune and self.iteration < self.opt.densify_until_iter
            render_result = self.sam3d_data.render_gaussian(
                xyz=xyz_cam,
                rotation=rot_cam,
                scaling=scale_cam,
                opacity=opacity,
                features_dc=features_dc_render,
                width=w,
                height=h,
                return_depth=True,
                return_meta=need_densify_meta,
            )
            if need_densify_meta:
                rendered, alpha, rendered_depth, render_meta = render_result
                render_meta['means2d'].retain_grad()
            else:
                rendered, alpha, rendered_depth = render_result
                render_meta = None

            rendered_norm = rendered / 255.0
            gt_norm = gt_image / 255.0

            sds_active = self.lambda_sds_rgb > 0
            if sds_active and mask_hw is not None:
                mask_3d = mask_hw.unsqueeze(-1).expand_as(gt_norm)  # (H, W, 3)
                masked_rendered = rendered_norm * mask_3d

                # Build target: GT inside mask, optionally use SAM3D render in occluded regions
                m_occ = self.precomputed_occ.get(target_idx)
                if self.occlusion_compositing and m_occ is not None and m_occ.any():
                    m_occ_3d = m_occ.unsqueeze(-1).expand_as(gt_norm).float()
                    sam3d_render_target = self.precomputed_sam3d_render[target_idx]
                    occ_weight = self.precomputed_occ_weight.get(target_idx)
                    if occ_weight is not None:
                        occ_weight_3d = occ_weight.unsqueeze(-1).expand_as(gt_norm)
                    else:
                        occ_weight_3d = m_occ_3d
                    # In occluded regions: use SAM3D render; elsewhere: use GT
                    target_image = sam3d_render_target * occ_weight_3d + gt_norm * (1.0 - occ_weight_3d)
                    masked_gt = target_image * mask_3d
                else:
                    masked_gt = gt_norm * mask_3d

                masked_rendered_chw = masked_rendered.permute(2, 0, 1)  # (3, H, W)
                masked_gt_chw = masked_gt.permute(2, 0, 1)
                Ll1 = l1_loss(masked_rendered_chw, masked_gt_chw)
                ssim_val = ssim(masked_rendered_chw, masked_gt_chw)
                lpips_val = lpips_metric(masked_rendered_chw.unsqueeze(0), masked_gt_chw.unsqueeze(0)).mean() if self.enable_lpips else torch.tensor(0.0, device='cuda')

                # Silhouette loss: rendered alpha should match GT mask + occluded regions
                pred_silhouette = alpha.squeeze(-1)  # (H, W)
                if self.occlusion_compositing and m_occ is not None and m_occ.any():
                    # Expand silhouette target to include occluded regions —
                    # otherwise Gaussians behind occluders get pushed to zero opacity and pruned
                    silhouette_target = torch.clamp(mask_hw + m_occ.float(), 0.0, 1.0)
                else:
                    silhouette_target = mask_hw
                loss_silhouette = F.mse_loss(pred_silhouette, silhouette_target)

                loss_render = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_val) + 0.1 * lpips_val + loss_silhouette
            else:
                # Compositing: rendered GS in SAM mask, SAM3D render in occluded region, GT elsewhere
                m_occ = self.precomputed_occ.get(target_idx)

                if mask_hw is not None:
                    mask_3d = mask_hw.unsqueeze(-1).expand_as(gt_norm)  # (H, W, 3)

                    if self.occlusion_compositing and m_occ is not None and m_occ.any():
                        m_occ_3d = m_occ.unsqueeze(-1).expand_as(gt_norm).float()  # (H, W, 3)
                        sam3d_render_target = self.precomputed_sam3d_render[target_idx]  # (H, W, 3)
                        combined_mask_3d = torch.clamp(mask_3d + m_occ_3d, 0.0, 1.0)
                        composited = rendered_norm + gt_norm * (1.0 - combined_mask_3d)
                        occ_weight = self.precomputed_occ_weight.get(target_idx)
                        if occ_weight is not None:
                            occ_weight_3d = occ_weight.unsqueeze(-1).expand_as(gt_norm)
                        else:
                            occ_weight_3d = m_occ_3d
                        target = sam3d_render_target * occ_weight_3d + gt_norm * (1.0 - occ_weight_3d)
                    else:
                        composited = rendered_norm + gt_norm * (1.0 - mask_3d)
                        target = gt_norm
                else:
                    composited = rendered_norm
                    target = gt_norm

                composited_chw = composited.permute(2, 0, 1)  # (3, H, W)
                target_chw = target.permute(2, 0, 1)
                Ll1 = l1_loss(composited_chw, target_chw)
                ssim_val = ssim(composited_chw, target_chw)
                lpips_val = lpips_metric(composited_chw.unsqueeze(0), target_chw.unsqueeze(0)).mean() if self.enable_lpips else torch.tensor(0.0, device='cuda')

                loss_render = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_val) + 0.1 * lpips_val


            if rendering_loss_active:
                loss = loss + self.lambda_render * loss_render

        # === RANDOM CAMERA LOSS (novel-view appearance + depth) ===
        loss_random_cam = torch.tensor(0.0, device='cuda')
        need_rc_densify_meta = self.enable_densify_and_prune_rc and self.iteration < self.opt.densify_until_iter
        rc_render_metas = []
        rc_enabled = self.lambda_rc > 0
        if rc_enabled:
            orbit_distance = getattr(self.args, 'rc_orbit_distance', None) or 5.0
            render_size = 512
            rc_batch_size = getattr(self.args, 'rc_batch_size', 2)

            sam3d_xyz = target_data['xyz']
            rc_sam3d_xyz = sam3d_xyz

            def_rot_rand = self.gaussians.get_rotation.detach()
            def_scale_rand = self.gaussians.get_scaling.detach()
            def_opacity_detached = self.gaussians.get_opacity.detach()

            features_dc_live = self.gaussians._features_dc  # (N, 1, 3)
            def_features = features_dc_live.permute(0, 2, 1)


            # Two-pass RC rendering when opacity needs gradients from RC (live opacity for RGB, detached for depth)
            rc_two_pass = self.enable_reset_opacity_rc

            batch_rc_losses = []
            for _rc_i in range(rc_batch_size):
                # Sample random camera on sphere
                azimuth = np.random.uniform(0, 2 * np.pi)
                elevation = np.random.uniform(-np.pi / 2, np.pi / 2)

                cos_el, sin_el = np.cos(elevation), np.sin(elevation)
                cos_az, sin_az = np.cos(azimuth), np.sin(azimuth)

                R_y = torch.tensor([
                    [cos_az, 0, sin_az], [0, 1, 0], [-sin_az, 0, cos_az],
                ], dtype=torch.float32, device='cuda')
                R_x = torch.tensor([
                    [1, 0, 0], [0, cos_el, -sin_el], [0, sin_el, cos_el],
                ], dtype=torch.float32, device='cuda')
                R_cam = R_x @ R_y

                def_xyz_rand = deformed_xyz @ R_cam.T
                def_xyz_rand = def_xyz_rand.clone()
                def_xyz_rand[:, 2] += orbit_distance

                # Single-pass RC rendering: render RGB+depth together with live opacity,
                # then detach depth/alpha for the depth loss to avoid FPE through opacity.
                rc_opacity = self.gaussians.get_opacity if rc_two_pass else def_opacity_detached
                if need_rc_densify_meta:
                    rc_result = self.sam3d_data.render_gaussian(
                        xyz=def_xyz_rand, rotation=def_rot_rand, scaling=def_scale_rand,
                        opacity=rc_opacity, features_dc=def_features,
                        width=render_size, height=render_size,
                        return_depth=True, return_meta=True,
                    )
                    def_rgb_live, _rc_alpha, def_depth_live, rc_meta = rc_result
                    rc_meta['means2d'].retain_grad()
                    rc_render_metas.append(rc_meta)
                else:
                    def_rgb_live, _rc_alpha, def_depth_live = self.sam3d_data.render_gaussian(
                        xyz=def_xyz_rand, rotation=def_rot_rand, scaling=def_scale_rand,
                        opacity=rc_opacity, features_dc=def_features,
                        width=render_size, height=render_size,
                        return_depth=True,
                    )
                # Detach depth and alpha so depth loss gradients don't flow through opacity
                def_depth = def_depth_live.detach()
                def_alpha_det = _rc_alpha.detach()

                # Render SAM3D target GS
                sam3d_xyz_rand = rc_sam3d_xyz @ R_cam.T
                sam3d_xyz_rand = sam3d_xyz_rand.clone()
                sam3d_xyz_rand[:, 2] += orbit_distance

                sam3d_rgb, sam3d_alpha, sam3d_depth = self.sam3d_data.render_gaussian(
                    xyz=sam3d_xyz_rand, rotation=target_data['rotation'],
                    scaling=target_data['scaling'], opacity=target_data['opacity'],
                    features_dc=target_data['features_dc'],
                    width=render_size, height=render_size,
                    return_depth=True,
                )

                # RGB loss (uses live opacity render)
                def_rgb_norm = def_rgb_live / 255.0
                sam3d_rgb_norm = sam3d_rgb / 255.0
                def_rgb_chw = def_rgb_norm.permute(2, 0, 1)
                sam3d_rgb_chw = sam3d_rgb_norm.permute(2, 0, 1)
                rc_Ll1 = l1_loss(def_rgb_chw, sam3d_rgb_chw)
                rc_ssim = ssim(def_rgb_chw, sam3d_rgb_chw)
                rc_lpips = lpips_metric(def_rgb_chw.unsqueeze(0), sam3d_rgb_chw.unsqueeze(0)).mean() if self.enable_lpips else torch.tensor(0.0, device='cuda')
                loss_rc_rgb = (1.0 - self.opt.lambda_dssim) * rc_Ll1 + self.opt.lambda_dssim * (1.0 - rc_ssim) + 0.1 * rc_lpips

                # Depth loss (uses detached opacity render — avoids FPE)
                # Charbonnier loss inside the object mask
                epsilon = 1e-3
                depth_diff_sq = (def_depth - sam3d_depth) ** 2
                charbonnier = torch.sqrt(depth_diff_sq + epsilon ** 2) - epsilon
                loss_rc_depth_inside = (sam3d_alpha * charbonnier).mean()

                # Depth smoothness loss: penalize large gradients in rendered depth
                def_d_2d = def_depth.squeeze(-1)  # (H, W)
                smooth_dx = (def_d_2d[:, 1:] - def_d_2d[:, :-1]) ** 2
                smooth_dy = (def_d_2d[1:, :] - def_d_2d[:-1, :]) ** 2
                loss_rc_smooth = smooth_dx.mean() + smooth_dy.mean()

                # Penalize deformed GS rendering outside the object mask
                # Use live (non-detached) alpha so gradients flow through opacity,
                # allowing this loss to actually push outlier Gaussians toward transparency.
                outside_mask = (1.0 - sam3d_alpha)  # regions where SAM3D has no content
                loss_rc_depth_outside = (outside_mask * _rc_alpha).mean()

                loss_rc_depth = loss_rc_depth_inside + 0.1 * loss_rc_smooth + loss_rc_depth_outside


                batch_rc_losses.append(self.lambda_rc * (loss_rc_depth + loss_rc_rgb))

            loss_random_cam = torch.stack(batch_rc_losses).mean()
            loss = loss + loss_random_cam

            # Debug: save RC renders (RGB, depth, normals) every 200 iterations
            # Grid layout: 2 rows (Deformed GS top, SAM3D bottom) × 3 cols (RGB, Depth, Normals)
            if getattr(self.args, 'save_rc_debug', False) and self.iteration % 200 == 0:
                rc_debug_dir = os.path.join(self.args.model_path, "rc_debug")
                os.makedirs(rc_debug_dir, exist_ok=True)
                with torch.no_grad():
                    import cv2 as _cv2
                    from pytorch3d.transforms import quaternion_to_matrix as _q2m
                    S = render_size  # panel size
                    fx_rc = self.sam3d_data.focal_length
                    _Ks_rc = torch.tensor([[fx_rc, 0, S/2.0], [0, fx_rc, S/2.0], [0, 0, 1]],
                                          device='cuda', dtype=torch.float32).unsqueeze(0)
                    _vm = torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0)

                    # RGB
                    def_np = (def_rgb_norm.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    sam3d_np = (sam3d_rgb_norm.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                    # Depth (shared normalization)
                    def_d = def_depth.squeeze(-1).cpu().numpy()
                    sam3d_d = sam3d_depth.squeeze(-1).cpu().numpy()
                    d_max = max(def_d[def_d > 0].max() if (def_d > 0).any() else 1,
                                sam3d_d[sam3d_d > 0].max() if (sam3d_d > 0).any() else 1)
                    def_d_color = _cv2.applyColorMap((def_d / d_max * 255).clip(0, 255).astype(np.uint8), _cv2.COLORMAP_VIRIDIS)
                    sam3d_d_color = _cv2.applyColorMap((sam3d_d / d_max * 255).clip(0, 255).astype(np.uint8), _cv2.COLORMAP_VIRIDIS)
                    def_d_rgb = _cv2.cvtColor(def_d_color, _cv2.COLOR_BGR2RGB)
                    sam3d_d_rgb = _cv2.cvtColor(sam3d_d_color, _cv2.COLOR_BGR2RGB)

                    # Depth-gradient normals: unproject depth to 3D, central differences, cross product
                    def _depth_to_normals(depth_hw, focal, size):
                        d = depth_hw.squeeze(-1)  # (S, S)
                        ys, xs = torch.meshgrid(torch.arange(size, device='cuda', dtype=torch.float32),
                                                torch.arange(size, device='cuda', dtype=torch.float32), indexing='ij')
                        xs_c = (xs - size / 2.0) / focal
                        ys_c = (ys - size / 2.0) / focal
                        pts = torch.stack([xs_c * d, ys_c * d, d], dim=-1)  # (S, S, 3)
                        # Central differences
                        dx = pts[2:, 1:-1] - pts[:-2, 1:-1]
                        dy = pts[1:-1, 2:] - pts[1:-1, :-2]
                        n = F.normalize(torch.cross(dx, dy, dim=-1), dim=-1, eps=1e-6)
                        out = torch.full((size, size, 3), 0.5, device='cuda')  # gray bg = (0,0,0) after *0.5+0.5
                        out[1:-1, 1:-1] = n * 0.5 + 0.5
                        return (out.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

                    def_normals_np = _depth_to_normals(def_depth, fx_rc, S)
                    sam3d_normals_np = _depth_to_normals(sam3d_depth, fx_rc, S)

                    # Grid: top row = Deformed (RGB | Depth | Normals), bottom = SAM3D
                    row_def = np.concatenate([def_np, def_d_rgb, def_normals_np], axis=1)
                    row_sam = np.concatenate([sam3d_np, sam3d_d_rgb, sam3d_normals_np], axis=1)
                    grid = np.concatenate([row_def, row_sam], axis=0)
                    _cv2.imwrite(
                        os.path.join(rc_debug_dir, f"rc_iter{self.iteration:06d}_az{np.degrees(azimuth):.0f}_el{np.degrees(elevation):.0f}.png"),
                        _cv2.cvtColor(grid, _cv2.COLOR_RGB2BGR))

        # === SDS LOSS (Zero123 Score Distillation Sampling) ===
        # Anneal timestep range if end values are specified
        if self.zero123_guidance is not None:
            sds_min_end = getattr(self.args, 'sds_min_step_end', None)
            sds_max_end = getattr(self.args, 'sds_max_step_end', None)
            if sds_min_end is not None or sds_max_end is not None:
                progress = self.iteration / max(1, self.opt.iterations)
                self.zero123_guidance.update_steps(progress, sds_min_end, sds_max_end)

        loss_sds_rgb = torch.tensor(0.0, device='cuda')
        if self.lambda_sds_rgb > 0 and self.zero123_guidance is not None:
            # Set reference frame: canonical only or current target
            sdsrgb_frame = target_idx
            self.zero123_guidance.set_active_frame(sdsrgb_frame)

            sdsrgb_az = np.random.uniform(0, 2 * np.pi)
            sdsrgb_el = np.random.uniform(-np.pi / 12, np.pi / 6)

            sdsrgb_deformed = deformed_xyz

            # Get SE(3) transform for projection (use sdsrgb_frame for canonical-only mode)
            # SDS RGB only affects colors — detach all geometry
            sdsrgb_transform = self.sam3d_data.get_transform(sdsrgb_frame)
            sdsrgb_rot = self.gaussians.get_rotation.detach()
            sdsrgb_scale = self.gaussians.get_scaling.detach()
            sdsrgb_xyz_in = sdsrgb_deformed.detach()
            if self.optimize_per_frame_compose_transforms_app:
                sdsrgb_xyz_in, sdsrgb_scale = self.apply_compose_transform_app(sdsrgb_xyz_in, sdsrgb_frame, sdsrgb_scale)
            sdsrgb_xyz_cam, sdsrgb_rot_cam, sdsrgb_scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                sdsrgb_xyz_in, sdsrgb_rot, sdsrgb_scale, sdsrgb_transform,
            )

            # Center + normalize once (view-independent; scalar scale commutes with rotation)
            centroid = sdsrgb_xyz_cam.mean(dim=0).detach()
            sdsrgb_xyz_centered = (sdsrgb_xyz_cam - centroid) * self.sds_norm_scale
            sdsrgb_scale_cam = sdsrgb_scale_cam * self.sds_norm_scale

            features_dc_live = self.gaussians._features_dc if self.optimize_canonical_color else self.gaussians._features_dc.detach()
            sdsrgb_features_dc = features_dc_live.permute(0, 2, 1)

            # Orbit + render one novel view (live — gradients flow back to Gaussians)
            _ce, _se = np.cos(sdsrgb_el), np.sin(sdsrgb_el)
            _ca, _sa = np.cos(sdsrgb_az), np.sin(sdsrgb_az)
            R_y = torch.tensor([
                [_ca, 0, _sa], [0, 1, 0], [-_sa, 0, _ca],
            ], dtype=torch.float32, device='cuda')
            R_x = torch.tensor([
                [1, 0, 0], [0, _ce, -_se], [0, _se, _ce],
            ], dtype=torch.float32, device='cuda')
            R_cam = R_x @ R_y
            view_xyz = (sdsrgb_xyz_centered @ R_cam.T).clone()
            view_xyz[:, 2] += self.sds_cond_distance
            sdsrgb_rendered, _ = self.sam3d_data.render_gaussian(
                xyz=view_xyz,
                rotation=sdsrgb_rot_cam,
                scaling=sdsrgb_scale_cam,
                opacity=self.gaussians.get_opacity.detach(),
                features_dc=sdsrgb_features_dc,
                width=256, height=256,
                bg_color=(1.0, 1.0, 1.0),
                focal_length=self.sds_focal_length,
            )

            # DDIM target: encode + noise + denoise (all no_grad, detached)
            ddim_target, ddim_w = self.zero123_guidance.ddim_target(
                sdsrgb_rendered.detach(),
                elevation_deg=np.degrees(sdsrgb_el),
                azimuth_deg=np.degrees(sdsrgb_az),
                distance=self.sds_cond_distance,
            )

            if ddim_target is not None:
                # L2 + optional LPIPS loss: live GS render vs detached DDIM prediction.
                rendered_norm = sdsrgb_rendered / 255.0            # (256, 256, 3) [0, 1]
                ddim_detached = ddim_target.detach()
                rendered_chw = rendered_norm.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 256, 256)
                ddim_chw = ddim_detached.permute(2, 0, 1).unsqueeze(0)
                loss_sds_rgb_mse = F.mse_loss(rendered_norm, ddim_detached)
                if self.enable_lpips:
                    loss_sds_rgb_lpips = lpips_metric(rendered_chw, ddim_chw).mean()
                else:
                    loss_sds_rgb_lpips = torch.tensor(0.0, device='cuda')
                loss_sds_rgb = 1 * (self.lambda_sds_rgb * loss_sds_rgb_mse + 0.1 * loss_sds_rgb_lpips)
                loss = loss + loss_sds_rgb

                # Debug: save reference | GS render | DDIM target side by side
                if getattr(self.args, 'save_sds_debug', False) and self.iteration % 10 == 0:
                    debug_dir = os.path.join(self.args.model_path, "sds_debug_render")
                    os.makedirs(debug_dir, exist_ok=True)
                    with torch.no_grad():
                        # Reference conditioning image (256x256)
                        ref_tensor = self._sds_reference_images.get(sdsrgb_frame)
                        if ref_tensor is not None:
                            ref_np = (ref_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        else:
                            ref_np = np.zeros((256, 256, 3), dtype=np.uint8)
                        # GS render
                        render_np = sdsrgb_rendered.detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                        # DDIM prediction
                        _ddim_vis = ddim_target
                        target_np = (_ddim_vis.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                        strip = np.concatenate([ref_np, render_np, target_np], axis=1)
                        from PIL import Image as _PILImage
                        _PILImage.fromarray(strip).save(
                            os.path.join(debug_dir, f"sdsrgb_iter{self.iteration:06d}_az{np.degrees(sdsrgb_az):.0f}_el{np.degrees(sdsrgb_el):.0f}.png")
                        )



        # NaN guard
        if torch.isnan(loss):
            nan_parts = []
            if torch.isnan(loss_chamfer): nan_parts.append('chamfer')
            if torch.is_tensor(loss_arap) and torch.isnan(loss_arap): nan_parts.append('arap')
            if torch.isnan(loss_temporal_tv): nan_parts.append('temporal_tv')
            if self.lambda_rc > 0 and torch.isnan(loss_random_cam): nan_parts.append('random_cam')
            if rendering_loss_active and torch.isnan(loss_render): nan_parts.append('render')
            if self.lambda_sds_rgb > 0 and torch.isnan(loss_sds_rgb): nan_parts.append('sds_rgb')
            if self.lambda_rigid > 0 and torch.isnan(loss_rigid): nan_parts.append('rigid')
            print(f"\n[WARN] NaN loss at iter {self.iteration}, sources: {nan_parts}")
            # Skip this iteration to prevent NaN from corrupting weights
            self.deform.optimizer.zero_grad()
            self.gaussians.optimizer.zero_grad(set_to_none=True)
            self.iteration += 1
            return

        # Backward
        loss.backward()

        # Accumulate densification stats from gsplat meta (packed format)
        if render_meta is not None and self.enable_densify_and_prune:
            means2d = render_meta['means2d']  # (M, 2) packed
            if means2d.grad is not None:
                gaussian_ids = render_meta['gaussian_ids']  # (M,)
                radii = render_meta['radii']  # (M, 2) int32
                N = self.gaussians.get_xyz.shape[0]
                # Resize accumulators if Gaussian count changed
                if self.gaussians.xyz_gradient_accum.shape[0] != N:
                    self.gaussians.xyz_gradient_accum = torch.zeros((N, 1), device='cuda')
                    self.gaussians.denom = torch.zeros((N, 1), device='cuda')
                    self.gaussians.max_radii2D = torch.zeros(N, device='cuda')
                # Accumulate 2D gradient norms
                grad_norms = means2d.grad.norm(dim=-1)  # (M,)
                self.gaussians.xyz_gradient_accum[gaussian_ids] += grad_norms.unsqueeze(-1)
                self.gaussians.denom[gaussian_ids] += 1
                # Update max screen-space radii
                max_radii = radii.max(dim=-1).values.float()  # (M,)
                self.gaussians.max_radii2D[gaussian_ids] = torch.max(
                    self.gaussians.max_radii2D[gaussian_ids], max_radii)

        # Accumulate densification stats from RC render metas
        if self.enable_densify_and_prune_rc and rc_render_metas:
            N = self.gaussians.get_xyz.shape[0]
            if self.gaussians.xyz_gradient_accum.shape[0] != N:
                self.gaussians.xyz_gradient_accum = torch.zeros((N, 1), device='cuda')
                self.gaussians.denom = torch.zeros((N, 1), device='cuda')
                self.gaussians.max_radii2D = torch.zeros(N, device='cuda')
            for rc_meta in rc_render_metas:
                means2d = rc_meta['means2d']
                if means2d.grad is not None:
                    gaussian_ids = rc_meta['gaussian_ids']
                    radii = rc_meta['radii']
                    grad_norms = means2d.grad.norm(dim=-1)
                    self.gaussians.xyz_gradient_accum[gaussian_ids] += grad_norms.unsqueeze(-1)
                    self.gaussians.denom[gaussian_ids] += 1
                    max_radii = radii.max(dim=-1).values.float()
                    self.gaussians.max_radii2D[gaussian_ids] = torch.max(
                        self.gaussians.max_radii2D[gaussian_ids], max_radii)


        self.iter_end.record()

        with torch.no_grad():
            # Progress tracking
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            if self.iteration % 10 == 0:
                postfix = {
                    "Loss": f"{self.ema_loss_for_log:.5f}",
                    "Chamfer": f"{loss_chamfer.item():.5f}",
                    "ARAP": f"{loss_arap.item():.5f}" if torch.is_tensor(loss_arap) else f"{loss_arap:.5f}",
                }
                if self.lambda_rc > 0:
                    postfix["RandCam"] = f"{loss_random_cam.item():.5f}"
                if self.lambda_sds_rgb > 0:
                    postfix["SDSrgb"] = f"{loss_sds_rgb.item():.5f}"
                if self.lambda_rigid > 0:
                    postfix["Rigid"] = f"{loss_rigid.item():.5f}"
                for pg in self.gaussians.optimizer.param_groups:
                    if pg["name"] == "xyz":
                        postfix["lr"] = f"{pg['lr']:.2e}"
                        break
                self.progress_bar.set_postfix(postfix)
                self.progress_bar.update(10)

            # Optimizer step — only step gaussians optimizer if canonical params should receive gradients
            if self.optimize_canonical_color or self.optimize_gs_xyz or self.occlusion_compositing or self.lambda_rc > 0 or self.lambda_sds_rgb > 0:
                self.gaussians.optimizer.step()
                self.gaussians.update_learning_rate(self.iteration - self._position_lr_iter_offset)
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            self.deform.optimizer.step()
            self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(self.iteration)



            if self.optimize_per_frame_compose_transforms_app:
                lr = self.compose_app_lr_func(self.iteration - self._compose_lr_iter_offset)
                for pg in self.compose_app_optimizer.param_groups:
                    pg['lr'] = lr
                self.compose_app_optimizer.step()
                self.compose_app_optimizer.zero_grad(set_to_none=True)


        self.deform.update(self.iteration)

        # Densification and pruning (opt-in via flags)
        if self.iteration < self.opt.densify_until_iter:
            if self.enable_densify_and_prune or self.enable_densify_and_prune_rc:
                if self.iteration > self.opt.densify_from_iter and self.iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    n_before = self.gaussians.get_xyz.shape[0]
                    self.gaussians.densify_and_prune(
                        self.opt.densify_grad_threshold,
                        0.005,  # min_opacity
                        self.scene.cameras_extent,  # = 1.0, prunes scales > 0.1
                        size_threshold
                    )

                    n_after = self.gaussians.get_xyz.shape[0]
                    if n_after != n_before:
                        self._knn_idx = None  # invalidate KNN cache

            if self.enable_reset_opacity_rc:
                if self.iteration % self.opt.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()

        self.iteration += 1

    def eval_metrics(self):
        """Evaluate PSNR/SSIM/LPIPS across all frames (masked region only)."""
        psnr_list, ssim_list, lpips_list = [], [], []

        with torch.no_grad():
            for frame_data in self.sam3d_data.frame_data:
                frame_idx = frame_data["frame_idx"]

                gt_image = self.sam3d_data.get_gt_image(frame_idx)
                if gt_image is None:
                    continue
                h, w = int(gt_image.shape[0]), int(gt_image.shape[1])

                transform = self.sam3d_data.get_transform(frame_idx)
                deformed_xyz = self._deform_at_frame(frame_idx)
                canonical_rotation = self.gaussians.get_rotation
                canonical_scaling = self.gaussians.get_scaling
                opacity = self.gaussians.get_opacity
                features_dc = self.gaussians._features_dc

                deformed_rotation = canonical_rotation
                deformed_scaling = canonical_scaling

                vis_xyz = deformed_xyz
                vis_rot = deformed_rotation
                vis_scale = deformed_scaling
                if self.optimize_per_frame_compose_transforms_app:
                    vis_xyz, vis_scale = self.apply_compose_transform_app(vis_xyz, frame_idx, vis_scale)
                xyz_cam, rot_cam, scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                    vis_xyz, vis_rot, vis_scale, transform
                )

                features_dc_render = features_dc.permute(0, 2, 1)
                rendered, _ = self.sam3d_data.render_gaussian(
                    xyz=xyz_cam, rotation=rot_cam, scaling=scale_cam,
                    opacity=opacity, features_dc=features_dc_render,
                    width=w, height=h,
                )

                # Normalize to [0, 1] and reshape to (1, 3, H, W) for metrics
                rendered_norm = torch.clamp(rendered / 255.0, 0.0, 1.0).permute(2, 0, 1).unsqueeze(0)
                gt_norm = torch.clamp(gt_image / 255.0, 0.0, 1.0).permute(2, 0, 1).unsqueeze(0)

                # Apply mask if available
                mask = self.sam3d_data.get_mask(frame_idx)
                if mask is not None:
                    mask_4d = mask.unsqueeze(0).unsqueeze(0).expand_as(rendered_norm)
                    rendered_norm = (rendered_norm * mask_4d).contiguous()
                    gt_norm = (gt_norm * mask_4d).contiguous()

                psnr_list.append(psnr_func(rendered_norm, gt_norm).mean().item())
                ssim_list.append(ssim_metric(rendered_norm, gt_norm, data_range=1.).mean().item())
                lpips_list.append(lpips_metric(rendered_norm, gt_norm).mean().item())

        avg_psnr = np.mean(psnr_list) if psnr_list else 0.0
        avg_ssim = np.mean(ssim_list) if ssim_list else 0.0
        avg_lpips = np.mean(lpips_list) if lpips_list else 0.0

        print(f"\n[ITER {self.iteration}] Eval: PSNR={avg_psnr:.4f}  SSIM={avg_ssim:.4f}  LPIPS={avg_lpips:.4f}")
        return avg_psnr, avg_ssim, avg_lpips

    def save_comparison_video(self):
        """Save 4-panel comparison video: GT, SAM3D GS, Canonical GS, Deformed GS."""
        import cv2

        output_dir = os.path.join(self.args.model_path, f"comparison_iter_{self.iteration:06d}")
        os.makedirs(output_dir, exist_ok=True)

        print(f"\nSaving comparison video at iteration {self.iteration} to {output_dir}")

        comparison_frames = []
        tracked_frames = []  # for point-tracking overlay video

        # Pick a single visible pixel inside the mask on the first frame
        # and track the Gaussian closest to it
        with torch.no_grad():
            first_frame = self.sam3d_data.frame_data[0]
            first_idx = first_frame["frame_idx"]
            first_gt = self.sam3d_data.get_gt_image(first_idx)
            fh, fw = int(first_gt.shape[0]), int(first_gt.shape[1])
            first_mask = self.sam3d_data.get_mask(first_idx)
            first_transform = self.sam3d_data.get_transform(first_idx)

            # Deform canonical Gaussians to first frame
            can_xyz = self.gaussians.get_xyz.detach()
            deformed0 = self._deform_at_frame(first_idx)

            # Project to image plane
            track_xyz0 = deformed0
            track_rot0 = self.gaussians.get_rotation
            track_scale0 = self.gaussians.get_scaling
            if self.optimize_per_frame_compose_transforms_app:
                track_xyz0, track_scale0 = self.apply_compose_transform_app(track_xyz0, first_idx, track_scale0)
            xyz_cam0, _, _ = self.sam3d_data.apply_transform_to_gaussian(
                track_xyz0, track_rot0, track_scale0, first_transform,
            )
            max_dim0 = max(fw, fh)
            fx0 = self.sam3d_data.focal_length
            cx0 = max_dim0 / 2.0
            gz = xyz_cam0[:, 2].clamp(min=1e-6)
            gu = fx0 * xyz_cam0[:, 0] / gz + cx0
            gv = fx0 * xyz_cam0[:, 1] / gz + cx0
            if fw > fh:
                gv = gv - (max_dim0 - fh) // 2
            else:
                gu = gu - (max_dim0 - fw) // 2

            # Find Gaussians that project inside the mask
            gu_int = gu.long().clamp(0, fw - 1)
            gv_int = gv.long().clamp(0, fh - 1)
            in_bounds = (gu >= 0) & (gu < fw) & (gv >= 0) & (gv < fh)
            in_mask = torch.zeros(can_xyz.shape[0], dtype=torch.bool, device='cuda')
            if first_mask is not None:
                in_mask[in_bounds] = first_mask[gv_int[in_bounds], gu_int[in_bounds]] > 0.5
            else:
                in_mask = in_bounds

            # Sample 20 visible pixels uniformly from the mask
            n_track_target = 20
            valid_indices = in_mask.nonzero(as_tuple=True)[0]
            if valid_indices.numel() > 0:
                if valid_indices.numel() <= n_track_target:
                    track_indices = valid_indices.tolist()
                else:
                    # Uniform random sample without replacement
                    perm = torch.randperm(valid_indices.numel(), device='cuda')[:n_track_target]
                    track_indices = valid_indices[perm].tolist()
            else:
                # Fallback: pick the Gaussian closest to image center
                du = gu - fw / 2.0
                dv = gv - fh / 2.0
                track_indices = [(du ** 2 + dv ** 2).argmin().item()]

            n_track = len(track_indices)
            # Distinct colors for each tracked point (HSV hue spread)
            track_colors = []
            for ci in range(n_track):
                hue = int(180 * ci / max(n_track, 1))
                bgr = cv2.cvtColor(np.array([[[hue, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
                track_colors.append(tuple(int(c) for c in bgr[0, 0]))
            track_trajectories = [[] for _ in range(n_track)]


        with torch.no_grad():
            for frame_data in tqdm.tqdm(self.sam3d_data.frame_data, desc="Rendering comparison"):
                frame_idx = frame_data["frame_idx"]

                # Get GT image
                gt_image = self.sam3d_data.get_gt_image(frame_idx)
                if gt_image is None:
                    continue
                h, w = int(gt_image.shape[0]), int(gt_image.shape[1])

                transform = self.sam3d_data.get_transform(frame_idx)

                # Load SAM3D Gaussian for this frame
                sam3d_data = self.sam3d_data.load_frame(frame_idx)

                # Get canonical Gaussians
                canonical_xyz = self.gaussians.get_xyz
                canonical_rotation = self.gaussians.get_rotation
                canonical_scaling = self.gaussians.get_scaling
                opacity = self.gaussians.get_opacity
                features_dc = self.gaussians._features_dc

                # Compute deformation for this frame
                deformed_xyz = self._deform_at_frame(frame_idx)
                deformed_rotation = canonical_rotation
                deformed_scaling = canonical_scaling

                # Prepare features_dc for rendering
                features_dc_render = features_dc.permute(0, 2, 1)  # (N, 3, 1) -> (N, 1, 3)

                # 1. Render SAM3D GS (ground truth geometry) — always use per-frame transform
                sam3d_transform = self.sam3d_data.get_transform(frame_idx)
                sam3d_xyz_cam, sam3d_rot_cam, sam3d_scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                    sam3d_data['xyz'], sam3d_data['rotation'], sam3d_data['scaling'], sam3d_transform
                )
                sam3d_rendered, _ = self.sam3d_data.render_gaussian(
                    xyz=sam3d_xyz_cam,
                    rotation=sam3d_rot_cam,
                    scaling=sam3d_scale_cam,
                    opacity=sam3d_data['opacity'],
                    features_dc=sam3d_data['features_dc'],
                    width=w, height=h,
                )

                # 2. Render Canonical GS (using canonical SE(3) transform)
                canonical_transform = self.sam3d_data.get_transform(self.sam3d_data.canonical_frame_idx)
                can_xyz_cam, can_rot_cam, can_scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                    canonical_xyz, canonical_rotation, canonical_scaling, canonical_transform
                )
                canonical_rendered, _ = self.sam3d_data.render_gaussian(
                    xyz=can_xyz_cam,
                    rotation=can_rot_cam,
                    scaling=can_scale_cam,
                    opacity=opacity,
                    features_dc=features_dc_render,
                    width=w, height=h,
                )

                # 3. Render Deformed GS (using target frame SE(3) transform + learned correction)
                def_xyz_for_render = deformed_xyz
                def_rot_for_render = deformed_rotation
                def_scale_for_render = deformed_scaling
                if self.optimize_per_frame_compose_transforms_app:
                    def_xyz_for_render, def_scale_for_render = self.apply_compose_transform_app(def_xyz_for_render, frame_idx, def_scale_for_render)
                def_xyz_cam, def_rot_cam, def_scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                    def_xyz_for_render, def_rot_for_render, def_scale_for_render, transform
                )
                deformed_rendered, _, deformed_depth = self.sam3d_data.render_gaussian(
                    xyz=def_xyz_cam,
                    rotation=def_rot_cam,
                    scaling=def_scale_cam,
                    opacity=opacity,
                    features_dc=features_dc_render,
                    width=w, height=h,
                    return_depth=True,
                )

                # Get precomputed m_occ
                m_occ = self.precomputed_occ.get(frame_idx)
                m_occ_np = m_occ.cpu().numpy().astype(np.uint8) if m_occ is not None else np.zeros((h, w), dtype=np.uint8)

                # Convert to numpy
                gt_np = gt_image.cpu().numpy().astype(np.uint8)
                sam3d_np = sam3d_rendered.cpu().numpy().clip(0, 255).astype(np.uint8)
                canonical_np = canonical_rendered.cpu().numpy().clip(0, 255).astype(np.uint8)
                deformed_np = deformed_rendered.cpu().numpy().clip(0, 255).astype(np.uint8)

                # m_occ as RGB (white = occluded)
                m_occ_rgb = np.stack([m_occ_np * 255] * 3, axis=-1).astype(np.uint8)

                # m_occ overlay on GT: soft-blended color-corrected SAM3D render
                m_occ_overlay = gt_np.copy().astype(np.float32)
                sam3d_render_cc = self.precomputed_sam3d_render.get(frame_idx)
                occ_weight = self.precomputed_occ_weight.get(frame_idx)
                if sam3d_render_cc is not None and occ_weight is not None:
                    cc_np = (sam3d_render_cc.cpu().numpy() * 255).clip(0, 255).astype(np.float32)
                    w_np = occ_weight.cpu().numpy()[:, :, np.newaxis]
                    m_occ_overlay = cc_np * w_np + m_occ_overlay * (1.0 - w_np)
                m_occ_overlay = m_occ_overlay.clip(0, 255).astype(np.uint8)

                # Add labels
                label_h = 30
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (255, 255, 255)
                bg_color = (0, 0, 0)

                def add_label(img, label):
                    labeled = np.zeros((img.shape[0] + label_h, img.shape[1], 3), dtype=np.uint8)
                    labeled[:label_h] = bg_color
                    labeled[label_h:] = img
                    cv2.putText(labeled, label, (10, 22), font, font_scale, font_color, 1)
                    return labeled

                gt_labeled = add_label(gt_np, f"GT Frame {frame_idx}")
                sam3d_labeled = add_label(sam3d_np, "SAM3D GS")
                canonical_labeled = add_label(canonical_np, f"Canonical (frame {self.sam3d_data.canonical_frame_idx})")
                deformed_labeled = add_label(deformed_np, f"Deformed to frame {frame_idx}")
                m_occ_labeled = add_label(m_occ_rgb, "m_occ")
                m_occ_overlay_labeled = add_label(m_occ_overlay, "GT + color-matched occ")

                # Create 2x3 grid
                top_row = np.concatenate([gt_labeled, sam3d_labeled, m_occ_labeled], axis=1)
                bottom_row = np.concatenate([canonical_labeled, deformed_labeled, m_occ_overlay_labeled], axis=1)
                comparison = np.concatenate([top_row, bottom_row], axis=0)

                comparison_frames.append(comparison)

                # Save individual frame
                cv2.imwrite(
                    os.path.join(output_dir, f"frame_{frame_idx:04d}.png"),
                    cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
                )

                # --- Point tracking overlay ---
                # Project tracked Gaussians to 2D using deformed camera-space positions
                max_dim = max(w, h)
                fx = self.sam3d_data.focal_length
                cx = max_dim / 2.0
                track_xyz = def_xyz_cam[track_indices]  # (n_track, 3)
                tz = track_xyz[:, 2].clamp(min=1e-6)
                tu = fx * track_xyz[:, 0] / tz + cx
                tv = fx * track_xyz[:, 1] / tz + cx
                if w > h:
                    tv = tv - (max_dim - h) // 2
                else:
                    tu = tu - (max_dim - w) // 2
                tu_np = tu.cpu().numpy()
                tv_np = tv.cpu().numpy()

                # Store trajectory history
                for pi in range(n_track):
                    if np.isnan(tu_np[pi]) or np.isnan(tv_np[pi]):
                        track_trajectories[pi].append(None)
                    else:
                        track_trajectories[pi].append((int(tu_np[pi]), int(tv_np[pi])))

                # Draw on copies of GT and deformed panels
                gt_tracked = gt_np.copy()
                def_tracked = deformed_np.copy()
                radius = max(3, min(h, w) // 150)
                line_thick = max(1, radius // 2)

                # Draw trajectory lines per point (each has its own color)
                for pi in range(n_track):
                    color = track_colors[pi]
                    traj = track_trajectories[pi]
                    for ti in range(1, len(traj)):
                        p0, p1 = traj[ti - 1], traj[ti]
                        if p0 is None or p1 is None:
                            continue
                        if (0 <= p0[0] < w and 0 <= p0[1] < h and
                            0 <= p1[0] < w and 0 <= p1[1] < h):
                            cv2.line(gt_tracked, p0, p1, color, line_thick, cv2.LINE_AA)
                            cv2.line(def_tracked, p0, p1, color, line_thick, cv2.LINE_AA)

                # Draw current position dots on top
                for pi in range(n_track):
                    if np.isnan(tu_np[pi]) or np.isnan(tv_np[pi]):
                        continue
                    px, py = int(tu_np[pi]), int(tv_np[pi])
                    if 0 <= px < w and 0 <= py < h:
                        color = track_colors[pi]
                        cv2.circle(gt_tracked, (px, py), radius, color, -1)
                        cv2.circle(gt_tracked, (px, py), radius, (0, 0, 0), 1)
                        cv2.circle(def_tracked, (px, py), radius, color, -1)
                        cv2.circle(def_tracked, (px, py), radius, (0, 0, 0), 1)

                gt_tracked_labeled = add_label(gt_tracked, f"GT Frame {frame_idx} (tracked)")
                def_tracked_labeled = add_label(def_tracked, f"Deformed (tracked)")
                panels = [gt_tracked_labeled, def_tracked_labeled]
                tracked_row = np.concatenate(panels, axis=1)
                tracked_frames.append(tracked_row)

        # Save as video
        if len(comparison_frames) > 0:
            video_path = os.path.join(output_dir, "comparison.mp4")
            writer = imageio.get_writer(video_path, format="FFMPEG", fps=5)
            for frame in comparison_frames:
                writer.append_data(frame)
            writer.close()
            print(f"  Saved comparison video: {video_path}")
            print(f"  Saved {len(comparison_frames)} individual frames")

        # Save point-tracking video
        if len(tracked_frames) > 0:
            tracked_path = os.path.join(output_dir, "tracking.mp4")
            writer = imageio.get_writer(tracked_path, format="FFMPEG", fps=5)
            for frame in tracked_frames:
                writer.append_data(frame)
            writer.close()
            print(f"  Saved tracking video: {tracked_path}")

    def save_orbit_video(self):
        """
        Render turntable orbit views of the deformed GS for every video frame.

        Orbits in object space around Z (Blender Z-up) about the object centroid,
        then applies per-frame compose + SAM3D transforms to camera space — the
        same camera-pose logic used for Consistent4D evaluation.

        Views: --num_orbit_views azimuths, evenly spaced clockwise from 75 deg.
        For 4 views this yields [75, -15, -105, 165] — the Consistent4D eval
        convention — so --num_orbit_views 4 makes renders_{iteration}/ directly
        consumable by the evaluation script.

        Saves PNGs to {model_path}/renders_{iteration}/eval_{v}/{frame_idx}.png plus a
        per-view mp4 (and a GT side-by-side comparison mp4 for C4D).
        """
        num_views = int(getattr(self.args, 'num_orbit_views', 0) or 0)
        if num_views <= 0:
            return

        output_root = os.path.join(self.args.model_path, f"renders_{self.iteration}")
        os.makedirs(output_root, exist_ok=True)

        # Turntable azimuths: clockwise from 75 deg
        eval_azimuths_deg = [75.0 - v * (360.0 / num_views) for v in range(num_views)]

        for v in range(num_views):
            os.makedirs(os.path.join(output_root, f"eval_{v}"), exist_ok=True)

        # Get render resolution from first GT image
        first_gt = self.sam3d_data.get_gt_image(self.sam3d_data.frame_data[0]["frame_idx"])
        render_h, render_w = int(first_gt.shape[0]), int(first_gt.shape[1])

        print(f"\nSaving orbit views ({num_views} views x {len(self.sam3d_data.frame_data)} frames) "
              f"at {render_w}x{render_h} to {output_root}")
        print(f"  Orbit azimuths: {eval_azimuths_deg}")

        # Precompute rotation matrix for each eval azimuth: object-space Z (Blender Z-up).
        orbit_rotations = []
        orbit_quats = []
        for azimuth_deg in eval_azimuths_deg:
            a = np.radians(azimuth_deg)
            cos_a, sin_a = np.cos(a), np.sin(a)
            R_z = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a,  cos_a, 0],
                [0,      0,     1],
            ], dtype=torch.float32, device='cuda')
            orbit_rotations.append(R_z)
            orbit_quats.append(matrix_to_quaternion(R_z))

        with torch.no_grad():
            for frame_data in tqdm.tqdm(self.sam3d_data.frame_data, desc="Eval views"):
                frame_idx = frame_data["frame_idx"]
                transform = self.sam3d_data.get_transform(frame_idx)

                canonical_xyz = self.gaussians.get_xyz
                canonical_rotation = self.gaussians.get_rotation
                canonical_scaling = self.gaussians.get_scaling
                opacity = self.gaussians.get_opacity
                features_dc = self.gaussians._features_dc

                deformed_xyz = self._deform_at_frame(frame_idx)
                deformed_rotation = canonical_rotation
                deformed_scaling = canonical_scaling

                # Object-space centroid for orbit center
                obj_center = deformed_xyz.mean(dim=0)

                features_dc_render = features_dc.permute(0, 2, 1)

                for v in range(num_views):
                    R_z = orbit_rotations[v]
                    q_z = orbit_quats[v]

                    # 1. Orbit in object space: rotate around Z (Blender Z-up)
                    xyz_centered = deformed_xyz - obj_center.unsqueeze(0)
                    xyz_rotated = (xyz_centered @ R_z.T) + obj_center.unsqueeze(0)
                    rot_rotated = quaternion_multiply(
                        q_z.unsqueeze(0).expand(deformed_rotation.shape[0], -1),
                        deformed_rotation,
                    )

                    # 2. Apply compose transforms + SAM3D transform to camera space
                    eval_scale = deformed_scaling
                    if self.optimize_per_frame_compose_transforms_app:
                        xyz_rotated, eval_scale = self.apply_compose_transform_app(xyz_rotated, frame_idx, eval_scale)
                    xyz_cam, rot_cam, scale_cam = self.sam3d_data.apply_transform_to_gaussian(
                        xyz_rotated, rot_rotated, eval_scale, transform
                    )

                    # 3. Apply learned per-frame corrections

                    rendered, _ = self.sam3d_data.render_gaussian(
                        xyz=xyz_cam,
                        rotation=rot_cam,
                        scaling=scale_cam,
                        opacity=opacity,
                        features_dc=features_dc_render,
                        width=render_w, height=render_h,
                        bg_color=(1.0, 1.0, 1.0),
                    )

                    rendered_np = rendered.cpu().numpy().clip(0, 255).astype(np.uint8)
                    out_path = os.path.join(output_root, f"eval_{v}", f"{frame_idx}.png")
                    imageio.imwrite(out_path, rendered_np)

        # Locate eval GT directory (Consistent4D multi-view ground truth, optional).
        eval_gt_root = None
        if getattr(self.args, 'dataset', None) == "consistent4d" and self.args.object_name:
            # e.g. <consistent4d_root>/eval_gt/<object>
            eval_gt_root = os.path.join(
                ds_registry.consistent4d_root(), "eval_gt", self.args.object_name)
            if not os.path.isdir(eval_gt_root):
                print(f"  Warning: eval GT dir not found: {eval_gt_root}")
                eval_gt_root = None

        # Save a video per eval view + comparison video
        for v in range(num_views):
            view_dir = os.path.join(output_root, f"eval_{v}")
            frame_files = sorted(
                [f for f in os.listdir(view_dir) if f.endswith(".png")],
                key=lambda x: int(os.path.splitext(x)[0]),
            )
            if frame_files:
                video_path = os.path.join(output_root, f"eval_{v}.mp4")
                writer = imageio.get_writer(video_path, format="FFMPEG", fps=8)
                for f in frame_files:
                    writer.append_data(imageio.imread(os.path.join(view_dir, f)))
                writer.close()
                print(f"  Saved eval_{v} video: {video_path}")

                # Side-by-side comparison: rendered | GT from eval novel views
                if eval_gt_root is not None:
                    gt_view_dir = os.path.join(eval_gt_root, f"eval_{v}")
                    if os.path.isdir(gt_view_dir):
                        comp_path = os.path.join(output_root, f"eval_{v}_comp.mp4")
                        comp_writer = imageio.get_writer(comp_path, format="FFMPEG", fps=8)
                        for f in frame_files:
                            render_frame = imageio.imread(os.path.join(view_dir, f))
                            gt_file = os.path.join(gt_view_dir, f)
                            if os.path.exists(gt_file):
                                gt_frame = imageio.imread(gt_file)
                                if gt_frame.ndim == 3 and gt_frame.shape[2] == 4:
                                    gt_frame = gt_frame[:, :, :3]
                                if gt_frame.shape[:2] != render_frame.shape[:2]:
                                    from PIL import Image as _PILImage
                                    gt_frame = np.array(_PILImage.fromarray(gt_frame).resize(
                                        (render_frame.shape[1], render_frame.shape[0]), _PILImage.BILINEAR))
                                comp_frame = np.concatenate([render_frame, gt_frame], axis=1)
                            else:
                                comp_frame = render_frame
                            comp_writer.append_data(comp_frame)
                        comp_writer.close()
                        print(f"  Saved eval_{v}_comp video: {comp_path}")

        print(f"  Eval views saved to {output_root}")



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    parser.add_argument("--deform-type", type=str, default='node',
                        help="Deformation type: node (stage 1) or node_delta (stage 2)")
    parser.add_argument("--mask_name", type=str, default=None,
                        help="For --dataset custom: the SAM3 object suffix (e.g., 'box_1') "
                             "used to match masks named frame_000000_box_1.png.")
    parser.add_argument("--da3_npz", type=str, default=None,
                        help="Optional DA3 da3_output.npz; overrides the SAM3D focal with DA3 "
                             "intrinsics and loads DA3 depth. If omitted, a da3/da3_output.npz "
                             "next to the reconstructions is used when present.")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=list(ds_registry.DATASETS),
                        help="Dataset type: consistent4d | davis | custom.")
    parser.add_argument("--object_name", type=str, default=None,
                        help="Object name (consistent4d/davis) or video folder name (custom). "
                             "Omit to run every object in the dataset.")
    parser.add_argument("--data_root", type=str, default=ds_registry.data_root(),
                        help="Base data directory holding consistent4d/, DAVIS/, custom/ "
                             "(default: $LIFT4D_DATA_ROOT or <repo>/data).")
    parser.add_argument("--sam3d_input_dir", type=str, default=None,
                        help="Root holding the SAM3D per-frame reconstructions (frame_*/result.ply). "
                             "Defaults to $LIFT4D_SAM3D_OUT (<repo>/sam3d_output).")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Root output directory for model checkpoints (default: ./output)")
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Max frames to load (0=all). Includes canonical + frames after it, backfills before if needed.")
    parser.add_argument("--lambda_chamfer", type=float, default=0.0,
                        help="Chamfer loss weight (default: 1.0)")
    parser.add_argument("--lambda_temporal_tv", type=float, default=1.0,
                        help="Temporal TV loss weight — penalizes velocity of control point deltas between consecutive frames (default: 0)")
    parser.add_argument("--lambda_rigid", type=float, default=1.0,
                        help="Local rigidity loss weight — neighbors should follow rigid-body transform between consecutive frames (default: 0)")
    parser.add_argument("--smooth_transforms_window", type=int, default=0,
                        help="Temporally low-pass the per-frame SAM3D SE(3) transforms with this moving-average window (0/1 = off). Removes per-frame pose-estimation noise from the render chain AND the RC/chamfer anchors; real motion survives")
    parser.add_argument("--knn_batch_size", type=int, default=4096,
                        help="Batch size for KNN precomputation to avoid OOM on large point clouds (default: 4096)")
    parser.add_argument("--lambda_rc", type=float, default=0,
                        help="Random-camera (novel view) anchoring loss weight — applies to both the depth and RGB terms (default: 0)")
    parser.add_argument("--rc_orbit_distance", type=float, default=5.0,
                        help="Random camera orbit distance from object centroid (default: auto per dataset)")
    parser.add_argument("--rc_batch_size", type=int, default=2,
                        help="Number of random cameras to sample per iteration for RC loss (default: 1)")
    parser.add_argument("--disable_rendering_loss_iter", type=int, default=0,
                        help="Disable L1+SSIM rendering loss from this iteration onward (0 = never disable)")
    parser.add_argument("--lambda_render", type=float, default=1.0,
                        help="Weight on the input-view rendering loss (L1+SSIM+LPIPS). Default 1.0")
    parser.add_argument("--vis_interval", type=int, default=2000,
                        help="Save comparison video every N iterations (default: 2000)")
    parser.add_argument("--init_subsample_ratio", type=float, default=1.0,
                        help="Fraction of the SAM3D canonical Gaussians to keep at init "
                             "(evenly strided). 1.0 = keep all; e.g. 0.5 halves the count "
                             "to speed up early training.")
    parser.add_argument("--optimize_gs_xyz", action="store_true",
                        help="Allow geometry/rendering losses to backprop to canonical xyz positions")
    parser.add_argument("--optimize_canonical_color", action="store_true",
                        help="Allow rendering loss to backprop to canonical features_dc (color) only")
    parser.add_argument("--densify_and_prune", action="store_true",
                        help="Enable Gaussian densification and pruning during rendering phase")
    parser.add_argument("--densify_and_prune_rc", action="store_true",
                        help="Enable Gaussian densification and pruning using RC (random camera) render gradients")
    parser.add_argument("--reset_opacity_rc", action="store_true",
                        help="Enable periodic opacity reset; live opacity in RC RGB loss only (render loss opacity stays detached)")
    parser.add_argument("--optimize_per_frame_compose_transforms_app", action="store_true",
                        help="Learn separate per-frame compose (scale, translation) for rendering/SDS/eval paths")
    parser.add_argument("--compose_lr_reset", action="store_true",
                        help="Reset compose LR schedule on --load_checkpoint: 1e-3 -> 1e-7 over remaining iters")
    parser.add_argument("--opt_deform_rot", action="store_true",
                        help="Enable per-frame rotation deformation in delta deform (adds control_point_rotations tensor)")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save Gaussian PLY, deform weights, and transform corrections at vis_interval")
    parser.add_argument("--save_occ_debug", action="store_true",
                        help="Save debug images for occlusion pipeline (masks, depths, color matching)")
    parser.add_argument("--load_checkpoint", type=int, default=-1,
                        help="Load checkpoint at this iteration (-1 = none, 0 = latest)")
    parser.add_argument("--position_lr_init_app", type=float, default=None,
                        help="Override position_lr_init after checkpoint load, with LR schedule reset to step 0 "
                             "(e.g. --load_checkpoint 50000 --iterations 60000 --position_lr_init_app 1.6e-4 "
                             "behaves like --iterations 10000 --position_lr_init 1.6e-4 from scratch)")
    parser.add_argument("--num_orbit_views", type=int, default=4,
                        help="Number of orbit views to render (default: 4, e.g. 4 for 90-degree increments)")
    parser.add_argument("--occlusion_compositing", action="store_true",
                        help="Blend precomputed SAM3D renders into the rendering-loss target in occluded regions of the input view (DAVIS scenes with occluders)")
    parser.add_argument("--enable_lpips", action="store_true",
                        help="Enable LPIPS perceptual loss in rendering loss (default: off)")
    parser.add_argument("--sds_min_step", type=float, default=0.3,
                        help="Min timestep ratio for SDS noise sampling (default: 0.3)")
    parser.add_argument("--sds_max_step", type=float, default=0.6,
                        help="Max timestep ratio for SDS noise sampling (default: 0.6)")
    parser.add_argument("--sds_min_step_end", type=float, default=0.2,
                        help="Anneal sds_min_step to this value by end of training (default: no annealing)")
    parser.add_argument("--sds_max_step_end", type=float, default=0.5,
                        help="Anneal sds_max_step to this value by end of training (default: no annealing)")
    parser.add_argument("--zero123_dir", type=str, default="extern/ldm_zero123",
                        help="Directory containing stable_zero123.ckpt and sd-objaverse-finetune-c_concat-256.yaml")
    parser.add_argument("--save_sds_debug", action="store_true",
                        help="Save Zero123 rendered + denoised debug images (keeps VAE decoder in VRAM)")
    parser.add_argument("--save_rc_debug", action="store_true",
                        help="Save random-camera debug images (RGB, depth, normals grid) every 200 iterations")
    parser.add_argument("--lambda_sds_rgb", type=float, default=0.0,
                        help="L2 loss weight for inline DDIM RGB supervision (renders GS, runs DDIM, takes L2)")
    parser.add_argument("--ddim_steps", type=int, default=5,
                        help="Number of DDIM denoising steps for the SDS-RGB target (default: 5; fewer = faster)")

    args = parser.parse_args(sys.argv[1:])

    # Point the shared dataset registry at the chosen data root.
    os.environ["LIFT4D_DATA_ROOT"] = os.path.abspath(args.data_root)

    def _run_training(args):
        """Run training for the current args configuration."""
        if not args.model_path.endswith(args.deform_type):
            args.model_path = os.path.join(
                os.path.dirname(os.path.normpath(args.model_path)),
                os.path.basename(os.path.normpath(args.model_path)) + f'_{args.deform_type}')

        print("Optimizing " + args.model_path)
        safe_state(False)
        gui = GUI(args=args, dataset=lp.extract(args), opt=op.extract(args))
        gui.train(args.iterations)

    if args.dataset in ("consistent4d", "davis"):
        # ---- Benchmark datasets (Consistent4D / DAVIS) ----
        if args.object_name is not None:
            object_names = [args.object_name]
        else:
            object_names = ds_registry.list_objects(args.dataset)
            print(f"\n=== Multi-object mode: {len(object_names)} objects in {args.dataset} ===")

        # DAVIS per-object defaults
        DAVIS_RC_ORBIT_DEFAULTS = {
            'parkour': 1.0,
            'hike': 2.0, 'blackswan': 2.0,
            'hockey': 4.0,
        }
        DAVIS_MAX_FRAMES_DEFAULTS = {
            'bear': 74,
        }
        user_rc_orbit_distance = args.rc_orbit_distance  # None if not explicitly set
        user_max_frames = getattr(args, 'max_frames', 0)

        for obj_idx, obj_name in enumerate(object_names):
            spec = ds_registry.resolve(args.dataset, obj_name)
            print(f"\n=== Lift4D {args.dataset}: {obj_name} ({obj_idx+1}/{len(object_names)}) ===")
            print(f"  Lambda chamfer: {args.lambda_chamfer}")

            # Set per-object DAVIS defaults if not explicitly provided
            if args.dataset == "davis":
                if user_rc_orbit_distance is None:
                    args.rc_orbit_distance = DAVIS_RC_ORBIT_DEFAULTS.get(obj_name, 5.0)
                    print(f"  RC orbit distance (auto): {args.rc_orbit_distance}")
                if user_max_frames <= 0 and obj_name in DAVIS_MAX_FRAMES_DEFAULTS:
                    args.max_frames = DAVIS_MAX_FRAMES_DEFAULTS[obj_name]
                    print(f"  Max frames (auto): {args.max_frames}")
                elif user_max_frames <= 0:
                    args.max_frames = 0  # reset for other objects (no limit)

            # Set model path
            if args.model_path == "" or len(object_names) > 1:
                args.model_path = os.path.join(
                    args.output_dir, f"{spec.tag}_{args.deform_type}")

            os.makedirs(args.model_path, exist_ok=True)

            # Stage-1 SAM3D outputs for this object (<sam3d_out>/<tag>/frame_*/...)
            per_obj_input_dir = spec.sam3d_output_dir(args.sam3d_input_dir)

            sam3d_data = SAM3DDataLoader(
                selector=obj_name,
                canonical_frame_idx=0,
                device="cuda",
                dataset=args.dataset,
                object_name=obj_name,
                input_dir=per_obj_input_dir,
                max_frames=getattr(args, 'max_frames', 0),
            )

            args.sam3d_data = sam3d_data
            args.object_name = obj_name

            _run_training(args)
            print(f"\nCompleted: {obj_name} ({obj_idx+1}/{len(object_names)})")

    elif args.dataset == "custom":
        # ---- Custom in-the-wild video ----
        # --dataset custom --object_name <video> --mask_name <name>; paths resolve
        # from the data root via the shared registry.
        if not args.object_name:
            parser.error("--dataset custom requires --object_name <video folder> "
                         "(and --mask_name for the segmented object)")
        spec = ds_registry.resolve("custom", args.object_name, mask_name=args.mask_name)
        frames_dir, masks_dir, mask_name = spec.frames_dir, spec.masks_dir, spec.mask_name
        input_dir = spec.sam3d_output_dir(args.sam3d_input_dir)
        video_tag = spec.tag

        print(f"\n=== Lift4D custom video: {video_tag} ===")
        print(f"  Frames dir: {frames_dir}")
        print(f"  Masks dir:  {masks_dir}")
        print(f"  Mask name:  {mask_name}")
        print(f"  SAM3D dir:  {input_dir}")

        if args.model_path == "":
            args.model_path = os.path.join(
                args.output_dir, f"{video_tag}_{args.deform_type}")
        os.makedirs(args.model_path, exist_ok=True)

        sam3d_data = SAM3DDataLoader(
            selector=video_tag,
            canonical_frame_idx=0,
            device="cuda",
            input_dir=input_dir,
            frames_dir=frames_dir,
            masks_dir=masks_dir,
            mask_name=mask_name,
            da3_npz=args.da3_npz,
            max_frames=getattr(args, 'max_frames', 0),
        )

        args.sam3d_data = sam3d_data

        _run_training(args)

    else:
        parser.error("--dataset {consistent4d,davis,custom} is required")

    print("\nTraining complete.")
