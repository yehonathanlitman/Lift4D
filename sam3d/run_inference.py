"""
SAM 3D Objects Video Inference Script

Reconstructs one 3D object per video frame with SDEdit temporal conditioning:
frames are processed starting at --initial_frame_index and propagate
bi-directionally, warm-starting each frame from its neighbor's latents.

Usage (benchmark datasets, paths resolved from --data_root):
    python run_inference.py --dataset consistent4d --object_name pistol --render_video
    python run_inference.py --dataset davis       --object_name rhino  --render_video

Usage (custom video):
    python run_inference.py --dataset custom --object_name my_clip --mask_name box \
        --subsampling 8 --video_consistency 0.1 --render_video

You can also point directly at a folder instead of using the registry:
    python run_inference.py --input_path <video_dir> --mask_name box --render_video
    where <video_dir> has frames/ and masks/ subdirs and masks are named
    <frame_stem>_<mask_name>.png.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional
from loguru import logger
import numpy as np

_SAM3D_DIR = str(Path(__file__).resolve().parent)
sys.path.append(os.path.join(_SAM3D_DIR, "notebook"))
from inference import Inference
from sam3d_objects.pipeline.video_utils import load_video_frames_and_masks
from da3_utils import run_da3_inference

# Shared dataset registry (lives at repo root, one level above sam3d/).
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
import lift4d_datasets as ds_registry


SH_C0 = 0.28209479177387814
R_PYTORCH3D_TO_CAM = None  # lazy-initialized


def _get_r_p3d_to_cam(device):
    """Lazy-init the PyTorch3D → camera rotation matrix."""
    global R_PYTORCH3D_TO_CAM
    if R_PYTORCH3D_TO_CAM is None:
        import torch
        R_PYTORCH3D_TO_CAM = torch.tensor(
            [[-1, 0, 0], [0, -1, 0], [0, 0, 1]],
            dtype=torch.float32, device=device,
        )
    return R_PYTORCH3D_TO_CAM.to(device)


def render_gs_frame(gs_result, transform, focal_length, width, height,
                    device="cuda"):
    """
    Render a SAM3D Gaussian result dict to an RGB image.

    Args:
        gs_result: The inference result dict (must contain 'gs' key)
        transform: dict with scale, translation, rotation tensors
                   (or None to render in object space)
        focal_length: Camera focal length in pixels
        width, height: Output image dimensions
        device: CUDA device

    Returns:
        rendered: (H, W, 3) numpy uint8 array
    """
    import torch
    from gsplat import rasterization
    from sam3d_objects.utils.visualization import SceneVisualizer
    from pytorch3d.transforms import quaternion_multiply, quaternion_invert

    gs = gs_result['gs']
    xyz = gs.get_xyz.to(device)
    rotation = gs.get_rotation.to(device)
    scaling = gs.get_scaling.to(device)
    opacity = gs.get_opacity.to(device)
    features_dc = gs._features_dc.to(device)

    # Apply object → camera transform
    if transform is not None:
        scale_t = transform["scale"].to(device)
        trans_t = transform["translation"].to(device)
        rot_t = transform["rotation"].to(device)

        PC = SceneVisualizer.object_pointcloud(
            points_local=xyz.unsqueeze(0),
            quat_l2c=rot_t,
            trans_l2c=trans_t,
            scale_l2c=scale_t,
        )
        xyz = PC.points_list()[0]
        xyz = xyz @ _get_r_p3d_to_cam(device).T

        rot_quat = rot_t.squeeze()
        rotation = quaternion_multiply(
            quaternion_invert(rot_quat.unsqueeze(0).expand(rotation.shape[0], -1)),
            rotation,
        )
        scaling = scaling * scale_t.squeeze()

    # Convert SH → RGB
    if features_dc.dim() == 3:
        dc = features_dc.squeeze(1)
    else:
        dc = features_dc
    rgb_colors = torch.clamp(dc * SH_C0 + 0.5, 0.0, 1.0)

    # Camera intrinsics
    max_dim = max(width, height)
    Ks = torch.tensor([
        [focal_length, 0, max_dim / 2.0],
        [0, focal_length, max_dim / 2.0],
        [0, 0, 1],
    ], dtype=torch.float32, device=device).unsqueeze(0)
    viewmat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)
    bg = torch.ones(3, dtype=torch.float32, device=device)

    render_colors, render_alphas, _ = rasterization(
        means=xyz, quats=rotation, scales=scaling,
        opacities=opacity.squeeze(-1), colors=rgb_colors,
        sh_degree=None, viewmats=viewmat, Ks=Ks,
        width=max_dim, height=max_dim, backgrounds=bg,
    )

    rendered = render_colors.squeeze(0) * 255.0

    # Crop to target resolution
    if width > height:
        crop = (max_dim - height) // 2
        rendered = rendered[crop:crop + height, :width]
    else:
        crop = (max_dim - width) // 2
        rendered = rendered[:height, crop:crop + width]

    return rendered.cpu().numpy().clip(0, 255).astype(np.uint8)


def _mask_to_square_canvas(mask, width, height, size, device):
    """GT mask -> float target on the square render canvas, at `size` px."""
    import torch
    m = torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=device)
    if m.ndim == 3:
        m = m[..., 0]
    m = (m > (127.0 if float(m.max()) > 1.5 else 0.5)).float()
    max_dim = max(width, height)
    canvas = torch.zeros((max_dim, max_dim), dtype=torch.float32, device=device)
    if width > height:
        top = (max_dim - height) // 2
        canvas[top:top + height, :width] = m
    else:
        left = (max_dim - width) // 2
        canvas[:height, left:left + width] = m
    return torch.nn.functional.interpolate(
        canvas[None, None], size=(size, size), mode="area")[0, 0]


def refine_pose_to_mask(gs, transform, mask, focal_length, width, height,
                        device="cuda", schedule=((256, 80), (512, 60))):
    """Refine one frame's object pose so its silhouette matches the input mask.

    SAM3D predicts each frame's pose from a scale/shift-invariant pointmap. The
    raw prediction places the object at the right depth but oversized, so the
    reconstruction does not reproject onto the video object. The released
    pipeline handles this for single images with `layout_post_optimization`
    (ICP + silhouette render-compare, enabled by default in `run()`), but that
    path is unavailable here: pipeline.yaml wires no
    `layout_post_optimization_method`, and its open3d ICP is not usable on all
    platforms.

    This is the equivalent step done with gsplat: optimize scale/translation
    (then also rotation) to maximize a soft IoU between the rendered silhouette
    and the mask, coarse-to-fine, under the SAME intrinsics the pipeline uses
    downstream. It consumes only pipeline inputs (the frame's own object mask).

    Returns (transform dict with the same keys/shapes, final hard IoU).
    """
    import torch
    from gsplat import rasterization
    from pytorch3d.transforms import (quaternion_multiply, quaternion_invert,
                                      quaternion_to_matrix)
    from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform

    xyz = gs.get_xyz.to(device).float().detach()
    quats0 = gs.get_rotation.to(device).float().detach()
    scaling0 = gs.get_scaling.to(device).float().detach()
    opacity = gs.get_opacity.to(device).float().detach().squeeze(-1)
    colors = torch.ones((xyz.shape[0], 3), dtype=torch.float32, device=device)

    scale0 = transform["scale"].to(device).float().detach()
    trans0 = transform["translation"].to(device).float().detach()
    quat0 = transform["rotation"].to(device).float().detach()

    max_dim = max(width, height)
    viewmat = torch.eye(4, dtype=torch.float32, device=device).unsqueeze(0)

    log_s = torch.zeros(1, device=device, requires_grad=True)
    dt = torch.zeros(3, device=device, requires_grad=True)
    dq = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, requires_grad=True)

    def current():
        scale_t = scale0 * torch.exp(log_s)
        quat_t = quaternion_multiply(dq / dq.norm(), quat0)
        return scale_t, quat_t, trans0 + dt

    def render_alpha(size):
        scale_t, quat_t, trans_t = current()
        tfm = compose_transform(scale=scale_t, rotation=quaternion_to_matrix(quat_t),
                                translation=trans_t)
        pts = tfm.transform_points(xyz.unsqueeze(0))[0] @ _get_r_p3d_to_cam(device).T
        g_quats = quaternion_multiply(
            quaternion_invert(quat_t.squeeze().unsqueeze(0).expand(quats0.shape[0], -1)),
            quats0)
        g_scales = scaling0 * scale_t.squeeze()
        f = focal_length * size / max_dim
        Ks = torch.tensor([[f, 0, size / 2.0], [0, f, size / 2.0], [0, 0, 1]],
                          dtype=torch.float32, device=device).unsqueeze(0)
        _, alphas, _ = rasterization(
            means=pts.float(), quats=g_quats.float(), scales=g_scales.float(),
            opacities=opacity, colors=colors, sh_degree=None, viewmats=viewmat,
            Ks=Ks, width=size, height=size)
        return alphas.squeeze(0).squeeze(-1)

    def soft_iou_loss(a, target):
        inter = (a * target).sum()
        return 1.0 - inter / (a.sum() + target.sum() - inter + 1e-6)

    for size, iters in schedule:
        target = _mask_to_square_canvas(mask, width, height, size, device)
        # stage 1 places the object (translation+scale), stage 2 adds rotation
        for params, lr, n in ((( dt, log_s), 1e-2, iters),
                              ((dq, dt, log_s), 4e-3, iters)):
            opt = torch.optim.Adam(list(params), lr=lr)
            prev = None
            for _ in range(n):
                opt.zero_grad()
                loss = soft_iou_loss(render_alpha(size), target)
                loss.backward()
                opt.step()
                if prev is not None and abs(loss.item() - prev) < 1e-7:
                    break
                prev = loss.item()

    with torch.no_grad():
        size = schedule[-1][0]
        target = _mask_to_square_canvas(mask, width, height, size, device)
        a = render_alpha(size) > 0.5
        g = target > 0.5
        final_iou = float((a & g).sum()) / max(float((a | g).sum()), 1.0)
        scale_t, quat_t, trans_t = current()
    return ({"scale": scale_t.detach(), "translation": trans_t.detach(),
             "rotation": quat_t.detach()}, final_iou)


def load_gaussian_from_ply(ply_path, device="cuda"):
    """Load a SAM3D Gaussian from a PLY file."""
    from sam3d_objects.model.backbone.tdfy_dit.representations.gaussian.gaussian_model import Gaussian

    gs = Gaussian(
        aabb=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        sh_degree=0, mininum_kernel_size=0.0,
        scaling_bias=0.01, opacity_bias=0.1,
        scaling_activation="exp", device=device,
    )
    gs.load_ply(ply_path)
    return gs


def save_comparison_video(
    output_base,
    frame_images,
    output_path,
    focal_length,
    frame_labels=None,
    fps=8,
):
    """
    Render all SAM3D GS from disk and save a side-by-side comparison video.

    Loads PLY + obj_transform.json from each frame directory under
    output_base. Each frame uses its own SAM3D-inferred transform.

    Args:
        output_base: Path containing per-frame subdirectories
        frame_images: List of original frame numpy arrays
        output_path: Path to write the .mp4 file
        focal_length: Camera focal length in pixels
        fps: Video frame rate
    """
    import torch
    import imageio
    import cv2

    # Find every frame dir that actually has a reconstruction (robust to naming),
    # and sort NUMERICALLY — dirs are named '0','1','10',... so a lexical sort would
    # order them 0,1,10,2,... and misalign the GT/rendered halves from frame 10 on.
    output_base = Path(output_base)
    frame_dirs = [d for d in output_base.iterdir()
                  if d.is_dir() and (d / "result.ply").exists()]
    if not frame_dirs:
        logger.warning(f"No frame dirs with result.ply under {output_base}; skipping render_video")
        return

    def _num_key(d):
        try:
            return (0, int(d.name))
        except ValueError:
            return (1, d.name)
    frame_dirs.sort(key=_num_key)

    # Map each frame label -> its GT image so every reconstruction pairs with the
    # correct frame (falls back to positional order if labels are unavailable).
    label_to_gt = {}
    if frame_labels is not None:
        for lbl, img in zip(frame_labels, frame_images):
            label_to_gt[str(lbl)] = img

    logger.info(f"Rendering comparison video ({len(frame_dirs)} frames) → {output_path}")
    writer = imageio.get_writer(str(output_path), format="FFMPEG", fps=fps)

    for pos, frame_dir in enumerate(frame_dirs):
        gt_img = label_to_gt.get(frame_dir.name)
        if gt_img is None:
            gt_img = frame_images[pos] if pos < len(frame_images) else frame_images[-1]
        h, w = gt_img.shape[:2]

        # Render this frame's Gaussian with SAM3D's own per-frame focal — the geometry
        # was reconstructed with it, so any other focal misaligns the overlay.
        gs_result = {'gs': load_gaussian_from_ply(str(frame_dir / "result.ply"))}
        transform = None
        frame_focal = focal_length
        transform_path = frame_dir / "obj_transform.json"
        if transform_path.exists():
            with open(transform_path) as f:
                tf_data = json.load(f)
            frame_focal = float(tf_data.get("focal", focal_length))
            transform = {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in tf_data.items() if k != "focal"
            }

        rendered = render_gs_frame(gs_result, transform, frame_focal, w, h)

        gt_rgb = gt_img[:, :, :3] if gt_img.ndim == 3 else gt_img
        if gt_rgb.dtype != np.uint8:
            gt_rgb = gt_rgb.astype(np.uint8)
        if rendered.shape[:2] != (h, w):
            rendered = cv2.resize(rendered, (w, h))

        writer.append_data(np.concatenate([gt_rgb, rendered], axis=1))

    writer.close()
    logger.info(f"Comparison video saved: {output_path}")


def _render_comparison(output_base, frame_images, frame_labels=None):
    """Render comparison video from saved PLY/transform files on disk.

    Each frame renders with its own SAM3D focal from obj_transform.json; the
    max(H,W)*1.2 value below is only a fallback for frames missing that estimate.
    """
    h, w = frame_images[0].shape[:2]
    fallback_focal = max(h, w) * 1.2

    video_path = output_base / "comparison.mp4"
    save_comparison_video(
        output_base=output_base,
        frame_images=frame_images,
        output_path=video_path,
        focal_length=fallback_focal,
        frame_labels=frame_labels,
    )


def run_video_inference_cli(
    input_path: Path,
    mask_input_path: Path,
    image_names: Optional[List[str]] = None,
    mask_names: Optional[List[str]] = None,
    seed: int = 42,
    stage1_steps: int = 50,
    stage2_steps: int = 25,
    decode_formats: List[str] = None,
    mesh_postprocess: bool = False,
    texture_baking: bool = False,
    vertex_color: bool = True,
    model_tag: str = "hf",
    initial_frame_index: int = 0,
    consistency_strength: float = 0.3,
    output_name: Optional[str] = None,
    render_video: bool = False,
    output_dir: str = "visualization",
    run_da3: bool = False,
    da3_npz_path: Optional[str] = None,
    refine_pose: bool = False,
):
    """
    Run video inference - generates one mesh per frame

    Args:
        input_path: Directory containing frame images
        mask_input_path: Directory containing mask images
        image_names: Optional list of frame names (without extension) to load in order.
        mask_names: Optional list of mask names (without extension) to load.
                   If None and image_names is provided, uses image_names for masks.
        initial_frame_index: Frame to start from (propagates bi-directionally).
        consistency_strength: SDEdit temporal consistency strength [0.0, 1.0].
        output_name: Optional subdirectory name under output_dir for results.
        render_video: If True, render a side-by-side comparison video after
                      inference.
        run_da3: If True (and no da3_npz_path), run Depth Anything 3 over the
                 selected frames first, saving <output>/da3/da3_output.npz.
        da3_npz_path: Existing DA3 npz whose pointmaps align each frame's pose
                 estimate (frame order must match the processed frames).
    """
    config_path = os.path.join(_SAM3D_DIR, "checkpoints", model_tag, "pipeline.yaml")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")

    # Load video frames and masks
    logger.info(f"Loading video frames from: {input_path}")
    logger.info(f"Loading masks from: {mask_input_path}")

    frame_images, frame_masks = load_video_frames_and_masks(
        frames_dir=input_path,
        masks_dir=mask_input_path,
        image_names=image_names,
        mask_names=mask_names,
    )

    num_frames = len(frame_images)
    logger.info(f"Loaded {num_frames} frames")

    decode_formats = decode_formats or ["gaussian", "mesh"]

    # Compute output directory
    visualization_dir = Path(output_dir)
    visualization_dir.mkdir(parents=True, exist_ok=True)
    if output_name:
        output_base = visualization_dir / output_name
    else:
        output_base = visualization_dir / f"{input_path.name}_video_streaming"
    output_base.mkdir(parents=True, exist_ok=True)

    if da3_npz_path is None and run_da3:
        da3_dir = output_base / "da3"
        selected_paths = _resolve_frame_paths(input_path, image_names)
        if selected_paths is not None:
            run_da3_inference(image_files=selected_paths, output_dir=str(da3_dir))
        else:
            # No explicit frame list: DA3's numeric-sorted dir glob matches the
            # frame loader's ordering; cap at the number of frames loaded.
            run_da3_inference(image_dir=str(input_path), output_dir=str(da3_dir),
                              max_frames=num_frames)
        da3_npz_path = str(da3_dir / "da3_output.npz")

    da3_pointmaps = None
    da3_intrinsics = None
    da3_focals = None
    if da3_npz_path and Path(da3_npz_path).exists():
        import torch
        da3_data = np.load(da3_npz_path, allow_pickle=True)
        da3_pm = da3_data["pointmaps"]  # (N, H, W, 3)
        if da3_pm.shape[0] != num_frames:
            logger.warning(
                f"DA3 npz has {da3_pm.shape[0]} frames but {num_frames} are being "
                f"processed; skipping DA3 pose alignment")
        else:
            da3_pointmaps = [
                torch.tensor(da3_pm[i], dtype=torch.float32)
                for i in range(da3_pm.shape[0])
            ]
            # Pixel-unit fy at the original frame resolution, one per frame.
            da3_focals = [float(K[1, 1]) for K in da3_data["intrinsics"]]
            # The pipeline works in NORMALIZED intrinsics (fx/W, fy/H, cx/W,
            # cy/H). Pass DA3's own — inferring them from the pointmap gives a
            # negative focal, since the pointmap is already axis-flipped.
            da3_intrinsics = []
            for i, K in enumerate(da3_data["intrinsics"]):
                fh, fw = da3_pm[i].shape[0], da3_pm[i].shape[1]
                Kn = np.array(K, dtype=np.float32).copy()
                Kn[0, 0] /= fw; Kn[0, 2] /= fw
                Kn[1, 1] /= fh; Kn[1, 2] /= fh
                da3_intrinsics.append(torch.tensor(Kn, dtype=torch.float32))
            logger.info(f"Loaded {len(da3_pointmaps)} DA3 pointmaps for pose alignment")

    logger.info(f"Loading model: {config_path}")
    inference = Inference(config_path, compile=False)

    if hasattr(inference._pipeline, 'rendering_engine'):
        if inference._pipeline.rendering_engine != "pytorch3d":
            logger.warning(f"Rendering engine is set to {inference._pipeline.rendering_engine}, changing to pytorch3d")
            inference._pipeline.rendering_engine = "pytorch3d"

    results = inference._pipeline.run_video_inference(
        frame_images=frame_images,
        frame_masks=frame_masks,
        seed=seed,
        stage1_inference_steps=stage1_steps,
        stage2_inference_steps=stage2_steps,
        decode_formats=decode_formats,
        with_mesh_postprocess=mesh_postprocess,
        with_texture_baking=texture_baking,
        use_vertex_color=vertex_color,
        initial_frame_index=initial_frame_index,
        consistency_strength=consistency_strength,
        pointmaps=da3_pointmaps,
        pointmap_intrinsics=da3_intrinsics,
    )

    print(f"\n{'='*60}")
    print(f"Video inference completed!")
    print(f"Processed {num_frames} frames")
    print(f"{'='*60}")

    def frame_focal(result):
        fi = result["frame_idx"]
        gt_h, gt_w = frame_images[fi].shape[:2]
        if da3_focals is not None:
            return da3_focals[fi]
        # No DA3: use the depth model's own normalized intrinsics
        # (f_px = fy_norm * H); fall back to a resolution-relative guess.
        intr = result.get("intrinsics", None)
        if intr is not None:
            f_px = float(np.asarray(intr.detach().cpu()).reshape(3, 3)[1, 1] * gt_h)
            if f_px > 0:
                return f_px
        return max(gt_h, gt_w) * 1.2

    # Silhouette pose refinement (see refine_pose_to_mask): the raw predicted
    # pose is systematically oversized, so without this the reconstruction does
    # not reproject onto the video object.
    refined = {}
    if refine_pose:
        ious = []
        for result in results:
            fi = result["frame_idx"]
            gs = result.get("gs") or (result.get("gaussian") or [None])[0]
            if gs is None or frame_masks[fi] is None:
                continue
            gt_h, gt_w = frame_images[fi].shape[:2]
            focal = frame_focal(result)
            try:
                tf = {k: result[k] for k in ("scale", "translation", "rotation")}
                new_tf, iou = refine_pose_to_mask(
                    gs, tf, frame_masks[fi], focal, gt_w, gt_h)
                refined[fi] = new_tf
                ious.append(iou)
            except Exception as e:
                logger.warning(f"Frame {fi}: pose refinement failed ({e}); keeping raw pose")
            if len(ious) % 20 == 0 and ious:
                logger.info(f"  refined {len(ious)}/{num_frames} frames "
                            f"(running mean IoU {np.mean(ious):.4f})")
        if ious:
            logger.info(f"Pose refinement: mean silhouette IoU {np.mean(ious):.4f} "
                        f"over {len(ious)} frames")

    # Save outputs for each frame
    for result in results:
        frame_idx = result["frame_idx"]
        if image_names is not None and 0 <= frame_idx < len(image_names):
            frame_label = image_names[frame_idx]
        else:
            frame_label = f"frame_{frame_idx:04d}"
        frame_dir = output_base / frame_label
        frame_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []

        if 'glb' in result and result['glb'] is not None:
            output_path = frame_dir / "result.glb"
            result['glb'].export(str(output_path))
            saved_files.append("result.glb")

        if 'gs' in result:
            output_path = frame_dir / "result.ply"
            result['gs'].save_ply(str(output_path))
            saved_files.append("result.ply")
        elif 'gaussian' in result:
            if isinstance(result['gaussian'], list) and len(result['gaussian']) > 0:
                output_path = frame_dir / "result.ply"
                result['gaussian'][0].save_ply(str(output_path))
                saved_files.append("result.ply")

        # Save object transforms JSON for this frame
        src = refined.get(frame_idx, result)
        obj_transforms = {}
        for key in ('scale', 'translation', 'rotation'):
            if key in src:
                obj_transforms[key] = src[key].cpu().numpy().tolist()

        obj_transforms["focal"] = frame_focal(result)

        if obj_transforms:
            transform_path = frame_dir / "obj_transform.json"
            with open(transform_path, 'w') as f:
                json.dump(obj_transforms, f, indent=4)
            saved_files.append("obj_transform.json")

        print(f"Frame {frame_label}: saved {', '.join(saved_files)} to {frame_dir}")

    # Render comparison video if requested
    if render_video:
        _render_comparison(output_base, frame_images, frame_labels=image_names)

    print(f"\n{'='*60}")
    print(f"All output files saved to: {output_base}")
    print(f"{'='*60}")


def _resolve_frame_paths(frames_dir: Path, image_names: Optional[List[str]]) -> Optional[List[str]]:
    """Resolve ordered frame stems to image paths (for DA3). None if no stems."""
    if image_names is None:
        return None
    paths = []
    for stem in image_names:
        for ext in ('.png', '.jpg', '.jpeg', '.webp', '.bmp'):
            p = Path(frames_dir) / f"{stem}{ext}"
            if p.exists():
                paths.append(str(p))
                break
        else:
            raise FileNotFoundError(f"No image file for frame stem '{stem}' in {frames_dir}")
    return paths


def _discover_stems(frames_dir: Path) -> List[str]:
    """Return frame filename stems in numeric order (falls back to lexical)."""
    frame_files = []
    for ext in ('.png', '.jpg', '.jpeg'):
        frame_files.extend(frames_dir.glob(f"*{ext}"))

    def _sort_key(p):
        try:
            return (0, int(p.stem))
        except ValueError:
            return (1, p.stem)
    frame_files = sorted(set(frame_files), key=_sort_key)
    return [f.stem for f in frame_files]


def _select_stems(stems: List[str], args, initial_frame_index: int):
    """Apply --max_frames, --subsampling, --num_frames and remap the start index."""
    if args.max_frames > 0:
        stems = stems[:args.max_frames]
    if args.subsampling > 1:
        stems = stems[::args.subsampling]
    if args.num_frames > 0:
        stems = stems[:args.num_frames]
    # --initial_frame_index is in source-frame coords; remap into subsampled list.
    if args.subsampling > 1 and initial_frame_index > 0:
        remapped = min(initial_frame_index // args.subsampling, len(stems) - 1)
        logger.info(f"Remapped --initial_frame_index {initial_frame_index} -> {remapped} "
                    f"(stem '{stems[remapped]}')")
        initial_frame_index = remapped
    return stems, initial_frame_index


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3D Objects Video Inference Script - one 3D reconstruction per frame with temporal conditioning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_inference.py --dataset consistent4d --object_name pistol --render_video
  python run_inference.py --dataset custom --object_name my_clip --mask_name box \\
      --subsampling 8 --video_consistency 0.1 --render_video
        """
    )

    parser.add_argument(
        "--video",
        action="store_true",
        help="Video mode (the only supported mode; kept for compatibility)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(ds_registry.DATASETS),
        help="Dataset type: consistent4d | davis | custom. When set, frames/masks "
             "and the output location are resolved automatically from --data_root."
    )
    parser.add_argument(
        "--object_name",
        type=str,
        default=None,
        help="Object name (consistent4d/davis) or video folder name (custom). "
             "Used with --dataset."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="Base data directory (overrides $LIFT4D_DATA_ROOT). "
             "Contains consistent4d/, DAVIS/, custom/."
    )
    parser.add_argument(
        "--sam3d_out",
        type=str,
        default=None,
        help="Root for stage-1 outputs (overrides $LIFT4D_SAM3D_OUT, "
             "default <repo>/sam3d_output). Stage 2 reads from here."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        default=None,
        help="Custom mode: directory containing frame images, or a video directory "
             "with frames/ and masks/ subdirectories. Use instead of --dataset."
    )
    parser.add_argument(
        "--mask_input_path",
        type=str,
        default=None,
        help="Directory containing mask images. Auto-set to <input_path>/masks "
             "when that subdirectory exists."
    )
    parser.add_argument(
        "--mask_name",
        type=str,
        default=None,
        help="Mask object name suffix (e.g., 'bulldozer_2'). Auto-discovers "
             "frames and matches masks as <frame_stem>_<mask_name>."
    )
    parser.add_argument(
        "--subsampling",
        type=int,
        default=1,
        help="Frame stride for auto-discovery (e.g., 8 keeps every 8th frame). "
             "--initial_frame_index is remapped accordingly."
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Cap on source frames considered before subsampling (0=all). "
             "With --max_frames 40 --subsampling 4 you get 10 outputs."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=100,
        help="Cap on total frames processed, applied after subsampling "
             "(0=all). Default 100 keeps runs bounded on long videos."
    )
    parser.add_argument(
        "--initial_frame_index",
        type=int,
        default=0,
        help="Source-frame index to start processing from "
             "(propagates bi-directionally). Default: 0"
    )
    parser.add_argument(
        "--video_consistency",
        type=float,
        default=0.3,
        help="Strength of temporal consistency [0.0, 1.0]. Higher = more "
             "structure preserved, Lower = more deformation allowed. Default: 0.3"
    )
    parser.add_argument(
        "--render_video",
        action="store_true",
        default=False,
        help="After inference, render all SAM3D GS and save a side-by-side "
             "comparison video (original | rendered) to the output directory. "
             "If all frame PLYs already exist, skips inference and only renders."
    )

    parser.add_argument(
        "--run_da3",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Depth Anything 3 over the selected frames before SAM3D and "
             "save <output>/da3/da3_output.npz (cached across runs). The DA3 "
             "pointmaps align each frame's pose estimate, and the npz gives "
             "stage 3 its camera focal and per-frame scene depths for "
             "--occlusion_compositing. Use --no-run_da3 to disable."
    )
    parser.add_argument(
        "--da3_npz",
        type=str,
        default=None,
        help="Use an existing da3_output.npz instead of running DA3 (frame "
             "order must match the processed frames)."
    )
    parser.add_argument(
        "--refine_pose",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Refine each frame's pose so the reconstruction reprojects onto "
             "its own input mask (silhouette render-compare; the equivalent of "
             "SAM3D's layout post-optimization). Off by default; enable it when "
             "the predicted pose is oversized and the render does not line up "
             "with the video."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--stage1_steps",
        type=int,
        default=50,
        help="Stage 1 inference steps (default: 50)"
    )
    parser.add_argument(
        "--stage2_steps",
        type=int,
        default=25,
        help="Stage 2 inference steps (default: 25)"
    )
    parser.add_argument(
        "--decode_formats",
        type=str,
        default="gaussian,mesh",
        help="Decode formats, comma-separated, e.g., 'gaussian,mesh' or 'gaussian' (default: gaussian,mesh)"
    )
    parser.add_argument(
        "--mesh_postprocess",
        action="store_true",
        help="Enable mesh post-processing (simplification, hole filling, etc.)"
    )
    parser.add_argument(
        "--texture_baking",
        action="store_true",
        help="Enable texture baking (generate textured GLB)"
    )
    parser.add_argument(
        "--vertex_color",
        action="store_true",
        default=True,
        help="Use vertex colors (default: True, mutually exclusive with --no_vertex_color)"
    )
    parser.add_argument(
        "--no_vertex_color",
        action="store_false",
        dest="vertex_color",
        help="Do not use vertex colors (mutually exclusive with --vertex_color)"
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="hf",
        help="Model tag (default: hf)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="visualization",
        help="Base output directory for results (default: visualization/). "
             "Overridden to the video directory when frames/ + masks/ "
             "subdirectories are detected."
    )

    args = parser.parse_args()

    # Point the shared registry at the user's roots, if given.
    if args.data_root:
        os.environ["LIFT4D_DATA_ROOT"] = os.path.abspath(args.data_root)
    if args.sam3d_out:
        os.environ["LIFT4D_SAM3D_OUT"] = os.path.abspath(args.sam3d_out)

    image_names = None
    mask_names = None
    initial_frame_index = args.initial_frame_index

    if args.dataset is not None:
        # ---- Dataset mode: resolve frames/masks/output from the registry ----
        if not args.object_name:
            parser.error("--dataset requires --object_name")
        spec = ds_registry.resolve(args.dataset, args.object_name, mask_name=args.mask_name)
        input_path = Path(spec.frames_dir)
        if not input_path.is_dir():
            raise FileNotFoundError(f"Frames dir not found: {input_path}")

        all_stems = _discover_stems(input_path)
        if not all_stems:
            raise FileNotFoundError(f"No frames found in {input_path}")
        frame_stems, initial_frame_index = _select_stems(all_stems, args, initial_frame_index)
        image_names = frame_stems

        # Wire masks according to how this dataset stores them.
        if spec.mask_mode == ds_registry.MASK_ALPHA:
            mask_input_path = input_path            # mask is the RGBA alpha channel
            mask_names = frame_stems
        elif spec.mask_mode == ds_registry.MASK_DIR:
            mask_input_path = Path(spec.masks_dir)  # parallel Annotations dir
            mask_names = frame_stems
        else:  # MASK_SUFFIX (custom)
            if not args.mask_name:
                parser.error("--dataset custom requires --mask_name")
            mask_input_path = Path(spec.masks_dir)
            mask_names = [f"{s}_{args.mask_name}" for s in frame_stems]

        # Write where stage 2 (train_lift4d_scgs.py) will look for it.
        output_base_dir = spec.sam3d_output_dir()
        output_dir = str(Path(output_base_dir).parent)
        output_name = spec.tag
        logger.info(f"[{args.dataset}] {args.object_name}: {len(frame_stems)} frames, "
                    f"mask_mode={spec.mask_mode}, out={output_base_dir}")
    else:
        # ---- Custom --input_path mode ----
        if not args.input_path:
            parser.error("provide --dataset (+ --object_name) or --input_path")
        input_path = Path(args.input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input path does not exist: {input_path}")

        mask_input_path = Path(args.mask_input_path) if args.mask_input_path else None
        output_dir = args.output_dir
        output_name = None

        # If input_path has frames/ + masks/ subdirs, redirect to frames/ and
        # auto-set masks/. Anchor outputs under the video dir as sdedit_{consistency}/.
        frames_subdir = input_path / "frames"
        masks_subdir = input_path / "masks"
        if frames_subdir.is_dir():
            logger.info(f"Detected frames/ subdir, using {frames_subdir}")
            video_root = input_path
            input_path = frames_subdir
            if mask_input_path is None and masks_subdir.is_dir():
                mask_input_path = masks_subdir
                logger.info(f"Auto-set --mask_input_path to {masks_subdir}")
            output_dir = str(video_root)
            output_name = f"sdedit_{args.video_consistency}"
            if args.mask_name:
                output_name = f"{output_name}/{args.mask_name}"
            logger.info(f"Outputs will go to {video_root / output_name}")

        # --mask_name: auto-generate image_names and mask_names from frames dir
        if args.mask_name:
            frame_stems = _discover_stems(input_path)
            if not frame_stems:
                raise FileNotFoundError(f"No frames found in {input_path}")
            frame_stems, initial_frame_index = _select_stems(
                frame_stems, args, initial_frame_index)
            image_names = frame_stems
            mask_names = [f"{stem}_{args.mask_name}" for stem in frame_stems]
            logger.info(f"--mask_name '{args.mask_name}': {len(frame_stems)} frames "
                        f"(max_frames={args.max_frames}, subsampling={args.subsampling}), "
                        f"masks like '{mask_names[0]}' ... '{mask_names[-1]}'")

    decode_formats = [fmt.strip() for fmt in args.decode_formats.split(",") if fmt.strip()]
    if not decode_formats:
        decode_formats = ["gaussian", "mesh"]

    # Check if we can skip inference and just render from existing PLYs
    if args.render_video and output_name:
        output_base = Path(output_dir) / output_name
        if output_base.exists():
            frame_dirs = sorted(
                d for d in output_base.iterdir()
                if d.is_dir() and (d / "result.ply").exists()
                and (d / "obj_transform.json").exists()
            )
            expected = len(image_names) if image_names is not None else len(frame_dirs)
            if frame_dirs and len(frame_dirs) < expected:
                logger.warning(
                    f"Found only {len(frame_dirs)}/{expected} complete frames in "
                    f"{output_base}; re-running inference for the full sequence"
                )
                frame_dirs = []
            if frame_dirs:
                logger.info(
                    f"Found {len(frame_dirs)} existing frame PLYs in "
                    f"{output_base}, skipping inference"
                )
                # Load GT frames for side-by-side
                frame_images, _ = load_video_frames_and_masks(
                    frames_dir=input_path,
                    masks_dir=mask_input_path or input_path,
                    image_names=image_names,
                    mask_names=mask_names,
                )
                _render_comparison(output_base, frame_images, frame_labels=image_names)
                print(f"\nComparison video saved to: {output_base / 'comparison.mp4'}")
                return

    if mask_input_path is None:
        parser.error("Video mode requires --mask_input_path "
                     "(or a masks/ subdirectory next to frames/)")

    run_video_inference_cli(
        input_path=input_path,
        mask_input_path=mask_input_path,
        image_names=image_names,
        mask_names=mask_names,
        seed=args.seed,
        stage1_steps=args.stage1_steps,
        stage2_steps=args.stage2_steps,
        decode_formats=decode_formats,
        mesh_postprocess=args.mesh_postprocess,
        texture_baking=args.texture_baking,
        vertex_color=args.vertex_color,
        model_tag=args.model_tag,
        initial_frame_index=initial_frame_index,
        consistency_strength=args.video_consistency,
        output_name=output_name,
        render_video=args.render_video,
        output_dir=output_dir,
        run_da3=args.run_da3,
        da3_npz_path=args.da3_npz,
        refine_pose=args.refine_pose,
    )


if __name__ == "__main__":
    main()
