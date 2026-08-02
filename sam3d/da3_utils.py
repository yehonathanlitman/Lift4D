"""
Depth Anything 3 (DA3) inference utilities.

Extracted from demo_video_multi_object.ipynb for reuse in run_inference.py
and other scripts.
"""
import os
import copy
import cv2
import numpy as np
import imageio
from pathlib import Path
from loguru import logger


def depth_to_pointmap(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth

    pointmap = np.stack([-x, -y, z], axis=-1)
    return pointmap


def run_da3_inference(
    image_dir: str = None,
    output_dir: str = None,
    process_res: int = 504,
    max_frames: int = 32,
    save_visualization: bool = False,
    device: str = "cuda",
    image_files: list = None,
):
    """
    Run DA3 on a folder (or explicit list) of images. Produces a
    da3_output.npz with depth, pointmaps, extrinsics, and intrinsics.

    Args:
        image_dir: Path to folder containing input images
        output_dir: Path to output directory
        process_res: Processing resolution (default: 504)
        max_frames: Maximum number of frames to process (dir mode only)
        save_visualization: Whether to save GLB and depth visualizations
        device: Device to run on ('cuda' or 'cpu')
        image_files: Explicit ordered list of image paths. When given,
            image_dir/max_frames are ignored and the npz frame order follows
            this list exactly — pass the same frame selection the SAM3D
            pipeline will process so npz index == pipeline frame index.

    Returns:
        Dictionary with depth, pointmaps, pointmaps_sam3d, extrinsics,
        intrinsics, image_files keys.
    """
    import torch
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.export import export_to_glb

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Resolve the frame list first so a cached result is validated against it.
    if image_files is not None:
        image_files = [Path(f) for f in image_files]
    else:
        if image_dir is None:
            raise ValueError("run_da3_inference requires image_dir or image_files")
        image_dir = Path(image_dir)
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.webp', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(ext))

        def natural_sort_key(path):
            stem = path.stem
            try:
                return (0, int(stem), stem)
            except ValueError:
                return (1, 0, stem)

        image_files = sorted(image_files, key=natural_sort_key)[:max_frames]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Cached result: reuse only if it covers exactly the requested frames.
    output_file = output_path / "da3_output.npz"
    if output_file.exists():
        data = np.load(output_file, allow_pickle=True)
        wanted = [os.path.basename(str(f)) for f in image_files]
        cached = ([os.path.basename(str(f)) for f in data["image_files"]]
                  if "image_files" in data.files else None)
        if cached == wanted:
            logger.info(f"Found cached DA3 output at {output_file}")
            return {k: data[k] for k in data.files}
        logger.info(f"Cached DA3 output at {output_file} covers different frames; recomputing")

    model = DepthAnything3.from_pretrained(
        "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    ).to(device)

    logger.info(f"Running DA3 on {len(image_files)} images")

    # Build export format
    export_format = "mini_npz"
    if save_visualization:
        export_format += "-glb-depth_vis"

    # Run inference
    prediction = model.inference(
        image=[str(f) for f in image_files],
        process_res=process_res,
        export_dir=str(output_path),
        export_format=export_format,
        ref_view_strategy="middle",
        show_cameras=True,
        num_max_points=1_000_000,
        conf_thresh_percentile=10,
    )

    # Read original image dimensions (handle RGBA by only using first 3 channels)
    first_img = imageio.imread(image_files[0])
    h, w = first_img.shape[0], first_img.shape[1]
    n_images, h_pred, w_pred = prediction.depth.shape

    sx = w / w_pred
    sy = h / h_pred

    intrinsics_rescaled = prediction.intrinsics.copy()
    for i in range(intrinsics_rescaled.shape[0]):
        K = intrinsics_rescaled[i]
        K[0, 0] *= sx  # fx
        K[1, 1] *= sy  # fy
        # Force principal point to image center at original resolution
        K[0, 2] = w / 2.0  # cx
        K[1, 2] = h / 2.0  # cy

    depth_resized = np.zeros((n_images, h, w), dtype=np.float32)
    conf_resized = np.zeros((n_images, h, w), dtype=np.float32)
    for i in range(n_images):
        depth_resized[i] = cv2.resize(
            prediction.depth[i], (w, h), interpolation=cv2.INTER_NEAREST
        )
        conf_resized[i] = cv2.resize(
            prediction.conf[i], (w, h), interpolation=cv2.INTER_NEAREST
        )
    prediction.intrinsics = intrinsics_rescaled
    prediction.depth = depth_resized
    prediction.conf = conf_resized

    if save_visualization:
        # Read images at original resolution for per-frame GLB export
        imgs_resized = np.zeros((n_images, h, w, 3), dtype=np.float32)
        for i in range(n_images):
            img = imageio.imread(image_files[i])
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            imgs_resized[i] = img
        prediction.processed_images = imgs_resized

        for img_idx, img_path in enumerate(image_files):
            frame_name = os.path.basename(img_path)
            frame_dir = os.path.join(output_dir, frame_name.split(".")[0])
            os.makedirs(frame_dir, exist_ok=True)

            prediction_individual = copy.deepcopy(prediction)
            prediction_individual.depth = prediction_individual.depth[img_idx:img_idx+1]
            prediction_individual.conf = prediction_individual.conf[img_idx:img_idx+1]
            prediction_individual.extrinsics = prediction_individual.extrinsics[img_idx:img_idx+1]
            prediction_individual.intrinsics = prediction_individual.intrinsics[img_idx:img_idx+1]
            prediction_individual.processed_images = prediction_individual.processed_images[img_idx:img_idx+1]
            export_to_glb(
                prediction_individual, frame_dir,
                num_max_points=1_000_000, conf_thresh_percentile=10,
            )

    # Extract results
    depth = prediction.depth
    extrinsics = prediction.extrinsics
    intrinsics = prediction.intrinsics

    N = depth.shape[0]
    pointmaps = []
    pointmaps_sam3d = []
    for i in range(N):
        pm = depth_to_pointmap(depth[i], intrinsics[i])
        pointmaps.append(pm)
        pointmaps_sam3d.append(pm.transpose(2, 0, 1))

    pointmaps = np.stack(pointmaps, axis=0)
    pointmaps_sam3d = np.stack(pointmaps_sam3d, axis=0)

    logger.info(f"DA3 Output: depth={depth.shape}, intrinsics={intrinsics.shape}, "
                f"focal_length={intrinsics[0, 1, 1]:.1f}")

    # Save
    np.savez(
        output_file,
        depth=depth,
        pointmaps=pointmaps,
        pointmaps_sam3d=pointmaps_sam3d,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        image_files=np.array([str(f) for f in image_files]),
        process_res=process_res,
    )
    logger.info(f"DA3 results saved to: {output_file}")

    # Clean up model to free GPU memory
    del model
    torch.cuda.empty_cache()

    return {
        "depth": depth,
        "pointmaps": pointmaps,
        "pointmaps_sam3d": pointmaps_sam3d,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "image_files": [str(f) for f in image_files],
    }
