# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Video frame loading utilities for SAM 3D Objects
Loads paired frame images and masks from separate directories.
"""
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from loguru import logger


def load_image(path: Path) -> np.ndarray:
    """Load image as numpy array (always returns RGB, strips alpha if present)."""
    img = Image.open(path)
    img_array = np.array(img).astype(np.uint8)
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        # RGBA: extract RGB, composite on white background where alpha=0
        alpha = img_array[..., 3:4].astype(np.float32) / 255.0
        rgb = img_array[..., :3].astype(np.float32)
        img_array = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    return img_array


def load_mask(path: Path) -> np.ndarray:
    """
    Load mask from image file
    
    Args:
        path: Mask image file path
        
    Returns:
        mask: Binary mask, shape (H, W), bool format
    """
    img = Image.open(path)
    img_array = np.array(img)
    
    if img.mode == 'RGBA' and img_array.ndim == 3 and img_array.shape[2] >= 4:
        mask = img_array[..., 3] > 0
    elif img.mode == 'L':
        mask = img_array > 0
    elif img.mode == 'RGB' and img_array.ndim == 3:
        # Grayscale stored as RGB
        mask = img_array[..., 0] > 0
    else:
        # Fallback: use any non-zero value
        if img_array.ndim == 3:
            mask = img_array.any(axis=-1)
        else:
            mask = img_array > 0
    
    return mask


def load_video_frames_and_masks(
    frames_dir: Path,
    masks_dir: Path,
    image_names: Optional[List[str]] = None,
    mask_names: Optional[List[str]] = None,
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg'),
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load video frames and corresponding masks from directories.
    
    Files are matched by filename (without extension) and sorted alphabetically.
    This ensures frame_0000.png matches with frame_0000.png in masks directory.
    
    Args:
        frames_dir: Directory containing frame images
        masks_dir: Directory containing mask images (same filenames as frames)
        image_names: Optional list of frame names (without extension) to load in order.
                    If provided, only these frames are loaded in the given order.
        mask_names: Optional list of mask names (without extension) to load.
                   If None and image_names is provided, uses image_names for masks.
        extensions: Allowed file extensions
        
    Returns:
        frames: List of frame images (numpy arrays, uint8)
        masks: List of mask images (numpy arrays, bool)
    """
    frames_dir = Path(frames_dir)
    masks_dir = Path(masks_dir)
    
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory does not exist: {masks_dir}")
    
    # If image_names provided, use them; otherwise discover all frames
    if image_names is not None:
        logger.info(f"Loading {len(image_names)} specified frames")
        frame_stems = image_names
    else:
        # Get all frame files and extract stems
        frame_files = []
        for ext in extensions:
            frame_files.extend(frames_dir.glob(f"*{ext}"))
            frame_files.extend(frames_dir.glob(f"*{ext.upper()}"))
        
        # Sort numerically if stems are digits, otherwise alphabetically
        def _sort_key(p):
            try:
                return (0, int(p.stem))
            except ValueError:
                return (1, p.stem)
        frame_files = sorted(set(frame_files), key=_sort_key)
        
        if len(frame_files) == 0:
            raise ValueError(f"No image files found in {frames_dir}")
        
        frame_stems = [f.stem for f in frame_files]
        logger.info(f"Found {len(frame_stems)} frames in {frames_dir}")
    
    # Use mask_names if provided, otherwise use frame stems
    if mask_names is not None:
        if len(mask_names) != len(frame_stems):
            raise ValueError(f"Number of mask_names ({len(mask_names)}) must match number of image_names ({len(frame_stems)})")
        mask_stem_list = mask_names
    else:
        mask_stem_list = frame_stems
    
    frames = []
    masks = []
    
    for frame_stem, mask_stem in zip(frame_stems, mask_stem_list):
        # Find frame file
        frame_path = None
        for ext in extensions:
            candidate = frames_dir / f"{frame_stem}{ext}"
            if candidate.exists():
                frame_path = candidate
                break
            candidate = frames_dir / f"{frame_stem}{ext.upper()}"
            if candidate.exists():
                frame_path = candidate
                break
        
        if frame_path is None:
            logger.warning(f"Frame file not found for '{frame_stem}', skipping")
            continue
        
        # Find matching mask file
        mask_path = None
        for ext in extensions:
            candidate = masks_dir / f"{mask_stem}{ext}"
            if candidate.exists():
                mask_path = candidate
                break
            candidate = masks_dir / f"{mask_stem}{ext.upper()}"
            if candidate.exists():
                mask_path = candidate
                break
        
        if mask_path is None:
            logger.warning(f"No mask found for '{mask_stem}', skipping")
            continue
        
        try:
            frame = load_image(frame_path)
            mask = load_mask(mask_path)
            
            frames.append(frame)
            masks.append(mask)
            
            logger.debug(f"Loaded frame '{frame_stem}': image={frame.shape}, mask={mask.shape}")
            
        except Exception as e:
            logger.error(f"Failed to load frame '{frame_stem}': {e}")
            continue
    
    if len(frames) == 0:
        raise ValueError(f"No valid frame/mask pairs found")
    
    logger.info(f"Successfully loaded {len(frames)} frame/mask pairs")
    return frames, masks
