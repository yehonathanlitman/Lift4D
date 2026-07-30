"""
Shared dataset registry for the Lift4D pipeline.

Both pipeline stages import this module so they agree on where data lives and
how masks are stored:

    stage 1  sam3d/run_inference.py   (per-frame SAM3D reconstruction)
    stage 2  lift4d_scgs/train_lift4d_scgs.py   (4D optimization)

Three input types are supported, selected with ``--dataset``:

    consistent4d   RGBA frames, object mask = alpha channel
    davis          RGB frames + separate binary mask PNGs
    custom         your own video: frames/ + masks/ (from segment_video.py)

Data roots are configured with environment variables (or the ``--data_root``
flag), so nothing is hard-coded to a particular machine:

    LIFT4D_DATA_ROOT     base data dir            (default: <repo>/data)
    CONSISTENT4D_ROOT    consistent4d frames      (default: $LIFT4D_DATA_ROOT/consistent4d)
    DAVIS_ROOT           DAVIS dataset            (default: $LIFT4D_DATA_ROOT/DAVIS)
    LIFT4D_SAM3D_OUT     stage-1 output root      (default: <repo>/sam3d_output)

Expected on-disk layout::

    <DATA_ROOT>/consistent4d/<object>/0.png 1.png ...        # RGBA
    <DATA_ROOT>/DAVIS/JPEGImages/480p/<object>/00000.jpg ...
    <DATA_ROOT>/DAVIS/Annotations/480p/<object>/00000.png ...
    <DATA_ROOT>/custom/<video>/frames/frame_000000.jpg ...
    <DATA_ROOT>/custom/<video>/masks/frame_000000_<maskname>.png ...
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

# Repo root = directory that contains this file.
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

# Mask storage conventions.
MASK_ALPHA = "alpha"        # mask is the alpha channel of an RGBA frame (consistent4d)
MASK_DIR = "mask_dir"       # mask is a separate PNG in a parallel directory (davis)
MASK_SUFFIX = "suffix"      # mask is <frame_stem>_<mask_name>.png next to frames (custom)


def data_root() -> str:
    """Base directory holding all datasets (override with LIFT4D_DATA_ROOT)."""
    return os.environ.get("LIFT4D_DATA_ROOT", os.path.join(REPO_ROOT, "data"))


def consistent4d_root() -> str:
    return os.environ.get("CONSISTENT4D_ROOT", os.path.join(data_root(), "consistent4d"))


def davis_root() -> str:
    return os.environ.get("DAVIS_ROOT", os.path.join(data_root(), "DAVIS"))


def custom_root() -> str:
    return os.environ.get("CUSTOM_ROOT", os.path.join(data_root(), "custom"))


def sam3d_out_root() -> str:
    """Root where stage-1 (SAM3D) writes its per-frame reconstructions."""
    return os.environ.get("LIFT4D_SAM3D_OUT", os.path.join(REPO_ROOT, "sam3d_output"))


@dataclass
class DatasetSpec:
    """Everything the two stages need to locate frames and masks for one object."""
    dataset: str               # "consistent4d" | "davis" | "custom"
    name: str                  # object / video name (used in output paths)
    frames_dir: str            # directory holding the RGB(A) frames
    mask_mode: str             # MASK_ALPHA | MASK_DIR | MASK_SUFFIX
    masks_dir: Optional[str] = None   # for MASK_DIR / MASK_SUFFIX
    mask_name: Optional[str] = None   # for MASK_SUFFIX (e.g. "box_1")
    image_exts: tuple = (".png", ".jpg", ".jpeg")

    @property
    def tag(self) -> str:
        """Stable identifier used to name this object's stage-1 output dir."""
        if self.dataset == "custom" and self.mask_name:
            return f"custom_{self.name}_{self.mask_name}"
        return f"{self.dataset}_{self.name}"

    def sam3d_output_dir(self, out_root: Optional[str] = None) -> str:
        """Where stage 1 writes (and stage 2 reads) frame_*/result.ply for this object."""
        return os.path.join(out_root or sam3d_out_root(), self.tag)


DATASETS = ("consistent4d", "davis", "custom")


def list_objects(dataset: str) -> List[str]:
    """Object/video names available on disk for a dataset (empty list if none)."""
    if dataset == "consistent4d":
        # Scan actual object folders under consistent4d/ (+ synthetic/, in-the-wild/).
        root = consistent4d_root()
        objs = set()
        for sub in ("", "synthetic", "in-the-wild"):
            d = os.path.join(root, sub) if sub else root
            if os.path.isdir(d):
                objs.update(o for o in os.listdir(d)
                            if os.path.isdir(os.path.join(d, o))
                            and o not in ("synthetic", "in-the-wild"))
        return sorted(objs)
    if dataset == "davis":
        root = os.path.join(davis_root(), "JPEGImages", "480p")
        if os.path.isdir(root):
            return sorted(o for o in os.listdir(root)
                          if os.path.isdir(os.path.join(root, o)))
        return []
    if dataset == "custom":
        root = custom_root()
        if os.path.isdir(root):
            return sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)))
        return []
    raise ValueError(f"Unknown dataset '{dataset}'. Choose from {DATASETS}.")


def resolve(dataset: str, name: str, mask_name: Optional[str] = None) -> DatasetSpec:
    """
    Resolve one object/video into a DatasetSpec (paths + mask convention).

    Args:
        dataset:   one of DATASETS.
        name:      object name (consistent4d/davis) or video folder name (custom).
        mask_name: required for custom videos (the SAM3 prompt/object suffix).
    """
    if dataset == "consistent4d":
        # The Consistent4D archive nests objects under synthetic/ and in-the-wild/;
        # accept the plain object folder too. First existing match wins.
        root = consistent4d_root()
        frames_dir = next(
            (c for c in (os.path.join(root, name),
                         os.path.join(root, "synthetic", name),
                         os.path.join(root, "in-the-wild", name))
             if os.path.isdir(c)),
            None,
        )
        if frames_dir is None:
            raise FileNotFoundError(
                f"consistent4d object '{name}' not found under {root} "
                f"(looked in ./, synthetic/, in-the-wild/). "
                f"Available objects: {', '.join(list_objects('consistent4d')) or '(none)'}"
            )
        return DatasetSpec(dataset, name, frames_dir, MASK_ALPHA, image_exts=(".png",))

    if dataset == "davis":
        frames_dir = os.path.join(davis_root(), "JPEGImages", "480p", name)
        masks_dir = os.path.join(davis_root(), "Annotations", "480p", name)
        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(
                f"davis object '{name}' not found at {frames_dir}. "
                f"Available objects: {', '.join(list_objects('davis')) or '(none)'}"
            )
        return DatasetSpec(dataset, name, frames_dir, MASK_DIR,
                           masks_dir=masks_dir, image_exts=(".jpg",))

    if dataset == "custom":
        video_dir = name if os.path.isabs(name) else os.path.join(custom_root(), name)
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(
                f"custom video '{name}' not found at {video_dir}. "
                f"Available videos: {', '.join(list_objects('custom')) or '(none)'}"
            )
        frames_dir = os.path.join(video_dir, "frames")
        masks_dir = os.path.join(video_dir, "masks")
        # Allow a flat layout too (frames directly in the video dir).
        if not os.path.isdir(frames_dir):
            frames_dir = video_dir
        if not os.path.isdir(masks_dir):
            masks_dir = video_dir
        return DatasetSpec(dataset, os.path.basename(os.path.normpath(video_dir)),
                           frames_dir, MASK_SUFFIX, masks_dir=masks_dir,
                           mask_name=mask_name)

    raise ValueError(f"Unknown dataset '{dataset}'. Choose from {DATASETS}.")
