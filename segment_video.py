#!/usr/bin/env python3
"""
Non-interactive SAM3 video segmentation.
Segments the first frame with a text prompt, then propagates masks to all frames.

Input can be EITHER a video file (decoded here into frames) OR a directory of
pre-extracted frames named frame_000000.jpg, frame_000001.jpg, ... (the format
stage 2 / run_inference.py expects).

Usage:
    # (a) from a video file -> frames are written to --frames_dir first
    python segment_video.py \
        --video /path/to/slime.mp4 \
        --frames_dir data/custom/slime/frames \
        --output_dir data/custom/slime/masks \
        --prompt "slime" --subsampling 1

    # (b) from pre-extracted frames already in --frames_dir
    python segment_video.py \
        --frames_dir data/custom/slime/frames \
        --output_dir data/custom/slime/masks \
        --prompt "slime"

Masks are written as <frame_stem>_<prompt>_<instance>.png, so stage 2/3 use
--mask_name <prompt>_<instance> (e.g. --mask_name slime_1).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def extract_frames_from_video(video_path, frames_dir, subsampling=1, max_frames=0):
    """Decode a video into frames_dir/frame_%06d.jpg (the format stage 2 expects)."""
    import cv2, glob
    os.makedirs(frames_dir, exist_ok=True)
    for old in glob.glob(os.path.join(frames_dir, "frame_*.jpg")):
        os.remove(old)  # avoid mixing with frames from a previous decode
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or -1
    print(f"Decoding {video_path} ({total} frames, subsampling={subsampling}) "
          f"-> {frames_dir}/frame_%06d.jpg")
    src_idx, written = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if src_idx % max(1, subsampling) == 0:
            out = os.path.join(frames_dir, f"frame_{written:06d}.jpg")
            cv2.imwrite(out, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
            if max_frames and written >= max_frames:
                break
        src_idx += 1
    cap.release()
    if written == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    print(f"  Wrote {written} frames: frame_000000.jpg ... frame_{written - 1:06d}.jpg")
    return written


def _resolve_sam3_ckpt(arg):
    """Local SAM3 checkpoint to use (mirror weights), else None to download from HF."""
    if arg:
        return arg
    here = os.path.dirname(os.path.abspath(__file__))   # repo root — segment_video.py lives here
    default = os.path.join(here, "sam3", "checkpoints", "sam3", "sam3.pt")
    return default if os.path.exists(default) else None


def main():
    parser = argparse.ArgumentParser(description="SAM3 video segmentation")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file (e.g. slime.mp4). If given, it is "
                             "decoded into --frames_dir as frame_000000.jpg, ... first. "
                             "Omit to use frames already present in --frames_dir.")
    parser.add_argument("--frames_dir", type=str, required=True,
                        help="Directory of frames (frame_000000.jpg, ...). Written here "
                             "when --video is given, otherwise read from here.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for masks")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Text prompt for segmentation (e.g., 'box')")
    parser.add_argument("--subsampling", type=int, default=1,
                        help="When decoding --video, keep every Nth frame (default: 1)")
    parser.add_argument("--num_frames", type=int, default=0,
                        help="Limit to first N frames (0=all)")
    parser.add_argument("--instance", type=str, default="1",
                        help="Which detected instance(s) to segment: an integer (1-based, "
                             "default '1'), or 'all' to segment every instance of the prompt. "
                             "Each is saved separately as <prompt>_1.png, <prompt>_2.png, ... "
                             "so multiple instances of the same object stay distinct.")
    parser.add_argument("--sam3_checkpoint", type=str, default=None,
                        help="Path to a local SAM3 checkpoint (sam3.pt). Defaults to "
                             "sam3/checkpoints/sam3/sam3.pt if present, else downloads from "
                             "HuggingFace (facebook/sam3, gated).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # If a video file is given, decode it into --frames_dir first (the format
    # stage 2 expects). Otherwise assume frames are already there.
    if args.video:
        if not os.path.isfile(args.video):
            parser.error(f"--video file not found: {args.video}")
        extract_frames_from_video(args.video, args.frames_dir,
                                  subsampling=args.subsampling, max_frames=args.num_frames)
    else:
        print(f"No --video given; using pre-extracted frames in {args.frames_dir} "
              f"(expected names: frame_000000.jpg, frame_000001.jpg, ...).")

    # Discover frames — prefer JPEGs (required by SAM3 video model)
    # If only PNGs exist, convert to JPEG
    jpg_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    png_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.png")))

    if len(jpg_paths) == 0 and len(png_paths) > 0:
        print("Converting PNGs to JPEGs (SAM3 video model requires JPEG)...")
        from PIL import Image as PILImage
        for p in png_paths:
            img = PILImage.open(p)
            jpg_path = os.path.splitext(p)[0] + ".jpg"
            img.save(jpg_path, "JPEG", quality=95)
        jpg_paths = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
        print(f"  Converted {len(jpg_paths)} files")

    # Use JPEGs for frame_paths (video model needs them), but also keep PNGs for image model
    frame_paths = jpg_paths if len(jpg_paths) > 0 else png_paths
    if args.num_frames > 0:
        frame_paths = frame_paths[:args.num_frames]
    print(f"Found {len(frame_paths)} frames")

    # --- Stage 1: Segment first frame with SAM3 image model ---
    print(f"Segmenting first frame with prompt: '{args.prompt}'")
    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
    # BPE vocab location varies by install (assets/ or clip/); None lets sam3 auto-resolve.
    bpe_path = next((p for p in (
        os.path.join(sam3_root, "assets", "bpe_simple_vocab_16e6.txt.gz"),
        os.path.join(sam3_root, "clip", "bpe_simple_vocab_16e6.txt.gz"),
    ) if os.path.exists(p)), None)
    # Use a local SAM3 checkpoint (e.g. mirror weights) if available; else HF download.
    sam3_ckpt = _resolve_sam3_ckpt(args.sam3_checkpoint)
    load_hf = sam3_ckpt is None
    print(f"SAM3 weights: {sam3_ckpt if sam3_ckpt else 'HuggingFace (facebook/sam3)'}")

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path=sam3_ckpt,
                                           load_from_HF=load_hf, device="cuda")
            processor = Sam3Processor(model)

            first_frame = Image.open(frame_paths[0]).convert("RGB")
            state = processor.set_image(first_frame)
            state = processor.set_text_prompt(args.prompt, state)

            masks = state.get("masks", [])
            scores = state.get("scores", [])
            if len(masks) == 0:
                print("ERROR: No objects detected in first frame!")
                sys.exit(1)

            print(f"Detected {len(masks)} object(s)")
            for i, (m, s) in enumerate(zip(masks, scores)):
                score_val = s.item() if hasattr(s, 'item') else float(s)
                print(f"  Object {i+1}: score={score_val:.3f}")

            # Select which instance(s) to segment: a single 1-based index, or all.
            if str(args.instance).strip().lower() == "all":
                sel_instances = list(range(1, len(masks) + 1))
            else:
                sel_instances = [min(max(int(args.instance), 1), len(masks))]
            print(f"Segmenting instance(s): {sel_instances} of {len(masks)} detected")

            prompt_clean = args.prompt.replace(" ", "_")
            first_stem = os.path.splitext(os.path.basename(frame_paths[0]))[0]

            # first-frame mask per instance: obj_id (1-based instance) -> mask (float 0/1)
            first_masks = {}
            for inst in sel_instances:
                m = (masks[inst - 1][0].cpu().numpy() > 0.5).astype(np.float32)
                first_masks[inst] = m
                out_name = f"{first_stem}_{prompt_clean}_{inst}.png"
                Image.fromarray((m * 255).astype(np.uint8)).save(
                    os.path.join(args.output_dir, out_name))
                print(f"  Saved first-frame mask: {out_name}")

    # Free image model memory
    del model, processor, state
    torch.cuda.empty_cache()

    if len(frame_paths) <= 1:
        print("Only one frame, done.")
        return

    # --- Stage 2: Propagate with SAM3 video model ---
    print("Loading SAM3 video model for propagation...")
    from sam3.model_builder import build_sam3_video_model

    # SAM3's video loader requires integer frame filenames (0.jpg, 1.jpg, ...), while
    # the pipeline uses frame_000000.jpg. Point the video model at a temp dir of
    # integer-named symlinks (in frame order); masks are saved back under the real stems.
    import tempfile
    tmp_frames = tempfile.mkdtemp(prefix="sam3_vid_")
    for i, fp in enumerate(frame_paths):
        os.symlink(os.path.abspath(fp), os.path.join(tmp_frames, f"{i}.jpg"))

    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.inference_mode():
            sam3_video = build_sam3_video_model(checkpoint_path=sam3_ckpt, load_from_HF=load_hf)
            predictor = sam3_video.tracker
            predictor.backbone = sam3_video.detector.backbone

            inference_state = predictor.init_state(video_path=tmp_frames)
            predictor.clear_all_points_in_video(inference_state)

            # Add each selected instance's first-frame mask under its own obj_id
            # (= the 1-based instance number), so they propagate independently.
            for inst, m in first_masks.items():
                predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=inst,
                    mask=torch.tensor(m, dtype=torch.float32),
                )

            # Propagate
            print("Propagating masks...")
            video_segments = {}
            for frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores in predictor.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=len(frame_paths),
                reverse=False,
                propagate_preflight=True,
            ):
                video_segments[frame_idx] = {
                    out_obj_ids[i]: (video_res_masks[i] > 0).cpu().numpy()
                    for i in range(len(out_obj_ids))
                }

            print(f"Propagated through {len(video_segments)} frames")

            # Save propagated masks — one file per instance per frame.
            h, w = next(iter(first_masks.values())).shape[:2]
            for frame_idx in range(len(frame_paths)):
                if frame_idx == 0:
                    continue  # already saved
                stem = os.path.splitext(os.path.basename(frame_paths[frame_idx]))[0]
                segs = video_segments.get(frame_idx, {})
                for inst in first_masks:
                    if inst in segs:
                        mask_img = (np.squeeze(segs[inst]) > 0.5).astype(np.uint8) * 255
                    else:
                        mask_img = np.zeros((h, w), dtype=np.uint8)  # no mask this frame
                    out_name = f"{stem}_{prompt_clean}_{inst}.png"
                    Image.fromarray(mask_img).save(os.path.join(args.output_dir, out_name))

            print(f"All masks saved to {args.output_dir}")

    import shutil
    shutil.rmtree(tmp_frames, ignore_errors=True)

    # Summarize the on-disk format + the exact stage-2 command(s) to run next —
    # one per instance (each instance is a separate object to reconstruct).
    mask_names = [f"{prompt_clean}_{i}" for i in sel_instances]
    fd = os.path.normpath(args.frames_dir)
    video_name = os.path.basename(os.path.dirname(fd)) if os.path.basename(fd) == "frames" \
        else os.path.basename(fd)
    print("\n" + "=" * 64)
    print(f"Ready for stage 2. {len(mask_names)} instance(s) segmented. On-disk format:")
    print(f"  frames: {args.frames_dir}/frame_000000.jpg, frame_000001.jpg, ...")
    for mn in mask_names:
        print(f"  masks : {args.output_dir}/frame_000000_{mn}.png, ...")
    print("Next (stage 2 — SAM3D per-frame reconstruction), one run per instance:")
    for mn in mask_names:
        print(f"  cd sam3d && python run_inference.py --dataset custom --object_name {video_name} \\")
        print(f"      --mask_name {mn} --video_consistency 0.2 --render_video")
    print("=" * 64)


if __name__ == "__main__":
    main()
