
# Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild

<p align="center">
  <a href="https://yehonathanlitman.github.io/">Yehonathan Litman</a>&emsp;
  <a href="https://shirleymaxx.github.io/">Xiaoxuan Ma</a>&emsp;
  <a href="https://cs-mshah.github.io/">Manan Shah</a>&emsp;
  <a href="https://nicolasugrinovic.github.io/">Nicolás Ugrinovic</a>&emsp;
  <a href="https://kriskitani.github.io/">Kris Kitani</a>&emsp;
  <a href="https://www.cs.cmu.edu/~ftorre/">Fernando De la Torre</a>&emsp;
  <a href="https://shubhtuls.github.io/">Shubham Tulsiani</a>
</p>

<p align="center">Carnegie Mellon University</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.23688"><img src="https://img.shields.io/badge/arXiv-Lift4D-d55c5c?logo=arxiv&style=flat" alt="arXiv"></a>
  &nbsp;
  <a href="https://lift4d.github.io"><img src="https://img.shields.io/badge/Project-Page-6bbf59?logo=googlechrome&style=flat" alt="Project Page"></a>
  &nbsp;
  <a href="https://lift4d.github.io/#reconstructions"><img src="https://img.shields.io/badge/Interactive-4D-f2c14e?logo=unity&style=flat" alt="Interactive 4D"></a>
</p>

<h3 align="center">⚡️ TL;DR: Recover complete 4D asset reconstructions from videos in the wild.</h3>
<img width="1207" height="782" alt="teaser" src="https://github.com/user-attachments/assets/3ca0d9f2-bd1e-4627-b653-ea382ecedbab" />

## How it works

Lift4D runs in three stages:

```
        video ──▶  1. Segment  ──▶  2. Causal SAM3D  ──▶  3. Lift4D training  ──▶  4D asset
                      (SAM 3)         (per-frame 3D)         (fuse into deformable model)
```

1. **Segment** the object in the video with a text prompt (SAM 3) for in-the-wild videos.
2. **Causal SAM3D** reconstructs one object with causal latent propagation for temporal coherence.
3. **Lift4D training** fuses those per-frame reconstructions into one canonical shape
   plus a learned deformation, giving a temporally consistent 4D asset.

## 1. Installation

**Tested on an A100 40GB GPU**

```bash
git clone https://github.com/yehonathanlitman/Lift4D && cd Lift4D

#  Create the conda env `lift4d` (one env, shared by all three stages)
conda env create -f sam3d/environments/default.yml
conda activate lift4d

cd sam3d
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e .
pip install -e '.[p3d]'
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install hatchling hatch-requirements-txt ninja editables
pip install -r requirements.inference.txt --no-build-isolation
./patching/hydra
cd ..

cd lift4d_scgs
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
cd ..
pip install git+https://github.com/facebookresearch/sam3.git pycocotools
pip install 'huggingface-hub[cli]<1.0'
```

### Download the model weights

Model weights are installed to the following locations:

```bash
hf auth login          # SAM3D and SAM3 weights are gated — accept the license first

hf download --repo-type model --local-dir sam3d/checkpoints/hf-dl facebook/sam-3d-objects
mv sam3d/checkpoints/hf-dl/checkpoints sam3d/checkpoints/hf && rm -rf sam3d/checkpoints/hf-dl

# SAM3 ->  sam3/checkpoints/sam3/
hf download facebook/sam3 sam3.pt config.json --local-dir sam3/checkpoints/sam3

# Stable Zero123  ->  lift4d_scgs/extern/ldm_zero123/stable_zero123.ckpt
hf download --repo-type model stabilityai/stable-zero123 stable_zero123.ckpt \
  --local-dir lift4d_scgs/extern/ldm_zero123
```

## 2. Dataset setup

Put datasets under `data/` or set `LIFT4D_DATA_ROOT` to point elsewhere. Layout:

```
data/
├── DAVIS/JPEGImages/480p/<object>/00000.jpg ...  # RGB frames
│   └── Annotations/480p/<object>/00000.png ...   # binary masks
├── consistent4d/
│   ├── synthetic/<object>/0.png 1.png ...        # RGBA frames (mask = alpha)
│   └── in-the-wild/<object>/0.png 1.png ...      # RGBA frames
└── custom/<video>/frames/frame_000000.jpg ...    # your own video
    └── masks/                                    # created by SAM3
```

- **DAVIS** — download and unzip:
  ```bash
  wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip
  unzip DAVIS-data.zip -d data/
  ```
- **Consistent4D** - download and unzip [Consistent4D](https://drive.google.com/file/d/1mJNhFKvzZ-8icAw6KC-W-sf7JmmmMUkx/view)
  to `data/consistent4d/`.
  ```bash
  python -m zipfile -e data/CONSISTENT4D_DATA.zip data/consistent4d/
  ```
- **custom** — grab any clip in the wild, e.g.
  [playful goat](https://www.pexels.com/video/playful-goat-in-blue-barrel-outdoors-36075632/)
  (download the 960×540 file) and place in `data/custom`, or place frames as
  `data/custom/<video>/frames/frame_%06d.jpg`.

Paths are configurable via environment variables (or the `--data_root` flag):
`LIFT4D_DATA_ROOT`, `CUSTOM_ROOT`, `CONSISTENT4D_ROOT`, `DAVIS_ROOT`.



## 3. Running the pipeline

The code provided can currently read three dataset types, `davis`, `consistent4d`, and `custom`, which are selected with
`--dataset` at every stage. Consistent4D and DAVIS already have masks, so they **skip stage 1**.

### Stage 1 — Segment (in-the-wild videos only)

Segment your object with a text prompt: Masks are written to `data/custom/<video>/masks/`
as `frame_XXXXXX_<prompt>_<instance>.png`, so the mask name for later stages is
`<prompt>_<instance>` (e.g. `goat_1`).

```bash
# (a) straight from a video file — frames are decoded into --frames_dir for you
python segment_video.py \
    --video data/custom/goat.mp4 \
    --frames_dir data/custom/goat/frames \
    --output_dir data/custom/goat/masks \
    --prompt "goat"

# (b) or from frames you already extracted into --frames_dir (omit --video)
python segment_video.py \
    --frames_dir data/custom/my_clip/frames \
    --output_dir data/custom/my_clip/masks \
    --prompt "box"
```

### Stage 2 — SAM3D per-frame reconstruction

Reconstructs one 3D object per frame into `sam3d_output/<tag>/frame_XXXX/`.

```bash
cd sam3d
# DAVIS
python run_inference.py --dataset davis --video_consistency 0.2 --object_name rhino --render_video
# consistent4d
python run_inference.py --dataset consistent4d --video_consistency 0.1 --object_name scaring_skull --render_video
# custom (mask_name = <prompt>_<instance> from stage 1, e.g. goat_1)
python run_inference.py --dataset custom --object_name goat --mask_name goat_1 \
     --video_consistency 0.2 --render_video #--subsampling 4 --num_frames 40
```

Knobs: `--subsampling N` (keep every Nth frame), `--num_frames N` (total frames to
reconstruct after subsampling; default `100`, use `0` for all), `--video_consistency` in
`[0,1]` (higher = preserve more structure, for example for rigid objects), `--render_video` (writes a
side-by-side `comparison.mp4`).

### Stage 3 — Lift4D training

Two-step optimization process for geometry and appearance. Run both, in order, with the same `--dataset`/`--object_name`.

```bash
cd ../lift4d_scgs
DS="--dataset consistent4d --object_name scaring_skull"
# or: DS="--dataset davis --object_name rhino"
# or: DS="--dataset custom --object_name goat --mask_name goat_1"

# Step A — geometry
python train_lift4d_scgs.py $DS \
    --deform_type node --iterations 9999 --save_checkpoints \
    --optimize_canonical_color --opt_deform_rot --optimize_gs_xyz \
    --lambda_rc 1 --lambda_chamfer 1 --disable_rendering_loss_iter 1 \
    --reset_opacity_rc --densify_and_prune_rc \
    --position_lr_max_steps 10000 --deform_lr_max_steps 10000 \
    --opacity_reset_interval 2000 --smooth_transforms_window 5

# Step B — appearance, resumes from step A.
python train_lift4d_scgs.py $DS \
    --deform_type node_delta --load_checkpoint 10000 --iterations 19999 --save_checkpoints \
    --optimize_canonical_color --opt_deform_rot --densify_and_prune \
    --lambda_sds_rgb 0.1 --enable_lpips \
    --optimize_per_frame_compose_transforms_app --compose_lr_reset \
    --position_lr_init_app 1.6e-4 \
    --position_lr_max_steps 20000 --deform_lr_max_steps 13000 --smooth_transforms_window 5 #--occlusion_compositing #inpaint occlusions
```

**For videos > 50 frames:** double the iteration counts:

```bash
DS="--dataset davis --object_name rhino"
python train_lift4d_scgs.py $DS \
    --deform_type node --iterations 19999 --save_checkpoints \
    --optimize_canonical_color --opt_deform_rot --optimize_gs_xyz \
    --lambda_rc 1 --lambda_chamfer 1 --disable_rendering_loss_iter 1 \
    --reset_opacity_rc --densify_and_prune_rc \
    --position_lr_max_steps 20000 --deform_lr_max_steps 20000 \
    --opacity_reset_interval 2000 --smooth_transforms_window 5

# Step B — appearance, resumes from step A.
python train_lift4d_scgs.py $DS \
    --deform_type node_delta --load_checkpoint 20000 --iterations 29999 --save_checkpoints \
    --optimize_canonical_color --opt_deform_rot --densify_and_prune \
    --lambda_sds_rgb 0.1 --enable_lpips \
    --optimize_per_frame_compose_transforms_app --compose_lr_reset \
    --position_lr_init_app 1.6e-4 \
    --position_lr_max_steps 30000 --deform_lr_max_steps 26000 --smooth_transforms_window 5 #--occlusion_compositing #inpaint occlusions
```

**Efficiency:** step A needs ~10k iters (<50 frames) or ~20k (>50 frames). Step B reaches
max quality by ~10k iters (diminishing returns beyond). For faster runs, use these flags:

**Step A**
- `--position_lr_max_steps 5000 --deform_lr_max_steps 5000 --opacity_reset_interval 1000`
  - For videos without many frames.
- `--init_subsample_ratio 0.5`
  - Uniformly subsample the first frame's GS.
- `--rc_batch_size 1`
  - Randomly sample only one camera, which can lead to degraded quality.

**Step B**
- `--iterations 5999`
  - ~6k iterations provide most of the appearance gain.
- `--ddim_steps 3`
  - Reduce DDIM steps for faster SDS.

### Outputs

Trained assets land in `lift4d_scgs/output/<tag>_<deform_type>/` and comprise of the deformable Gaussian
checkpoint, and comparison/orbit videos.

## Citation

If you find this work useful, please cite:

```bibtex
@article{litman2026lift4d,
  author  = {Litman, Yehonathan and Ma, Xiaoxuan and Shah, Manan and Ugrinovic, Nicol\'{a}s and Kitani, Kris and De la Torre, Fernando and Tulsiani, Shubham},
  title   = {Lift4D: Harmonizing Single-View 3D Estimation for 4D Reconstruction In-the-Wild},
  journal = {arXiv preprint arXiv:2606.23688},
  year    = {2026},
}
```

## Acknowledgements

This project builds on [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) and [SC-GS](https://github.com/yihua7/SC-GS). Many thanks!
