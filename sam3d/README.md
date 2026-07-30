# sam3d/ — Stage 2 of Lift4D (per-frame SAM 3D reconstruction)

This directory is the **SAM3D stage** of the Lift4D pipeline: it reconstructs one 3D
object per video frame (`run_inference.py`). It is a trimmed fork of
[SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects).

**Setup and usage live in the top-level [`../README.md`](../README.md).** Usage:

```bash
python run_inference.py --dataset consistent4d --object_name pistol --render_video
python run_inference.py --dataset davis        --object_name rhino  --render_video
python run_inference.py --dataset custom --object_name my_clip --mask_name box --render_video
```

## Citation

```bibtex
@inproceedings{sam3dteam2025sam3d3dfyimages,
  title={SAM 3D: 3Dfy Anything in Images},
  author={Xingyu Chen and Fu-Jen Chu and Pierre Gleize and Kevin J Liang and Alexander Sax and Hao Tang and Weiyao Wang and Michelle Guo and Thibaut Hardin and Xiang Li and Aohan Lin and Jiawei Liu and Ziqi Ma and Anushka Sagar and Bowen Song and Xiaodong Wang and Jianing Yang and Bowen Zhang and Piotr Dollár and Georgia Gkioxari and Matt Feiszli and Jitendra Malik},
  booktitle={CVPR},
  year={2026}
}
```
