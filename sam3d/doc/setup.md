# Setup

## Prerequisites

* A linux 64-bits architecture (i.e. `linux-64` platform in `mamba info`).
* A NVIDIA GPU with at least 32 Gb of VRAM.

## 1. Setup Python Environment

The following will install the default environment. If you use `conda` instead of `mamba`, replace its name in the first two lines. Note that you may have to build the environment on a compute node with GPU (e.g., you may get a `RuntimeError: Not compiled with GPU support` error when running certain parts of the code that use Pytorch3D).

```bash
# create the `lift4d` conda env (from this file's name: field)
conda env create -f environments/default.yml     # env is named "lift4d"

# one-time so `conda activate` works in your shell, then re-open the shell
conda init bash            # (or your shell); skip if already initialized
conda activate lift4d
# if activation errors in the current shell, source conda first:
#   source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate lift4d

# for pytorch/cuda dependencies
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"

# install sam3d-objects and core dependencies
pip install -e .            # core deps (add '.[dev]' if you want the dev tooling)
pip install -e '.[p3d]' # pytorch3d dependency on pytorch is broken, this 2-step approach solves it

# for inference
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

# patch things that aren't yet in official pip packages
./patching/hydra # https://github.com/facebookresearch/hydra/pull/2863
```

## 2. Getting Checkpoints

### From HuggingFace

⚠️ Before using SAM 3D Objects, please request access to the checkpoints on the SAM 3D Objects
Hugging Face [repo](https://huggingface.co/facebook/sam-3d-objects). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token).

⚠️ SAM 3D Objects is available via HuggingFace globally, **except** in comprehensively sanctioned jurisdictions.
Sanctioned jurisdiction will result in requests being **rejected**.

```bash
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
hf download \
  --repo-type model \
  --local-dir checkpoints/${TAG}-download \
  --max-workers 1 \
  facebook/sam-3d-objects
mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
rm -rf checkpoints/${TAG}-download
```


