#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-flowbp}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ENV_ROOT="${ENV_ROOT:-/apdcephfs_fsgm3/share_305110755/hunyuan/boye/envs}"
ENV_PATH="$ENV_ROOT/$ENV_NAME"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$REPO_ROOT"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$HOME/miniconda3"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    CONDA_BASE="$HOME/anaconda3"
else
    echo "conda was not found. Install Miniconda/Anaconda or source conda.sh first." >&2
    exit 1
fi

# `conda activate` is a shell function; ensure it is available in non-interactive shells.
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

mkdir -p "$ENV_ROOT"

if ! conda env list | awk '{print $NF}' | grep -Fxq "$ENV_PATH"; then
    conda create -p "$ENV_PATH" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_PATH"

python -m pip install --upgrade pip
python -m pip install "setuptools<70" setuptools_scm wheel packaging ninja Cython numpy==1.26.4

# Install the torch stack before source-built extensions such as flash-attn.
python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# flash-attn imports torch during build, so build isolation often fails.
python -m pip install flash_attn==2.7.4.post1 --no-build-isolation

# Keep optional eval/detector stacks out of the default training install. They
# have mutually incompatible transitive pins (for example hpsv3 vs ImageReward
# on timm, and openxlab vs datasets on requests). The pinned diffusers git
# revision also declares a newer safetensors pre-release than the paper
# environment pins, so install those two explicitly and keep pip's resolver away
# from that metadata conflict.
REQ_FILE="$(mktemp)"
grep -v -E 'huggingface/diffusers|open-mmlab/(mmcv|mmdetection)|#egg=(diffusers|mmcv_full|mmdet)|^safetensors==|^hpsv3$|^image-reward$|^clean-fid==|^clip-benchmark==|^opendatalab==|^openmim==|^openxlab==|^paddleocr==|^paddlepaddle-gpu==' requirements.txt > "$REQ_FILE"
python -m pip install -r "$REQ_FILE" --no-build-isolation
rm -f "$REQ_FILE"
python -m pip install safetensors==0.7.0
python -m pip install "git+https://github.com/huggingface/diffusers.git@77f8cf8bf557a0136d269baea773cef26eb5991a#egg=diffusers" --no-deps --no-build-isolation

# requirements.txt is the pinned paper environment. Avoid letting the shorter
# dependency list in pyproject.toml re-resolve or overwrite that environment.
python -m pip install -e . --no-deps

echo "FlowBP setup complete."
echo "Activate the environment with: conda activate $ENV_PATH"
