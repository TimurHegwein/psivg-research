#!/bin/bash
# Setup script for PSIVG on a fresh RunPod instance.
# Run from the repo root: bash research_timur/setup_runpod.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="/root/miniconda3"

echo "================================================================"
echo " PSIVG RunPod Setup"
echo " Repo: $REPO_ROOT"
echo "================================================================"

cd "$REPO_ROOT"

# ── 1. Miniconda ────────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo ""
    echo "── [1/7] Installing Miniconda ──"
    curl -sO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
    rm Miniconda3-latest-Linux-x86_64.sh
else
    echo "── [1/7] Miniconda already installed, skipping"
fi

source "$CONDA_DIR/etc/profile.d/conda.sh"

# ── 2. Accept TOS + install mamba ────────────────────────────────────────────
echo ""
echo "── [2/7] Accepting conda TOS and installing mamba ──"
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   || true
conda install -n base -c conda-forge mamba -y

# ── 3. PSIVG_env1 ────────────────────────────────────────────────────────────
echo ""
echo "── [3/7] Creating PSIVG_env1 ──"
if conda env list | grep -q "PSIVG_env1"; then
    echo "  already exists, skipping"
else
    conda env create -f envs/PSIVG_env1.yml
fi

conda activate PSIVG_env1

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.7;8.9}"
export CUDA_INCLUDE_DIR="$CONDA_PREFIX/targets/x86_64-linux/include"
export CUDA_LIB_DIR="$CONDA_PREFIX/targets/x86_64-linux/lib"
export CPATH="$CUDA_INCLUDE_DIR:${CPATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_INCLUDE_DIR:${CPLUS_INCLUDE_PATH:-}"
export LIBRARY_PATH="$CUDA_LIB_DIR:${LIBRARY_PATH:-}"

echo "  Installing nvdiffrast..."
python -m pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

echo "  Installing GroundingDINO..."
python -m pip install --no-build-isolation --no-deps git+https://github.com/IDEA-Research/GroundingDINO.git

conda deactivate

# ── 4. PSIVG_env2 ────────────────────────────────────────────────────────────
echo ""
echo "── [4/7] Creating PSIVG_env2 ──"
if conda env list | grep -q "PSIVG_env2"; then
    echo "  already exists, skipping"
else
    conda env create -f envs/PSIVG_env2.yml
fi

conda activate PSIVG_env2
pip install -r envs/PSIVG_env2.txt

# ViPE's model download uses the gdown 4.x API (download(..., fuzzy=True)).
# gdown 6+ removed the fuzzy kwarg, so pin the older version or ViPE's
# weight download (triggered by main_part2.py) fails with a TypeError.
echo "  Pinning gdown==4.7.3 for ViPE..."
pip install "gdown==4.7.3"

echo "  Installing ViPE..."
cd psivg/perception/vipe
pip install --no-build-isolation -e .
cd "$REPO_ROOT"

conda deactivate

# ── 5. PSIVG_env3 ────────────────────────────────────────────────────────────
echo ""
echo "── [5/7] Creating PSIVG_env3 ──"
if conda env list | grep -q "PSIVG_env3"; then
    echo "  already exists, skipping"
else
    conda create -n PSIVG_env3 python=3.10 -y
fi

conda activate PSIVG_env3
pip install -r envs/PSIVG_env3.txt
conda deactivate

# ── 6. langsam ───────────────────────────────────────────────────────────────
echo ""
echo "── [6/7] Creating langsam environment ──"
if conda env list | grep -q "langsam"; then
    echo "  already exists, skipping"
else
    conda create -n langsam python=3.10 -y
fi

conda activate langsam
pip install git+https://github.com/luca-medeiros/lang-segment-anything.git
conda deactivate

# ── 7. Download pretrained models ────────────────────────────────────────────
echo ""
echo "── [7/7] Downloading pretrained models ──"
conda activate PSIVG_env1
python3 envs/download_pretrained.py
conda deactivate

echo ""
echo "================================================================"
echo " Setup complete. Run the pipeline with:"
echo "   conda activate PSIVG_env1"
echo "   CUDA_VISIBLE_DEVICES=0 python3 main_part1.py --video assets/0001.mp4"
echo "================================================================"
