#!/bin/bash
# Minimal setup for running ONLY Step 4 (video generation / TTCO) on a fresh pod.
# Skips the Step 1-3 environments (env1/env2/langsam) and only downloads the two
# models Step 4 needs: CogVideoX-5b-I2V + the Go-with-the-Flow LoRA.
# Run from the repo root:  bash research_timur/setup_step4_only.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_DIR="/root/miniconda3"
cd "$REPO_ROOT"
echo "=== Step-4-only setup. Repo: $REPO_ROOT ==="

# 1. Miniconda
if [ ! -f "$CONDA_DIR/bin/conda" ]; then
    echo "-- Installing Miniconda --"
    curl -sO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_DIR"
    rm -f Miniconda3-latest-Linux-x86_64.sh
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

# 2. mamba + TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r   || true
conda install -n base -c conda-forge mamba -y

# 3. PSIVG_env3 only
if ! conda env list | grep -q "PSIVG_env3"; then
    conda create -n PSIVG_env3 python=3.10 -y
fi
conda activate PSIVG_env3
pip install -r envs/PSIVG_env3.txt
# ffmpeg for the mp4 video output (validation_epoch*.mp4)
mamba install -n PSIVG_env3 -c conda-forge ffmpeg -y
pip install huggingface_hub

# 4. Download ONLY the two models Step 4 needs
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import urllib.request
pm = Path("pretrained_models"); pm.mkdir(parents=True, exist_ok=True)
print("Downloading CogVideoX-5b-I2V ...")
snapshot_download(repo_id="THUDM/CogVideoX-5b-I2V",
                  local_dir=str(pm / "CogVideoX-5b-I2V"),
                  local_dir_use_symlinks=False)
lora = pm / "I2V5B_final_i38800_nearest_lora_weights.safetensors"
if not lora.exists():
    print("Downloading LoRA weights ...")
    urllib.request.urlretrieve(
        "https://huggingface.co/Eyeline-Labs/Go-with-the-Flow/resolve/main/"
        "I2V5B_final_i38800_nearest_lora_weights.safetensors", str(lora))
print("Models done.")
PY
conda deactivate

echo "SETUP_STEP4_DONE"
