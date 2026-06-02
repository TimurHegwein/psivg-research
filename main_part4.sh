#!/bin/bash -l

# Support both conda and mamba; prefer conda since mamba needs explicit shell hook
# (bash script.sh runs a non-login shell that never sources .bashrc, so the
#  mamba/conda shell functions are not available unless we source conda.sh here)
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
  || source /root/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate PSIVG_env3

### input video id and processed dataset

# Override with environment variable: VIDEOS=0001 bash main_part4.sh
export VIDEOS="${VIDEOS:-0000}"

export DATA_ROOT="data_root/datasets/generated_data_example"

### output directory and setting name
export output_dir="./outputs"
export SETTING_NAME="generated_data_example"

### gwtf model
export LORA_WEIGHTS_PATH="./pretrained_models/I2V5B_final_i38800_nearest_lora_weights.safetensors"
export BASE_MODEL_NAME="./pretrained_models/CogVideoX-5b-I2V"

### hyperparameters
export DEGRADATION="0.5"
export LR_SCHEDULES="cosine"
export OPTIMIZERS="adamw"

export LEARNING_RATES="2e-4"  
export NOISE_STEP_THRESH="700"
export TIMESTEP_SAMPLING="noisy_steps"
export TTCO_LOSS_LAMBDA="10"
export MAX_TRAIN_STEPS="50"
export VALIDATION_EPOCHS="50"
export CHECKPOINTING_STEPS="10"


export USE_TTCO="false"

# Set to "false" for static-camera videos (e.g. 0001 tennis ball).
# NOTE: dataset.py hardcodes noises_column="merged_noises.txt" when this is
# "true"; that file only exists for moving-camera videos, so leaving this on
# for a static video raises "Expected --noises_column ...".
export USE_MOVING_CAMERA="${USE_MOVING_CAMERA:-false}"


./psivg/video_generation/video_gen_i2v.sh



