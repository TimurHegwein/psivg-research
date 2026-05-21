# PSIVG RunPod Troubleshooting

Collected from a full end-to-end run on a fresh RunPod Community Cloud instance (RTX A6000 48GB, Ubuntu).

---

## Environment / Shell

### `mamba activate` fails with "Shell not initialized"
**Symptom:** `critical libmamba Shell not initialized` when running any `main_part*.sh`  
**Cause:** `bash script.sh` starts a non-login, non-interactive subprocess. Neither `.bashrc` nor `.bash_profile` is sourced, so mamba's shell hook never loads.  
**Fix:** Add this line to the top of every script (after the shebang):
```bash
source ~/miniconda3/etc/profile.d/conda.sh
```
Then replace `mamba activate` → `conda activate` and `mamba deactivate` → `conda deactivate` throughout:
```bash
sed -i 's/mamba activate/conda activate/g; s/mamba deactivate/conda deactivate/g' main_part3.sh
sed -i 's/mamba activate/conda activate/g; s/mamba deactivate/conda deactivate/g' main_part4.sh
```

### `conda init` not recognized in subshell
**Symptom:** `CondaError: Run 'conda init' before 'conda activate'`  
**Cause:** Same as above — conda shell functions not sourced in subprocess.  
**Fix:** Same — explicitly source conda.sh at the top of the script.

---

## Step 3 (`main_part3.sh`)

### Wrong `VIDEO_ID`
**Symptom:** `find: 'data_root/OUT_Rendering/0000': No such file or directory`  
**Cause:** `VIDEO_ID` is hardcoded to `"0000"` in the script.  
**Fix:**
```bash
sed -i 's/VIDEO_ID="0000"/VIDEO_ID="0001"/' main_part3.sh
```

### Wrong `USE_MOVING_CAMERA`
**Symptom:** Script tries to run RAFT optical flow on a static-camera video, wastes time.  
**Cause:** `USE_MOVING_CAMERA="true"` is the default. For videos without camera movement (e.g. `0001` tennis ball), this is wrong.  
**Fix:**
```bash
sed -i 's/USE_MOVING_CAMERA="true"/USE_MOVING_CAMERA="false"/' main_part3.sh
```

### Missing prompt files
**Symptom:** `ValueError: Prompt file 'data_root/INPUT_DATA/Prompts/0001.txt' does not exist`  
**Cause:** The pipeline never auto-creates these — they must be written manually from the asset JSON.  
**Fix:** Create them from `assets/0001.json`:
```bash
mkdir -p data_root/INPUT_DATA/Prompts
echo "A yellow tennis ball is in midair, bouncing on a tennis court. There are no humans. The scene is realistic, with good lighting and no camera movement." \
  > data_root/INPUT_DATA/Prompts/0001.txt
echo "yellow tennis ball" > data_root/INPUT_DATA/Prompts/0001_fg.txt
```

### `ffmpeg` not installed
**Symptom:** `save_video_mp4: Can't use backend=='ffmpeg' because ffmpeg is not installed`  
**Cause:** RunPod base images don't include ffmpeg; `make_warped_noise.py` tries `sudo apt install` which also fails because there is no sudo.  
**Fix:**
```bash
conda install -c conda-forge ffmpeg -y
```

### `noises.npy` not created / `noises/` directory empty
**Symptom:** Step 4 fails with `ValueError: Expected --noises_column ...`  
**Cause:** `make_warped_noise.py` silently skipped (earlier mamba env issue), so `transfer_to_dataset.py` ran with nothing to transfer.  
**Fix:** Run `make_warped_noise.py` manually after fixing the environment, then re-run `transfer_to_dataset.py`:
```bash
conda activate PSIVG_env3
python psivg/utils/make_warped_noise.py \
  --selected_vids_file data_root/OUT_Flow/rendering_path/0001.txt \
  --input_folder data_root/OUT_Rendering \
  --output_folder data_root/OUT_Flow/computed_noises \
  --first_frame_folder data_root/OUT_Flow/Firstframe_PNG \
  --mask_firstframe_folder data_root/OUT_Flow/segmaps_firstframe/masks_npy

python psivg/utils/transfer_to_dataset.py \
  --input_dir data_root/OUT_Flow/computed_noises \
  --output_dataset_dir data_root/datasets/generated_data_example \
  --prompt_file data_root/INPUT_DATA/Prompts/0001.txt \
  --prompt_fg_file data_root/INPUT_DATA/Prompts/0001_fg.txt \
  --selected_vids_file data_root/OUT_Flow/rendering_path/0001.txt \
  --image_folder data_root/OUT_Flow/Firstframe_PNG \
  --with_correspondences
```

---

## Step 4 (`main_part4.sh`)

### `accelerate: command not found`
**Symptom:** `./psivg/video_generation/video_gen_i2v.sh: line 90: accelerate: command not found`  
**Cause:** Script ran in `base` env because `mamba activate PSIVG_env3` failed.  
**Fix:** Same conda.sh source fix as above.

### Wrong `VIDEO_ID` / `VIDEOS`
**Symptom:** Tries to load dataset for `0000` instead of `0001`.  
**Fix:**
```bash
sed -i 's/export VIDEOS="0000"/export VIDEOS="0001"/' main_part4.sh
```

### Wrong `USE_MOVING_CAMERA` — looks for `merged_noises.txt`
**Symptom:** `ValueError: Expected --noises_column ...` (even though `noises.txt` exists)  
**Cause:** `dataset.py` line 101 hardcodes `self.noises_column = "merged_noises.txt"` when `use_moving_camera=True`. `merged_noises.txt` doesn't exist for static-camera videos.  
**Fix:**
```bash
sed -i 's/export USE_MOVING_CAMERA="true"/export USE_MOVING_CAMERA="false"/' main_part4.sh
```

### `videos/0001.mp4` missing
**Symptom:** `ValueError: ...video data but found atleast one path that is not a valid file`  
**Cause:** `transfer_to_dataset.py` did not copy the input video into the dataset `videos/` folder.  
**Fix:**
```bash
cp data_root/INPUT_DATA/Videos/0001.mp4 \
   data_root/datasets/generated_data_example/0001/videos/0001.mp4
```

### `masks/0001.mp4` missing
**Symptom:** `RuntimeError: Error reading data_root/datasets/.../masks/0001.mp4`  
**Cause:** `transfer_to_dataset.py` did not copy the mask video.  
**Fix:**
```bash
cp data_root/OUT_Flow/computed_noises/0001/2026-05-21_run/original_length/obj_mask.mp4 \
   data_root/datasets/generated_data_example/0001/masks/0001.mp4
```

---

## General

### `gdown` `fuzzy` argument error
**Symptom:** `TypeError: download() got an unexpected keyword argument 'fuzzy'` during model download  
**Cause:** gdown v6+ removed the `fuzzy` parameter; the ViPE code uses `gdown==4.7.3` API.  
**Fix:**
```bash
conda activate PSIVG_env2
pip install "gdown==4.7.3"
```

### Input data directories missing
**Symptom:** `cp: cannot create regular file 'data_root/INPUT_DATA/Videos/'`  
**Fix:**
```bash
mkdir -p data_root/INPUT_DATA/Videos data_root/INPUT_DATA/Metadata \
          data_root/INPUT_DATA/Frames data_root/INPUT_DATA/Prompts
```

### conda TOS not accepted (blocks mamba install)
**Symptom:** `CondaToSNonInteractiveError` when running `conda install mamba`  
**Fix:**
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```
