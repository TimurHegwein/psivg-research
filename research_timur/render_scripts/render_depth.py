#!/usr/bin/env python3
"""Visualize ViPE's per-frame monocular DEPTH (Step 2) as a color-mapped mp4.

Reads the ViPE depth archive (one EXR per frame, channel "Z" = metric depth),
applies a single GLOBAL 2-98 percentile normalization (so the video does not
flicker) and the matplotlib "turbo_r" colormap (near = warm/red, far = blue).
The depth VALUES are unedited; only the color mapping + one global range are
cosmetic.

Paths default to the tennisball example; override with env vars:
  PSIVG_DEPTH_ZIP   path to the ViPE depth zip (EXR frames inside)
  PSIVG_RENDER_OUT  output directory
  PSIVG_DEPTH_FPS   frames per second (default 15)

Usage:
  PSIVG_DEPTH_ZIP=.../OUT_ViPE_Raw/depth/0002.zip \
  PSIVG_RENDER_OUT=.../_paper_scratch/step2_vipe \
  python research_timur/render_scripts/render_depth.py
"""
import os
import io
import glob
import zipfile
import tempfile
import numpy as np
import imageio.v2 as imageio
import matplotlib.cm as cm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ZIP = os.environ.get("PSIVG_DEPTH_ZIP", os.path.join(
    REPO, "research_timur/pod_results/run_tennisball/OUT_ViPE_Raw/depth/0001.zip"))
OUT = os.environ.get("PSIVG_RENDER_OUT", os.path.join(
    REPO, "research_timur/pod_results/presentation_renders/step2_vipe"))
FPS = int(os.environ.get("PSIVG_DEPTH_FPS", "15"))
os.makedirs(OUT, exist_ok=True)


def read_exr_depth(path):
    """Return a 2D float32 depth array from an EXR file.

    Prefers the OpenEXR library reading channel 'Z'; falls back to OpenCV.
    """
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(path)
        hdr = f.header()
        dw = hdr["dataWindow"]
        w = dw.max.x - dw.min.x + 1
        h = dw.max.y - dw.min.y + 1
        chans = hdr["channels"].keys()
        ch = "Z" if "Z" in chans else list(chans)[0]
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        raw = f.channel(ch, pt)
        arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w).copy()
        return arr
    except Exception:
        import cv2
        arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise RuntimeError(f"could not read EXR: {path}")
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.float32)


# ---- extract frames ---------------------------------------------------------
tmp = tempfile.mkdtemp(prefix="vipe_depth_")
with zipfile.ZipFile(ZIP) as z:
    z.extractall(tmp)
exr_files = sorted(glob.glob(os.path.join(tmp, "**", "*.exr"), recursive=True))
print(f"{len(exr_files)} depth frames in {ZIP}")

depths = [read_exr_depth(p) for p in exr_files]
stack = np.stack(depths, 0)
finite = stack[np.isfinite(stack)]
lo, hi = np.percentile(finite, 2), np.percentile(finite, 98)
print(f"global depth range (2-98 pct): {lo:.4f} .. {hi:.4f}  (units)")

cmap = cm.get_cmap("turbo_r")
frames = []
for d in depths:
    dd = np.clip(d, lo, hi)
    norm = (dd - lo) / (hi - lo + 1e-8)
    norm[~np.isfinite(d)] = 0.0
    rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    frames.append(rgb)

mp4 = os.path.join(OUT, "vipe_depth.mp4")
imageio.mimsave(mp4, frames, fps=FPS, quality=8)
print("wrote", mp4)
gif = os.path.join(OUT, "vipe_depth.gif")
imageio.mimsave(gif, frames, duration=1.0 / FPS, loop=0)
print("wrote", gif)
print("DONE ->", OUT)
