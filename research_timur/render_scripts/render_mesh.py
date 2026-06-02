#!/usr/bin/env python3
"""Render the Step-1 reconstructed mesh (ball.obj, textured) for the slides.

Outputs (into research_timur/pod_results/presentation_renders/step1_mesh/):
  - view_front.png / view_side.png / view_top.png : clean static angles
  - turntable.mp4 + turntable.gif                 : 360deg rotation

Usage:
  python research_timur/render_scripts/render_mesh.py
"""
import os
import numpy as np
import pyvista as pv
import imageio.v2 as imageio

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Paths default to the tennisball example but can be overridden via env vars
# (PSIVG_MESH / PSIVG_RENDER_OUT) so the same script renders other examples.
MESH = os.environ.get("PSIVG_MESH", os.path.join(
    REPO,
    "research_timur/pod_results/run_tennisball/OUT_Perception/0001/00000/meshes/ball.obj",
))
OUT = os.environ.get("PSIVG_RENDER_OUT", os.path.join(
    REPO, "research_timur/pod_results/presentation_renders/step1_mesh"))
os.makedirs(OUT, exist_ok=True)

WINDOW = (1280, 1024)
BG = "white"
N_TURN = 72  # frames for a full 360 turntable


def new_plotter():
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background(BG)
    # vtkOBJImporter reads geometry + mtl + texture together and sets up the
    # textured actor for us; this is the reliable path for a textured OBJ.
    pl.import_obj(MESH)
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    pl.add_light(pv.Light(position=(3, 3, 4), intensity=0.6, light_type="scene light"))
    pl.add_light(pv.Light(position=(-3, -1, 2), intensity=0.4, light_type="scene light"))
    return pl


def reset_cam(pl, azimuth=0.0, elevation=15.0, zoom=1.3):
    pl.camera_position = "xz"     # look along -y, ball upright
    pl.camera.azimuth = azimuth
    pl.camera.elevation = elevation
    pl.reset_camera()
    pl.camera.zoom(zoom)


# ---- static views -----------------------------------------------------------
for name, az, el in [("front", 0, 12), ("side", 90, 12), ("top", 0, 75)]:
    pl = new_plotter()
    reset_cam(pl, azimuth=az, elevation=el)
    path = os.path.join(OUT, f"view_{name}.png")
    pl.screenshot(path)
    pl.close()
    print("wrote", path)

# ---- turntable ---------------------------------------------------------------
pl = new_plotter()
reset_cam(pl, azimuth=0, elevation=12)
frames = []
for i in range(N_TURN):
    pl.camera.azimuth = i * (360.0 / N_TURN)
    pl.render()
    frames.append(np.asarray(pl.screenshot(return_img=True)))
pl.close()

gif_path = os.path.join(OUT, "turntable.gif")
imageio.mimsave(gif_path, frames, duration=1 / 24, loop=0)
print("wrote", gif_path)
try:
    mp4_path = os.path.join(OUT, "turntable.mp4")
    imageio.mimsave(mp4_path, frames, fps=24, quality=8)
    print("wrote", mp4_path)
except Exception as e:
    print("mp4 skipped:", e)

print("DONE ->", OUT)
