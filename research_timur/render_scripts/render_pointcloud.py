#!/usr/bin/env python3
"""Render the Step-2 MPM simulation point clouds (196 .ply frames).

Outputs (into research_timur/pod_results/presentation_renders/step2_pointcloud/):
  - trajectory_overlay.png : every Nth frame overlaid, colored by time
                             (the whole bounce arc frozen in one still)  <-- best slide
  - simulation.mp4 / .gif  : the simulation played back with real colors
  - turntable_mid.mp4      : 360deg orbit around one mid-bounce frame

Usage:
  python research_timur/render_scripts/render_pointcloud.py
"""
import os
import glob
import numpy as np
import pyvista as pv
import imageio.v2 as imageio
import matplotlib.cm as cm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Defaults to the tennisball example; override with env vars to render others:
#   PSIVG_PLY_GLOB    glob that resolves to a point_cloud dir
#   PSIVG_RENDER_OUT  output directory
PLY_DIR = glob.glob(os.environ.get("PSIVG_PLY_GLOB", os.path.join(
    REPO,
    "research_timur/pod_results/run_tennisball/OUT_Simulation/0001/*/point_cloud",
)))[0]
OUT = os.environ.get("PSIVG_RENDER_OUT", os.path.join(
    REPO, "research_timur/pod_results/presentation_renders/step2_pointcloud"))
os.makedirs(OUT, exist_ok=True)

WINDOW = (1280, 1024)
BG = "white"
POINT_SIZE = 4

ply_files = sorted(glob.glob(os.path.join(PLY_DIR, "*.ply")))
print(f"{len(ply_files)} ply frames in {PLY_DIR}")


def load(path):
    return pv.read(path)


def color_array_name(cloud):
    """Find the per-point uint8 RGB(A) array name, if any."""
    for key in cloud.point_data.keys():
        arr = cloud.point_data[key]
        if arr.ndim == 2 and arr.shape[1] in (3, 4) and arr.dtype == np.uint8:
            return key
    return None


# ---- global bounds across all frames -> fixed camera ------------------------
mins = np.array([np.inf, np.inf, np.inf])
maxs = -mins.copy()
centroids = []
for p in ply_files:
    pts = load(p).points
    mins = np.minimum(mins, pts.min(0))
    maxs = np.maximum(maxs, pts.max(0))
    centroids.append(pts.mean(0))
centroids = np.array(centroids)
span = maxs - mins
print("axis span (x,y,z):", np.round(span, 3))
print("centroid travel  :", np.round(centroids.max(0) - centroids.min(0), 3))
combined_bounds = [mins[0], maxs[0], mins[1], maxs[1], mins[2], maxs[2]]


# y-up camera so the vertical bounce (the ball moves along +y) reads as
# vertical on screen. The old camera_position="yz" put z up and y sideways,
# which made the bounce look like a horizontal "middle-to-left" drift.
_center = (mins + maxs) / 2.0
_diag = float(np.linalg.norm(maxs - mins))


def setup_cam(pl, az_deg=25.0):
    a = np.radians(az_deg)
    d = _diag * 2.2
    pos = _center + np.array([np.sin(a) * d, 0.25 * d, np.cos(a) * d])
    # (position, focal_point, view_up) with y as the up axis
    pl.camera_position = [tuple(pos), tuple(_center), (0.0, 1.0, 0.0)]


# ---- 1) trajectory overlay (time-colored) -----------------------------------
pl = pv.Plotter(off_screen=True, window_size=WINDOW)
pl.set_background(BG)
n_overlay = 16
idxs = np.linspace(0, len(ply_files) - 1, n_overlay).astype(int)
cmap = cm.get_cmap("turbo")
for k, i in enumerate(idxs):
    cloud = load(ply_files[i])
    rgb = np.array(cmap(k / (n_overlay - 1))[:3]) * 255
    cloud["t_color"] = np.tile(rgb, (cloud.n_points, 1)).astype(np.uint8)
    pl.add_mesh(cloud, scalars="t_color", rgb=True,
                render_points_as_spheres=True, point_size=POINT_SIZE,
                opacity=0.55)
setup_cam(pl)
try:
    pl.enable_anti_aliasing("ssaa")
except Exception:
    pass
path = os.path.join(OUT, "trajectory_overlay.png")
pl.screenshot(path)
pl.close()
print("wrote", path)

# ---- 2) simulation playback (real colors) -----------------------------------
frames = []
cname0 = color_array_name(load(ply_files[0]))
for p in ply_files:
    cloud = load(p)
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background(BG)
    if cname0:
        pl.add_mesh(cloud, scalars=cname0, rgb=True,
                    render_points_as_spheres=True, point_size=POINT_SIZE)
    else:
        pl.add_mesh(cloud, color="#d4e600",
                    render_points_as_spheres=True, point_size=POINT_SIZE)
    setup_cam(pl)
    frames.append(np.asarray(pl.screenshot(return_img=True)))
    pl.close()

imageio.mimsave(os.path.join(OUT, "simulation.gif"), frames, duration=1 / 24, loop=0)
print("wrote simulation.gif")
try:
    imageio.mimsave(os.path.join(OUT, "simulation.mp4"), frames, fps=24, quality=8)
    print("wrote simulation.mp4")
except Exception as e:
    print("mp4 skipped:", e)

# ---- 3) turntable around a mid-bounce frame ---------------------------------
mid = ply_files[len(ply_files) // 2]
cloud = load(mid)
cname = color_array_name(cloud)
tt = []
N = 60
for i in range(N):
    pl = pv.Plotter(off_screen=True, window_size=WINDOW)
    pl.set_background(BG)
    if cname:
        pl.add_mesh(cloud, scalars=cname, rgb=True,
                    render_points_as_spheres=True, point_size=POINT_SIZE)
    else:
        pl.add_mesh(cloud, color="#d4e600",
                    render_points_as_spheres=True, point_size=POINT_SIZE)
    pl.camera_position = "yz"
    pl.camera.azimuth = i * (360.0 / N)
    pl.camera.elevation = 15
    pl.reset_camera()
    pl.camera.zoom(1.1)
    tt.append(np.asarray(pl.screenshot(return_img=True)))
    pl.close()
try:
    imageio.mimsave(os.path.join(OUT, "turntable_mid.mp4"), tt, fps=24, quality=8)
    print("wrote turntable_mid.mp4")
except Exception as e:
    imageio.mimsave(os.path.join(OUT, "turntable_mid.gif"), tt, duration=1 / 24, loop=0)
    print("wrote turntable_mid.gif (mp4 skipped:", e, ")")

print("DONE ->", OUT)
