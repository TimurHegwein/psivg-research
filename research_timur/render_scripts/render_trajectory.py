#!/usr/bin/env python3
"""Step-2 trajectory stills from the MPM point clouds (clean, slide-ready).

The ball bounces along the +y axis (vertical). Outputs into
research_timur/pod_results/presentation_renders/step2_pointcloud/:
  - trajectory_height.png  : vertical position vs time -> the damped bounce
  - trajectory_3d.png      : centroid path as a time-colored tube in 3D,
                             with ball snapshots at key frames
  - trajectory_overlay2.png: 8 evenly spaced ball snapshots, y-up view
"""
import os, glob
import numpy as np
import pyvista as pv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.signal import argrelextrema

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Defaults to tennisball; override via PSIVG_PLY_GLOB / PSIVG_RENDER_OUT.
PLY_DIR = glob.glob(os.environ.get("PSIVG_PLY_GLOB", os.path.join(
    REPO, "research_timur/pod_results/run_tennisball/OUT_Simulation/0001/*/point_cloud")))[0]
OUT = os.environ.get("PSIVG_RENDER_OUT", os.path.join(
    REPO, "research_timur/pod_results/presentation_renders/step2_pointcloud"))
os.makedirs(OUT, exist_ok=True)
# Object noun used in plot labels (e.g. "ball", "paper sheet"); cosmetic only.
OBJ_LABEL = os.environ.get("PSIVG_OBJECT_LABEL", "ball")
WINDOW = (1280, 1024)

fs = sorted(glob.glob(os.path.join(PLY_DIR, "*.ply")))
clouds = [pv.read(p) for p in fs]
C = np.array([c.points.mean(0) for c in clouds])
n = len(fs)
print(f"{n} frames")

# detect the per-point RGB(A) array name from the ORIGINAL data (before we add
# any helper arrays like "col" below)
CN = None
for k in clouds[1].point_data.keys():
    a = clouds[1].point_data[k]
    if a.ndim == 2 and a.shape[1] in (3, 4) and a.dtype == np.uint8:
        CN = k
        break
print("color array:", CN)

# ---- 1) bounce height curve -------------------------------------------------
y = C[:, 1]
t = np.arange(n)
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(t, y, color="#1f6f3f", lw=2.4)
# mark local minima (impacts)
mins = argrelextrema(y, np.less, order=4)[0]
ax.scatter(mins, y[mins], color="#d62728", zorder=5, s=45,
           label="ground contact (impact)")
ax.set_xlabel("Simulation frame", fontsize=12)
ax.set_ylabel(f"Vertical position of {OBJ_LABEL} (sim. units)", fontsize=12)
ax.set_title("Step 2 — Simulated physics: damped bouncing trajectory",
             fontsize=13, fontweight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
p1 = os.path.join(OUT, "trajectory_height.png")
fig.savefig(p1, dpi=140); plt.close(fig)
print("wrote", p1)

# ---- shared 3D camera (y up) ------------------------------------------------
mins_b = np.min([c.points.min(0) for c in clouds], 0)
maxs_b = np.max([c.points.max(0) for c in clouds], 0)
center = (mins_b + maxs_b) / 2
diag = np.linalg.norm(maxs_b - mins_b)


def y_up_cam(pl, az_deg=25):
    # look mostly along -z, slightly orbited; y stays vertical on screen
    a = np.radians(az_deg)
    d = diag * 2.2
    pos = center + np.array([np.sin(a) * d, 0.25 * d, np.cos(a) * d])
    pl.camera_position = [tuple(pos), tuple(center), (0, 1, 0)]


# ---- 2) 3D centroid tube colored by time + snapshots ------------------------
pl = pv.Plotter(off_screen=True, window_size=WINDOW)
pl.set_background("white")
path = pv.Spline(C, 400)
path["time"] = np.linspace(0, 1, path.n_points)
pl.add_mesh(path.tube(radius=diag * 0.006), scalars="time", cmap="turbo",
            show_scalar_bar=False)
# snapshots: start, first impact, first apex, end
key = [0, int(mins[0]) if len(mins) else n // 5, n // 4, n - 1]
cmap = cm.get_cmap("turbo")
for k in key:
    c = clouds[k]
    col = (np.array(cmap(k / (n - 1))[:3]) * 255).astype(np.uint8)
    c["col"] = np.tile(col, (c.n_points, 1))
    pl.add_mesh(c, scalars="col", rgb=True, render_points_as_spheres=True,
                point_size=3, opacity=0.85)
y_up_cam(pl)
try: pl.enable_anti_aliasing("ssaa")
except Exception: pass
p2 = os.path.join(OUT, "trajectory_3d.png")
pl.screenshot(p2); pl.close()
print("wrote", p2)

# ---- 3) clean overlay: 8 snapshots, true motion, y-up -----------------------
pl = pv.Plotter(off_screen=True, window_size=WINDOW)
pl.set_background("white")
idxs = np.linspace(0, n - 1, 8).astype(int)
for j, i in enumerate(idxs):
    c = clouds[i]
    op = 0.35 + 0.65 * (j / (len(idxs) - 1))  # later = more opaque
    if CN:
        pl.add_mesh(c, scalars=CN, rgb=True, render_points_as_spheres=True,
                    point_size=3, opacity=op)
    else:
        pl.add_mesh(c, color="#cde000", render_points_as_spheres=True,
                    point_size=3, opacity=op)
y_up_cam(pl)
try: pl.enable_anti_aliasing("ssaa")
except Exception: pass
p3 = os.path.join(OUT, "trajectory_overlay2.png")
pl.screenshot(p3); pl.close()
print("wrote", p3)
print("DONE ->", OUT)
