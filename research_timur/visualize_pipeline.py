#!/usr/bin/env python3
"""
PSIVG Pipeline Visualization
Generates per-step I/O figures for a good and a bad example.

Usage:
    python visualize_pipeline.py
    python visualize_pipeline.py --good 0001 --bad 0002 --out figures/
"""

import argparse
import sys
from io import BytesIO
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJ_ROOT = Path(__file__).parent.parent   # repo root
DATA_ROOT = PROJ_ROOT / "data_root"
OUTPUTS_DIR = PROJ_ROOT / "outputs"

DARK_BG = "#111111"
TITLE_COLOR = "#eeeeee"
LABEL_COLOR = "#aaaaaa"
PLACEHOLDER_BG = "#1e1e1e"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,
    "axes.facecolor": DARK_BG,
    "text.color": TITLE_COLOR,
})


# ── image helpers ──────────────────────────────────────────────────────────────

def load_image(path, size=None):
    if path is None or not Path(path).exists():
        return None
    img = Image.open(path).convert("RGB")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def load_rgba(path, size=None):
    if path is None or not Path(path).exists():
        return None
    img = Image.open(path).convert("RGBA")
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img)


def composite_rgba_on_white(rgba):
    if rgba is None:
        return None
    bg = np.ones((*rgba.shape[:2], 3), dtype=np.uint8) * 255
    alpha = rgba[:, :, 3:] / 255.0
    return (rgba[:, :, :3] * alpha + bg * (1 - alpha)).astype(np.uint8)


def fig_to_array(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def placeholder(label, w=480, h=360):
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100,
                           facecolor=PLACEHOLDER_BG)
    ax.set_facecolor(PLACEHOLDER_BG)
    ax.text(0.5, 0.5, f"[ {label} ]\nnot yet available",
            ha="center", va="center", color="#555555",
            fontsize=8, transform=ax.transAxes, linespacing=1.8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
    return fig_to_array(fig)


def imshow(ax, img, title=None, cmap=None):
    if img is None:
        img = placeholder(title or "missing")
    ax.imshow(img, cmap=cmap, aspect="auto")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, color=LABEL_COLOR, pad=4)


# ── 3D render helpers ──────────────────────────────────────────────────────────

def render_mesh(obj_path, size=(400, 400)):
    try:
        import trimesh
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        mesh = trimesh.load(str(obj_path), force="mesh")
        v, f = mesh.vertices, mesh.faces

        fig = plt.figure(figsize=(4, 4), facecolor="#0d0d0d")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")

        step = max(1, len(f) // 4000)
        tri = v[f[::step]]
        poly = Poly3DCollection(tri, alpha=0.8, linewidths=0)
        poly.set_facecolor("#4a90d9")
        poly.set_edgecolor("none")
        ax.add_collection3d(poly)

        for dim, setter in enumerate([ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
            setter([v[:, dim].min(), v[:, dim].max()])
        ax.set_axis_off()
        ax.view_init(elev=20, azim=135)

        return fig_to_array(fig)
    except Exception:
        return None


def render_particles(npz_path, size=(400, 400)):
    try:
        sys.path.insert(0, str(PROJ_ROOT))
        from psivg.rendering.particle_io import ParticleIO

        x, _, color = ParticleIO.read_particles_3d(str(npz_path))
        idx = np.random.choice(len(x), min(6000, len(x)), replace=False)
        x, c = x[idx], color[idx] / 255.0

        fig = plt.figure(figsize=(4, 4), facecolor="#0d0d0d")
        ax = fig.add_subplot(111, projection="3d", facecolor="#0d0d0d")
        ax.scatter(x[:, 0], x[:, 2], x[:, 1], c=c, s=0.8, alpha=0.7,
                   linewidths=0)
        ax.set_axis_off()
        ax.view_init(elev=20, azim=45)

        return fig_to_array(fig)
    except Exception:
        return None


# ── path helpers ───────────────────────────────────────────────────────────────

def perception_dir(sid):
    return DATA_ROOT / "OUT_Perception" / sid / "00000"


def get_multiview_images(sid, n=4):
    img_dir = perception_dir(sid) / "images"
    if not img_dir.exists():
        return [None] * n
    imgs = sorted(img_dir.glob("*.png"))[:n]
    return [load_image(p) for p in imgs] + [None] * (n - len(imgs))


def get_mesh_path(sid):
    mesh_dir = perception_dir(sid) / "meshes"
    if not mesh_dir.exists():
        return None
    objs = list(mesh_dir.glob("*.obj"))
    return objs[0] if objs else None


def _latest_run(root_dir):
    if not root_dir.exists():
        return None
    runs = sorted(d for d in root_dir.iterdir() if d.is_dir())
    return runs[-1] if runs else None


def get_particle_snapshots(sid, n=3):
    sim_dir = _latest_run(DATA_ROOT / "OUT_Simulation" / sid)
    if sim_dir is None:
        return [None] * n
    npzs = sorted(sim_dir.glob("*.npz"))
    if not npzs:
        return [None] * n
    idxs = np.linspace(0, len(npzs) - 1, n, dtype=int)
    return [render_particles(npzs[i]) for i in idxs]


def _get_frames(directory, n):
    frames = sorted(directory.glob("*.jpg")) + sorted(directory.glob("*.png"))
    if not frames:
        return [None] * n
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [load_image(frames[i]) for i in idxs]


def get_rendered_frames(sid, n=3):
    rend_dir = _latest_run(DATA_ROOT / "OUT_Rendering" / sid)
    if rend_dir is None:
        return [None] * n
    # Real pipeline output: original_length/final/  (fallback: rgb/ for test data)
    for candidate in [rend_dir / "original_length" / "final",
                      rend_dir / "more_frames" / "final",
                      rend_dir / "rgb",
                      rend_dir]:
        if candidate.exists():
            frames = _get_frames(candidate, n)
            if any(f is not None for f in frames):
                return frames
    return [None] * n


def get_flow_frames(sid, n=3):
    rend_dir = _latest_run(DATA_ROOT / "OUT_Rendering" / sid)
    if rend_dir is None:
        return [None] * n
    # Real pipeline output: original_length/flow_visual/  (fallback: flow_visual/)
    for candidate in [rend_dir / "original_length" / "flow_visual",
                      rend_dir / "more_frames" / "flow_visual",
                      rend_dir / "flow_visual"]:
        if candidate.exists():
            frames = _get_frames(candidate, n)
            if any(f is not None for f in frames):
                return frames
    return [None] * n


def get_output_frames(sid, n=3):
    for run_dir in sorted(OUTPUTS_DIR.glob(f"*/*/{sid}")):
        return _get_frames(run_dir, n)
    return [None] * n


def get_first_frame(sid):
    return load_image(DATA_ROOT / "INPUT_DATA" / "Frames" / sid / "00000.jpg")


# ── figure 1: perception ───────────────────────────────────────────────────────

def fig_perception(sid, out_path):
    pd = perception_dir(sid)

    raw       = load_image(pd / "raw_image.jpg")
    dino_sam  = load_image(pd / "grounded_sam_output.jpg")
    mask      = load_image(pd / "mask" / "mask.jpg")

    obj_dir = pd / "objects"
    transparent = None
    if obj_dir.exists():
        pngs = list(obj_dir.glob("*_transparent.png"))
        if pngs:
            transparent = composite_rgba_on_white(load_rgba(pngs[0]))

    inpainted = load_image(pd / "inpaint" / "inpainted_all.jpg")
    mv = get_multiview_images(sid, n=4)
    mesh_img = render_mesh(get_mesh_path(sid)) if get_mesh_path(sid) else None

    fig = plt.figure(figsize=(18, 11), facecolor=DARK_BG)
    fig.suptitle(f"Step 1 — Perception Pipeline   [{sid}]",
                 fontsize=14, color=TITLE_COLOR, y=0.99, fontweight="bold")

    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.38, wspace=0.06,
                           top=0.94, bottom=0.02, left=0.02, right=0.98)

    # row 0 — detection & segmentation
    imshow(fig.add_subplot(gs[0, :2]), raw,      "Input frame (first frame)")
    imshow(fig.add_subplot(gs[0, 2:4]), dino_sam, "GroundingDINO + SAM overlay")
    imshow(fig.add_subplot(gs[0, 4:]), mask,      "Segmentation mask")

    # row 1 — object crop, inpainted bg, multi-view
    imshow(fig.add_subplot(gs[1, :2]), transparent, "Cropped foreground object")
    imshow(fig.add_subplot(gs[1, 2:4]), inpainted,  "LaMa — inpainted background")
    for i in range(2):
        imshow(fig.add_subplot(gs[1, 4 + i]), mv[i], f"InstantMesh multi-view {i+1}")

    # row 2 — remaining multi-views + 3D mesh
    for i in range(2):
        imshow(fig.add_subplot(gs[2, i * 2: i * 2 + 2]), mv[2 + i],
               f"InstantMesh multi-view {i+3}")
    imshow(fig.add_subplot(gs[2, 4:]), mesh_img, "3D mesh reconstruction")

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── figure 2: simulation ───────────────────────────────────────────────────────

def fig_simulation(sid, out_path):
    particles = get_particle_snapshots(sid, n=3)
    rendered  = get_rendered_frames(sid, n=3)
    flow      = get_flow_frames(sid, n=3)

    fig = plt.figure(figsize=(15, 9), facecolor=DARK_BG)
    fig.suptitle(f"Step 2 — Physics Simulation   [{sid}]",
                 fontsize=14, color=TITLE_COLOR, y=0.99, fontweight="bold")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.06,
                           top=0.94, bottom=0.02, left=0.02, right=0.98)

    t_labels = ["t = 0  (start)", "t = 0.5  (mid)", "t = 1  (end)"]
    row_labels = ["Taichi MPM particles", "Rendered RGB", "Optical flow"]
    rows = [particles, rendered, flow]

    for row, (row_data, row_label) in enumerate(zip(rows, row_labels)):
        for col, (img, t_label) in enumerate(zip(row_data, t_labels)):
            ax = fig.add_subplot(gs[row, col])
            title = f"{row_label} — {t_label}" if row == 0 else t_label
            imshow(ax, img, title)
            if col == 0:
                ax.set_ylabel(row_label, fontsize=8, color=LABEL_COLOR)

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── figure 3: good vs bad comparison ──────────────────────────────────────────

def fig_comparison(good_id, bad_id, out_path):
    def first_or_none(lst):
        return lst[0] if lst else None

    steps = [
        ("Input\nframe", [
            get_first_frame(good_id),
            get_first_frame(bad_id),
        ]),
        ("GroundingDINO\n+ SAM", [
            load_image(perception_dir(good_id) / "grounded_sam_output.jpg"),
            load_image(perception_dir(bad_id)  / "grounded_sam_output.jpg"),
        ]),
        ("Inpainted\nbackground", [
            load_image(perception_dir(good_id) / "inpaint" / "inpainted_all.jpg"),
            load_image(perception_dir(bad_id)  / "inpaint" / "inpainted_all.jpg"),
        ]),
        ("3D mesh\n(InstantMesh)", [
            render_mesh(get_mesh_path(good_id)) if get_mesh_path(good_id) else None,
            render_mesh(get_mesh_path(bad_id))  if get_mesh_path(bad_id)  else None,
        ]),
        ("Simulation\n(rendered)", [
            first_or_none(get_rendered_frames(good_id, 1)),
            first_or_none(get_rendered_frames(bad_id,  1)),
        ]),
        ("Output\nvideo frame", [
            first_or_none(get_output_frames(good_id, 1)),
            first_or_none(get_output_frames(bad_id,  1)),
        ]),
    ]

    n = len(steps)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7), facecolor=DARK_BG)
    fig.suptitle(
        f"Pipeline Overview — Good [{good_id}] vs Bad [{bad_id}]",
        fontsize=13, color=TITLE_COLOR, y=1.01, fontweight="bold",
    )

    row_labels = [f"Good  [{good_id}]", f"Bad   [{bad_id}]"]
    for row in range(2):
        for col, (step_name, imgs) in enumerate(steps):
            ax = axes[row, col]
            imshow(ax, imgs[row], step_name if row == 0 else None)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9, color=LABEL_COLOR)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    print(f"  Saved {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize PSIVG pipeline outputs")
    parser.add_argument("--good", default="0001", help="Good example sample ID")
    parser.add_argument("--bad",  default="0002", help="Bad example sample ID")
    parser.add_argument("--out",  default="figures", help="Output directory")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n── Perception figures ──")
    fig_perception(args.good, out / f"perception_{args.good}.png")
    fig_perception(args.bad,  out / f"perception_{args.bad}.png")

    print("\n── Simulation figures ──")
    fig_simulation(args.good, out / f"simulation_{args.good}.png")
    fig_simulation(args.bad,  out / f"simulation_{args.bad}.png")

    print("\n── Comparison figure ──")
    fig_comparison(args.good, args.bad, out / "comparison.png")

    print(f"\nDone. All figures in ./{out}/")


if __name__ == "__main__":
    main()
