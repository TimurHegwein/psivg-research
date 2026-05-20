#!/usr/bin/env python3
"""
Generate synthetic pipeline outputs for local layout testing.

Creates fake-but-realistic images in the exact directory structure
the pipeline produces, so visualize_pipeline.py can be tested
without running on a GPU cluster.

Usage:
    python generate_test_data.py
    python generate_test_data.py --samples 0001 0002
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJ_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJ_ROOT / "data_root"
W, H = 720, 480  # matches the pipeline's expected video resolution

# ── drawing helpers ────────────────────────────────────────────────────────────

def new_img(bg_color=(30, 30, 40)):
    return Image.new("RGB", (W, H), bg_color)


def draw_text(img, text, xy=(20, 20), color=(200, 200, 200), size=18):
    draw = ImageDraw.Draw(img)
    draw.text(xy, text, fill=color)
    return img


def draw_ball(img, cx, cy, r, color=(220, 220, 220), pattern=True):
    draw = ImageDraw.Draw(img)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=(50, 50, 50), width=2)
    if pattern:
        for angle in range(0, 360, 60):
            rad = np.deg2rad(angle)
            px = int(cx + r * 0.5 * np.cos(rad))
            py = int(cy + r * 0.5 * np.sin(rad))
            draw.ellipse([px - r // 5, py - r // 5, px + r // 5, py + r // 5],
                         fill=(30, 30, 30))
    return img


def save(img, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


# ── per-step synthetic images ──────────────────────────────────────────────────

def make_input_frame(sid, frame_idx=0, ball_x=None, ball_y=None, label=""):
    """Realistic-looking input frame: gradient background + ball."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    # sky gradient
    for row in range(H // 2):
        t = row / (H // 2)
        arr[row] = [int(60 + 100 * t), int(100 + 80 * t), int(160 + 60 * t)]
    # ground
    for row in range(H // 2, H):
        t = (row - H // 2) / (H // 2)
        arr[row] = [int(80 * (1 - t)), int(60 + 40 * (1 - t)), int(20)]
    img = Image.fromarray(arr)
    bx = ball_x or W // 2
    by = ball_y or H // 3
    draw_ball(img, bx, by, 55)
    draw_text(img, f"[{sid}] frame {frame_idx:05d}  {label}", color=(180, 180, 180))
    return img


def make_dino_sam(sid, ball_x=None, ball_y=None):
    bx = ball_x or W // 2
    by = ball_y or H // 3
    img = make_input_frame(sid, ball_x=bx, ball_y=by)
    draw = ImageDraw.Draw(img, "RGBA")
    r = 55
    # SAM mask overlay
    draw.ellipse([bx - r, by - r, bx + r, by + r],
                 fill=(0, 120, 255, 80), outline=(0, 180, 255, 220), width=2)
    # bounding box
    pad = 15
    draw.rectangle([bx - r - pad, by - r - pad, bx + r + pad, by + r + pad],
                   outline=(0, 255, 0, 255), width=3)
    draw.text((bx - r - pad, by - r - pad - 20), "ball (0.94)", fill=(0, 255, 0))
    return img.convert("RGB")


def make_mask(ball_x=None, ball_y=None):
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    bx = ball_x or W // 2
    by = ball_y or H // 3
    draw.ellipse([bx - 60, by - 60, bx + 60, by + 60], fill=(255, 255, 255))
    return img


def make_transparent_crop(color=(210, 210, 210)):
    size = 200
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([10, 10, size - 10, size - 10], fill=(*color, 255),
                 outline=(50, 50, 50, 255), width=2)
    for angle in range(0, 360, 60):
        rad = np.deg2rad(angle)
        cx, cy = size // 2, size // 2
        px = int(cx + 60 * np.cos(rad))
        py = int(cy + 60 * np.sin(rad))
        draw.ellipse([px - 15, py - 15, px + 15, py + 15], fill=(30, 30, 30, 255))
    return img


def make_inpainted(ball_x=None, ball_y=None):
    """Background with object removed (smeared patch)."""
    img = make_input_frame("bg", ball_x=-200, ball_y=-200)  # ball off-screen
    draw = ImageDraw.Draw(img)
    bx = ball_x or W // 2
    by = ball_y or H // 3
    # simulate inpaint patch — slightly different color
    for dr in range(70, 0, -10):
        alpha = int(255 * (1 - dr / 70))
        draw.ellipse([bx - dr, by - dr, bx + dr, by + dr],
                     fill=(100, 130, 160) if by < H // 2 else (70, 55, 20))
    return img


def make_multiview(view_idx, color=(210, 210, 210)):
    """One of InstantMesh's 6 multi-view renders."""
    angles = [0, 60, 120, 180, 240, 300]
    angle = np.deg2rad(angles[view_idx % len(angles)])
    img = Image.new("RGB", (256, 256), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    # simulate a slightly different oval for each viewpoint
    rx, ry = 90, int(90 * abs(np.cos(angle)) + 30 * abs(np.sin(angle)))
    cx, cy = 128, 128
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=color,
                 outline=(50, 50, 50), width=2)
    for a2 in range(0, 360, 60):
        r2 = np.deg2rad(a2)
        px = int(cx + rx * 0.55 * np.cos(r2))
        py = int(cy + ry * 0.55 * np.sin(r2))
        draw.ellipse([px - 14, py - 10, px + 14, py + 10], fill=(30, 30, 30))
    draw.text((5, 5), f"view {view_idx + 1}", fill=(150, 150, 150))
    return img


def make_simple_obj(path):
    """Write a minimal UV-sphere .obj for mesh rendering tests."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    verts, faces = [], []
    stacks, slices = 12, 16
    for i in range(stacks + 1):
        phi = np.pi * i / stacks
        for j in range(slices):
            theta = 2 * np.pi * j / slices
            x = np.sin(phi) * np.cos(theta)
            y = np.cos(phi)
            z = np.sin(phi) * np.sin(theta)
            verts.append((x, y, z))
    for i in range(stacks):
        for j in range(slices):
            a = i * slices + j
            b = a + slices
            c = i * slices + (j + 1) % slices
            d = b + (j + 1) % slices - j
            faces.append((a + 1, b + 1, c + 1))
            faces.append((b + 1, d + 1, c + 1))
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


def make_particle_npz(path, n=8000):
    """Write a fake particle snapshot in the format ParticleIO expects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # sphere of particles
    phi = np.random.uniform(0, np.pi, n)
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = np.random.uniform(0, 0.08, n)
    x = np.stack([
        r * np.sin(phi) * np.cos(theta),
        r * np.cos(phi) + 0.5,
        r * np.sin(phi) * np.sin(theta),
    ], axis=1).astype(np.float32)

    x_bits, v_bits = 24, 8
    ranges = np.zeros((2, 3, 2), dtype=np.float32)
    for d in range(3):
        ranges[0, d] = [x[:, d].min(), x[:, d].max() + 1e-5]
    x_norm = ((x - ranges[0, :, 0]) /
              (ranges[0, :, 1] - ranges[0, :, 0]) *
              (2 ** x_bits - 1)).astype(np.uint32)
    x_and_v = (x_norm << v_bits)

    color = np.tile([210, 210, 210], (n, 1)).astype(np.uint8)
    # tint by height
    color[:, 0] = np.clip(180 + (x[:, 1] * 200).astype(int), 0, 255)
    color[:, 2] = np.clip(100 - (x[:, 1] * 100).astype(int), 0, 255)

    np.savez(path, ranges=ranges, x_and_v=x_and_v, color=color)


def make_rendered_frame(t, ball_x_start=150, ball_x_end=570):
    """Simulate a rendered ball-in-flight frame at time t ∈ [0,1]."""
    img = new_img((15, 15, 20))
    draw = ImageDraw.Draw(img)
    # simple floor
    draw.rectangle([0, H * 2 // 3, W, H], fill=(30, 40, 30))
    bx = int(ball_x_start + (ball_x_end - ball_x_start) * t)
    by = int(H // 3 + 80 * np.sin(np.pi * t))  # arc
    draw_ball(img, bx, by, 50)
    draw_text(img, f"rendered  t={t:.2f}", color=(140, 140, 140))
    return img


def make_flow_frame(t):
    """HSV-encoded optical flow visualization."""
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    cx, cy = W // 2, H // 2
    for y in range(0, H, 4):
        for x in range(0, W, 4):
            dx = (x - cx) / W * 2 + np.cos(t * np.pi) * 0.3
            dy = (y - cy) / H * 2 + np.sin(t * np.pi) * 0.2
            angle = (np.arctan2(dy, dx) / np.pi + 1) / 2
            mag = min(1.0, np.sqrt(dx ** 2 + dy ** 2) * 2)
            h = int(angle * 179)
            s = int(mag * 255)
            v = 200
            # HSV → RGB via numpy
            hi = (h // 30) % 6
            f2 = (h % 30) / 30.0
            p = int(v * (1 - s / 255))
            q = int(v * (1 - f2 * s / 255))
            t2 = int(v * (1 - (1 - f2) * s / 255))
            rgb = [(v, t2, p), (q, v, p), (p, v, t2), (p, q, v), (t2, p, v), (v, p, q)][hi]
            arr[y:y+4, x:x+4] = rgb
    img = Image.fromarray(arr)
    draw_text(img, f"optical flow  t={t:.2f}", color=(180, 180, 180))
    return img


# ── scaffold builder ───────────────────────────────────────────────────────────

SAMPLE_CONFIGS = {
    "0001": {"label": "tennis ball (static cam)", "ball_x": 360, "ball_y": 160,
             "ball_color": (220, 190, 50), "bad": False},
    "0002": {"label": "football (moving cam)",    "ball_x": 280, "ball_y": 200,
             "ball_color": (40,  40,  40), "bad": True},
}


def build_sample(sid):
    cfg = SAMPLE_CONFIGS.get(sid, {"label": sid, "ball_x": W//2, "ball_y": H//3,
                                    "ball_color": (200, 200, 200), "bad": False})
    bx, by, bc = cfg["ball_x"], cfg["ball_y"], cfg["ball_color"]
    label = cfg["label"]
    print(f"  [{sid}] {label}")

    # ── Step 0: metadata + input ──────────────────────────────────────────
    meta_path = DATA_ROOT / "INPUT_DATA" / "Metadata" / f"{sid}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    existing = PROJ_ROOT / "assets" / f"{sid}.json"
    if existing.exists():
        import shutil
        shutil.copy(existing, meta_path)
    else:
        meta_path.write_text(json.dumps({"primary": "ball", "fg_prompt": "ball",
                                          "video_prompt": label}))

    frames_dir = DATA_ROOT / "INPUT_DATA" / "Frames" / sid
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        save(make_input_frame(sid, i, bx, by + i * 10, label),
             frames_dir / f"{i:05d}.jpg")

    # ── Step 1: perception ────────────────────────────────────────────────
    pd = DATA_ROOT / "OUT_Perception" / sid / "00000"

    save(make_input_frame(sid, 0, bx, by, label), pd / "raw_image.jpg")
    save(make_dino_sam(sid, bx, by),               pd / "grounded_sam_output.jpg")
    save(make_mask(bx, by),                        pd / "mask" / "mask.jpg")

    # object crops
    obj_dir = pd / "objects"
    save(make_transparent_crop(bc), obj_dir / "ball_transparent.png")

    # inpainted background
    save(make_inpainted(bx, by), pd / "inpaint" / "inpainted_all.jpg")

    # multi-view renders
    img_dir = pd / "images"
    for i in range(4):
        save(make_multiview(i, bc), img_dir / f"mv_{i:02d}.png")

    # 3D mesh
    make_simple_obj(pd / "meshes" / "ball.obj")

    # ── Step 2: simulation ────────────────────────────────────────────────
    sim_dir = DATA_ROOT / "OUT_Simulation" / sid / "2026-01-01_run"
    for i, t in enumerate(np.linspace(0, 1, 9)):
        make_particle_npz(sim_dir / f"particles_{i:04d}.npz")

    rend_dir = DATA_ROOT / "OUT_Rendering" / sid / "2026-01-01_run"
    rgb_dir = rend_dir / "rgb"
    flow_dir = rend_dir / "flow_visual"
    for i, t in enumerate(np.linspace(0, 1, 9)):
        x_start = bx - 120 if not cfg["bad"] else bx - 200
        save(make_rendered_frame(t, x_start, bx + 120), rgb_dir  / f"{i:05d}.jpg")
        save(make_flow_frame(t),                          flow_dir / f"{i:05d}.png")

    # ── Step 4: output video frames ───────────────────────────────────────
    out_dir = PROJ_ROOT / "outputs" / "generated_data_example" / \
              "expt_LRcosine_lr0.0002_iter50" / sid
    for i, t in enumerate(np.linspace(0, 1, 9)):
        save(make_rendered_frame(t, bx - 180, bx + 180), out_dir / f"{i:05d}.jpg")

    (pd / "success.txt").touch()
    print(f"     → {pd}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data")
    parser.add_argument("--samples", nargs="+", default=["0001", "0002"])
    args = parser.parse_args()

    print("Generating synthetic pipeline outputs...")
    for sid in args.samples:
        build_sample(sid)
    print("\nDone. Now run:")
    print("  python research_timur/visualize_pipeline.py --out figures/")


if __name__ == "__main__":
    main()
