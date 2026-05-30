"""Render a single PNG with two 10×10 grids of grad heatmaps — top grid
is DINOv3-B v1, bottom grid is MegaDescriptor-L. Same 100 random test cows
in the same positions in both grids so you can compare attention directly.

Avoids the 200× python+model-load overhead of calling visualize_attention.py
in a loop — loads each model once.

Usage:
    uv run python scripts/heatmap_grid.py \\
        --out heatmaps/_grid_100_cows.png \\
        --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reuse the heatmap machinery instead of duplicating it.
from visualize_attention import (  # noqa: E402
    _dinov3_heatmaps,
    _grid_to_overlay,
    _load_backbone,
    _megadescriptor_heatmaps,
    _pick_checkpoint,
    _preprocess,
    _read_env,
    _read_norm,
)


def _render_overlays(run_dir: Path, dispatch: str, images: list[Path],
                     device: torch.device) -> list[Image.Image]:
    """Load model once, return grad-overlay PIL for each image."""
    env = _read_env(run_dir)
    ckpt = _pick_checkpoint(run_dir, None)
    force_eager = env["BACKBONE"] == "dinov3"
    backbone, bb_cfg, _ = _load_backbone(env, ckpt, device, force_eager_attn=force_eager)
    mean, std = _read_norm(env)
    input_size = tuple(int(s) for s in bb_cfg.input_size)

    overlays: list[Image.Image] = []
    for img_path in tqdm(images, desc=f"{dispatch:<14}"):
        img_tensor, img_pil = _preprocess(img_path, input_size, mean, std)
        img_tensor = img_tensor.to(device)
        if dispatch == "dinov3":
            heatmaps = _dinov3_heatmaps(backbone, img_tensor, method="grad", ref_emb=None)
        elif dispatch == "megadescriptor":
            heatmaps = _megadescriptor_heatmaps(backbone, img_tensor, ref_emb=None)
        else:
            raise ValueError(f"unknown dispatch {dispatch!r}")
        # Single entry — the grad heatmap.
        grid = next(iter(heatmaps.values()))
        overlay = _grid_to_overlay(grid, img_pil)
        overlays.append(overlay)
    return overlays


def _tile(overlays: list[Image.Image], rows: int, cols: int, cell: int) -> Image.Image:
    """Lay overlays into a `rows × cols` grid of `cell × cell` squares."""
    canvas = Image.new("RGB", (cols * cell, rows * cell), "white")
    for i, im in enumerate(overlays):
        r, c = divmod(i, cols)
        if r >= rows:
            break
        thumb = im.resize((cell, cell), Image.LANCZOS)
        canvas.paste(thumb, (c * cell, r * cell))
    return canvas


def _add_title(img: Image.Image, text: str, height: int = 40) -> Image.Image:
    """Add a white title bar on top of `img`."""
    out = Image.new("RGB", (img.size[0], img.size[1] + height), "white")
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    draw.text(((img.size[0] - tw) // 2, (height - 22) // 2), text, fill="black", font=font)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-root", type=Path,
                   default=REPO_ROOT / "datasets/ceia_motion_detection_faces_split/test")
    p.add_argument("--dinov3-run",
                   default="2026-05-17T02-11-52_dinov3_cow_faces_v1")
    p.add_argument("--mega-run",
                   default="2026-05-17T14-20-09_megadescriptor_l384_arcface_sota")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "heatmaps/_grid_100_cows.png")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--cell", type=int, default=160, help="thumbnail side in pixels")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    # Square-ish grid: rows × cols ≈ n.
    cols = int(round(args.n**0.5))
    rows = (args.n + cols - 1) // cols
    print(f"[info] grid layout {rows} × {cols} = {rows * cols} cells (asked for {args.n})")

    # Sample N images from the test split.
    all_imgs = sorted(
        p for cls in args.test_root.iterdir() if cls.is_dir()
        for p in cls.iterdir() if p.suffix.lower() == ".png"
    )
    if len(all_imgs) < args.n:
        raise SystemExit(f"only {len(all_imgs)} test images, need {args.n}")
    rng = random.Random(args.seed)
    images = rng.sample(all_imgs, args.n)
    print(f"[info] sampled {len(images)} images from {args.test_root}")

    device = torch.device(args.device)
    runs_dir = REPO_ROOT / "runs/trains"

    dinov3_overlays = _render_overlays(runs_dir / args.dinov3_run, "dinov3", images, device)
    mega_overlays = _render_overlays(runs_dir / args.mega_run, "megadescriptor", images, device)

    grid_d = _tile(dinov3_overlays, rows, cols, args.cell)
    grid_m = _tile(mega_overlays, rows, cols, args.cell)
    grid_d = _add_title(grid_d, "DINOv3-B + CosFace (rank-1 83.9%) — grad ||emb||² overlay")
    grid_m = _add_title(grid_m, "MegaDescriptor-L + ArcFace (rank-1 77.8% peak) — Grad-CAM overlay")

    # Stack the two titled grids vertically.
    final = Image.new("RGB", (grid_d.size[0], grid_d.size[1] + grid_m.size[1]), "white")
    final.paste(grid_d, (0, 0))
    final.paste(grid_m, (0, grid_d.size[1]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    final.save(args.out)
    print(f"[ok] wrote {args.out} ({final.size[0]}×{final.size[1]})")


if __name__ == "__main__":
    main()
