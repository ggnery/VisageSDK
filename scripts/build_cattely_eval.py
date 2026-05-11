"""Build eval-only layouts for the Cattely cattle face dataset.

Source (read-only): ./datasets/Cattely-Cattle-Face-Images-Dataset/
   contains 47 cattle-ID folders (sXXX/, nXXXX/, snXX/) + a `valid/` YOLO
   detection folder we ignore. Only 45 IDs are kept after filtering:
   `s1806` ships only 1 image and is dropped (a single-image identity
   trivially gives rank-1=100% via gallery==probe duplication).

Produces four sibling directories under ./datasets/ :
   1. cattely_clean/                    symlinks to the 45 retained ID folders
   2. cattely_split/{train,val}/        50/50 stratified per-class split
   3. cattely_identification_eval/      gallery + probe symlinks (uses split)
   4. cattely_verification_pairs/       symlinks + pairs.txt (exhaustive)

Idempotent — safe to re-run. Existing destinations are removed first by
the underlying split / build helpers.

Usage:
    uv run python scripts/build_cattely_eval.py
    uv run python scripts/build_cattely_eval.py \
        --src ./datasets/Cattely-Cattle-Face-Images-Dataset \
        --out-dir ./datasets --seed 42 --val-frac 0.5 --n-folds 10
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from build_eval_layout import build as build_eval_layout
from build_verification_pairs import build as build_verification_pairs
from split_dataset import split as split_dataset

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_SKIP_DIRS = {"valid", ".git"}


def build_clean(src: Path, dst: Path) -> tuple[int, int]:
    """Symlink every cattle-ID folder with >=2 images into `dst`.

    Returns (n_classes, n_images). Returns image count via the symlinks so
    the user sees the real number that downstream scripts will consume.
    """
    if not src.exists():
        raise FileNotFoundError(
            f"Expected {src} (the upstream Cattely dataset) to exist."
        )

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    n_classes = 0
    n_images = 0
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        if d.name in _SKIP_DIRS:
            continue
        images = [p for p in d.iterdir() if p.is_file() and p.suffix.lower() in _VALID_EXTS]
        if len(images) < 2:
            print(f"   skip {d.name} (only {len(images)} image)")
            continue
        (dst / d.name).symlink_to(d.resolve(), target_is_directory=True)
        n_classes += 1
        n_images += len(images)
    return n_classes, n_images


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo_root = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--src",
        type=Path,
        default=repo_root / "datasets" / "Cattely-Cattle-Face-Images-Dataset",
        help="Upstream Cattely dataset directory.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "datasets",
        help="Parent directory for the four generated layouts.",
    )
    p.add_argument("--val-frac", type=float, default=0.5)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    clean = out_dir / "cattely_clean"
    split = out_dir / "cattely_split"
    ident = out_dir / "cattely_identification_eval"
    verif = out_dir / "cattely_verification_pairs"

    print(f"==> Building {clean} (symlinks to >=2-image cattle IDs only)")
    n_classes, n_images = build_clean(args.src, clean)
    print(f"   {n_classes} classes / {n_images} images")

    print()
    print(f"==> {int((1 - args.val_frac) * 100)}/{int(args.val_frac * 100)} stratified split (gallery <- train, probe <- val)")
    split_dataset(src=clean, dst=split, val_frac=args.val_frac, test_frac=0.0, seed=args.seed)

    print()
    print("==> Identification layout (gallery + probe symlinks)")
    build_eval_layout(src=split, dst=ident, gallery_split="train", probe_split="val")

    print()
    print("==> Verification pairs (exhaustive positives + exhaustive negatives)")
    build_verification_pairs(
        src=clean,
        dst=verif,
        n_folds=args.n_folds,
        n_pairs_per_fold=0,  # ignored in --exhaustive mode
        seed=args.seed,
        ext="jpg",
        exhaustive=True,
        exhaustive_negatives=True,
    )

    print()
    print("Done. Point eval.py at:")
    print("  configs/dataset/eval/cattely_identification.yaml  (identification)")
    print("  configs/dataset/eval/cattely_verification.yaml    (verification)")


if __name__ == "__main__":
    main()
