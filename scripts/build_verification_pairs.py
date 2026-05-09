"""Build an LFW-style pairs.txt + image tree from a held-out split for the
verification evaluator (TAR@FAR / ROC-AUC / EER).

The framework's lfw_pairs dataset hardcodes image paths to
`<eval_dir>/<class>/<class>_<NNNN>.<ext>`, so we symlink the source images
under that exact convention. Then we sample N folds × M positive (same
cow) and M negative (different cow) pairs at random.

Output:
    <dst>/<class>/<class>_0001.jpg, _0002.jpg, ...   (symlinks)
    <dst>/pairs.txt

Usage:
    uv run python scripts/build_verification_pairs.py \
        --src ./datasets/ceia_similarity_split/test \
        --dst ./datasets/ceia_similarity_pairs \
        --n-folds 10 --n-pairs-per-fold 100
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in _VALID_EXTS)


def build(
    src: Path, dst: Path, n_folds: int, n_pairs_per_fold: int, seed: int, ext: str
) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Symlink images under <dst>/<class>/<class>_NNNN.<ext>; remember per-class
    # index ranges so pair sampling can refer to images by (name, idx).
    class_indices: dict[str, list[int]] = {}
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        images = _list_images(class_dir)
        if not images:
            continue
        out_dir = dst / class_dir.name
        out_dir.mkdir()
        idxs: list[int] = []
        for i, img in enumerate(images, start=1):
            link = out_dir / f"{class_dir.name}_{i:04d}.{ext}"
            link.symlink_to(img.resolve())
            idxs.append(i)
        class_indices[class_dir.name] = idxs

    rng = random.Random(seed)
    classes_with_pos = [c for c, ix in class_indices.items() if len(ix) >= 2]
    all_classes = list(class_indices.keys())

    if len(classes_with_pos) < 2:
        raise ValueError(
            "Not enough classes with >=2 images to sample positive pairs. "
            f"Got {len(classes_with_pos)}."
        )

    pairs_path = dst / "pairs.txt"
    with open(pairs_path, "w") as f:
        f.write(f"{n_folds} {n_pairs_per_fold}\n")
        for _ in range(n_folds):
            # Positive pairs: same class, two distinct image indices.
            for _ in range(n_pairs_per_fold):
                cls = rng.choice(classes_with_pos)
                a, b = rng.sample(class_indices[cls], 2)
                f.write(f"{cls} {a} {b}\n")
            # Negative pairs: two distinct classes.
            for _ in range(n_pairs_per_fold):
                ca, cb = rng.sample(all_classes, 2)
                a = rng.choice(class_indices[ca])
                b = rng.choice(class_indices[cb])
                f.write(f"{ca} {a} {cb} {b}\n")

    n_imgs = sum(len(ix) for ix in class_indices.values())
    n_pairs = n_folds * n_pairs_per_fold * 2
    print(f"Verification layout written to {dst}")
    print(f"  classes        : {len(class_indices)} ({len(classes_with_pos)} with >=2 imgs)")
    print(f"  image symlinks : {n_imgs}")
    print(f"  pairs file     : {pairs_path}")
    print(f"  pairs total    : {n_pairs} ({n_folds} folds × {n_pairs_per_fold} same + {n_pairs_per_fold} diff)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--n-folds", type=int, default=10)
    p.add_argument("--n-pairs-per-fold", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ext", default="jpg", help="Image extension symlinks use")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.src, args.dst, args.n_folds, args.n_pairs_per_fold, args.seed, args.ext)
