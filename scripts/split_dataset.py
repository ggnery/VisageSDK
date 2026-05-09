"""Split a flat <class>/<img> tree into train/, val/, and (optionally) test/.

The framework's `image_folder` dataset rebuilds `label_to_idx` per split, so
for label IDs to line up across splits each non-empty class must appear in
every produced split. This script enforces that:

- empty class folders are skipped entirely,
- singleton classes (1 image) duplicate that image into every split,
- 2-image classes (only when --test-frac > 0) place one image in train and
  one in val, then re-use the train image for test (test for those is a
  biased upper bound — flagged in the summary),
- 3+ image classes get a stratified split with at least 1 image in each
  split.

Pass `--test-frac 0` (default) for a 2-way train/val split. Any positive
`--test-frac` produces a 3-way split.

Usage:
    # 2-way (train/val) for the annotated dataset
    uv run python scripts/split_dataset.py \
        --src ./datasets/ceia_softsystem_annotated \
        --dst ./datasets/ceia_softsystem_annotated_split \
        --val-frac 0.2

    # 3-way (train/val/test) for the similarity dataset
    uv run python scripts/split_dataset.py \
        --src ./datasets/ceia_similarity_not_annotated \
        --dst ./datasets/ceia_similarity_split \
        --val-frac 0.15 \
        --test-frac 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in _VALID_EXTS)


def _assign(
    images: list[Path], val_frac: float, test_frac: float, rng: random.Random
) -> tuple[dict[str, list[Path]], bool]:
    """Return ({split_name: [imgs]}, duplicated_flag).

    `duplicated_flag` is True when at least one image had to be duplicated
    across splits because the class was too small for a clean partition.
    """
    n = len(images)
    has_test = test_frac > 0

    if n == 1:
        out = {"train": list(images), "val": list(images)}
        if has_test:
            out["test"] = list(images)
        return out, True

    shuffled = list(images)
    rng.shuffle(shuffled)

    if not has_test:
        n_val = max(1, min(n - 1, round(n * val_frac)))
        return {"val": shuffled[:n_val], "train": shuffled[n_val:]}, False

    # 3-way
    if n == 2:
        # 1 train, 1 val, dup train into test
        return {"train": shuffled[:1], "val": shuffled[1:2], "test": shuffled[:1]}, True

    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_train = n - n_test - n_val
    if n_train < 1:
        # n == 3 (or odd rounding): 1 each, no duplication
        return {
            "train": shuffled[0:1],
            "val": shuffled[1:2],
            "test": shuffled[2:3],
        }, False

    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }, False


def split(src: Path, dst: Path, val_frac: float, test_frac: float, seed: int) -> dict:
    if not src.exists():
        raise FileNotFoundError(src)
    if test_frac < 0 or val_frac <= 0:
        raise ValueError("val-frac must be > 0; test-frac must be >= 0")
    if val_frac + test_frac >= 1:
        raise ValueError("val-frac + test-frac must be < 1")

    if dst.exists():
        shutil.rmtree(dst)
    has_test = test_frac > 0
    splits = ["train", "val"] + (["test"] if has_test else [])
    for s in splits:
        (dst / s).mkdir(parents=True)

    rng = random.Random(seed)
    skipped_empty: list[str] = []
    duplicated: list[str] = []
    summary: list[tuple[str, int, dict[str, int]]] = []

    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        images = _list_images(class_dir)
        n = len(images)
        if n == 0:
            skipped_empty.append(class_dir.name)
            continue

        for s in splits:
            (dst / s / class_dir.name).mkdir()

        assignment, was_dup = _assign(images, val_frac, test_frac, rng)
        if was_dup:
            duplicated.append(class_dir.name)

        counts = {}
        for s, imgs in assignment.items():
            for f in imgs:
                shutil.copy2(f, dst / s / class_dir.name / f.name)
            counts[s] = len(imgs)
        summary.append((class_dir.name, n, counts))

    totals = {s: sum(c.get(s, 0) for _, _, c in summary) for s in splits}
    kept_classes = len(summary)

    print(f"Split written to {dst}")
    print(f"  num_classes (non-empty)  : {kept_classes}")
    print(f"  skipped (empty folders)  : {len(skipped_empty)} -> {skipped_empty}")
    print(f"  duplicated (small class) : {len(duplicated)} -> {duplicated}")
    print(f"  totals                   : {totals}")
    if duplicated:
        print(
            "  NOTE: duplicated classes have the same image in 2+ splits; "
            "metrics on those classes are a biased upper bound."
        )
    print()
    print("  per-class breakdown (orig / " + " / ".join(splits) + "):")
    for name, orig, counts in summary:
        cells = " / ".join(f"{counts.get(s, 0):>3d}" for s in splits)
        print(f"    {name:<14s} {orig:>3d} / {cells}")

    return {
        "num_classes": kept_classes,
        "skipped_empty": skipped_empty,
        "duplicated": duplicated,
        "totals": totals,
        "splits": splits,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--test-frac", type=float, default=0.0, help="0 disables test split (default)"
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split(args.src, args.dst, args.val_frac, args.test_frac, args.seed)
