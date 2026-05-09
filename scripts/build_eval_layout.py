"""Build a gallery/probe layout for the IdentificationEvaluator from a
train/val[/test] tree produced by scripts/split_dataset.py.

The eval dataset expects:
    <eval_dir>/gallery/<class>/<img>
    <eval_dir>/probe/<class>/<img>

We symlink the chosen splits in place — no copies.

Usage:
    # Annotated (no test split): gallery=train, probe=val
    uv run python scripts/build_eval_layout.py \
        --src ./datasets/ceia_softsystem_annotated_split \
        --dst ./datasets/ceia_softsystem_annotated_eval \
        --gallery-split train --probe-split val

    # Similarity (held-out test): gallery=train, probe=test
    uv run python scripts/build_eval_layout.py \
        --src ./datasets/ceia_similarity_split \
        --dst ./datasets/ceia_similarity_eval \
        --gallery-split train --probe-split test
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def build(src: Path, dst: Path, gallery_split: str, probe_split: str) -> None:
    gallery_dir = src / gallery_split
    probe_dir = src / probe_split
    for d in (gallery_dir, probe_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Expected {d}. Run scripts/split_dataset.py first."
            )

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    (dst / "gallery").symlink_to(gallery_dir.resolve(), target_is_directory=True)
    (dst / "probe").symlink_to(probe_dir.resolve(), target_is_directory=True)

    n_gallery = sum(1 for _ in (dst / "gallery").rglob("*") if _.is_file())
    n_probe = sum(1 for _ in (dst / "probe").rglob("*") if _.is_file())
    n_classes_g = sum(1 for _ in (dst / "gallery").iterdir() if _.is_dir())
    n_classes_p = sum(1 for _ in (dst / "probe").iterdir() if _.is_dir())
    print(f"Eval layout written to {dst}")
    print(f"  gallery ({gallery_split}): {n_classes_g} classes, {n_gallery} images")
    print(f"  probe   ({probe_split}): {n_classes_p} classes, {n_probe} images")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--gallery-split", default="train")
    p.add_argument("--probe-split", default="val")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(args.src, args.dst, args.gallery_split, args.probe_split)
