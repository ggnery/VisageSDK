"""Dedupe-aware train/val/test split.

Many cattle-id / face-id datasets are built from video frames extracted
at high FPS. Two frames captured 33 ms apart look nearly identical at the
embedding-resolution scale (e.g. 224×224). The standard `split_dataset.py`
sorts frames into splits at random, so adjacent frames leak between train
and test — the resulting metrics measure "does the model recognize this
specific frame" rather than "does the model generalize to a new view of
this individual".

This script fixes that by clustering near-duplicate frames within each
class via perceptual hashing (dHash), then splitting CLUSTERS (not
individual images) across train/val/test. Frames from the same moment
stay together, eliminating the leak.

A cluster is a connected component under the relation
    `Hamming(dHash(a), dHash(b)) <= --hamming-threshold`.

Usage:
    uv run python scripts/dedupe_and_split.py \\
        --src ./datasets/ceia_motion_detection \\
        --dst ./datasets/ceia_motion_detection_clean_split \\
        --val-frac 0.15 --test-frac 0.15 \\
        --hamming-threshold 5
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def _list_images(class_dir: Path) -> list[Path]:
    return sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in _VALID_EXTS)


def _dhash(path: Path, size: int = 8) -> int:
    """Return a `size*size`-bit difference hash packed as int.

    Resize to (size+1, size) → grayscale → row-wise pairwise diff sign → bits.
    Distance = popcount(hash1 ^ hash2).
    """
    img = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _cluster(hashes: list[int], threshold: int) -> list[list[int]]:
    """Union-find clustering of hash indices by Hamming distance ≤ threshold."""
    n = len(hashes)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n):
        for j in range(i + 1, n):
            if _hamming(hashes[i], hashes[j]) <= threshold:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    # Sort clusters by descending size so downstream allocation is deterministic.
    return sorted(groups.values(), key=len, reverse=True)


def _assign_clusters(
    clusters: list[list[Path]],
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> tuple[dict[str, list[Path]], bool]:
    """Allocate whole clusters to splits, approximating frac targets by image count.

    Returns ({split: [paths]}, undersplit_flag). `undersplit_flag` is True
    when the class has fewer clusters than the requested number of splits;
    everything falls into `train` in that case (and the user gets a warning).
    """
    has_test = test_frac > 0
    splits = ["train", "val"] + (["test"] if has_test else [])
    n_splits_needed = 3 if has_test else 2

    if len(clusters) < n_splits_needed:
        out: dict[str, list[Path]] = {s: [] for s in splits}
        for c in clusters:
            out["train"].extend(c)
        return out, True

    # Reserve the smallest clusters for the smallest splits first; keeps the
    # diverse, large clusters available for train where they matter most.
    clusters_sorted = sorted(clusters, key=len)
    assignments: dict[str, list[list[Path]]] = {s: [] for s in splits}
    counts = {s: 0 for s in splits}

    # Force at least one cluster into each non-train split.
    forced = ["val"] + (["test"] if has_test else [])
    for split in forced:
        c = clusters_sorted.pop(0)
        assignments[split].append(c)
        counts[split] += len(c)

    n_imgs = sum(len(c) for c in clusters)
    targets = {"train": n_imgs * (1.0 - val_frac - test_frac), "val": n_imgs * val_frac}
    if has_test:
        targets["test"] = n_imgs * test_frac

    # Greedy: place each remaining cluster into whichever split has the
    # largest absolute under-allocation. This converges to the target
    # fractions in image count while keeping cluster atoms intact.
    remaining = clusters_sorted[:]
    rng.shuffle(remaining)
    for c in remaining:
        choice = max(splits, key=lambda s: targets[s] - counts[s])
        assignments[choice].append(c)
        counts[choice] += len(c)

    flat = {s: [p for c in assignments[s] for p in c] for s in splits}
    return flat, False


def dedupe_split(
    src: Path,
    dst: Path,
    val_frac: float,
    test_frac: float,
    seed: int,
    hamming_threshold: int,
    hash_size: int = 8,
) -> dict:
    if not src.exists():
        raise FileNotFoundError(src)
    if val_frac <= 0 or test_frac < 0:
        raise ValueError("val-frac must be > 0; test-frac must be >= 0")
    if val_frac + test_frac >= 1:
        raise ValueError("val-frac + test-frac must be < 1")

    has_test = test_frac > 0
    splits = ["train", "val"] + (["test"] if has_test else [])

    if dst.exists():
        shutil.rmtree(dst)
    for s in splits:
        (dst / s).mkdir(parents=True)

    rng = random.Random(seed)

    skipped_empty: list[str] = []
    undersplit: list[str] = []
    per_class: list[tuple[str, int, int, dict[str, int]]] = []
    pre_total = 0
    cluster_counts: list[int] = []

    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        images = _list_images(class_dir)
        n = len(images)
        if n == 0:
            skipped_empty.append(class_dir.name)
            continue

        hashes = [_dhash(p, hash_size) for p in images]
        cluster_idx_groups = _cluster(hashes, hamming_threshold)
        clusters: list[list[Path]] = [[images[i] for i in g] for g in cluster_idx_groups]
        cluster_counts.append(len(clusters))

        for s in splits:
            (dst / s / class_dir.name).mkdir()

        assignments, was_under = _assign_clusters(clusters, val_frac, test_frac, rng)
        if was_under:
            suffix = "s" if len(clusters) != 1 else ""
            undersplit.append(f"{class_dir.name} ({len(clusters)} cluster{suffix})")

        counts: dict[str, int] = {}
        for s, imgs in assignments.items():
            for f in imgs:
                shutil.copy2(f, dst / s / class_dir.name / f.name)
            counts[s] = len(imgs)

        per_class.append((class_dir.name, n, len(clusters), counts))
        pre_total += n

    totals = {s: sum(c.get(s, 0) for _, _, _, c in per_class) for s in splits}
    kept_classes = len(per_class)

    print(f"Dedupe-aware split written to {dst}")
    print(f"  num_classes (non-empty)      : {kept_classes}")
    print(f"  skipped (empty folders)      : {len(skipped_empty)}")
    if skipped_empty:
        print(f"                                 {skipped_empty}")
    print(f"  pre-dedupe total images      : {pre_total}")
    if cluster_counts:
        print(
            f"  clusters per class           : "
            f"min={min(cluster_counts)} "
            f"median={int(np.median(cluster_counts))} "
            f"max={max(cluster_counts)} "
            f"mean={float(np.mean(cluster_counts)):.1f} "
            f"total={sum(cluster_counts)}"
        )
    print(
        f"  dedup params                 : Hamming≤{hamming_threshold} on "
        f"{hash_size}x{hash_size} dHash ({hash_size * hash_size}-bit)"
    )
    print(f"  under-split classes          : {len(undersplit)}")
    if undersplit:
        for u in undersplit[:15]:
            print(f"    - {u}")
        if len(undersplit) > 15:
            print(f"    ... and {len(undersplit) - 15} more")
    print(f"  totals per split (images)    : {totals}")

    if undersplit:
        print()
        print("  Under-split classes were dumped entirely into train — they have")
        print("  fewer distinct visual clusters than splits requested. Consider:")
        print("    * raising --hamming-threshold (stricter dedup, fewer false-positive merges)")
        print("    * collecting more diverse images for those classes")
        print("    * excluding them from eval datasets manually")

    return {
        "num_classes": kept_classes,
        "skipped_empty": skipped_empty,
        "undersplit": undersplit,
        "totals": totals,
        "pre_total": pre_total,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument(
        "--test-frac", type=float, default=0.15,
        help="0 disables test split (default 0.15 — 3-way split)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--hamming-threshold", type=int, default=5,
        help="Pairs with dHash Hamming distance ≤ this are merged into one "
             "cluster (default 5). 0 = byte-identical only; ~10 = very lenient.",
    )
    p.add_argument(
        "--hash-size", type=int, default=8,
        help="dHash grid (8 → 64-bit hash; 16 → 256-bit). Larger = finer-grained "
             "but slower (default 8).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dedupe_split(
        args.src, args.dst, args.val_frac, args.test_frac, args.seed,
        args.hamming_threshold, args.hash_size,
    )
