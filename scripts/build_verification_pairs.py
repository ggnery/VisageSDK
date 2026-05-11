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
    src: Path,
    dst: Path,
    n_folds: int,
    n_pairs_per_fold: int,
    seed: int,
    ext: str,
    exhaustive: bool = False,
    n_neg_per_fold: int | None = None,
    exhaustive_negatives: bool = False,
) -> None:
    if exhaustive_negatives:
        exhaustive = True  # implies positive enumeration too
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

    if exhaustive:
        # Enumerate every unique positive pair, then sample N unique
        # negative pairs without replacement. Distribute across folds
        # round-robin so per-fold threshold tuning sees the same class
        # diversity throughout. With `n_neg_per_fold` set, the file uses
        # the asymmetric `<n_folds> <n_same> <n_diff>` header so far more
        # negatives than positives can ship — necessary for TAR@FAR≤1e-5
        # resolution when unique positives bottleneck the dataset.
        all_positives: list[tuple[str, int, int]] = []
        for cls, idxs in class_indices.items():
            if len(idxs) < 2:
                continue
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    all_positives.append((cls, idxs[i], idxs[j]))
        rng.shuffle(all_positives)
        n_total_same = len(all_positives)

        same_per_fold = n_total_same // n_folds
        if same_per_fold == 0:
            raise ValueError(
                f"Only {n_total_same} unique positive pairs available, less "
                f"than {n_folds} folds. Reduce --n-folds."
            )

        # Build the negative pair pool. Two strategies:
        #
        #   1. Rejection sampling (default and --n-neg-per-fold modes):
        #      cheap when the requested count is well below the full pool,
        #      degrades near saturation as collisions explode.
        #
        #   2. Exhaustive enumeration (--exhaustive-negatives): walk every
        #      pair of distinct classes, then every cross-class image pair.
        #      Touches ~C(N,2) - sum(C(k,2)) tuples — guarantees the
        #      absolute max possible without any collision risk. Memory
        #      cost is ~150 B per pair, so ~550 MB for 3.7M neg pairs;
        #      acceptable on a workstation.
        all_negatives: list[tuple[str, int, str, int]] = []
        if exhaustive_negatives:
            for ia in range(len(all_classes)):
                ca = all_classes[ia]
                for ib in range(ia + 1, len(all_classes)):
                    cb = all_classes[ib]
                    for a in class_indices[ca]:
                        for b in class_indices[cb]:
                            all_negatives.append((ca, a, cb, b))
            rng.shuffle(all_negatives)
            n_total_diff = len(all_negatives)
            diff_per_fold = n_total_diff // n_folds
        else:
            # Default (balanced exhaustive): same many unique negs as positives.
            diff_per_fold = n_neg_per_fold if n_neg_per_fold is not None else same_per_fold
            n_total_diff = diff_per_fold * n_folds

            seen: set[tuple[str, int, str, int]] = set()
            max_attempts = n_total_diff * 10
            attempts = 0
            while len(all_negatives) < n_total_diff and attempts < max_attempts:
                ca, cb = rng.sample(all_classes, 2)
                a = rng.choice(class_indices[ca])
                b = rng.choice(class_indices[cb])
                key = (ca, a, cb, b)
                if key not in seen:
                    seen.add(key)
                    all_negatives.append(key)
                attempts += 1
            if len(all_negatives) < n_total_diff:
                raise RuntimeError(
                    f"Could not sample {n_total_diff:,} unique negative pairs after "
                    f"{max_attempts:,} attempts. The class-image distribution may "
                    "be too thin to support this many unique negatives."
                )

        pairs_path = dst / "pairs.txt"
        asymmetric = (
            exhaustive_negatives
            or (n_neg_per_fold is not None and n_neg_per_fold != same_per_fold)
        )
        with open(pairs_path, "w") as f:
            if asymmetric:
                # Extended header: parsed by LFWPairsDataset's optional 3-token mode.
                f.write(f"{n_folds} {same_per_fold} {diff_per_fold}\n")
            else:
                f.write(f"{n_folds} {same_per_fold}\n")
            for fold in range(n_folds):
                lo_s, hi_s = fold * same_per_fold, (fold + 1) * same_per_fold
                lo_d, hi_d = fold * diff_per_fold, (fold + 1) * diff_per_fold
                for cls, a, b in all_positives[lo_s:hi_s]:
                    f.write(f"{cls} {a} {b}\n")
                for ca, a, cb, b in all_negatives[lo_d:hi_d]:
                    f.write(f"{ca} {a} {cb} {b}\n")

        n_pairs = n_folds * (same_per_fold + diff_per_fold)
        n_imgs = sum(len(ix) for ix in class_indices.values())
        print(f"Verification layout written to {dst} (EXHAUSTIVE mode)")
        print(f"  classes              : {len(class_indices)} ({len(classes_with_pos)} with >=2 imgs)")
        print(f"  image symlinks       : {n_imgs}")
        print(f"  unique positives     : {n_total_same:,}")
        print(f"  unique negatives kept: {len(all_negatives):,}")
        print(f"  pairs file           : {pairs_path}")
        if asymmetric:
            print(
                f"  pairs in folds       : {n_pairs:,} "
                f"(asymmetric: {n_folds} folds × {same_per_fold:,} same + "
                f"{diff_per_fold:,} diff)"
            )
        else:
            print(
                f"  pairs in folds       : {n_pairs:,} "
                f"({n_folds} folds × {same_per_fold:,} same + {same_per_fold:,} diff)"
            )
        if n_total_same % n_folds:
            print(
                f"  NOTE: {n_total_same % n_folds} positive pairs dropped to keep folds balanced"
            )
        return

    pairs_path = dst / "pairs.txt"
    # Track already-emitted negatives so the rejection-sampled legacy
    # mode doesn't emit the same impostor pair more than once per file.
    # (Positives in the legacy LFW convention are intentionally allowed
    # to repeat across folds — matches the upstream protocol.)
    seen_neg: set[tuple[str, int, str, int]] = set()
    max_attempts_per_neg = 100
    with open(pairs_path, "w") as f:
        f.write(f"{n_folds} {n_pairs_per_fold}\n")
        for _ in range(n_folds):
            # Positive pairs: same class, two distinct image indices.
            for _ in range(n_pairs_per_fold):
                cls = rng.choice(classes_with_pos)
                a, b = rng.sample(class_indices[cls], 2)
                f.write(f"{cls} {a} {b}\n")
            # Negative pairs: two distinct classes; reject duplicates.
            for _ in range(n_pairs_per_fold):
                for _attempt in range(max_attempts_per_neg):
                    ca, cb = rng.sample(all_classes, 2)
                    a = rng.choice(class_indices[ca])
                    b = rng.choice(class_indices[cb])
                    key = (ca, a, cb, b)
                    if key not in seen_neg:
                        seen_neg.add(key)
                        f.write(f"{ca} {a} {cb} {b}\n")
                        break
                else:
                    # Pool of unique negatives is genuinely exhausted —
                    # fail loudly so the user picks --exhaustive or a
                    # smaller --n-pairs-per-fold rather than silently
                    # shipping fewer pairs than the header claims.
                    raise RuntimeError(
                        f"Could not sample a unique negative pair after "
                        f"{max_attempts_per_neg} attempts. Pool likely exhausted; "
                        "switch to --exhaustive or reduce --n-pairs-per-fold."
                    )

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
    p.add_argument(
        "--exhaustive",
        action="store_true",
        help=(
            "Enumerate every unique positive pair + sample matching count of "
            "unique negative pairs (no with-replacement sampling). The "
            "principled 'maximum useful' setting — beyond this, extra pairs "
            "only re-sample positives that already appeared. Ignores "
            "--n-pairs-per-fold."
        ),
    )
    p.add_argument(
        "--n-neg-per-fold",
        type=int,
        default=None,
        help=(
            "When combined with --exhaustive, override the negative count "
            "per fold (defaults to matching the positive count for a "
            "balanced LFW protocol). Use this to ship many more unique "
            "negatives than positives so TAR@FAR≤1e-5 is measurable. "
            "Outputs the 3-token asymmetric header that LFWPairsDataset "
            "parses transparently."
        ),
    )
    p.add_argument(
        "--exhaustive-negatives",
        action="store_true",
        help=(
            "Enumerate EVERY unique cross-class image pair as negatives "
            "(implies --exhaustive). Guarantees the absolute max without "
            "rejection-sampling collisions, at the cost of building a list "
            "of all C(N,2) - same-class pairs in memory. The principled "
            "ceiling for TAR@FAR resolution on a fixed test set."
        ),
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build(
        args.src, args.dst, args.n_folds, args.n_pairs_per_fold, args.seed,
        args.ext, exhaustive=args.exhaustive, n_neg_per_fold=args.n_neg_per_fold,
        exhaustive_negatives=args.exhaustive_negatives,
    )
