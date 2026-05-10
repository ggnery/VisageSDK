"""LFW verification dataset.

Reads the standard `pairs.txt` format with optional asymmetric extension:

    Balanced (legacy LFW):
        <n_folds> <n_pairs_per_fold>          # 2-token header
        <name> <idx_a> <idx_b>                # n_pairs_per_fold same lines
        <name_a> <idx_a> <name_b> <idx_b>     # n_pairs_per_fold diff lines

    Asymmetric (extended for low-FAR resolution):
        <n_folds> <n_same_per_fold> <n_diff_per_fold>   # 3-token header
        ...n_same_per_fold same lines...
        ...n_diff_per_fold diff lines...

The asymmetric format lets you ship many more negatives than positives
when the test set has a hard ceiling on unique positive pairs but a much
larger pool of unique negatives — the typical situation when you want
TAR@FAR≤1e-5 measurable. Image paths are resolved as
    <eval_dir>/<name>/<name>_<idx:04d>.jpg

The dataset exposes a flat list of unique images (so each image is encoded
once even if it appears in multiple pairs) plus a list of pairs as
(idx_a, idx_b, is_same, fold_id).
"""

from pathlib import Path
from typing import override

from config.dataset.eval.base_eval_dataset_config import EvalDatasetConfig
from dataset.eval.base_eval_dataset import BaseEvalDataset
from transformation.base_transformation import BaseTransformation


def _img_path(root: Path, name: str, idx: int, ext: str) -> Path:
    return root / name / f"{name}_{idx:04d}.{ext}"


class LFWPairsDataset(BaseEvalDataset):
    pairs: list[tuple[int, int, int, int]]  # (img_idx_a, img_idx_b, is_same, fold)
    n_folds: int
    n_same_per_fold: int
    n_diff_per_fold: int

    # Backward-compat alias — pre-asymmetric callers read `n_pairs_per_fold`.
    @property
    def n_pairs_per_fold(self) -> int:
        return self.n_same_per_fold

    @override
    def __init__(self, config: EvalDatasetConfig, transformation: BaseTransformation) -> None:
        super().__init__(config, transformation)

        eval_dir = Path(config.eval_dir)
        pairs_path = Path(config.pairs_path)
        ext = getattr(config, "image_ext", "jpg")

        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        path_to_idx: dict = {}

        def register(name: str, idx: int) -> int:
            path = _img_path(eval_dir, name, idx, ext)
            key = str(path)
            if key not in path_to_idx:
                path_to_idx[key] = len(self.data)
                self.data.append((name, key))
            return path_to_idx[key]

        with open(pairs_path) as f:
            header = f.readline().strip().split()
            self.n_folds = int(header[0])
            if len(header) >= 3:
                # Asymmetric format: separate counts for same and diff.
                self.n_same_per_fold = int(header[1])
                self.n_diff_per_fold = int(header[2])
            else:
                # Legacy LFW balanced format: one count, used for both.
                self.n_same_per_fold = int(header[1])
                self.n_diff_per_fold = int(header[1])

            self.pairs = []
            for fold in range(self.n_folds):
                for _ in range(self.n_same_per_fold):
                    parts = f.readline().strip().split()
                    if len(parts) != 3:
                        raise ValueError(f"Expected same-pair line, got: {parts}")
                    name, ia, ib = parts[0], int(parts[1]), int(parts[2])
                    a = register(name, ia)
                    b = register(name, ib)
                    self.pairs.append((a, b, 1, fold))
                for _ in range(self.n_diff_per_fold):
                    parts = f.readline().strip().split()
                    if len(parts) != 4:
                        raise ValueError(f"Expected diff-pair line, got: {parts}")
                    name_a, ia, name_b, ib = parts[0], int(parts[1]), parts[2], int(parts[3])
                    a = register(name_a, ia)
                    b = register(name_b, ib)
                    self.pairs.append((a, b, 0, fold))

        for _, path in self.data:
            if not Path(path).exists():
                raise FileNotFoundError(f"Image referenced by pairs.txt not found: {path}")
