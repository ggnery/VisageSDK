"""LFW verification dataset.

Reads `pairs.txt` in either the standard 2-token header (`<n_folds> <n_pairs>`)
or the asymmetric 3-token header (`<n_folds> <n_same> <n_diff>`) for low-FAR
resolution. Image paths follow `<eval_dir>/<name>/<name>_<idx:04d>.<ext>`.

Each unique image is encoded once; pairs are exposed as
`(img_idx_a, img_idx_b, is_same, fold_id)`.
"""

from pathlib import Path
from typing import override

from config.eval_dataset_config import EvalDatasetConfig
from dataset.eval.base_eval_dataset import BaseEvalDataset
from transformation.base_transformation import BaseTransformation

_DEFAULT_EXTS = ("jpg", "jpeg", "png", "bmp")


def _resolve_img_path(root: Path, name: str, idx: int, exts: tuple[str, ...]) -> Path:
    """Return the first existing path matching any of the candidate extensions."""
    for ext in exts:
        candidate = root / name / f"{name}_{idx:04d}.{ext}"
        if candidate.exists():
            return candidate
    # No match found — return the first candidate so the downstream existence
    # check (in __init__) raises a FileNotFoundError pointing at a real path.
    return root / name / f"{name}_{idx:04d}.{exts[0]}"


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
        ext_cfg = getattr(config, "image_ext", None)
        if isinstance(ext_cfg, str):
            exts: tuple[str, ...] = (ext_cfg,)
        elif isinstance(ext_cfg, (list, tuple)) and ext_cfg:
            exts = tuple(ext_cfg)
        else:
            exts = _DEFAULT_EXTS

        if not pairs_path.exists():
            raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

        path_to_idx: dict = {}

        def register(name: str, idx: int) -> int:
            path = _resolve_img_path(eval_dir, name, idx, exts)
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
