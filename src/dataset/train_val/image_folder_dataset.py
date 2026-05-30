from pathlib import Path
from typing import override

from config.train_val_dataset_config import TrainValDatasetConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_SPLITS = ("train", "val")


def _scan_dir(root: Path) -> list[tuple[str, str]]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    pairs: list[tuple[str, str]] = []
    # Sort for reproducible label IDs across machines.
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_file in sorted(class_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in _VALID_EXTS:
                pairs.append((class_dir.name, str(img_file.absolute())))
    return pairs


def _scan_class_names(root: Path) -> list[str]:
    """Return the sorted class-dir names directly under `root`."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    return sorted(d.name for d in root.iterdir() if d.is_dir())


class ImageFolderDataset(BaseTrainValDataset):
    """Single dataset class for both train and val splits.

    The split is selected at construction time (`split="train"` / `"val"`) and
    decides which directory in the YAML (`train_dir` / `val_dir`) to scan.
    """

    def __init__(self, dataset_config: TrainValDatasetConfig, transformation, split: str = "train") -> None:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}, got {split!r}")
        self._split = split
        super().__init__(dataset_config, transformation)

    @override
    def read_data(self, dataset_config: TrainValDatasetConfig) -> list[tuple[str, str]]:
        target = dataset_config.train_dir if self._split == "train" else dataset_config.val_dir
        return _scan_dir(Path(target))

    @override
    def read_all_labels(self, dataset_config: TrainValDatasetConfig) -> list[str]:
        labels: set[str] = set()
        for split_dir in (dataset_config.train_dir, dataset_config.val_dir):
            labels.update(_scan_class_names(Path(split_dir)))
        return sorted(labels)
