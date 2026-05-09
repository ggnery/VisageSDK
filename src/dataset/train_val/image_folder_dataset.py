from pathlib import Path
from typing import List, Tuple
from typing_extensions import override

from config.dataset.train_val.base_train_val_dataset_config import TrainValDatasetConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset


_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _scan_dir(root: Path) -> List[Tuple[str, str]]:
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    pairs: List[Tuple[str, str]] = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        for img_file in class_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in _VALID_EXTS:
                pairs.append((class_dir.name, str(img_file.absolute())))
    return pairs


class ImageFolderTrainDataset(BaseTrainValDataset):
    @override
    def read_data(self, dataset_config: TrainValDatasetConfig) -> List[Tuple[str, str]]:
        return _scan_dir(Path(dataset_config.train_dir))


class ImageFolderValDataset(BaseTrainValDataset):
    @override
    def read_data(self, dataset_config: TrainValDatasetConfig) -> List[Tuple[str, str]]:
        return _scan_dir(Path(dataset_config.val_dir))
