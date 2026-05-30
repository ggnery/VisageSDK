from collections import defaultdict

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from config.train_val_dataset_config import TrainValDatasetConfig
from transformation.base_transformation import BaseTransformation


class BaseTrainValDataset(Dataset):
    data: list[tuple[str, str]]
    transform: transforms.Compose
    label_to_idx: dict[str, int]
    label_map: dict[int, list[int]]

    def __init__(self, dataset_config: TrainValDatasetConfig, transformation: BaseTransformation) -> None:
        super().__init__()

        # Build label_to_idx from the UNION of train + val class names so the
        # same class always maps to the same integer ID across splits.
        all_labels = self.read_all_labels(dataset_config)
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}

        self.data = self.read_data(dataset_config)
        self.transform = transformation.transform

        self.label_map = defaultdict(list)
        for idx, (label, _) in enumerate(self.data):
            if label not in self.label_to_idx:
                # `read_all_labels` MUST return the sorted union of all labels
                # across splits; if it misses one, label IDs become inconsistent
                # across runs. Fail fast rather than silently appending.
                raise KeyError(
                    f"Label {label!r} found in data but not in label_to_idx — "
                    "the subclass's read_all_labels is incomplete."
                )
            self.label_map[self.label_to_idx[label]].append(idx)

        # Reject `num_classes` smaller than the actual count — loss head would
        # be sized too small and indexing would go out of bounds.
        num_classes = getattr(dataset_config, "num_classes", None)
        if num_classes is not None and num_classes < len(self.label_to_idx):
            raise ValueError(
                f"num_classes={num_classes} in dataset config is smaller than "
                f"the actual {len(self.label_to_idx)} classes discovered across "
                f"train + val. Update `num_classes` to at least {len(self.label_to_idx)}."
            )

    def __getitem__(self, idx):
        img_class, img_path = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return self.label_to_idx[img_class], image

    def __len__(self):
        return len(self.data)

    def read_data(self, dataset_config: TrainValDatasetConfig) -> list[tuple[str, str]]:
        """Override to return list of (label, image_path) tuples."""
        raise NotImplementedError()

    def read_all_labels(self, dataset_config: TrainValDatasetConfig) -> list[str]:
        """Override to return the sorted UNION of class labels across splits."""
        raise NotImplementedError()
