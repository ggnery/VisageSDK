from collections import defaultdict

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from config.dataset.train_val.base_train_val_dataset_config import TrainValDatasetConfig
from transformation.base_transformation import BaseTransformation


class BaseTrainValDataset(Dataset):
    data: list[tuple[str, str]]
    transform: transforms.Compose
    label_to_idx: dict[str, int]
    label_map: dict[int, list[int]]

    def __init__(self, dataset_config: TrainValDatasetConfig, transformation: BaseTransformation) -> None:
        super().__init__()

        # Build `label_to_idx` from the UNION of class names across all
        # splits (train + val), sorted alphabetically. Without this, each
        # split would index classes independently — so a class that
        # appears in train but not val (or vice versa) would land on
        # different integer IDs, and the cross-entropy / margin head
        # trained on `train` would mispredict on `val`. Subclasses
        # implement `read_all_labels` to enumerate the union.
        all_labels = self.read_all_labels(dataset_config)
        self.label_to_idx = {label: i for i, label in enumerate(all_labels)}

        self.data = self.read_data(dataset_config)
        self.transform = transformation.transform

        self.label_map = defaultdict(list)
        for idx, (label, _) in enumerate(self.data):
            if label not in self.label_to_idx:
                # Defensive: `read_all_labels` should have returned the
                # full union. If a label slips through (custom subclass
                # mismatch), assign a fresh index so __getitem__ doesn't
                # KeyError — but the loss head sized to len(label_to_idx)
                # at construction time will be one slot short.
                self.label_to_idx[label] = len(self.label_to_idx)
            self.label_map[self.label_to_idx[label]].append(idx)

        # Sanity check against the dataset's declared `num_classes`: the
        # loss head is sized from this value, and if it's smaller than
        # the actual count the trainer indexes out of bounds.
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
        """Override to return the sorted UNION of class labels across all
        splits the dataset consumes. The base `__init__` uses this to
        align integer IDs between train and val so the loss head's
        outputs stay coherent across splits."""
        raise NotImplementedError()
