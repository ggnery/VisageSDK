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
        self.data = self.read_data(dataset_config)
        self.transform = transformation.transform

        self.label_to_idx = {}
        self.label_map = defaultdict(list)
        for idx, (label, _) in enumerate(self.data):
            if label not in self.label_to_idx:
                self.label_to_idx[label] = len(self.label_to_idx)
            label_idx = self.label_to_idx[label]
            self.label_map[label_idx].append(idx)

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
