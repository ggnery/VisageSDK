from registry import DATASETS

from .base_train_val_dataset import BaseTrainValDataset
from .image_folder_dataset import ImageFolderTrainDataset, ImageFolderValDataset

DATASETS.register("image_folder_train", ImageFolderTrainDataset)
DATASETS.register("image_folder_val", ImageFolderValDataset)

__all__ = [
    "BaseTrainValDataset",
    "ImageFolderTrainDataset",
    "ImageFolderValDataset",
]
