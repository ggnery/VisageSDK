from registry import DATASETS

from .base_train_val_dataset import BaseTrainValDataset
from .image_folder_dataset import ImageFolderDataset

DATASETS.register("image_folder", ImageFolderDataset)

__all__ = ["BaseTrainValDataset", "ImageFolderDataset"]
