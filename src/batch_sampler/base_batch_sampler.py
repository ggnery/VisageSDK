from collections.abc import Iterator
from typing import override

from torch.utils.data import BatchSampler

from dataset.train_val.base_train_val_dataset import BaseTrainValDataset


class BaseBatchSampler(BatchSampler):
    def __init__(self, dataset: BaseTrainValDataset):
        self.dataset = dataset

    @override
    def __iter__(self) -> Iterator[int]:
        """Yield batches of dataset indices (list[int])."""
        raise NotImplementedError()

    @override
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        raise NotImplementedError()
