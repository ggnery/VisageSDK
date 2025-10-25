from typing import Iterator, override
from torch.utils.data import BatchSampler

from dataset.train_val.base_train_val_dataset import BaseTrainValDataset

class BaseBatchSampler(BatchSampler):
    def __init__(self, dataset: BaseTrainValDataset):
        self.dataset = dataset
    
    @override 
    def __iter__(self) -> Iterator[int]:
        """
        Returns an iterator over batches of dataset indices.
        
        Returns:
            Iterator[List[int]]: Iterator yielding lists of integer indices (batches)
        """
        raise NotImplementedError()
        
    
    @override
    def __len__(self) -> int:
        """
        Returns the number of batches in the sampler.
        
        Returns:
            int: Total number of batches
        """
        raise NotImplementedError()