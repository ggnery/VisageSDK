from typing import override
from torch.utils.data import Sampler
import random

from config.batch_sampler.facenet_batch_sampler_config import FacenetBatchSamplerConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset
from batch_sampler.base_batch_sampler import BaseBatchSampler

class FacenetBatchSampler(BaseBatchSampler):
    """
    Custom batch sampler for FaceNet training.
    """
    
    def __init__(self, config: FacenetBatchSamplerConfig, train_dataset: BaseTrainValDataset):
        """
        Initialize the sampler.
        
        Args:
            dataset: BaseDataset instance
            faces_per_identity: Number of faces per identity in batch
            num_identities_per_batch: Number of identities per batch
        """
        super().__init__(train_dataset)
        self.faces_per_identity = config.faces_per_identity
        self.num_identities_per_batch = config.num_identities_per_batch
        self.batch_size = self.faces_per_identity * self.num_identities_per_batch
        
        # Filter identities that have enough samples
        self.valid_identities = [
            label for label, indices in train_dataset.label_map.items()
            if len(indices) >= self.faces_per_identity
        ]
        
        print(f"Found {len(self.valid_identities)} identities with >= {self.faces_per_identity} samples")
        
        self.num_valid_identities = len(self.valid_identities)
    
    @override
    def __iter__(self):
        """Generate batches according to FaceNet sampling strategy."""
        # Shuffle identities for each epoch
        identity_order = self.valid_identities.copy()
        random.shuffle(identity_order)
        
        # Generate batches
        for i in range(0, len(identity_order), self.num_identities_per_batch):
            batch_identities = identity_order[i:i + self.num_identities_per_batch]
            
            if len(batch_identities) < self.num_identities_per_batch:
                # For the last incomplete batch, add random identities
                remaining = self.num_identities_per_batch - len(batch_identities)
                extra_identities = random.sample(self.valid_identities, remaining)
                batch_identities.extend(extra_identities)
            
            # Sample faces for each identity
            batch_indices = []
            for identity in batch_identities:
                identity_samples = self.dataset.label_map[identity]
                
                # Sample with replacement if needed
                if len(identity_samples) >= self.faces_per_identity:
                    selected = random.sample(identity_samples, self.faces_per_identity)
                else:
                    selected = random.choices(identity_samples, k=self.faces_per_identity)
                    
                batch_indices.extend(selected)
            
            # Shuffle within batch for better mixing
            random.shuffle(batch_indices)
            yield batch_indices
    
    @override
    def __len__(self):
        """Number of batches per epoch."""
        return (self.num_valid_identities + self.num_identities_per_batch - 1) // self.num_identities_per_batch