import random
from typing import override

from batch_sampler.base_batch_sampler import BaseBatchSampler
from config.batch_sampler.base_batch_sampler_config import BatchSamplerConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset


class FacenetBatchSampler(BaseBatchSampler):
    """Custom batch sampler for FaceNet training."""

    def __init__(self, config: BatchSamplerConfig, train_dataset: BaseTrainValDataset):
        """
        Args:
            config: BatchSamplerConfig exposing `faces_per_identity` and
                `num_identities_per_batch` from YAML.
            train_dataset: BaseTrainValDataset providing `label_map`.
        """
        super().__init__(train_dataset)
        self.faces_per_identity = config.faces_per_identity
        self.num_identities_per_batch = config.num_identities_per_batch
        self.batch_size = self.faces_per_identity * self.num_identities_per_batch

        # Filter identities that have enough samples
        self.valid_identities = [
            label
            for label, indices in train_dataset.label_map.items()
            if len(indices) >= self.faces_per_identity
        ]

        print(f"Found {len(self.valid_identities)} identities with >= {self.faces_per_identity} samples")

        self.num_valid_identities = len(self.valid_identities)

        # Per-instance RNG so the sampler is isolated from the global random state.
        seed = getattr(config, "seed", None)
        self._rng = random.Random(int(seed)) if seed is not None else random.Random()

    @override
    def __iter__(self):
        identity_order = self.valid_identities.copy()
        self._rng.shuffle(identity_order)

        for i in range(0, len(identity_order), self.num_identities_per_batch):
            batch_identities = identity_order[i : i + self.num_identities_per_batch]

            if len(batch_identities) < self.num_identities_per_batch:
                # Pad last partial batch with replacement (sample would fail when
                # remaining > num_valid_identities).
                remaining = self.num_identities_per_batch - len(batch_identities)
                extra_identities = self._rng.choices(self.valid_identities, k=remaining)
                batch_identities.extend(extra_identities)

            batch_indices = []
            for identity in batch_identities:
                identity_samples = self.dataset.label_map[identity]
                selected = self._rng.sample(identity_samples, self.faces_per_identity)
                batch_indices.extend(selected)

            self._rng.shuffle(batch_indices)
            yield batch_indices

    @override
    def __len__(self):
        return (
            self.num_valid_identities + self.num_identities_per_batch - 1
        ) // self.num_identities_per_batch
