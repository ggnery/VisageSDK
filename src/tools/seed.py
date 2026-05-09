"""Determinism helpers.

Seeding fixes torch (CPU+CUDA), numpy, and Python's random module. When
`deterministic=True` is requested, cuDNN is also forced into deterministic
mode at the cost of some throughput.
"""

import logging
import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: Optional[int], deterministic: bool = False) -> None:
    if seed is None:
        return
    logger = logging.getLogger(__name__)
    logger.info(f"Setting seed={seed} (deterministic={deterministic})")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dataloader_generator(seed: Optional[int]) -> Optional[torch.Generator]:
    """Generator for shuffling DataLoaders reproducibly."""
    if seed is None:
        return None
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(_worker_id: int) -> None:
    """worker_init_fn that re-seeds numpy/random per DataLoader worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
