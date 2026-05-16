"""Determinism helpers: seed torch/numpy/random, optionally force cuDNN deterministic mode."""

import logging
import os
import random

import numpy as np
import torch


def set_seed(seed: int | None, deterministic: bool = False) -> None:
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
        # `cudnn.deterministic` only covers cuDNN kernels — scatter/index ops
        # need use_deterministic_algorithms, which in turn requires the
        # CUBLAS workspace env var to avoid a runtime error.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)


def make_dataloader_generator(seed: int | None) -> torch.Generator | None:
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
