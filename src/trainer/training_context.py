from dataclasses import dataclass
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

@dataclass
class TrainingContext:
    """Context data in Trainer"""
    optimizer: Optimizer
    scheduler: LRScheduler
    train_loader : DataLoader
    val_loader: DataLoader
    num_epochs: int

@dataclass
class BatchContext:
    """Context data in each batch"""
    images: torch.Tensor
    labels: torch.Tensor
    embeddings: torch.Tensor
    loss: torch.Tensor
    batch_idx: int

@dataclass
class EpochContext:
    """Context data in each epoch"""
    epoch: int
    train_loss: float
    val_loss: float