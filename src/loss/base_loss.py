import torch
import torch.nn as nn

from config.loss_config import LossConfig


class BaseLoss(nn.Module):
    device: torch.device
    num_classes: int
    embedding_size: int

    def __init__(self, loss_config: LossConfig):
        super().__init__()
        self.device = torch.device(loss_config.device)
        self.num_classes = loss_config.num_classes
        self.embedding_size = loss_config.embedding_size
        # Note: builder calls `.to(device)` on the fully-constructed subclass —
        # don't call `.to()` here because subclass layers haven't been added yet.

    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Override to return (loss_tensor, loss_stats_dict)."""
        raise NotImplementedError()
