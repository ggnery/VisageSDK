import torch
import torch.nn as nn

from config.loss.base_loss_config import LossConfig


class BaseLoss(nn.Module):
    device: torch.device
    num_classes: int
    embedding_size: int

    def __init__(self, loss_config: LossConfig):
        super().__init__()
        self.device = torch.device(loss_config.device)
        self.num_classes = loss_config.num_classes
        self.embedding_size = loss_config.embedding_size
        self.to(self.device)

    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Override to compute the loss.

        Args:
            embeddings: tensor (batch_size x embedding_size)
            y_true: ground-truth class index per embedding

        Returns:
            (loss_tensor, loss_stats_dict)
        """
        raise NotImplementedError()
