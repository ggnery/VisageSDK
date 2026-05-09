import torch
import torch.nn as nn

from config.backbone.base_backbone_config import BackboneConfig


class BaseBackbone(nn.Module):
    embedding_size: int
    device: torch.device

    def __init__(self, backbone_config: BackboneConfig) -> None:
        super().__init__()
        self.embedding_size = backbone_config.embedding_size
        self.device = torch.device(backbone_config.device)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override to implement custom forward.

        Args:
            x: input batch (B x C x H x W)
        Returns:
            embeddings (B x embedding_size)
        """
        raise NotImplementedError()
