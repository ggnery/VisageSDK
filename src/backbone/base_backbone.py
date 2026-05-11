import torch
import torch.nn as nn

from config.backbone.base_backbone_config import BackboneConfig


class BaseBackbone(nn.Module):
    embedding_size: int
    input_size: list[int]
    device: torch.device

    def __init__(self, backbone_config: BackboneConfig) -> None:
        super().__init__()
        self.embedding_size = backbone_config.embedding_size
        self.input_size = list(backbone_config.input_size)
        self.device = torch.device(backbone_config.device)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override to map (B, C, H, W) → (B, embedding_size)."""
        raise NotImplementedError()
