import torch
import torch.nn as nn

from config.backbone_config import BackboneConfig


class BaseBackbone(nn.Module):
    embedding_size: int
    input_size: list[int]
    device: torch.device

    def __init__(self, backbone_config: BackboneConfig) -> None:
        super().__init__()
        self.embedding_size = backbone_config.embedding_size
        self.input_size = list(backbone_config.input_size)
        self.device = torch.device(backbone_config.device)
        # Note: builder calls `.to(device)` on the fully-constructed subclass —
        # don't call `.to()` here because subclass layers haven't been added yet.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override to map (B, C, H, W) → (B, embedding_size)."""
        raise NotImplementedError()
