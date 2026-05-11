from typing import override

import torch
from torch import nn

from config.loss.base_loss_config import LossConfig
from loss.base_loss import BaseLoss


class CenterLoss(BaseLoss):
    """Implementation of CenterLoss from:
    - https://github.com/KaiyangZhou/pytorch-center-loss/tree/master (Original code)
    - http://ydwen.github.io/papers/WenECCV16.pdf (Original paper)
    """

    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        # Use getattr with documented defaults instead of direct attribute
        # access — `BaseConfig.__getattr__` raises AttributeError for
        # missing keys, which gives a confusing trace deep in the loss
        # module instead of a friendly "your YAML is missing X".
        self.alpha = float(getattr(loss_config, "alpha", 0.5))
        self.use_bias = bool(getattr(loss_config, "use_bias", True))
        self.linear = nn.Linear(self.embedding_size, self.num_classes, bias=self.use_bias)

        # Add proper initialization
        nn.init.xavier_normal_(self.linear.weight)
        if self.use_bias:
            nn.init.constant_(self.linear.bias, 0)

        self.criterion = nn.CrossEntropyLoss()

        # Construct on `self.device` directly. `BaseLoss.__init__` already
        # ran `self.to(self.device)` before this assignment, so a plain
        # `torch.randn(...)` would register `self.centers` on the CPU.
        # The trainer happens to call `.to(device)` a second time after
        # building, but isolating the loss (tests, eval scripts) without
        # that extra call would crash here with a device mismatch.
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.embedding_size, device=self.device)
        )

    @override
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Compute cross entropy loss
        logits = self.linear(embeddings)
        cross_entropy_loss = self.criterion(logits, y_true)

        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            correct = (predicted == y_true).float()
            accuracy = correct.mean()

        # Compute center loss

        batch_size = embeddings.size(0)
        distmat = (
            torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(embeddings, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        y_true = y_true.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = y_true.eq(classes.expand(batch_size, self.num_classes))

        # Clamp BEFORE masking so the per-sample contributions of off-target
        # classes (zeroed by `mask`) stay zero in the sum. The previous order
        # (clamp after masking) bumped every off-target entry from 0 to 1e-12,
        # adding a tiny but non-zero bias of 1e-12 * (num_classes - 1) per
        # sample to the reported center loss.
        dist = distmat.clamp(min=1e-12, max=1e12) * mask.float()
        center_loss = dist.sum() / batch_size

        loss = cross_entropy_loss + self.alpha * center_loss
        additional_info = {
            "cross_entropy_loss": cross_entropy_loss.item(),
            "center_loss": center_loss.item(),
            "cls_accuracy": accuracy.item(),
        }
        return loss, additional_info
