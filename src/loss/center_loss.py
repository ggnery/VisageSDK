from typing import override

import torch
from torch import nn

from config.loss.base_loss_config import LossConfig
from loss.base_loss import BaseLoss


class CenterLoss(BaseLoss):
    """Center loss (Wen et al., ECCV 2016) jointly with cross-entropy."""

    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        self.alpha = float(getattr(loss_config, "alpha", 0.5))
        self.use_bias = bool(getattr(loss_config, "use_bias", True))
        self.linear = nn.Linear(self.embedding_size, self.num_classes, bias=self.use_bias)

        nn.init.xavier_normal_(self.linear.weight)
        if self.use_bias:
            nn.init.constant_(self.linear.bias, 0)

        self.criterion = nn.CrossEntropyLoss()

        # Construct on self.device so isolated use (tests) doesn't crash from
        # mismatched device — BaseLoss already ran `.to(self.device)`.
        self.centers = nn.Parameter(
            torch.randn(self.num_classes, self.embedding_size, device=self.device)
        )

    @override
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits = self.linear(embeddings)
        cross_entropy_loss = self.criterion(logits, y_true)

        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            correct = (predicted == y_true).float()
            accuracy = correct.mean()

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

        # Clamp BEFORE masking — otherwise zeroed off-target entries get bumped
        # up to 1e-12 and bias the center loss.
        dist = distmat.clamp(min=1e-12, max=1e12) * mask.float()
        center_loss = dist.sum() / batch_size

        loss = cross_entropy_loss + self.alpha * center_loss
        additional_info = {
            "cross_entropy_loss": cross_entropy_loss.item(),
            "center_loss": center_loss.item(),
            "cls_accuracy": accuracy.item(),
        }
        return loss, additional_info
