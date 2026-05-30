from typing import override

import torch
from torch import nn

from config.loss_config import LossConfig
from loss.base_loss import BaseLoss


class CenterLoss(BaseLoss):
    """Center loss (Wen et al., ECCV 2016) jointly with cross-entropy.

    `alpha` weights the center term in the joint loss; `center_lr` is the
    rate of the manual center update (Wen's α in the paper). Centers are a
    BUFFER, not a parameter — they're updated under no_grad on every train
    forward, so the main optimizer's LR doesn't touch them.
    """

    centers: torch.Tensor

    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        self.alpha = float(getattr(loss_config, "alpha", 0.5))
        self.center_lr = float(getattr(loss_config, "center_lr", 0.5))
        self.use_bias = bool(getattr(loss_config, "use_bias", True))
        self.linear = nn.Linear(self.embedding_size, self.num_classes, bias=self.use_bias)

        nn.init.xavier_normal_(self.linear.weight)
        if self.use_bias:
            nn.init.constant_(self.linear.bias, 0)

        self.criterion = nn.CrossEntropyLoss()

        # Centers as a buffer (persists in state_dict, excluded from .parameters())
        # so they're driven by the manual Wen-style update only.
        self.register_buffer("centers", torch.randn(self.num_classes, self.embedding_size))

    @override
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        logits = self.linear(embeddings)
        cross_entropy_loss = self.criterion(logits, y_true)

        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            correct = (predicted == y_true).float()
            accuracy = correct.mean()

        batch_size = embeddings.size(0)
        # Centers are non-trainable (buffer); gradient flows only through embeddings.
        centers = self.centers
        distmat = (
            torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat = distmat.addmm(embeddings, centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes, device=self.centers.device)
        y_expanded = y_true.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = y_expanded.eq(classes.expand(batch_size, self.num_classes))

        # Clamp BEFORE masking — otherwise zeroed off-target entries get bumped
        # up to 1e-12 and bias the center loss.
        dist = distmat.clamp(min=1e-12, max=1e12) * mask.float()
        center_loss = dist.sum() / batch_size

        # Wen et al. manual update under no_grad, only during training:
        # Δc_k = (Σ_{i: y_i=k}(c_k - x_i)) / (1 + count_k); c_k -= center_lr · Δc_k
        if self.training:
            self._update_centers(embeddings.detach(), y_true)

        loss = cross_entropy_loss + self.alpha * center_loss
        additional_info = {
            "cross_entropy_loss": cross_entropy_loss.item(),
            "center_loss": center_loss.item(),
            "cls_accuracy": accuracy.item(),
        }
        return loss, additional_info

    @torch.no_grad()
    def _update_centers(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> None:
        # Vectorize the per-class diff sum via scatter_add: for each class k,
        # delta_k = Σ_{i:y_i=k}(c_k - x_i) = count_k·c_k - Σ_{i:y_i=k} x_i.
        counts = torch.zeros(self.num_classes, dtype=torch.long, device=self.centers.device)
        counts.scatter_add_(0, y_true, torch.ones_like(y_true))
        sums = torch.zeros_like(self.centers)
        sums.index_add_(0, y_true, embeddings)
        delta = counts.unsqueeze(1).float() * self.centers - sums
        denom = (1 + counts).float().unsqueeze(1)
        self.centers -= self.center_lr * delta / denom
