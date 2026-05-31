"""ArcFace (additive angular margin) classification head.

Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
(CVPR 2019). Modern face-recognition SOTA: adds the margin in the *angle*
domain (theta + m) instead of CosFace's cosine domain (cos(theta) - m). The
angular formulation gives a more consistent geometric penalty and typically
edges out CosFace by 0.5-2pp on face-recognition benchmarks.
"""

import math
from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.loss_config import LossConfig
from loss.base_loss import BaseLoss


class ArcFaceLoss(BaseLoss):
    """ArcFace head — adds an additive angular margin to the target class.

    YAML keys (LossConfig):
        device: str
        s: float    logit scale (typical: 30-64 for face)
        m: float    angular margin in *radians* (typical: 0.5; 0.5 rad ≈ 28.6°)
        easy_margin: bool   (default False) — fall back to plain cosine when
                            cos(theta) < 0, avoiding the angle wrap-around.
                            Helps stability at the very start of training.
    """

    # Registered as buffers in __init__; annotated here so type-checkers treat
    # attribute access as Tensor (not Tensor | Module) in the forward math.
    cos_m: torch.Tensor
    sin_m: torch.Tensor
    th: torch.Tensor
    mm: torch.Tensor

    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)

        self.in_features = self.embedding_size
        self.out_features = self.num_classes
        self.s = float(loss_config.s)
        self.m = float(loss_config.m)
        self.easy_margin = bool(getattr(loss_config, "easy_margin", False))

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.criterion = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.weight)

        # Precompute trig constants used in the closed-form cos(theta + m)
        # expansion. Stored as buffers (not parameters) so they move with .to(device)
        # but don't end up in the optimizer.
        self.register_buffer("cos_m", torch.tensor(math.cos(self.m)))
        self.register_buffer("sin_m", torch.tensor(math.sin(self.m)))
        # When cos(theta) < threshold, theta + m > pi → cos wraps. Replace with
        # a linear penalty (cos(theta) - m*sin(m)) so gradients stay well-behaved.
        self.register_buffer("th", torch.tensor(math.cos(math.pi - self.m)))
        self.register_buffer("mm", torch.tensor(math.sin(math.pi - self.m) * self.m))

    @override
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(
            -1.0 + 1e-7, 1.0 - 1e-7
        )
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(min=0.0))

        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            # Outside [0, pi - m] the cosine wraps; clamp to a linear extension.
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Apply the angular margin only to the true class column.
        one_hot = F.one_hot(y_true.long(), num_classes=self.out_features).float()
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = self.s * output
        loss = self.criterion(output, y_true)

        with torch.no_grad():
            # Measure accuracy on the *un-marginated* cosine so the metric tracks
            # what a classifier without the margin penalty would predict.
            _, predicted = torch.max(cosine, 1)
            accuracy = (predicted == y_true).float().mean()

        return loss, {"cls_accuracy": accuracy.item()}
