from typing import override

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.loss_config import LossConfig
from loss.base_loss import BaseLoss


class MarginCosineProductLoss(BaseLoss):
    """CosFace-style angular margin classification head."""

    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)

        self.in_features = self.embedding_size
        self.out_features = self.num_classes
        self.s = loss_config.s
        self.m = loss_config.m

        self.weight = nn.Parameter(torch.empty(self.out_features, self.in_features))
        self.criterion = nn.CrossEntropyLoss()
        nn.init.xavier_uniform_(self.weight)

    @override
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # Clamp guards against tiny [-1, 1] excursions from F.normalize that get
        # amplified by `self.s` (e.g. 64) when the cross-entropy is applied.
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight)).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        one_hot = F.one_hot(y_true.long(), num_classes=self.out_features).float()

        output = self.s * (cosine - one_hot * self.m)
        loss = self.criterion(output, y_true)

        with torch.no_grad():
            # Accuracy is measured on the raw cosine (no margin), otherwise the
            # margin penalty on the true class artificially lowers argmax recall.
            _, predicted = torch.max(cosine, 1)
            correct = (predicted == y_true).float()
            accuracy = correct.mean()

        return loss, {"cls_accuracy": accuracy.item()}
