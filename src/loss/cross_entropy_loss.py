from typing import Dict, Tuple
from config.loss.base_loss_config import LossConfig
from loss.base_loss import BaseLoss

from torch import nn
import torch.nn.functional as F
import torch

class CrossEntropyLoss(BaseLoss):
    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        
        self.linear = nn.Linear(self.embedding_size, self.num_classes, bias=loss_config.use_bias)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=loss_config.label_smoothing)
        
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        logits = self.linear(embeddings)
        loss = self.criterion(logits, y_true)
        
        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            correct = (predicted == y_true).float()
            accuracy = correct.mean()
        
        loss_stats = {"cls_accuracy": accuracy.item()}
        return loss, loss_stats
        
        