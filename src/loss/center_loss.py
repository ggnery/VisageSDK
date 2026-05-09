from typing import Dict, Tuple, override

from config.loss.base_loss_config import LossConfig
from loss.base_loss import BaseLoss

from torch import nn
import torch

class CenterLoss(BaseLoss):
    """Implementation of CenterLoss from:
    - https://github.com/KaiyangZhou/pytorch-center-loss/tree/master (Original code)
    - http://ydwen.github.io/papers/WenECCV16.pdf (Original paper)
    """
    
    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        self.alpha = loss_config.alpha
        self.use_bias = loss_config.use_bias
        self.linear = nn.Linear(self.embedding_size, self.num_classes, bias=loss_config.use_bias)
        
        # Add proper initialization
        nn.init.xavier_normal_(self.linear.weight)
        if loss_config.use_bias:
            nn.init.constant_(self.linear.bias, 0)
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.centers = nn.Parameter(torch.randn(self.num_classes, self.embedding_size))


    @override
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # Compute cross entropy loss
        logits = self.linear(embeddings)
        cross_entropy_loss = self.criterion(logits, y_true)
        
        with torch.no_grad():
            _, predicted = torch.max(logits, 1)
            correct = (predicted == y_true).float()
            accuracy = correct.mean()
        
        # Compute center loss
        
        batch_size = embeddings.size(0)
        distmat = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(embeddings, self.centers.t(), beta=1, alpha=-2)

        classes = torch.arange(self.num_classes).long()
        classes = classes.to(self.device)
        y_true = y_true.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = y_true.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        center_loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        loss = cross_entropy_loss + self.alpha * center_loss
        additional_info = {
                "loss": loss.item(),
                "cross_entropy_loss": cross_entropy_loss.item(),
                "center_loss": center_loss.item(),
                "accuracy": accuracy.item()
            }

        return loss, additional_info