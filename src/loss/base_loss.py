from typing import Dict, Tuple
import torch
import torch.nn as nn

from config.loss.base_loss_config import BaseLossConfig
from trainer.training_context import EpochContext, TrainingContext, BatchContext
class BaseLoss(nn.Module):
    device: torch.device
    num_classes: int
    
    def __init__(self, loss_config: BaseLossConfig):
        super().__init__()
        
        self.device = torch.device(loss_config.device)
        self.num_classes = loss_config.num_classes
        self.embedding_size = loss_config.embedding_size
        self.to(self.device)
        
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Override this method to compute a personalized loss

        Args:
            embeddings (torch.Tensor): embeddings tensor (batch_size x embedding_size)
            y_true (torch.Tensor): true classe of each embedding

        Returns:
            Tuple[torch.Tensor, Dict]: returns (loss, loss_stats) where 
                loss is the loss tensor\n
                loss_stats is a dict with loss statistics\n
        """
        raise NotImplementedError()
    
    def before_train(self, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in loss before the train process
              
        Args:
            train_ctx (TrainingContext): Current training context 
        """
        pass
    
    def after_backward(self, batch_ctx: BatchContext, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in loss after each backward pass

        Args:
            batch_ctx (BatchContext): Current batch context
            train_ctx (TrainingContext): Current training context 
        """
        pass
    
    def after_step(self, batch_ctx: BatchContext, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in loss after each step/params update

        Args:
            batch_ctx (BatchContext): Current batch context
            train_ctx (TrainingContext): Current training context 
        """
        pass
    
    def after_epoch(self, epoch_ctx: EpochContext, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in loss after an epoch has finished
        
        Args:
            epoch_ctx (EpochContext): Current epoch context 
            train_ctx (TrainingContext): Current training context 
        """
        pass