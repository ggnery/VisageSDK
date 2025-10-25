import torch.nn as nn
import torch

from config.backbone.base_backbone_config import BaseBackboneConfig
from trainer.training_context import EpochContext, TrainingContext, BatchContext

class BaseBackbone(nn.Module):
    embedding_size: int
    device: torch.device
    num_classes: int
    
    def __init__(self, backbone_config: BaseBackboneConfig) -> None:
        super().__init__()
        self.embedding_size = backbone_config.embedding_size
        self.device = torch.device(backbone_config.device)
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Override this function to implement custom forward for your backbone

        Args:
            x (torch.Tensor): input batch (B x C x H x W)
        Returns:
            torch.Tensor: output_embeddings (B x embedding_size)
        """
        raise NotImplementedError()
   
    def before_train(self, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in backbone before the train process
              
        Args:
            train_ctx (TrainingContext): Current training context 
        """
        pass
    
    def after_backward(self, batch_ctx: BatchContext, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in backbone after each backward pass

        Args:
            batch_ctx (BatchContext): Current batch context
            train_ctx (TrainingContext): Current training context 
        """
        pass
    
    def after_step(self, batch_ctx: BatchContext, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in backbone after each step/params update

        Args:
            batch_ctx (BatchContext): Current batch context
            train_ctx (TrainingContext): Current training context 
        """
        pass
    
    def after_epoch(self, epoch_ctx: EpochContext, train_ctx: TrainingContext):
        """Override this method if you want to do some behavior in backbone after an epoch has finished
        
        Args:
            epoch_ctx (EpochContext): Current epoch context 
            train_ctx (TrainingContext): Current training context 
        """
        pass