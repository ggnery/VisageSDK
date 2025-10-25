from backbone.base_backbone import BaseBackbone
from config.trainer.trainer_config import TrainerConfig
from torch.optim import Optimizer, RMSprop, Adam, SGD, AdamW

from loss.base_loss import BaseLoss

def build_optimizer(model: BaseBackbone, loss: BaseLoss, config: TrainerConfig) -> Optimizer:
    optimizer_type = config.optmizer_type
    optimizer_params = config.optmizer_params
    match optimizer_type:
        case "RMSprop":
            return RMSprop([
                    {"params": model.parameters()}, 
                    {"params": loss.parameters()}
                ], **optimizer_params)
        case "Adam":
            return Adam([
                    {"params": model.parameters()}, 
                    {"params": loss.parameters()}
                ], **optimizer_params)
        case "SGD":
            return SGD([
                    {"params": model.parameters()}, 
                    {"params": loss.parameters()}
                ], **optimizer_params)
        case "AdamW":
            return AdamW([
                    {"params": model.parameters()}, 
                    {"params": loss.parameters()}
                ], **optimizer_params)
        case _:
            raise Exception(f"Optimizer {optimizer_type} not implemented")
         
    