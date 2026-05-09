from backbone.base_backbone import BaseBackbone
from config.trainer.trainer_config import TrainerConfig
from torch.optim import Optimizer, RMSprop, Adam, SGD, AdamW

from loss.base_loss import BaseLoss


def build_optimizer(model: BaseBackbone, loss: BaseLoss, config: TrainerConfig) -> Optimizer:
    optimizer_type = config.optimizer_type
    optimizer_params = config.optimizer_params
    param_groups = [{"params": model.parameters()}, {"params": loss.parameters()}]

    match optimizer_type:
        case "RMSprop":
            return RMSprop(param_groups, **optimizer_params)
        case "Adam":
            return Adam(param_groups, **optimizer_params)
        case "SGD":
            return SGD(param_groups, **optimizer_params)
        case "AdamW":
            return AdamW(param_groups, **optimizer_params)
        case _:
            raise ValueError(f"Optimizer {optimizer_type} not implemented")
