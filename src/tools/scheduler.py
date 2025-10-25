from typing import Dict
from config.trainer.trainer_config import TrainerConfig
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, MultiStepLR, StepLR, ReduceLROnPlateau
from torch.optim import Optimizer

def build_scheduler(optimizer: Optimizer, config: TrainerConfig) -> LRScheduler:
    scheduler_type = config.lr_schedule_type
    scheduler_params = config.lr_schedule_params
    
    match scheduler_type:
        case "StairLR":
            return build_stair_lr(optimizer, scheduler_params)
        case "MultiStepLR":
            return build_multi_step_lr(optimizer, scheduler_params)
        case "StepLR":
            return build_step_lr(optimizer, scheduler_params)
        case "ReduceLROnPlateau":
            return build_reduce_lr_on_plateau(optimizer, scheduler_params)
        case _:
            raise Exception(f"Scheduler {scheduler_type} not implemented")
        
    
def build_stair_lr(optimizer: Optimizer, scheduler_params: Dict[int, float]) -> LambdaLR:
    epochs = sorted(scheduler_params.keys())  # Sort epochs for proper traversal
    initial_lr = optimizer.param_groups[0]["lr"]
    
    def lr_lambda(epoch: int):
        target_lr = initial_lr
        
        for epoch_threshold in epochs:
            if epoch >= epoch_threshold:
                target_lr = scheduler_params[epoch_threshold]
            else:
                break
        
        return target_lr / initial_lr
            
    return LambdaLR(optimizer, lr_lambda)


def build_multi_step_lr(optimizer: Optimizer, scheduler_params: Dict) -> MultiStepLR:

    milestones = scheduler_params.get('milestones', [])
    gamma = scheduler_params.get('gamma', 0.1)
    
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

def build_step_lr(optimizer: Optimizer, scheduler_params: Dict) -> StepLR:
    gamma = scheduler_params["gamma"]
    step_size = scheduler_params["step_size"]
    
    return StepLR(optimizer, step_size, gamma)

def build_reduce_lr_on_plateau(optimizer: Optimizer, scheduler_params: Dict) -> ReduceLROnPlateau:
    mode = scheduler_params["mode"]
    factor = scheduler_params["factor"]
    patience = scheduler_params["patience"]
    
    return ReduceLROnPlateau(optimizer, mode, factor, patience)