from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    LambdaLR,
    LRScheduler,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)

from config.trainer.trainer_config import TrainerConfig


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
            raise ValueError(f"Scheduler {scheduler_type} not implemented")


def build_stair_lr(optimizer: Optimizer, scheduler_params: dict[int, float]) -> LambdaLR:
    """Stair LR: YAML maps `epoch -> absolute_lr`. Per-group lambdas so the
    absolute target applies regardless of each group's base LR."""
    epochs = sorted(scheduler_params.keys())

    def make_lambda(base_lr: float):
        def lr_lambda(epoch: int) -> float:
            target = base_lr
            for ep in epochs:
                if epoch >= ep:
                    target = scheduler_params[ep]
                else:
                    break
            return (target / base_lr) if base_lr else 1.0

        return lr_lambda

    lambdas = [make_lambda(g["lr"]) for g in optimizer.param_groups]
    return LambdaLR(optimizer, lambdas)


def build_multi_step_lr(optimizer: Optimizer, scheduler_params: dict) -> MultiStepLR:
    milestones = scheduler_params.get("milestones", [])
    gamma = scheduler_params.get("gamma", 0.1)
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def build_step_lr(optimizer: Optimizer, scheduler_params: dict) -> StepLR:
    return StepLR(optimizer, scheduler_params["step_size"], scheduler_params["gamma"])


def build_reduce_lr_on_plateau(optimizer: Optimizer, scheduler_params: dict) -> ReduceLROnPlateau:
    return ReduceLROnPlateau(
        optimizer,
        scheduler_params["mode"],
        scheduler_params["factor"],
        scheduler_params["patience"],
    )
