from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

from config.trainer_config import TrainerConfig


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
        case "CosineAnnealingLR":
            return build_cosine_annealing_lr(optimizer, scheduler_params)
        case "CosineWarmup":
            return build_cosine_warmup(optimizer, scheduler_params)
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


def build_cosine_annealing_lr(optimizer: Optimizer, scheduler_params: dict) -> CosineAnnealingLR:
    """Plain cosine annealing over `T_max` epochs to `eta_min`."""
    return CosineAnnealingLR(
        optimizer,
        T_max=int(scheduler_params["T_max"]),
        eta_min=float(scheduler_params.get("eta_min", 0.0)),
    )


def build_cosine_warmup(optimizer: Optimizer, scheduler_params: dict) -> SequentialLR:
    """Linear warmup then cosine anneal — common modern recipe for ViT fine-tunes.

    YAML keys:
        warmup_epochs: int      epochs to warm up from start_factor*lr → lr
        total_epochs:  int      total schedule length; cosine runs for the rest
        start_factor:  float    fraction of base lr at epoch 0 (default 0.01)
        eta_min:       float    floor lr the cosine decays to (default 0.0)
    """
    warmup_epochs = int(scheduler_params["warmup_epochs"])
    total_epochs = int(scheduler_params["total_epochs"])
    if warmup_epochs >= total_epochs:
        raise ValueError(
            f"warmup_epochs ({warmup_epochs}) must be < total_epochs ({total_epochs})"
        )
    start_factor = float(scheduler_params.get("start_factor", 0.01))
    eta_min = float(scheduler_params.get("eta_min", 0.0))

    warmup = LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
