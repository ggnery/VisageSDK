from fnmatch import fnmatch
from typing import Any

from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop

from backbone.base_backbone import BaseBackbone
from config.trainer.trainer_config import TrainerConfig
from loss.base_loss import BaseLoss

_OPTIMIZER_CLASSES = {
    "RMSprop": RMSprop,
    "Adam": Adam,
    "SGD": SGD,
    "AdamW": AdamW,
}


def _named_with_prefix(module, prefix: str):
    for name, p in module.named_parameters():
        yield f"{prefix}.{name}", p


def _build_param_groups(
    model: BaseBackbone,
    loss: BaseLoss,
    group_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign each (named) parameter to the first matching group spec.

    Each spec is a dict with a `pattern` (fnmatch on prefixed name) plus any
    optimizer overrides (lr, weight_decay, ...). Parameters that match no
    pattern fall into a default group with no overrides (the optimizer's
    base lr/weight_decay applies).
    """
    groups: list[dict[str, Any]] = []
    for spec in group_specs:
        if "pattern" not in spec:
            raise ValueError("Each param_group spec must include a `pattern`")
        g = {k: v for k, v in spec.items() if k != "pattern"}
        g["params"] = []
        g["_pattern"] = spec["pattern"]
        groups.append(g)

    default_group: dict[str, Any] = {"params": []}

    def assign(name: str, p):
        for g in groups:
            if fnmatch(name, g["_pattern"]):
                g["params"].append(p)
                return
        default_group["params"].append(p)

    for name, p in _named_with_prefix(model, "backbone"):
        assign(name, p)
    for name, p in _named_with_prefix(loss, "loss"):
        assign(name, p)

    for g in groups:
        g.pop("_pattern", None)

    out = [g for g in groups if g["params"]]
    if default_group["params"]:
        out.append(default_group)
    return out


def build_optimizer(model: BaseBackbone, loss: BaseLoss, config: TrainerConfig) -> Optimizer:
    cls = _OPTIMIZER_CLASSES.get(config.optimizer_type)
    if cls is None:
        raise ValueError(f"Optimizer {config.optimizer_type} not implemented")

    if config.optimizer_param_groups:
        groups = _build_param_groups(model, loss, config.optimizer_param_groups)
    else:
        groups = [{"params": list(model.parameters())}, {"params": list(loss.parameters())}]

    return cls(groups, **config.optimizer_params)
