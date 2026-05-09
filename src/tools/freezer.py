from fnmatch import fnmatch
from typing import Dict, Iterable, List, Optional, Tuple
import logging

import torch.nn as nn


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, p) for p in patterns)


def freeze_by_patterns(
    module: nn.Module,
    patterns: Optional[List[str]] = None,
    except_patterns: Optional[List[str]] = None,
) -> List[str]:
    """Set requires_grad=False on parameters matching the policy.

    Exactly one of `patterns` / `except_patterns` should be provided.
    - patterns: freeze params whose name matches any pattern.
    - except_patterns: freeze params whose name does NOT match any pattern.

    Patterns use fnmatch syntax against the full named_parameters() key
    (e.g. 'features.0.conv.weight', 'features.[0-2].*', 'last_linear*').

    Returns the list of frozen parameter names.
    """
    if (patterns is None) == (except_patterns is None):
        raise ValueError("Provide exactly one of `patterns` or `except_patterns`")

    frozen: List[str] = []
    for name, p in module.named_parameters():
        if patterns is not None:
            should_freeze = _match_any(name, patterns)
        else:
            should_freeze = not _match_any(name, except_patterns)
        if should_freeze:
            p.requires_grad = False
            frozen.append(name)
    return frozen


def unfreeze_by_patterns(module: nn.Module, patterns: List[str]) -> List[str]:
    """Set requires_grad=True on parameters matching any pattern.

    Returns the list of unfrozen parameter names.
    """
    unfrozen: List[str] = []
    for name, p in module.named_parameters():
        if _match_any(name, patterns) and not p.requires_grad:
            p.requires_grad = True
            unfrozen.append(name)
    return unfrozen


def freeze_summary(module: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params) counts."""
    total = 0
    trainable = 0
    for p in module.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def log_freeze_state(module: nn.Module, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or logging.getLogger(__name__)
    trainable, total = freeze_summary(module)
    pct = 100.0 * trainable / total if total else 0.0
    logger.info(
        f"{type(module).__name__}: {trainable:,}/{total:,} trainable params ({pct:.1f}%)"
    )


def parse_unfreeze_schedule(raw: Optional[Dict]) -> Dict[int, List[str]]:
    """Normalize unfreeze schedule from YAML. Keys are epoch numbers (int),
    values are pattern lists.
    """
    if not raw:
        return {}
    schedule: Dict[int, List[str]] = {}
    for epoch, patterns in raw.items():
        schedule[int(epoch)] = list(patterns)
    return schedule
