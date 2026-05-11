"""LoRA / PEFT integration helpers.

Wraps a backbone with PEFT's LoraConfig. Apply AFTER loading the base
checkpoint — wrapping renames parameters to `base_model.model.*`.
"""

from __future__ import annotations

from typing import cast

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.peft_model import PeftModel


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_modules: list[str],
    dropout: float = 0.0,
    modules_to_save: list[str] | None = None,
) -> PeftModel:
    """Wrap `model` with a LoRA adapter (PEFT). Effective scaling = alpha / rank.

    `modules_to_save` lists modules to fully fine-tune (not LoRA-adapt) —
    useful for the embedding head when adapter deltas aren't enough.
    """
    # PEFT stubs over-restrict lora_alpha type and the model type; runtime is fine.
    config = LoraConfig(
        r=rank,
        lora_alpha=int(alpha) if alpha == int(alpha) else alpha,  # type: ignore[arg-type]
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        modules_to_save=list(modules_to_save) if modules_to_save else None,
    )
    wrapped = get_peft_model(model, config)  # type: ignore[arg-type]
    return cast(PeftModel, wrapped)


def lora_trainable_summary(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
