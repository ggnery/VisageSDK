"""LoRA / PEFT integration helpers.

Wraps the framework's existing backbone with a `peft` LoraConfig adapter.
The underlying base weights are frozen by PEFT; only the lora_A / lora_B
parameters per target module are trainable.

Why a thin wrapper instead of writing LoRA from scratch:
- `peft` is the de facto standard, well-tested by HuggingFace.
- Saving / composing adapters comes for free (`save_pretrained`,
  `set_adapter`, `add_adapter`).
- We keep a single integration point so the rest of the trainer pipeline
  doesn't need to know about LoRA.

LoRA must be applied AFTER the base checkpoint is loaded — once the model
is wrapped, parameter keys gain `base_model.model.` prefixes that don't
match the source state_dict. The Trainer hooks LoRA in right after
load_checkpoint and rebuilds the optimizer to capture the new params.
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
) -> PeftModel:
    """Wrap `model` with a LoRA adapter and return the PEFT-wrapped model.

    Args:
        model: A built backbone (any nn.Module). Its weights stay; PEFT
            inserts LoRA layers on the targeted submodules and freezes
            everything else.
        rank: The LoRA rank (typical: 4–16). Lower rank = fewer trainable
            params, slightly less expressivity.
        alpha: LoRA scaling factor; the effective LoRA contribution is
            alpha / rank. Common defaults: alpha = 2 * rank.
        target_modules: Module names or fnmatch-style patterns that PEFT
            should wrap. Examples: ["last_linear", "block8.branch1.0.conv"].
        dropout: Dropout applied to the LoRA path. 0 disables.

    Returns:
        A `peft.PeftModel`. Attribute access proxies to the base model, so
        existing code that reads `backbone.embedding_size` etc. keeps
        working transparently.
    """
    # peft's stubs declare lora_alpha as int and only accept transformers'
    # PreTrainedModel, but at runtime both float and any nn.Module work
    # fine (the latter is the documented "wrap any model" path). Casting
    # silences the static checker without changing behavior.
    config = LoraConfig(
        r=rank,
        lora_alpha=int(alpha) if alpha == int(alpha) else alpha,  # type: ignore[arg-type]
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
    )
    wrapped = get_peft_model(model, config)  # type: ignore[arg-type]
    return cast(PeftModel, wrapped)


def lora_trainable_summary(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts for a (possibly LoRA-
    wrapped) model. Mirrors the freezer's `freeze_summary` semantics so
    log lines stay consistent with the rest of the framework."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
