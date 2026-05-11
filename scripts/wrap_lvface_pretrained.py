"""Wrap the official LVFace state_dict into the framework's checkpoint format.

LVFace publishes weights as a flat OrderedDict at
    models/base/LVFace-B_Glint360K.pt
which the trainer's `load_checkpoint` cannot consume directly — it expects
a wrapped dict with `backbone_state_dict`, `epoch`, etc. This script
repackages the file. Only the backbone tensors are needed for fine-tuning;
the wrapper inserts empty dicts for loss/optimizer/scheduler so the
trainer skips them when `checkpoint.load.loss/optimizer/scheduler: False`.

Usage:
    uv run python scripts/wrap_lvface_pretrained.py \
        --src ./models/base/LVFace-B_Glint360K.pt \
        --dst ./models/base/LVFace-B_Glint360K_wrapped.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def wrap(src: Path, dst: Path) -> None:
    state_dict = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict (dict), got {type(state_dict).__name__}")

    # Unwrap upstream "state_dict" / "model" containers FIRST so the
    # `backbone_state_dict` check below operates on the actual tensors,
    # not the wrapper. A checkpoint like `{"model": {"backbone_state_dict": ...}}`
    # would otherwise slip past the guard.
    for outer in ("state_dict", "model"):
        if outer in state_dict and isinstance(state_dict[outer], dict):
            state_dict = state_dict[outer]
            break

    if "backbone_state_dict" in state_dict:
        raise ValueError(
            f"{src} already looks like a wrapped framework checkpoint "
            "(has 'backbone_state_dict' key). Nothing to do."
        )

    checkpoint = {
        "epoch": 0,
        "train_loss": 0.0,
        "val_loss": float("inf"),
        "backbone_state_dict": state_dict,
        "loss_state_dict": {},
        "optimizer_state_dict": {},
        "scheduler_state_dict": {},
        "scaler_state_dict": None,
    }

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, dst)

    n_params = sum(t.numel() for t in state_dict.values() if torch.is_tensor(t))
    n_keys = len(state_dict)
    print(f"Wrapped {src} -> {dst}")
    print(f"  {n_keys} tensors, {n_params:,} parameters")
    print(
        "  Trainer will start at epoch=1; pair with `checkpoint.load.loss: False` "
        "in the trainer YAML so the head reinitializes for the new class set."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("./models/base/LVFace-B_Glint360K.pt"))
    p.add_argument("--dst", type=Path, default=Path("./models/base/LVFace-B_Glint360K_wrapped.pth"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wrap(args.src, args.dst)
