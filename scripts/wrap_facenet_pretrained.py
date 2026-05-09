"""Wrap a raw facenet-pytorch state_dict into the framework's checkpoint format.

The trainer's `load_checkpoint` expects a dict with keys
`backbone_state_dict`, `epoch`, `val_loss`, ... — a third-party checkpoint
like `vggface2.pt` from https://github.com/timesler/facenet-pytorch is a
flat OrderedDict, so it cannot be passed directly. This script repackages
it.

The trainer loads with `strict=False`, so the `logits.*` head weights from
the VGGFace2 classification training are silently ignored — only the
shared backbone tensors (everything up to `last_linear` / `last_bn`) are
transferred.

Usage:
    uv run python scripts/wrap_facenet_pretrained.py \
        --src ./models/base/vggface2.pt \
        --dst ./models/base/vggface2_wrapped.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def wrap(src: Path, dst: Path) -> None:
    state_dict = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Expected a state_dict (dict), got {type(state_dict).__name__}")

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
        "  Trainer will start at epoch=1; logits.* (if present) is dropped "
        "by strict=False during load_state_dict."
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("./models/base/vggface2.pt"))
    p.add_argument("--dst", type=Path, default=Path("./models/base/vggface2_wrapped.pth"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wrap(args.src, args.dst)
