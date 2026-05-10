"""Backfill `lora_config` into checkpoints saved before the
`EvaluatorBuilder` LoRA-load fix.

Old PEFT-wrapped checkpoints have `base_model.model.*` keys in their
state_dict but no `lora_config` metadata, so the eval flow loaded them
into a bare backbone and silently dropped every key. This script reads
the trainer YAML the run snapshotted under `<run>/configs/trainer.yaml`,
extracts the `lora` block, and rewrites every `*.pth` in the run with
the missing metadata.

Usage:
    uv run python scripts/backfill_lora_config.py --run-dir runs/trains/<timestamp>_<name>
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml


def backfill(run_dir: Path) -> None:
    trainer_yaml_path = run_dir / "configs" / "trainer.yaml"
    if not trainer_yaml_path.exists():
        raise FileNotFoundError(
            f"Trainer YAML snapshot not found at {trainer_yaml_path}. "
            "This script needs the run's snapshotted trainer.yaml to recover "
            "LoRA hyperparameters."
        )
    cfg = yaml.safe_load(trainer_yaml_path.read_text())
    lora_block = cfg.get("lora") or {}
    if not lora_block.get("enabled"):
        print(f"trainer.yaml in {run_dir} has lora.enabled=false; nothing to do.")
        return

    lora_config = {
        "rank": int(lora_block.get("rank", 8)),
        "alpha": float(lora_block.get("alpha", 16.0)),
        "dropout": float(lora_block.get("dropout", 0.0)),
        "target_modules": list(lora_block.get("target_modules", [])),
        "modules_to_save": list(lora_block.get("modules_to_save") or []),
    }
    print(f"Backfill lora_config: {lora_config}")

    ckpt_dir = run_dir / "checkpoints"
    pth_files = sorted(ckpt_dir.glob("*.pth"))
    if not pth_files:
        print(f"No .pth files in {ckpt_dir}")
        return

    n_patched = 0
    n_skipped = 0
    for pth in pth_files:
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            print(f"  {pth.name}: not a dict, skipping")
            n_skipped += 1
            continue
        if "lora_config" in ckpt:
            print(f"  {pth.name}: already has lora_config, skipping")
            n_skipped += 1
            continue
        sd = ckpt.get("backbone_state_dict", {})
        if not any(k.startswith("base_model.model.") for k in sd):
            print(f"  {pth.name}: no PEFT prefixes detected, skipping")
            n_skipped += 1
            continue

        # Atomic-ish rewrite: write to .new, then move over.
        ckpt["lora_config"] = lora_config
        tmp = pth.with_suffix(".pth.new")
        torch.save(ckpt, tmp)
        shutil.move(str(tmp), str(pth))
        print(f"  {pth.name}: lora_config injected ({pth.stat().st_size // 1_000_000} MB)")
        n_patched += 1

    print(f"Done. Patched {n_patched}, skipped {n_skipped}.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir", type=Path, required=True,
        help="A run directory under runs/trains/ containing configs/trainer.yaml",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    backfill(args.run_dir)
