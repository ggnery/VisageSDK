import importlib

import torch

from config.backbone.base_backbone_config import BackboneConfig
from config.dataset.eval.base_eval_dataset_config import EvalDatasetConfig
from config.env_eval_config import ENVEvalConfig
from config.evaluator.base_evaluator_config import EvaluatorConfig
from config.transformation.base_transformation_config import TransformationConfig
from registry import (
    BACKBONES,
    EVAL_DATASETS,
    EVALUATORS,
    TRANSFORMATIONS,
)

# Trigger registry population by importing each component package for side effects.
for _component_pkg in ("backbone", "dataset.eval", "evaluator", "transformation"):
    importlib.import_module(_component_pkg)


class EvaluatorBuilder:
    def __init__(self, env: ENVEvalConfig):
        self.env = env
        self.config_str = ""
        self._build_configs()
        self._build_instances()

    def _build_configs(self) -> None:
        self.backbone_config = BackboneConfig(self.env.backbone_config)
        self.config_str += self.backbone_config.get_config_string()
        backbone_info = {
            "input_size": self.backbone_config.input_size,
            "embedding_size": self.backbone_config.embedding_size,
        }

        self.transformation_config = TransformationConfig(self.env.eval_transformation_config, backbone_info)
        self.config_str += self.transformation_config.get_config_string()

        self.dataset_config = EvalDatasetConfig(self.env.eval_dataset_config, backbone_info)
        self.config_str += self.dataset_config.get_config_string()

        self.evaluator_config = EvaluatorConfig(self.env.evaluator_config)
        self.config_str += self.evaluator_config.get_config_string()

    def _build_instances(self) -> None:
        device = torch.device(self.evaluator_config.device)

        backbone_cls = BACKBONES.get(self.env.backbone)
        self.backbone = backbone_cls(self.backbone_config).to(device)

        ckpt = torch.load(self.env.checkpoint_path, map_location=device, weights_only=False)

        # If the checkpoint was saved from a PEFT-wrapped backbone, every
        # tensor key has a `base_model.model.*` prefix. Without rebuilding
        # the wrap before loading, strict=False would silently drop every
        # key and we'd score a randomly-initialized backbone — which used
        # to pass for "trained model" with a deceptive ~85% LFW accuracy.
        # The trainer persists `lora_config` precisely to make this
        # reconstruction possible.
        lora_config = ckpt.get("lora_config") if isinstance(ckpt, dict) else None
        if lora_config:
            from tools.lora import apply_lora

            target_modules = list(lora_config.get("target_modules", []))

            # If the user picks a backbone variant in the GUI that doesn't
            # contain ANY of the targeted modules, PEFT raises a cryptic
            # `Target modules X not found in the base model.` deep in its
            # internals. Detect the obvious mismatch up front so the error
            # message points at the actual fix (pick the right backbone).
            backbone_module_names = {name for name, _ in self.backbone.named_modules()}
            matched = any(
                any(
                    name == target or name.endswith(f".{target}")
                    for target in target_modules
                )
                for name in backbone_module_names
            )
            if not matched:
                raise ValueError(
                    f"Checkpoint at {self.env.checkpoint_path} was saved with "
                    f"lora_config.target_modules={target_modules}, but the "
                    f"selected backbone variant '{self.env.backbone}' (class "
                    f"{type(self.backbone).__name__}) contains no matching "
                    "modules. Pick the backbone variant the checkpoint was "
                    "trained on (likely lvface_vit_b vs inception_resnet_v1)."
                )

            self.backbone = apply_lora(
                self.backbone,
                rank=lora_config["rank"],
                alpha=lora_config["alpha"],
                target_modules=target_modules,
                dropout=lora_config.get("dropout", 0.0),
                modules_to_save=list(lora_config.get("modules_to_save") or []) or None,
            )
            self.backbone.to(device)

        if "backbone_state_dict" in ckpt:
            sd_to_load = ckpt["backbone_state_dict"]
        else:
            # raw state dict
            sd_to_load = ckpt
        # Catch the silent-drop trap: a checkpoint saved from a PEFT-wrapped
        # backbone has keys like `base_model.model.*.weight`. Without
        # `lora_config` metadata (i.e., before scripts/backfill_lora_config.py
        # was run or before the wrap-aware save), we land here without
        # having rebuilt the wrap — and strict=False would happily drop
        # every key, leaving the backbone at random init while reporting
        # a deceptive ~85% LFW accuracy. Detect the prefix explicitly and
        # tell the user how to recover.
        if lora_config is None and any(
            k.startswith("base_model.model.") for k in sd_to_load
        ):
            raise ValueError(
                f"Checkpoint at {self.env.checkpoint_path} has PEFT-prefixed "
                "state_dict keys (`base_model.model.*`) but no `lora_config` "
                "metadata. Loading with strict=False would silently drop "
                "every weight and leave the backbone at random init. "
                "Run `scripts/backfill_lora_config.py --run-dir <run>` to "
                "inject the missing metadata, or re-save the checkpoint "
                "with the LoRA-aware Trainer."
            )
        result = self.backbone.load_state_dict(sd_to_load, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(
                f"[EvaluatorBuilder] state_dict load: "
                f"{len(result.missing_keys)} missing, "
                f"{len(result.unexpected_keys)} unexpected keys"
            )

        tx_cls = TRANSFORMATIONS.get(self.env.eval_transformation)
        self.transformation = tx_cls(self.transformation_config)

        ds_cls = EVAL_DATASETS.get(self.env.eval_dataset)
        self.dataset = ds_cls(self.dataset_config, self.transformation)

        ev_cls = EVALUATORS.get(self.env.evaluator)
        self.evaluator = ev_cls(self.evaluator_config, self.dataset, self.backbone)

    def build(self):
        return self.evaluator
