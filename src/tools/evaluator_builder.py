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

        # Rebuild the PEFT wrap before loading so `base_model.model.*` keys match.
        # Without this, strict=False silently drops every weight (random init →
        # deceptive ~85% LFW accuracy).
        lora_config = ckpt.get("lora_config") if isinstance(ckpt, dict) else None
        if lora_config:
            from tools.lora import apply_lora

            target_modules = list(lora_config.get("target_modules", []))

            # Detect backbone/target_modules mismatch up front (PEFT's own error
            # message is cryptic).
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
                    f"Checkpoint {self.env.checkpoint_path} has "
                    f"lora_config.target_modules={target_modules}, but selected "
                    f"backbone '{self.env.backbone}' contains no matching modules. "
                    "Pick the backbone variant the checkpoint was trained on."
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

        # `backbone_state_dict` present → framework-wrapped; absent → raw state_dict.
        sd_to_load = ckpt.get("backbone_state_dict", ckpt)
        # Detect PEFT-prefixed keys without `lora_config` metadata — strict=False
        # would silently drop every weight here.
        if lora_config is None and any(
            k.startswith("base_model.model.") for k in sd_to_load
        ):
            raise ValueError(
                f"Checkpoint {self.env.checkpoint_path} has PEFT-prefixed keys "
                "but no `lora_config` metadata. Run "
                "`scripts/backfill_lora_config.py --run-dir <run>` to inject it."
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
