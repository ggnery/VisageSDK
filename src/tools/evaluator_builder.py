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

# Side-effect imports populate the registry.
import backbone  # noqa: F401
import dataset.eval  # noqa: F401
import evaluator  # noqa: F401
import transformation  # noqa: F401


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

        self.transformation_config = TransformationConfig(
            self.env.eval_transformation_config, backbone_info
        )
        self.config_str += self.transformation_config.get_config_string()

        self.dataset_config = EvalDatasetConfig(
            self.env.eval_dataset_config, backbone_info
        )
        self.config_str += self.dataset_config.get_config_string()

        self.evaluator_config = EvaluatorConfig(self.env.evaluator_config)
        self.config_str += self.evaluator_config.get_config_string()

    def _build_instances(self) -> None:
        device = torch.device(self.evaluator_config.device)

        backbone_cls = BACKBONES.get(self.env.backbone)
        self.backbone = backbone_cls(self.backbone_config).to(device)

        ckpt = torch.load(self.env.checkpoint_path, map_location=device, weights_only=False)
        if "backbone_state_dict" in ckpt:
            self.backbone.load_state_dict(ckpt["backbone_state_dict"], strict=False)
        else:
            # raw state dict
            self.backbone.load_state_dict(ckpt, strict=False)

        tx_cls = TRANSFORMATIONS.get(self.env.eval_transformation)
        self.transformation = tx_cls(self.transformation_config)

        ds_cls = EVAL_DATASETS.get(self.env.eval_dataset)
        self.dataset = ds_cls(self.dataset_config, self.transformation)

        ev_cls = EVALUATORS.get(self.env.evaluator)
        self.evaluator = ev_cls(self.evaluator_config, self.dataset, self.backbone)

    def build(self):
        return self.evaluator
