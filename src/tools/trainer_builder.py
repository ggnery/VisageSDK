import importlib
from datetime import datetime
from pathlib import Path

from config.backbone.base_backbone_config import BackboneConfig
from config.batch_sampler.base_batch_sampler_config import BatchSamplerConfig
from config.dataset.eval.base_eval_dataset_config import EvalDatasetConfig
from config.dataset.train_val.base_train_val_dataset_config import TrainValDatasetConfig
from config.early_stopper.base_early_stopper_config import EarlyStopperConfig
from config.env_config import ENVConfig
from config.evaluator.base_evaluator_config import EvaluatorConfig
from config.loss.base_loss_config import LossConfig
from config.trainer.trainer_config import TrainerConfig
from config.transformation.base_transformation_config import TransformationConfig
from registry import (
    BACKBONES,
    DATASETS,
    EARLY_STOPPERS,
    EVAL_DATASETS,
    EVALUATORS,
    LOSSES,
    SAMPLERS,
    TRANSFORMATIONS,
)
from tools.freezer import freeze_by_patterns, log_freeze_state
from tools.optimizer import build_optimizer
from tools.scheduler import build_scheduler
from tools.seed import set_seed
from trainer.trainer import Trainer

# Trigger registry population by importing each component package for side effects.
for _component_pkg in (
    "backbone",
    "batch_sampler",
    "dataset.eval",
    "dataset.train_val",
    "early_stopper",
    "evaluator",
    "loss",
    "transformation",
):
    importlib.import_module(_component_pkg)


class TrainerBuilder:
    def __init__(self, env_config: ENVConfig):
        self.env = env_config
        self.config_str = ""
        self._build_configs()
        # Seed BEFORE instantiating any model so weight initialization is reproducible.
        set_seed(self.trainer_config.seed, deterministic=self.trainer_config.deterministic)
        self._build_instances()

    def _build_configs(self) -> None:
        self.backbone_config = BackboneConfig(self.env.backbone_config)
        self.config_str += self.backbone_config.get_config_string()
        backbone_info = {
            "input_size": self.backbone_config.input_size,
            "embedding_size": self.backbone_config.embedding_size,
        }

        self.transformation_config = TransformationConfig(
            self.env.train_val_transformation_config, backbone_info
        )
        self.config_str += self.transformation_config.get_config_string()

        self.dataset_config = TrainValDatasetConfig(self.env.train_val_dataset_config, backbone_info)
        self.config_str += self.dataset_config.get_config_string()
        dataset_info = {"num_classes": self.dataset_config.num_classes}

        self.loss_config = LossConfig(self.env.loss_config, backbone_info, dataset_info)
        self.config_str += self.loss_config.get_config_string()

        self.early_stopper_config = None
        if self.env.early_stopper and self.env.early_stopper_config:
            self.early_stopper_config = EarlyStopperConfig(self.env.early_stopper_config)
            self.config_str += self.early_stopper_config.get_config_string()

        self.sampler_config = None
        if self.env.sampler and self.env.sampler_config:
            self.sampler_config = BatchSamplerConfig(self.env.sampler_config)
            self.config_str += self.sampler_config.get_config_string()

        self.trainer_config = TrainerConfig(self.env.trainer_config)
        self.config_str += self.trainer_config.get_config_string()

        save_dir = Path(self.trainer_config.checkpoint_save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        snapshot = save_dir / f"train_config_{datetime.now().isoformat(timespec='seconds')}.txt"
        snapshot.write_text(self.config_str)
        print(self.config_str)

    def _build_instances(self) -> None:
        backbone_cls = BACKBONES.get(self.env.backbone)
        self.backbone = backbone_cls(self.backbone_config).to(self.backbone_config.device)

        train_tx_cls = TRANSFORMATIONS.get(self.env.train_transformation)
        val_tx_cls = TRANSFORMATIONS.get(self.env.val_transformation)
        self.train_transformation = train_tx_cls(self.transformation_config)
        self.val_transformation = val_tx_cls(self.transformation_config)

        ds_cls = DATASETS.get(self.env.train_val_dataset)
        self.train_dataset = ds_cls(self.dataset_config, self.train_transformation, split="train")
        self.val_dataset = ds_cls(self.dataset_config, self.val_transformation, split="val")

        loss_cls = LOSSES.get(self.env.loss)
        self.loss = loss_cls(self.loss_config).to(self.loss_config.device)

        self.early_stopper = None
        if self.early_stopper_config is not None and self.env.early_stopper:
            self.early_stopper = EARLY_STOPPERS.get(self.env.early_stopper)(self.early_stopper_config)

        self.sampler = None
        if self.sampler_config is not None and self.env.sampler:
            self.sampler = SAMPLERS.get(self.env.sampler)(self.sampler_config, self.train_dataset)

        if self.trainer_config.freeze_patterns or self.trainer_config.freeze_except:
            freeze_by_patterns(
                self.backbone,
                patterns=self.trainer_config.freeze_patterns,
                except_patterns=self.trainer_config.freeze_except,
            )
            log_freeze_state(self.backbone)

        self.optimizer = build_optimizer(self.backbone, self.loss, self.trainer_config)
        self.scheduler = build_scheduler(self.optimizer, self.trainer_config)

        self.periodic_evaluator = self._build_periodic_evaluator()

    def _build_periodic_evaluator(self):
        block = self.trainer_config.periodic_eval
        if not block or not block.get("enabled", True):
            return None

        backbone_info = {
            "input_size": self.backbone_config.input_size,
            "embedding_size": self.backbone_config.embedding_size,
        }

        eval_tx_cls = TRANSFORMATIONS.get(block["transformation"])
        eval_tx_cfg = TransformationConfig(block["transformation_config"], backbone_info)
        eval_tx = eval_tx_cls(eval_tx_cfg)

        eval_ds_cls = EVAL_DATASETS.get(block["dataset"])
        eval_ds_cfg = EvalDatasetConfig(block["dataset_config"], backbone_info)
        eval_ds = eval_ds_cls(eval_ds_cfg, eval_tx)

        evaluator_cls = EVALUATORS.get(block["evaluator"])
        evaluator_cfg = EvaluatorConfig(block["evaluator_config"])
        return evaluator_cls(evaluator_cfg, eval_ds, self.backbone)

    def build_trainer(self) -> Trainer:
        return Trainer(
            self.trainer_config,
            self.train_dataset,
            self.val_dataset,
            self.backbone,
            self.loss,
            self.optimizer,
            self.scheduler,
            self.sampler,
            self.early_stopper,
            self.periodic_evaluator,
        )
