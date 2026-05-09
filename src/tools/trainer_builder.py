from datetime import datetime
from pathlib import Path

from config.backbone.base_backbone_config import BackboneConfig
from config.batch_sampler.base_batch_sampler_config import BatchSamplerConfig
from config.dataset.train_val.base_train_val_dataset_config import TrainValDatasetConfig
from config.early_stopper.base_early_stopper_config import EarlyStopperConfig
from config.env_config import ENVConfig
from config.loss.base_loss_config import LossConfig
from config.trainer.trainer_config import TrainerConfig
from config.transformation.base_transformation_config import TransformationConfig
from registry import (
    BACKBONES,
    DATASETS,
    EARLY_STOPPERS,
    LOSSES,
    SAMPLERS,
    TRAIN_TRANSFORMATIONS,
    VAL_TRANSFORMATIONS,
)
from tools.optimizer import build_optimizer
from tools.scheduler import build_scheduler
from trainer.trainer import Trainer

# Trigger registry population. Side-effect imports — keep them.
import backbone  # noqa: F401
import loss  # noqa: F401
import dataset.train_val  # noqa: F401
import early_stopper  # noqa: F401
import batch_sampler  # noqa: F401
import transformation  # noqa: F401


class TrainerBuilder:
    def __init__(self, env_config: ENVConfig):
        self.env = env_config
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
            self.env.train_val_transformation_config, backbone_info
        )
        self.config_str += self.transformation_config.get_config_string()

        self.dataset_config = TrainValDatasetConfig(
            self.env.train_val_dataset_config, backbone_info
        )
        self.config_str += self.dataset_config.get_config_string()
        dataset_info = {"num_classes": self.dataset_config.num_classes}

        self.loss_config = LossConfig(self.env.loss_config, backbone_info, dataset_info)
        self.config_str += self.loss_config.get_config_string()

        self.early_stopper_config = None
        if self.env.early_stopper:
            self.early_stopper_config = EarlyStopperConfig(self.env.early_stopper_config)
            self.config_str += self.early_stopper_config.get_config_string()

        self.sampler_config = None
        if self.env.sampler:
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

        train_tx_cls = TRAIN_TRANSFORMATIONS.get(self.env.train_transformation)
        val_tx_cls = VAL_TRANSFORMATIONS.get(self.env.val_transformation)
        self.train_transformation = train_tx_cls(self.transformation_config)
        self.val_transformation = val_tx_cls(self.transformation_config)

        train_ds_cls = DATASETS.get(f"{self.env.train_val_dataset}_train")
        val_ds_cls = DATASETS.get(f"{self.env.train_val_dataset}_val")
        self.train_dataset = train_ds_cls(self.dataset_config, self.train_transformation)
        self.val_dataset = val_ds_cls(self.dataset_config, self.val_transformation)

        loss_cls = LOSSES.get(self.env.loss)
        self.loss = loss_cls(self.loss_config).to(self.loss_config.device)

        self.early_stopper = None
        if self.early_stopper_config is not None:
            self.early_stopper = EARLY_STOPPERS.get(self.env.early_stopper)(self.early_stopper_config)

        self.sampler = None
        if self.sampler_config is not None:
            self.sampler = SAMPLERS.get(self.env.sampler)(self.sampler_config, self.train_dataset)

        self.optimizer = build_optimizer(self.backbone, self.loss, self.trainer_config)
        self.scheduler = build_scheduler(self.optimizer, self.trainer_config)

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
        )
