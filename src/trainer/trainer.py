from typing import Dict, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm
from backbone.base_backbone import BaseBackbone
from batch_sampler.base_batch_sampler import BaseBatchSampler
from config.trainer.trainer_config import TrainerConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset
from early_stopper.base_early_stopper import BaseEarlyStopper
from loss.base_loss import BaseLoss
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
import torch
import logging
import json
from pathlib import Path

from tools.freezer import log_freeze_state, unfreeze_by_patterns


class Trainer:
    device: torch.device

    train_loader: DataLoader
    val_loader: DataLoader
    backbone: BaseBackbone
    loss: BaseLoss
    optimizer: Optimizer
    scheduler: LRScheduler
    config: TrainerConfig

    num_epochs: int
    epoch: int = 1
    best_val_loss: float = float("inf")
    dataset_class_name: str

    checkpoint_load_path: Path
    checkpoint_save_dir: Path
    checkpoint_frequency: int

    def __init__(self, config: TrainerConfig,
                 train_dataset: BaseTrainValDataset,
                 val_dataset: BaseTrainValDataset,
                 backbone: BaseBackbone,
                 loss: BaseLoss,
                 optimizer: Optimizer,
                 scheduler: LRScheduler,
                 sampler: BaseBatchSampler = None,
                 early_stopper: BaseEarlyStopper = None):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.device = torch.device(config.device)
        self.config = config
        self.early_stopper = early_stopper

        if sampler is not None:
            self.logger.info(
                f"{sampler.__class__.__name__} is being used. batch_size and shuffle deactivated in training"
            )
            self.train_loader = DataLoader(
                dataset=train_dataset,
                num_workers=config.train_workers,
                batch_sampler=sampler,
                pin_memory=True,
            )
        else:
            self.logger.info(
                "Sampler is NOT being used. batch_size and shuffle are default from config in training"
            )
            self.train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=config.train_batch_size,
                num_workers=config.train_workers,
                shuffle=config.train_shuffle,
                pin_memory=True,
            )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.val_batch_size,
            num_workers=config.val_workers,
            shuffle=config.val_shuffle,
            pin_memory=True,
        )

        self.backbone = backbone
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_epochs = config.num_epochs
        self.checkpoint_frequency = config.checkpoint_save_frequency

        if config.checkpoint_load_path is not None:
            self.load_checkpoint(
                Path(config.checkpoint_load_path),
                config.checkpoint_load_backbone,
                config.checkpoint_load_loss,
                config.checkpoint_load_scheduler,
                config.checkpoint_load_optimizer,
            )

        self.checkpoint_save_dir = Path(config.checkpoint_save_dir)
        self.dataset_class_name = train_dataset.__class__.__name__.replace("Train", "")

    def train(self):
        for self.epoch in range(self.epoch, self.num_epochs + 1):
            self._apply_unfreeze_schedule()
            train_loss, train_stats = self.train_epoch()
            val_loss, val_stats = self.validate_epoch()

            self.logger.info(
                f"Epoch {self.epoch}/{self.num_epochs} - "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f}"
            )

            if self.epoch % self.checkpoint_frequency == 0 or self.epoch == self.num_epochs:
                name = self._checkpoint_name(f"epoch_{self.epoch}")
                self.save_checkpoint(train_loss, val_loss, name)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                name = self._checkpoint_name("best")
                self.save_checkpoint(train_loss, val_loss, name)
                self.logger.info(f"New best validation loss: {val_loss:.4f}")

            self.save_stats(train_loss, val_loss, train_stats, val_stats)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if self.early_stopper is not None and self.early_stopper.early_stop(val_loss):
                self.logger.info(f"Early stopping {self.early_stopper.__class__.__name__} triggered")
                break

    def train_epoch(self) -> Tuple[float, Dict]:
        total_loss = 0.0
        total_samples = 0
        running_stats: Dict[str, float] = {}
        n_batches = 0

        self.backbone.train()
        self.loss.train()

        pbar = tqdm(self.train_loader, desc=f"Train epoch {self.epoch}")
        for labels, images in pbar:
            self.optimizer.zero_grad()

            images = images.to(self.device)
            labels = labels.to(self.device)

            embeddings = self.backbone(images)
            loss, batch_stats = self.loss(embeddings, labels)

            loss.backward()
            self.optimizer.step()

            batch_size = embeddings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            n_batches += 1

            for k, v in batch_stats.items():
                if isinstance(v, (int, float)):
                    running_stats[k] = running_stats.get(k, 0.0) + float(v)

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / total_samples
        epoch_stats = {k: v / n_batches for k, v in running_stats.items()}
        return avg_loss, epoch_stats

    def validate_epoch(self) -> Tuple[float, Dict]:
        total_loss = 0.0
        total_samples = 0
        running_stats: Dict[str, float] = {}
        n_batches = 0

        self.backbone.eval()
        self.loss.eval()

        pbar = tqdm(self.val_loader, desc=f"Val epoch {self.epoch}")
        with torch.no_grad():
            for labels, images in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                embeddings = self.backbone(images)
                loss, batch_stats = self.loss(embeddings, labels)

                batch_size = embeddings.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                n_batches += 1

                for k, v in batch_stats.items():
                    if isinstance(v, (int, float)):
                        running_stats[k] = running_stats.get(k, 0.0) + float(v)

                pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / total_samples
        epoch_stats = {k: v / n_batches for k, v in running_stats.items()}
        return avg_loss, epoch_stats

    def _apply_unfreeze_schedule(self) -> None:
        patterns = self.config.unfreeze_at_epoch.get(self.epoch)
        if not patterns:
            return
        unfrozen = unfreeze_by_patterns(self.backbone, patterns)
        if unfrozen:
            self.logger.info(
                f"Epoch {self.epoch}: unfroze {len(unfrozen)} params matching {patterns}"
            )
            log_freeze_state(self.backbone, self.logger)

    def _checkpoint_name(self, suffix: str) -> str:
        return f"{self.backbone.__class__.__name__}_{self.loss.__class__.__name__}_{self.dataset_class_name}_{suffix}.pth"

    def save_checkpoint(self, train_loss: float, val_loss: float, checkpoint_name: str):
        checkpoint = {
            "epoch": self.epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "backbone_state_dict": self.backbone.state_dict(),
            "loss_state_dict": self.loss.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        path = self.checkpoint_save_dir / checkpoint_name
        self.checkpoint_save_dir.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: Path,
                        load_backbone: bool,
                        load_loss: bool,
                        load_scheduler: bool,
                        load_optimizer: bool):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if load_backbone:
            self.backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=False)
        if load_loss:
            self.loss.load_state_dict(checkpoint["loss_state_dict"], strict=False)
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["val_loss"]

        self.logger.info(f"Checkpoint {checkpoint_path} for backbone {self.backbone.__class__.__name__} successfully loaded")
        self.logger.info(f"Resuming train in epoch {self.epoch}")

    def save_stats(self, train_loss: float, val_loss: float, train_stats: Dict, val_stats: Dict):
        history_path = self.checkpoint_save_dir / self._checkpoint_name("training_history").replace(".pth", ".json")

        full_history: Dict = {}
        if history_path.exists():
            with open(history_path, "r") as f:
                full_history = json.load(f)

        full_history[f"epoch_{self.epoch}"] = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        with open(history_path, "w") as f:
            json.dump(full_history, f, indent=2)
