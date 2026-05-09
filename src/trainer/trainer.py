from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import logging
import json
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from tqdm import tqdm

from backbone.base_backbone import BaseBackbone
from batch_sampler.base_batch_sampler import BaseBatchSampler
from config.trainer.trainer_config import TrainerConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset
from early_stopper.base_early_stopper import BaseEarlyStopper
from evaluator.base_evaluator import BaseEvaluator
from loss.base_loss import BaseLoss
from tools.freezer import log_freeze_state, unfreeze_by_patterns
from tools.seed import make_dataloader_generator, seed_worker


_DTYPE_MAP = {"float16": torch.float16, "bfloat16": torch.bfloat16, "fp16": torch.float16, "bf16": torch.bfloat16}


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
                 sampler: Optional[BaseBatchSampler] = None,
                 early_stopper: Optional[BaseEarlyStopper] = None,
                 periodic_evaluator: Optional[BaseEvaluator] = None):

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # NOTE: set_seed is called in TrainerBuilder before model instantiation;
        # by the time we get here, weight init is already deterministic.

        self.device = torch.device(config.device)
        self.config = config
        self.early_stopper = early_stopper
        self.periodic_evaluator = periodic_evaluator

        gen = make_dataloader_generator(config.seed)
        worker_init = seed_worker if config.seed is not None else None

        if sampler is not None:
            self.logger.info(
                f"{sampler.__class__.__name__} is being used. batch_size and shuffle deactivated in training"
            )
            self.train_loader = DataLoader(
                dataset=train_dataset,
                num_workers=config.train_workers,
                batch_sampler=sampler,
                pin_memory=True,
                worker_init_fn=worker_init,
                generator=gen,
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
                worker_init_fn=worker_init,
                generator=gen,
            )

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=config.val_batch_size,
            num_workers=config.val_workers,
            shuffle=config.val_shuffle,
            pin_memory=True,
            worker_init_fn=worker_init,
            generator=gen,
        )

        self.backbone = backbone
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.num_epochs = config.num_epochs
        self.checkpoint_frequency = config.checkpoint_save_frequency

        self.amp_enabled = config.amp_enabled and self.device.type == "cuda"
        if config.amp_enabled and self.device.type != "cuda":
            self.logger.info("AMP requested but device is not CUDA — disabling AMP.")
        self.amp_dtype = _DTYPE_MAP.get(config.amp_dtype.lower(), torch.float16)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled and self.amp_dtype == torch.float16)

        self.grad_clip_max_norm = config.grad_clip_max_norm
        self.grad_clip_norm_type = config.grad_clip_norm_type

        self.writer = self._build_tensorboard_writer()

        if config.checkpoint_load_path is not None:
            self.load_checkpoint(
                Path(config.checkpoint_load_path),
                config.checkpoint_load_backbone,
                config.checkpoint_load_loss,
                config.checkpoint_load_scheduler,
                config.checkpoint_load_optimizer,
            )
            # `requires_grad` is not part of state_dict, so the freeze state at this
            # point is whatever the builder set up before training started. Replay
            # unfreeze events that happened on or before the checkpoint epoch so
            # the trainable set matches what the run was using when it stopped.
            self._replay_unfreeze_up_to(self.epoch - 1)

        self.checkpoint_save_dir = Path(config.checkpoint_save_dir)
        self.dataset_class_name = train_dataset.__class__.__name__.replace("Train", "")

    def _build_tensorboard_writer(self):
        if not self.config.tensorboard_enabled:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            self.logger.warning(f"tensorboard not available ({e}); disabling TB logging")
            return None
        log_dir = Path(self.config.tensorboard_log_dir or (Path(self.config.checkpoint_save_dir) / "runs"))
        log_dir = log_dir / datetime.now().isoformat(timespec="seconds")
        log_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"TensorBoard logs: {log_dir}")
        return SummaryWriter(log_dir=str(log_dir))

    def train(self):
        for self.epoch in range(self.epoch, self.num_epochs + 1):
            self._apply_unfreeze_schedule(self.epoch)
            train_loss, train_stats = self.train_epoch()
            val_loss, val_stats = self.validate_epoch()
            current_lr = self.scheduler.get_last_lr()[0]

            self.logger.info(
                f"Epoch {self.epoch}/{self.num_epochs} - "
                f"LR: {current_lr:.6f} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f}"
            )

            self._tb_log_epoch(train_loss, val_loss, current_lr, train_stats, val_stats)
            eval_results = self._maybe_run_periodic_eval()

            if self.epoch % self.checkpoint_frequency == 0 or self.epoch == self.num_epochs:
                self.save_checkpoint(train_loss, val_loss, self._checkpoint_name(f"epoch_{self.epoch}"))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(train_loss, val_loss, self._checkpoint_name("best"))
                self.logger.info(f"New best validation loss: {val_loss:.4f}")

            self.save_stats(train_loss, val_loss, train_stats, val_stats, eval_results)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            if self.early_stopper is not None and self.early_stopper.early_stop(val_loss):
                self.logger.info(f"Early stopping {self.early_stopper.__class__.__name__} triggered")
                break

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def _apply_unfreeze_schedule(self, epoch: int) -> None:
        patterns = self.config.unfreeze_at_epoch.get(epoch)
        if not patterns:
            return
        unfrozen = unfreeze_by_patterns(self.backbone, patterns)
        if unfrozen:
            self.logger.info(
                f"Epoch {epoch}: unfroze {len(unfrozen)} params matching {patterns}"
            )
            log_freeze_state(self.backbone, self.logger)

    def _replay_unfreeze_up_to(self, last_completed_epoch: int) -> None:
        """Re-apply every unfreeze event scheduled at epoch <= last_completed_epoch.

        Used after `load_checkpoint` since `requires_grad` is not persisted in
        state_dict; without this, resuming a run wipes the progressive
        unfreezes that happened before the checkpoint was saved.
        """
        if last_completed_epoch < 1:
            return
        replayed = 0
        for ep in sorted(self.config.unfreeze_at_epoch):
            if ep <= last_completed_epoch:
                unfrozen = unfreeze_by_patterns(self.backbone, self.config.unfreeze_at_epoch[ep])
                replayed += len(unfrozen)
        if replayed:
            self.logger.info(f"Replayed {replayed} unfreeze events up to epoch {last_completed_epoch}")
            log_freeze_state(self.backbone, self.logger)

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=self.amp_dtype)

    def _step(self, use_scaler: bool) -> None:
        """Optimizer step with optional AMP scaler and gradient clipping.

        With AMP fp16: must `unscale_` before clipping so the clip threshold
        applies to true (post-unscale) gradient magnitudes.
        """
        if self.grad_clip_max_norm is not None:
            if use_scaler:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in self.optimizer.param_groups for p in g["params"] if p.requires_grad],
                max_norm=self.grad_clip_max_norm,
                norm_type=self.grad_clip_norm_type,
            )
        if use_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

    def train_epoch(self) -> Tuple[float, Dict]:
        total_loss = 0.0
        total_samples = 0
        running_stats: Dict[str, float] = {}
        n_batches = 0

        self.backbone.train()
        self.loss.train()
        use_scaler = self.amp_enabled and self.amp_dtype == torch.float16

        pbar = tqdm(self.train_loader, desc=f"Train epoch {self.epoch}")
        for labels, images in pbar:
            self.optimizer.zero_grad(set_to_none=True)

            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with self._autocast():
                embeddings = self.backbone(images)
                loss, batch_stats = self.loss(embeddings, labels)

            if use_scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            self._step(use_scaler)

            batch_size = embeddings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            n_batches += 1

            for k, v in batch_stats.items():
                if isinstance(v, (int, float)):
                    running_stats[k] = running_stats.get(k, 0.0) + float(v)

            pbar.set_postfix({"loss": loss.item()})

        if total_samples == 0:
            self.logger.warning("Train epoch saw zero samples; skipping averaging.")
            return 0.0, {}
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
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with self._autocast():
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

        if total_samples == 0:
            self.logger.warning("Val epoch saw zero samples; skipping averaging.")
            return 0.0, {}
        avg_loss = total_loss / total_samples
        epoch_stats = {k: v / n_batches for k, v in running_stats.items()}
        return avg_loss, epoch_stats

    def _maybe_run_periodic_eval(self) -> Optional[Dict[str, float]]:
        if self.periodic_evaluator is None:
            return None
        every_n = (self.config.periodic_eval or {}).get("every_n_epochs", 1)
        if self.epoch % every_n != 0 and self.epoch != self.num_epochs:
            return None
        self.logger.info(f"Running periodic eval at epoch {self.epoch}")
        results = self.periodic_evaluator.evaluate()
        for k, v in results.items():
            self.logger.info(f"  eval/{k} = {v:.6f}")
            if self.writer is not None:
                self.writer.add_scalar(f"eval/{k}", float(v), self.epoch)
        return results

    def _tb_log_epoch(self, train_loss: float, val_loss: float, lr: float,
                      train_stats: Dict, val_stats: Dict) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("loss/train", train_loss, self.epoch)
        self.writer.add_scalar("loss/val", val_loss, self.epoch)
        self.writer.add_scalar("lr", lr, self.epoch)
        for k, v in train_stats.items():
            self.writer.add_scalar(f"train_stats/{k}", v, self.epoch)
        for k, v in val_stats.items():
            self.writer.add_scalar(f"val_stats/{k}", v, self.epoch)

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
            "scaler_state_dict": self.scaler.state_dict() if self.amp_enabled else None,
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
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if load_backbone:
            self.backbone.load_state_dict(checkpoint["backbone_state_dict"], strict=False)
        if load_loss:
            self.loss.load_state_dict(checkpoint["loss_state_dict"], strict=False)
        if load_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.amp_enabled and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["val_loss"]

        self.logger.info(f"Checkpoint {checkpoint_path} for backbone {self.backbone.__class__.__name__} successfully loaded")
        self.logger.info(f"Resuming train in epoch {self.epoch}")

    def save_stats(self, train_loss: float, val_loss: float,
                   train_stats: Dict, val_stats: Dict,
                   eval_results: Optional[Dict[str, float]] = None):
        history_name = self._checkpoint_name("training_history").replace(".pth", ".json")
        history_path = self.checkpoint_save_dir / history_name
        full_history: Dict = {}
        if history_path.exists():
            with open(history_path, "r") as f:
                full_history = json.load(f)

        entry: Dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_stats": train_stats,
            "val_stats": val_stats,
        }
        if eval_results is not None:
            entry["eval"] = eval_results
        full_history[f"epoch_{self.epoch}"] = entry
        with open(history_path, "w") as f:
            json.dump(full_history, f, indent=2)
