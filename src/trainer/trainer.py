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
from trainer.training_context import BatchContext, EpochContext, TrainingContext
import torch
import logging
import json
from pathlib import Path

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

    train_ctx: TrainingContext
    
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
            self.logger.info(f"{sampler.__class__.__name__} is being used. batch_size and shuffle deactivated in training")
            self.train_loader = DataLoader(dataset=train_dataset, 
                                    num_workers=config.train_workers,
                                    batch_sampler = sampler,
                                    pin_memory=True)
        else:
            self.logger.info("Sampler is NOT being used. batch_size and shuffle are default from config in training")
            self.train_loader = DataLoader(dataset=train_dataset, 
                                    batch_size=config.train_batch_size,
                                    num_workers=config.train_workers,
                                    shuffle=config.train_shuffle,
                                    pin_memory=True)
        
        self.val_loader = DataLoader(dataset=val_dataset, 
                                batch_size=config.val_batch_size,
                                num_workers=config.val_workers,
                                shuffle=config.val_shuffle,
                                pin_memory=True)
           
        self.backbone = backbone
        self.loss= loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.num_epochs = config.num_epochs
        
        self.checkpoint_frequency = config.checkpoint_save_frequency
        if config.checkpoint_load_path != None:
            checkpoint_load_path = Path(config.checkpoint_load_path)
            self.load_checkpoint(checkpoint_load_path, 
                                 config.checkpoint_load_backbone, 
                                 config.checkpoint_load_loss, 
                                 config.checkpoint_load_scheduler, 
                                 config.checkpoint_load_optimizer)
            
        self.checkpoint_save_dir = Path(config.checkpoint_save_dir)
        
        self.dataset_class_name = train_dataset.__class__.__name__.replace("Train", "")    
        
        self.train_ctx = TrainingContext(self.optimizer, self.scheduler, self.train_loader, self.val_loader, self.num_epochs)
    
    def train(self):
        self.backbone.before_train(self.train_ctx)
        self.loss.before_train(self.train_ctx)
                
        for self.epoch in range(self.epoch, self.num_epochs + 1):
            train_loss, epoch_train_stats = self.train_epoch()
            val_loss, epoch_val_stats = self.validate_epoch()
            
            epoch_ctx = EpochContext(self.epoch, train_loss, val_loss)
            self.backbone.after_epoch(epoch_ctx, self.train_ctx)
            self.loss.after_epoch(epoch_ctx, self.train_ctx)  
            
            self.logger.info(
                f"Epoch {self.epoch}/{self.num_epochs} - "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f} - "
                f"Train Loss: {train_loss} - "
                f"Val Loss: {val_loss} - "
            )

            if self.epoch % self.checkpoint_frequency == 0 or self.epoch == self.num_epochs:
                checkpoint_name = f"{self.backbone.__class__.__name__}_{self.loss.__class__.__name__}_{self.dataset_class_name}_epoch_{self.epoch}.pth"
                self.save_checkpoint(train_loss, val_loss, checkpoint_name)       
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                checkpoint_name = f"{self.backbone.__class__.__name__}_{self.loss.__class__.__name__}_{self.dataset_class_name}_best.pth"
                self.save_checkpoint(train_loss, val_loss, checkpoint_name)  
                self.logger.info(f"New best validation loss: {val_loss:.4f}")

            self.save_stats(epoch_train_stats, epoch_val_stats, train_loss, val_loss)
            
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
                
            if self.early_stopper is not None and self.early_stopper.early_stop(epoch_ctx):
                self.logger.info(f"Early stopping {self.early_stopper.__class__.__name__} triggered")
                break      
            
    def train_epoch(self) -> Tuple[float, Dict]:
        total_loss = 0.0
        total_samples = 0
        epoch_train_stats = {}
        
        # train mode
        self.backbone.train()
        self.loss.train()
        
        pbar = tqdm(self.train_loader, desc=f"Train epoch {self.epoch}")
        for batch_idx, (labels, images) in enumerate(pbar):
            self.optimizer.zero_grad() #zero previous gradients

            images = images.to(self.device)
            labels = labels.to(self.device)
                    
            # Forward pass 
            embeddings = self.backbone.forward(images)
            loss, train_loss_stats = self.loss.forward(embeddings, labels)
            
            batch_ctx = BatchContext(images, labels, embeddings, loss, batch_idx)    
        
            #Backward pass
            loss.backward()
            
            self.loss.after_backward(batch_ctx, self.train_ctx)
            self.backbone.after_backward(batch_ctx, self.train_ctx)
            
            self.optimizer.step()                                  
                             
            self.loss.after_step(batch_ctx, self.train_ctx)
            self.backbone.after_step(batch_ctx, self.train_ctx)  
                   
            # Update statistics
            batch_size = embeddings.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            epoch_train_stats[f"batch_{batch_idx}"] = {"train_loss_stats": train_loss_stats}
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        #epoch statistics
        avg_loss = total_loss / total_samples
        
        return avg_loss, epoch_train_stats
    
    def validate_epoch(self) -> Tuple[float, Dict]:
        total_loss = 0.0
        total_samples = 0
        epoch_val_stats = {}
        
        # eval mode for inference
        self.backbone.eval() 
        self.loss.eval()
        
        pbar = tqdm(self.val_loader, desc=f"Val epoch {self.epoch}")
        with torch.no_grad():
            for batch_idx, (labels, images) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass 
                embeddings = self.backbone.forward(images)                
                loss, val_loss_stats = self.loss.forward(embeddings, labels)
                
                # Update statistics
                batch_size = embeddings.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                epoch_val_stats[f"batch_{batch_idx}"] = {"val_loss_stats": val_loss_stats}

                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})
        avg_loss = total_loss / total_samples
        
        return avg_loss, epoch_val_stats      
        
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
                        load_optimizer: bool
                        ):
        checkpoint = torch.load(checkpoint_path)
        
        if load_backbone:
            backbone_state_dict = checkpoint["backbone_state_dict"]
            self.backbone.load_state_dict(backbone_state_dict, strict=False)
        if load_loss:
            self.loss.load_state_dict(checkpoint["loss_state_dict"], strict=False)
        if load_scheduler:    
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"], strict=False)
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"], strict=False)
        self.epoch = checkpoint["epoch"] + 1
        self.best_val_loss = checkpoint["val_loss"]
        
        self.logger.info(f"Checkpoint {checkpoint_path} for backbone {self.backbone.__class__.__name__} successfully loaded")
        self.logger.info(f"Resuming train in epoch {self.epoch}")
    
    def save_stats(self, epoch_train_stats: Dict, epoch_val_stats: Dict, train_loss: float, val_loss: float):
        history_path = self.checkpoint_save_dir / f"{self.backbone.__class__.__name__}_{self.loss.__class__.__name__}_{self.dataset_class_name}_training_history.json"

        full_history = {}
        if history_path.exists():
            with open(history_path, 'r') as f:
                full_history = json.load(f)
            
        
        new_epoch_data = {
            f"epoch_{self.epoch}": {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_train_stats": epoch_train_stats,
                "epoch_val_stats": epoch_val_stats
            }
        }
        
        full_history.update(new_epoch_data)
        with open(history_path, 'w') as f:
            json.dump(full_history, f, indent=2)