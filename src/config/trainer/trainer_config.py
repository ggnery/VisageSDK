from typing import Any
from config.base_config import BaseConfig

class TrainerConfig(BaseConfig):
    optimizer_type: str
    optimizer_params: Any
    
    lr_schedule_type: str
    lr_schedule_params: Any
    
    train_batch_size: int
    train_workers: int
    train_shuffle: bool
    
    val_batch_size: int
    val_workers: int
    val_shuffle: bool

    checkpoint_save_frequency: int
    checkpoint_save_dir: str

    checkpoint_load_path: str
    checkpoint_load_backbone: bool
    checkpoint_load_loss: bool
    checkpoint_load_scheduler: bool
    checkpoint_load_optimizer: bool
    
    num_epochs: int
    
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        
        self.optmizer_type = self.config["optimizer"]["type"]
        self.optmizer_params = self.config["optimizer"]["params"]
      
        train_dataloader = self.config["dataloader"]["train"]
        val_dataloader = self.config["dataloader"]["val"]
        
        self.train_batch_size = train_dataloader["batch_size"]
        self.train_workers = train_dataloader["num_workers"]
        self.train_shuffle = train_dataloader["shuffle"]
        
        self.val_batch_size = val_dataloader["batch_size"]
        self.val_workers = val_dataloader["num_workers"]
        self.val_shuffle = val_dataloader["shuffle"]
        
        self.lr_schedule_type = self.config["lr_schedule"]["type"]
        self.lr_schedule_params = self.config["lr_schedule"]["params"]
        
        self.num_epochs = self.config["num_epochs"]
        
        self.device = self.config["device"]
        
        checkpoint_save = self.config["checkpoint"]["save"]
        checkpoint_load = self.config["checkpoint"]["load"]

        self.checkpoint_save_frequency = checkpoint_save["frequency"]
        self.checkpoint_save_dir = checkpoint_save["dir"]
        
        self.checkpoint_load_path = checkpoint_load["path"]
        self.checkpoint_load_backbone = checkpoint_load["backbone"]
        self.checkpoint_load_loss = checkpoint_load["loss"]
        self.checkpoint_load_scheduler = checkpoint_load["scheduler"]
        self.checkpoint_load_optimizer = checkpoint_load["optimizer"]

        self.build_config()