from typing import Any, Optional
from config.base_config import BaseConfig


class TrainerConfig(BaseConfig):
    optimizer_type: str
    optimizer_params: Any

    lr_schedule_type: str
    lr_schedule_params: Any

    train_batch_size: Optional[int]
    train_workers: int
    train_shuffle: Optional[bool]

    val_batch_size: int
    val_workers: int
    val_shuffle: bool

    checkpoint_save_frequency: int
    checkpoint_save_dir: str

    checkpoint_load_path: Optional[str]
    checkpoint_load_backbone: bool
    checkpoint_load_loss: bool
    checkpoint_load_scheduler: bool
    checkpoint_load_optimizer: bool

    num_epochs: int
    device: str

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)

        self.optimizer_type = self._params["optimizer"]["type"]
        self.optimizer_params = self._params["optimizer"]["params"]

        self.lr_schedule_type = self._params["lr_schedule"]["type"]
        self.lr_schedule_params = self._params["lr_schedule"]["params"]

        train_dl = self._params["dataloader"]["train"]
        val_dl = self._params["dataloader"]["val"]

        self.train_batch_size = train_dl["batch_size"]
        self.train_workers = train_dl["num_workers"]
        self.train_shuffle = train_dl["shuffle"]

        self.val_batch_size = val_dl["batch_size"]
        self.val_workers = val_dl["num_workers"]
        self.val_shuffle = val_dl["shuffle"]

        self.num_epochs = self._params["num_epochs"]
        self.device = self._params["device"]

        save = self._params["checkpoint"]["save"]
        load = self._params["checkpoint"]["load"]

        self.checkpoint_save_frequency = save["frequency"]
        self.checkpoint_save_dir = save["dir"]

        self.checkpoint_load_path = load["path"]
        self.checkpoint_load_backbone = load["backbone"]
        self.checkpoint_load_loss = load["loss"]
        self.checkpoint_load_scheduler = load["scheduler"]
        self.checkpoint_load_optimizer = load["optimizer"]
