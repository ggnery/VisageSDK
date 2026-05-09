from typing import Any

from config.base_config import BaseConfig


class TrainerConfig(BaseConfig):
    optimizer_type: str
    optimizer_params: Any
    optimizer_param_groups: list[dict[str, Any]] | None

    lr_schedule_type: str
    lr_schedule_params: Any

    train_batch_size: int | None
    train_workers: int
    train_shuffle: bool | None

    val_batch_size: int
    val_workers: int
    val_shuffle: bool

    checkpoint_save_frequency: int
    checkpoint_save_dir: str

    checkpoint_load_path: str | None
    checkpoint_load_backbone: bool
    checkpoint_load_loss: bool
    checkpoint_load_scheduler: bool
    checkpoint_load_optimizer: bool

    num_epochs: int
    device: str

    freeze_patterns: list[str] | None
    freeze_except: list[str] | None
    unfreeze_at_epoch: dict[int, list[str]]

    seed: int | None
    deterministic: bool

    amp_enabled: bool
    amp_dtype: str

    grad_clip_max_norm: float | None
    grad_clip_norm_type: float

    tensorboard_enabled: bool
    tensorboard_log_dir: str | None

    periodic_eval: dict[str, Any] | None

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)

        optimizer_block = self._params["optimizer"]
        self.optimizer_type = optimizer_block["type"]
        self.optimizer_params = optimizer_block["params"]
        self.optimizer_param_groups = optimizer_block.get("param_groups")

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

        freeze = self._params.get("freeze") or {}
        self.freeze_patterns = freeze.get("patterns")
        self.freeze_except = freeze.get("except")
        # Use presence (not truthiness) so `patterns: []` + `except: [...]`
        # is also rejected — almost certainly a user typo.
        if self.freeze_patterns is not None and self.freeze_except is not None:
            raise ValueError("freeze: provide either `patterns` or `except`, not both")
        raw_schedule = freeze.get("unfreeze_at_epoch") or {}
        self.unfreeze_at_epoch = {int(k): list(v) for k, v in raw_schedule.items()}

        self.seed = self._params.get("seed")
        self.deterministic = bool(self._params.get("deterministic", False))

        amp = self._params.get("amp") or {}
        self.amp_enabled = bool(amp.get("enabled", False))
        self.amp_dtype = str(amp.get("dtype", "float16"))

        grad_clip = self._params.get("gradient_clip") or {}
        self.grad_clip_max_norm = grad_clip.get("max_norm")
        # Use `or 2.0` so explicit YAML `norm_type: null` falls back to the default.
        self.grad_clip_norm_type = float(grad_clip.get("norm_type") or 2.0)

        logging_block = self._params.get("logging") or {}
        self.tensorboard_enabled = bool(logging_block.get("tensorboard", False))
        self.tensorboard_log_dir = logging_block.get("log_dir")

        self.periodic_eval = self._params.get("periodic_eval")
