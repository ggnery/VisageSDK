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

    onnx_export_enabled: bool
    onnx_export_opset: int
    onnx_export_dynamic_batch: bool

    lora_enabled: bool
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_target_modules: list[str]
    lora_modules_to_save: list[str]

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

        # Coerce non-positive `frequency` (typo `0`, accidental negative)
        # up to 1 — without this, `epoch % frequency` in the train loop
        # crashes mid-run with ZeroDivisionError.
        freq = int(save["frequency"])
        self.checkpoint_save_frequency = max(freq, 1)
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

        onnx_export = self._params.get("onnx_export") or {}
        self.onnx_export_enabled = bool(onnx_export.get("enabled", False))
        self.onnx_export_opset = int(onnx_export.get("opset", 17))
        self.onnx_export_dynamic_batch = bool(onnx_export.get("dynamic_batch", True))

        lora = self._params.get("lora") or {}
        self.lora_enabled = bool(lora.get("enabled", False))
        self.lora_rank = int(lora.get("rank", 8))
        self.lora_alpha = float(lora.get("alpha", 16.0))
        self.lora_dropout = float(lora.get("dropout", 0.0))
        self.lora_target_modules = list(lora.get("target_modules", []))
        # `modules_to_save` is the PEFT escape hatch for modules that
        # need full fine-tuning (not LoRA). Use it for components like the
        # final feature head where adapter-only updates aren't expressive
        # enough to re-shape the embedding space for a new domain.
        self.lora_modules_to_save = list(lora.get("modules_to_save", []))
        if self.lora_enabled and not self.lora_target_modules:
            raise ValueError("lora.enabled requires at least one entry in lora.target_modules")

        # When LoRA is enabled, PEFT wraps the backbone and renames every
        # parameter to `base_model.model.<original>` while marking only
        # `lora_A`/`lora_B` (and any `modules_to_save` entries) trainable.
        # This makes the following blocks silently no-op or incompatible:
        #   - `freeze.patterns` / `freeze.except`: PEFT already handles
        #     freezing — running freeze_by_patterns first only to have PEFT
        #     overwrite it leaves the user with the wrong mental model.
        #   - `freeze.unfreeze_at_epoch`: patterns are written against
        #     bare backbone names and never match PEFT-prefixed ones; the
        #     schedule is silently a no-op.
        #   - `optimizer.param_groups`: pattern strings like
        #     "backbone.last_linear*" don't match
        #     "backbone.base_model.model.last_linear.lora_A.default.weight",
        #     so discriminative LRs collapse into the default group.
        # Catch all three combinations up front with explicit errors.
        if self.lora_enabled:
            if self.freeze_patterns or self.freeze_except:
                raise ValueError(
                    "`lora.enabled` is incompatible with `freeze.patterns`/`freeze.except`: "
                    "PEFT handles all freezing of the base backbone — remove the freeze block."
                )
            if self.unfreeze_at_epoch:
                raise ValueError(
                    "`lora.enabled` is incompatible with `freeze.unfreeze_at_epoch`: "
                    "patterns won't match PEFT-prefixed names. Remove the schedule or "
                    "use full fine-tuning instead of LoRA."
                )
            if self.optimizer_param_groups:
                raise ValueError(
                    "`lora.enabled` is incompatible with `optimizer.param_groups`: "
                    "pattern strings target bare backbone names that don't match "
                    "PEFT-wrapped parameter names. Drop the param_groups block, or "
                    "switch off LoRA if you need discriminative learning rates."
                )

        self.periodic_eval = self._params.get("periodic_eval")
