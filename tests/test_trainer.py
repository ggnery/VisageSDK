"""Trainer integration tests, including the B1 resume-replay regression.

These run a tiny training loop on dummy data via the registry-driven
TrainerBuilder. They serve as both end-to-end smoke tests and regression
guards for the higher-impact bug fixes (B1 replay, B6 norm_type:null,
B8 zero-sample guard, etc.).
"""

from pathlib import Path

import pytest
import yaml


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def tiny_train_setup(tmp_imagefolder, tmp_path, populated_registries):
    """Create a complete YAML+ENV setup pointing at tmp_imagefolder.

    Returns a dict of env vars suitable for ENVConfig.from_env (or for
    direct instantiation of ENVConfig).
    """
    cfg_dir = tmp_path / "configs"
    ckpt_dir = tmp_path / "ckpt"

    _write_yaml(
        cfg_dir / "backbone.yaml",
        {
            # InceptionResNetV1 has fixed layer sizes targeting 160×160 input;
            # the transformation pipeline upsamples the small fixture images.
            "input_size": [160, 160],
            "embedding_size": 16,
            "device": "cpu",
            "dropout_keep": 0.8,
        },
    )
    _write_yaml(
        cfg_dir / "loss.yaml",
        {
            "device": "cpu",
            "label_smoothing": 0.0,
            "use_bias": True,
        },
    )
    _write_yaml(
        cfg_dir / "dataset.yaml",
        {
            "train_dir": str(tmp_imagefolder / "train"),
            "val_dir": str(tmp_imagefolder / "val"),
            "num_classes": 3,
        },
    )
    _write_yaml(
        cfg_dir / "tx.yaml",
        {
            "train": {
                "normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
                "random_horizontal_flip": 0.0,
            },
            "val": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
        },
    )
    _write_yaml(
        cfg_dir / "trainer.yaml",
        {
            "optimizer": {"type": "SGD", "params": {"lr": 0.01, "momentum": 0.9}},
            "lr_schedule": {"type": "StepLR", "params": {"step_size": 1, "gamma": 0.95}},
            "dataloader": {
                "train": {"batch_size": 4, "shuffle": True, "num_workers": 0},
                "val": {"batch_size": 4, "shuffle": False, "num_workers": 0},
            },
            "num_epochs": 1,
            "device": "cpu",
            "checkpoint": {
                "save": {"dir": str(ckpt_dir), "frequency": 1},
                "load": {"path": None, "backbone": True, "loss": True, "scheduler": True, "optimizer": True},
            },
            "seed": 42,
            "gradient_clip": {"max_norm": 1.0, "norm_type": None},  # B6 regression
        },
    )

    return {
        "BACKBONE": "inception_resnet_v1",
        "BACKBONE_CONFIG": str(cfg_dir / "backbone.yaml"),
        "TRAIN_VAL_DATASET": "image_folder",
        "TRAIN_VAL_DATASET_CONFIG": str(cfg_dir / "dataset.yaml"),
        "TRAIN_TRANSFORMATION": "casia_webface_train",
        "VAL_TRANSFORMATION": "casia_webface_val",
        "TRAIN_VAL_TRANSFORMATION_CONFIG": str(cfg_dir / "tx.yaml"),
        "LOSS": "cross_entropy",
        "LOSS_CONFIG": str(cfg_dir / "loss.yaml"),
        "TRAINER_CONFIG": str(cfg_dir / "trainer.yaml"),
        "_ckpt_dir": str(ckpt_dir),
        "_cfg_dir": str(cfg_dir),
    }


def _build_env(env_dict):
    """Apply env_dict to os.environ for the duration of a test using monkeypatch
    is cleaner, but since ENVConfig.from_env reads via os.getenv we can also
    instantiate ENVConfig directly to avoid global state."""
    from config.env_config import ENVConfig

    return ENVConfig(
        backbone=env_dict["BACKBONE"],
        backbone_config=env_dict["BACKBONE_CONFIG"],
        train_val_dataset=env_dict["TRAIN_VAL_DATASET"],
        train_val_dataset_config=env_dict["TRAIN_VAL_DATASET_CONFIG"],
        train_transformation=env_dict["TRAIN_TRANSFORMATION"],
        val_transformation=env_dict["VAL_TRANSFORMATION"],
        train_val_transformation_config=env_dict["TRAIN_VAL_TRANSFORMATION_CONFIG"],
        loss=env_dict["LOSS"],
        loss_config=env_dict["LOSS_CONFIG"],
        trainer_config=env_dict["TRAINER_CONFIG"],
    )


# =============================================================================
# End-to-end smoke
# =============================================================================


class TestTrainerSmoke:
    def test_one_epoch_run_completes(self, tiny_train_setup):
        from tools.trainer_builder import TrainerBuilder

        env = _build_env(tiny_train_setup)
        builder = TrainerBuilder(env)
        trainer = builder.build_trainer()
        trainer.train()

        ckpt_dir = Path(tiny_train_setup["_ckpt_dir"])
        # At least the per-epoch + best checkpoints + the history JSON
        assert any(ckpt_dir.glob("*.pth"))
        assert any(ckpt_dir.glob("*_training_history.json"))


# =============================================================================
# B1 — Resume + replay unfreeze
# =============================================================================


class TestResumeReplay:
    def test_replay_unfreezes_on_resume(self, tiny_train_setup, tmp_path):
        """B1: after `load_checkpoint`, every unfreeze whose epoch <= ckpt epoch
        must be re-applied so the trainable set matches what was running
        when the checkpoint was saved.
        """
        from tools.freezer import freeze_summary
        from tools.trainer_builder import TrainerBuilder

        # First: train 2 epochs with an unfreeze at epoch 2.
        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        trainer_yaml_path = cfg_dir / "trainer.yaml"
        first_yaml = yaml.safe_load(trainer_yaml_path.read_text())
        first_yaml["num_epochs"] = 2
        first_yaml["freeze"] = {
            "except": ["last_linear*", "last_bn*"],
            "unfreeze_at_epoch": {2: ["block8*"]},
        }
        trainer_yaml_path.write_text(yaml.safe_dump(first_yaml))

        env = _build_env(tiny_train_setup)
        builder = TrainerBuilder(env)
        trainer = builder.build_trainer()
        trainer.train()
        trainable_after_2_epochs, total = freeze_summary(builder.backbone)

        # Find the saved checkpoint
        ckpt_dir = Path(tiny_train_setup["_ckpt_dir"])
        epoch2_ckpts = list(ckpt_dir.glob("*epoch_2.pth"))
        assert epoch2_ckpts, "expected epoch_2.pth to be saved"
        ckpt_path = str(epoch2_ckpts[0])

        # Now: reuse the SAME YAML (with the unfreeze schedule), but load the
        # checkpoint and continue. The replay must restore the freeze state.
        resume_yaml = yaml.safe_load(trainer_yaml_path.read_text())
        resume_yaml["num_epochs"] = 3  # one more epoch to make the loop body reachable
        resume_yaml["checkpoint"]["save"]["dir"] = str(tmp_path / "ckpt2")
        resume_yaml["checkpoint"]["load"] = {
            "path": ckpt_path,
            "backbone": True,
            "loss": True,
            "scheduler": False,
            "optimizer": False,
        }
        trainer_yaml_path.write_text(yaml.safe_dump(resume_yaml))

        env = _build_env(tiny_train_setup)
        builder2 = TrainerBuilder(env)
        # build_trainer triggers Trainer.__init__ → load_checkpoint → replay
        builder2.build_trainer()

        trainable_after_resume, _ = freeze_summary(builder2.backbone)
        # Same freeze state as the original run after epoch 2
        assert trainable_after_resume == trainable_after_2_epochs, (
            "B1 regression: resume failed to replay past unfreeze events; "
            f"trainable={trainable_after_resume} vs expected={trainable_after_2_epochs}"
        )


# =============================================================================
# B6 — gradient_clip.norm_type: null does not crash
# =============================================================================


class TestGradientClipNullNormType:
    def test_run_with_norm_type_null(self, tiny_train_setup):
        """tiny_train_setup already sets gradient_clip.norm_type: None — the
        TrainerConfig must coerce to default (2.0) without raising TypeError.
        """
        from tools.trainer_builder import TrainerBuilder

        env = _build_env(tiny_train_setup)
        builder = TrainerBuilder(env)
        assert builder.trainer_config.grad_clip_max_norm == 1.0
        assert builder.trainer_config.grad_clip_norm_type == 2.0
        # Build_trainer + 1 epoch must not raise
        trainer = builder.build_trainer()
        trainer.train()


# =============================================================================
# ONNX export alongside .pth checkpoints
# =============================================================================


def _enable_onnx_export(cfg_dir: Path, **overrides) -> None:
    """Patch the trainer YAML in-place to enable onnx_export."""
    trainer_yaml_path = cfg_dir / "trainer.yaml"
    data = yaml.safe_load(trainer_yaml_path.read_text())
    data["onnx_export"] = {"enabled": True, **overrides}
    trainer_yaml_path.write_text(yaml.safe_dump(data))


class TestOnnxExport:
    def test_disabled_by_default(self, tiny_train_setup):
        """When the trainer YAML omits onnx_export, save_checkpoint writes
        only the .pth — no .onnx artifact appears."""
        from tools.trainer_builder import TrainerBuilder

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        trainer.save_checkpoint(0.0, 0.0, "ckpt.pth")

        ckpt_dir = Path(tiny_train_setup["_ckpt_dir"])
        assert (ckpt_dir / "ckpt.pth").exists()
        assert not (ckpt_dir / "ckpt.onnx").exists()

    def test_enabled_writes_valid_onnx(self, tiny_train_setup):
        """onnx_export.enabled produces a graph that passes onnx.checker
        with the expected named input/output."""
        import onnx

        from tools.trainer_builder import TrainerBuilder

        _enable_onnx_export(Path(tiny_train_setup["_cfg_dir"]))

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        trainer.save_checkpoint(0.0, 0.0, "ckpt.pth")

        onnx_path = Path(tiny_train_setup["_ckpt_dir"]) / "ckpt.onnx"
        assert onnx_path.exists()

        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        assert [i.name for i in model.graph.input] == ["input"]
        assert [o.name for o in model.graph.output] == ["embedding"]

    def test_dynamic_batch_dim_is_symbolic(self, tiny_train_setup):
        """dynamic_batch=true: dim 0 of input/output is a symbolic 'batch'
        param, dims 1..3 are concrete (3, H, W from backbone.input_size)."""
        import onnx

        from tools.trainer_builder import TrainerBuilder

        _enable_onnx_export(Path(tiny_train_setup["_cfg_dir"]), dynamic_batch=True)

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        trainer.save_checkpoint(0.0, 0.0, "ckpt.pth")

        model = onnx.load(str(Path(tiny_train_setup["_ckpt_dir"]) / "ckpt.onnx"))
        in_dims = model.graph.input[0].type.tensor_type.shape.dim
        out_dims = model.graph.output[0].type.tensor_type.shape.dim

        assert in_dims[0].dim_param == "batch"
        assert (in_dims[1].dim_value, in_dims[2].dim_value, in_dims[3].dim_value) == (3, 160, 160)
        assert out_dims[0].dim_param == "batch"
        assert out_dims[1].dim_value == 16  # embedding_size from tiny_train_setup

    def test_static_batch_dim_is_concrete(self, tiny_train_setup):
        """dynamic_batch=false: batch dim is fixed (= dummy tensor batch=1)."""
        import onnx

        from tools.trainer_builder import TrainerBuilder

        _enable_onnx_export(Path(tiny_train_setup["_cfg_dir"]), dynamic_batch=False)

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        trainer.save_checkpoint(0.0, 0.0, "ckpt.pth")

        model = onnx.load(str(Path(tiny_train_setup["_ckpt_dir"]) / "ckpt.onnx"))
        batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
        assert batch_dim.dim_value == 1
        assert batch_dim.dim_param == ""  # no symbolic name

    def test_backbone_input_size_propagated(self, tiny_train_setup):
        """Regression: BaseBackbone must store input_size for the dummy
        tensor in _maybe_export_onnx — without it the export crashes."""
        from tools.trainer_builder import TrainerBuilder

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        assert trainer.backbone.input_size == [160, 160]


# =============================================================================
# LoRA / PEFT integration
# =============================================================================


def _enable_lora(cfg_dir: Path, **overrides) -> None:
    """Patch the trainer YAML in-place to enable LoRA on `last_linear`."""
    trainer_yaml_path = cfg_dir / "trainer.yaml"
    data = yaml.safe_load(trainer_yaml_path.read_text())
    data["lora"] = {
        "enabled": True,
        "rank": 4,
        "alpha": 8.0,
        "target_modules": ["last_linear"],
        **overrides,
    }
    trainer_yaml_path.write_text(yaml.safe_dump(data))


class TestLoRAIntegration:
    def test_disabled_by_default(self, tiny_train_setup):
        """No `lora` block → backbone stays a plain BaseBackbone subclass,
        not a PeftModel. Existing trainer YAMLs keep working unchanged."""
        from peft.peft_model import PeftModel

        from tools.trainer_builder import TrainerBuilder

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        assert not isinstance(trainer.backbone, PeftModel)

    def test_enabled_wraps_backbone_and_freezes_base(self, tiny_train_setup):
        """LoRA wraps the backbone in a PeftModel and only the lora_A /
        lora_B params on `last_linear` end up trainable. The base
        InceptionResNetV1 weights are frozen by PEFT."""
        from peft.peft_model import PeftModel

        from tools.trainer_builder import TrainerBuilder

        _enable_lora(Path(tiny_train_setup["_cfg_dir"]))

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        assert isinstance(trainer.backbone, PeftModel)

        trainable = [n for n, p in trainer.backbone.named_parameters() if p.requires_grad]
        # Every trainable backbone tensor must be a LoRA tensor (PEFT names
        # them `*.lora_A.<adapter>.weight` etc).
        assert trainable, "no trainable LoRA params found"
        assert all("lora_" in n for n in trainable), f"non-LoRA trainable params: {trainable}"

        # Forward should still emit (B, embedding_size) since PeftModel
        # proxies attribute access to the wrapped backbone.
        import torch

        trainer.backbone.eval()
        x = torch.randn(2, 3, 160, 160)
        with torch.no_grad():
            emb = trainer.backbone(x)
        assert emb.shape == (2, 16)  # tiny_train_setup uses embedding_size=16

    def test_optimizer_rebuilt_with_lora_params(self, tiny_train_setup):
        """The optimizer was created BEFORE LoRA wrapping (in trainer_builder).
        After Trainer.__init__ wraps the backbone, the optimizer must be
        rebuilt so the new lora_A / lora_B parameters actually receive
        gradient updates — otherwise training would silently no-op."""
        from tools.trainer_builder import TrainerBuilder

        _enable_lora(Path(tiny_train_setup["_cfg_dir"]))

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()

        opt_params: set[int] = set()
        for group in trainer.optimizer.param_groups:
            for p in group["params"]:
                opt_params.add(id(p))

        lora_params = [p for n, p in trainer.backbone.named_parameters() if "lora_" in n]
        assert lora_params, "expected LoRA params on the wrapped backbone"
        assert all(id(p) in opt_params for p in lora_params), (
            "LoRA parameters missing from optimizer — _maybe_apply_lora forgot to rebuild it"
        )
