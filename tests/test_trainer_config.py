"""Tests for config.trainer.trainer_config.TrainerConfig parsing."""

from pathlib import Path

import pytest
import yaml

from config.trainer.trainer_config import TrainerConfig


def _minimal_trainer_yaml() -> dict:
    return {
        "optimizer": {"type": "SGD", "params": {"lr": 0.01}},
        "lr_schedule": {"type": "StepLR", "params": {"step_size": 1, "gamma": 0.5}},
        "dataloader": {
            "train": {"batch_size": 4, "shuffle": True, "num_workers": 0},
            "val": {"batch_size": 4, "shuffle": False, "num_workers": 0},
        },
        "num_epochs": 2,
        "device": "cpu",
        "checkpoint": {
            "save": {"dir": "/tmp/x", "frequency": 1},
            "load": {"path": None, "backbone": True, "loss": True, "scheduler": True, "optimizer": True},
        },
    }


@pytest.fixture
def trainer_yaml(tmp_path):
    def _make(extra: dict | None = None) -> Path:
        data = _minimal_trainer_yaml()
        if extra:
            data.update(extra)
        p = tmp_path / "trainer.yaml"
        p.write_text(yaml.safe_dump(data))
        return p

    return _make


class TestRequiredFields:
    def test_minimal_parses(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.optimizer_type == "SGD"
        assert cfg.optimizer_params == {"lr": 0.01}
        assert cfg.lr_schedule_type == "StepLR"
        assert cfg.train_batch_size == 4
        assert cfg.val_shuffle is False
        assert cfg.num_epochs == 2
        assert cfg.device == "cpu"
        assert cfg.checkpoint_load_path is None


class TestOptionalBlocks:
    def test_amp_defaults(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.amp_enabled is False
        assert cfg.amp_dtype == "float16"

    def test_amp_explicit(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml({"amp": {"enabled": True, "dtype": "bfloat16"}})))
        assert cfg.amp_enabled is True
        assert cfg.amp_dtype == "bfloat16"

    def test_seed_default_none(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.seed is None
        assert cfg.deterministic is False

    def test_seed_explicit(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml({"seed": 7, "deterministic": True})))
        assert cfg.seed == 7
        assert cfg.deterministic is True

    def test_gradient_clip_defaults(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.grad_clip_max_norm is None
        assert cfg.grad_clip_norm_type == 2.0

    def test_gradient_clip_norm_type_null_falls_back_to_default(self, trainer_yaml):
        """B6 regression: explicit `norm_type: null` must not crash on float(None)."""
        cfg = TrainerConfig(str(trainer_yaml({"gradient_clip": {"max_norm": 1.0, "norm_type": None}})))
        assert cfg.grad_clip_max_norm == 1.0
        assert cfg.grad_clip_norm_type == 2.0  # defaulted, not crashed

    def test_tensorboard_defaults(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.tensorboard_enabled is False
        assert cfg.tensorboard_log_dir is None

    def test_tensorboard_explicit(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml({"logging": {"tensorboard": True, "log_dir": "/tmp/tb"}})))
        assert cfg.tensorboard_enabled is True
        assert cfg.tensorboard_log_dir == "/tmp/tb"

    def test_periodic_eval_default_none(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.periodic_eval is None

    def test_periodic_eval_pass_through(self, trainer_yaml):
        block = {"enabled": True, "every_n_epochs": 5, "dataset": "lfw_pairs"}
        cfg = TrainerConfig(str(trainer_yaml({"periodic_eval": block})))
        assert cfg.periodic_eval == block

    def test_onnx_export_defaults(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        # Opt-in: missing block must NOT enable export, and existing trainer
        # YAMLs (without the block) must keep loading.
        assert cfg.onnx_export_enabled is False
        assert cfg.onnx_export_opset == 17
        assert cfg.onnx_export_dynamic_batch is True

    def test_onnx_export_explicit(self, trainer_yaml):
        cfg = TrainerConfig(
            str(
                trainer_yaml(
                    {"onnx_export": {"enabled": True, "opset": 14, "dynamic_batch": False}}
                )
            )
        )
        assert cfg.onnx_export_enabled is True
        assert cfg.onnx_export_opset == 14
        assert cfg.onnx_export_dynamic_batch is False

    def test_lora_defaults(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        # Opt-in: missing block stays disabled and existing trainer YAMLs
        # without `lora` keep loading.
        assert cfg.lora_enabled is False
        assert cfg.lora_rank == 8
        assert cfg.lora_alpha == 16.0
        assert cfg.lora_dropout == 0.0
        assert cfg.lora_target_modules == []

    def test_lora_explicit(self, trainer_yaml):
        cfg = TrainerConfig(
            str(
                trainer_yaml(
                    {
                        "lora": {
                            "enabled": True,
                            "rank": 4,
                            "alpha": 8.0,
                            "dropout": 0.1,
                            "target_modules": ["last_linear", "block8.branch0.conv"],
                        }
                    }
                )
            )
        )
        assert cfg.lora_enabled is True
        assert cfg.lora_rank == 4
        assert cfg.lora_alpha == 8.0
        assert cfg.lora_dropout == 0.1
        assert cfg.lora_target_modules == ["last_linear", "block8.branch0.conv"]
        # `modules_to_save` defaults to an empty list (PEFT escape hatch off).
        assert cfg.lora_modules_to_save == []

    def test_lora_modules_to_save(self, trainer_yaml):
        cfg = TrainerConfig(
            str(
                trainer_yaml(
                    {
                        "lora": {
                            "enabled": True,
                            "target_modules": ["qkv"],
                            "modules_to_save": ["feature", "norm"],
                        }
                    }
                )
            )
        )
        assert cfg.lora_modules_to_save == ["feature", "norm"]

    def test_lora_enabled_requires_target_modules(self, trainer_yaml):
        # Catching the typo at config-load time prevents a confusing
        # "no parameters found" error deeper in PEFT.
        with pytest.raises(ValueError, match="target_modules"):
            TrainerConfig(str(trainer_yaml({"lora": {"enabled": True}})))


class TestFreeze:
    def test_no_freeze_block(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.freeze_patterns is None
        assert cfg.freeze_except is None
        assert cfg.unfreeze_at_epoch == {}

    def test_freeze_patterns(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml({"freeze": {"patterns": ["features.0.*"]}})))
        assert cfg.freeze_patterns == ["features.0.*"]
        assert cfg.freeze_except is None

    def test_freeze_except(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml({"freeze": {"except": ["last_linear*"]}})))
        assert cfg.freeze_except == ["last_linear*"]

    def test_freeze_both_raises(self, trainer_yaml):
        with pytest.raises(ValueError, match="patterns.*except"):
            TrainerConfig(str(trainer_yaml({"freeze": {"patterns": ["a"], "except": ["b"]}})))

    def test_unfreeze_schedule_keys_normalized_to_int(self, trainer_yaml):
        cfg = TrainerConfig(
            str(trainer_yaml({"freeze": {"unfreeze_at_epoch": {3: ["features.*"], 5: ["all*"]}}}))
        )
        assert cfg.unfreeze_at_epoch == {3: ["features.*"], 5: ["all*"]}


class TestOptimizerParamGroups:
    def test_no_param_groups_default(self, trainer_yaml):
        cfg = TrainerConfig(str(trainer_yaml()))
        assert cfg.optimizer_param_groups is None

    def test_param_groups_passed_through(self, trainer_yaml):
        groups = [{"pattern": "loss.*", "lr": 1e-3}]
        cfg = TrainerConfig(
            str(trainer_yaml({"optimizer": {"type": "SGD", "params": {"lr": 0.01}, "param_groups": groups}}))
        )
        assert cfg.optimizer_param_groups == groups


class TestLoRAConflicts:
    """Regression: LoRA's PEFT wrap renames every parameter and freezes
    the base, so `freeze.*`, `freeze.unfreeze_at_epoch`, and
    `optimizer.param_groups` either silently no-op or get overwritten.
    The config layer rejects these combos up front (rather than letting
    them produce confusing runtime behavior)."""

    _LORA = {
        "enabled": True,
        "target_modules": ["last_linear"],
    }

    def test_lora_plus_freeze_patterns_raises(self, trainer_yaml):
        with pytest.raises(ValueError, match="freeze"):
            TrainerConfig(
                str(trainer_yaml({"lora": self._LORA, "freeze": {"patterns": ["features.*"]}}))
            )

    def test_lora_plus_freeze_except_raises(self, trainer_yaml):
        with pytest.raises(ValueError, match="freeze"):
            TrainerConfig(
                str(trainer_yaml({"lora": self._LORA, "freeze": {"except": ["last_linear*"]}}))
            )

    def test_lora_plus_unfreeze_schedule_raises(self, trainer_yaml):
        with pytest.raises(ValueError, match="unfreeze_at_epoch"):
            TrainerConfig(
                str(
                    trainer_yaml(
                        {"lora": self._LORA, "freeze": {"unfreeze_at_epoch": {3: ["features.*"]}}}
                    )
                )
            )

    def test_lora_plus_param_groups_raises(self, trainer_yaml):
        with pytest.raises(ValueError, match="param_groups"):
            TrainerConfig(
                str(
                    trainer_yaml(
                        {
                            "lora": self._LORA,
                            "optimizer": {
                                "type": "SGD",
                                "params": {"lr": 0.01},
                                "param_groups": [{"pattern": "loss.*", "lr": 1e-3}],
                            },
                        }
                    )
                )
            )

    def test_lora_without_conflicts_passes(self, trainer_yaml):
        # The existing canonical lora_finetune.yaml shape: no freeze, no
        # param_groups, just LoRA. Must still parse cleanly.
        cfg = TrainerConfig(str(trainer_yaml({"lora": self._LORA})))
        assert cfg.lora_enabled is True
