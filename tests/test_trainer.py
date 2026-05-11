"""Trainer integration tests, including the B1 resume-replay regression.

These run a tiny training loop on dummy data via the registry-driven
TrainerBuilder. They serve as both end-to-end smoke tests and regression
guards for the higher-impact bug fixes (B1 replay, B6 norm_type:null,
B8 zero-sample guard, etc.).
"""

from pathlib import Path

import pytest
import torch
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
# B-10 — device-mismatch detection across YAMLs
# =============================================================================


class TestDeviceMismatch:
    def test_backbone_cuda_trainer_cpu_raises(self, tiny_train_setup):
        """Pre-fix, mismatching `device` across backbone/loss/trainer YAMLs
        silently produced a "tensors on different devices" runtime error
        deep in the forward pass. Validate up front."""
        from tools.trainer_builder import TrainerBuilder

        # Patch backbone YAML to claim cuda while the trainer stays on cpu.
        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        bb_path = cfg_dir / "backbone.yaml"
        bb = yaml.safe_load(bb_path.read_text())
        bb["device"] = "cuda"
        bb_path.write_text(yaml.safe_dump(bb))

        env = _build_env(tiny_train_setup)
        with pytest.raises(ValueError, match="device mismatch"):
            TrainerBuilder(env)


# =============================================================================
# B-1 — LR scheduler is "ready for next epoch" at checkpoint time
# =============================================================================


class TestSchedulerStateOnSave:
    def test_saved_scheduler_state_is_post_step(self, tiny_train_setup):
        """B-1 regression: scheduler.step() must run BEFORE save_checkpoint
        so that the persisted state matches the LR the next epoch should
        train at. Pre-fix the order was reversed, and resuming silently
        trained one schedule-step behind."""
        from tools.trainer_builder import TrainerBuilder

        # Force the LR to actually decay after epoch 1 so the test catches
        # a missing step (StepLR with step_size=1, gamma=0.5 → LR halves).
        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        trainer_yaml_path = cfg_dir / "trainer.yaml"
        data = yaml.safe_load(trainer_yaml_path.read_text())
        data["num_epochs"] = 1
        # StepLR with step_size=1, gamma=0.5: starts at 0.01 → after first
        # step the LR is 0.005.
        trainer_yaml_path.write_text(yaml.safe_dump(data))

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        initial_lr = float(trainer.scheduler.get_last_lr()[0])
        trainer.train()

        # After 1 epoch (and the in-loop scheduler.step()), the LR must
        # have moved off the initial value: post-fix the step runs before
        # save_checkpoint, so the trainer state already reflects the new
        # LR by the time `train()` returns.
        post_train_lr = float(trainer.scheduler.get_last_lr()[0])
        assert post_train_lr != initial_lr, (
            f"scheduler didn't step (initial={initial_lr}, after={post_train_lr})"
        )

        # And the saved checkpoint's scheduler state must reflect that
        # post-step LR, not the pre-step value.
        ckpt_dir = Path(tiny_train_setup["_ckpt_dir"])
        ckpts = sorted(ckpt_dir.glob("*epoch_1.pth"))
        assert ckpts, "expected epoch_1 checkpoint"
        saved = torch.load(ckpts[0], map_location="cpu", weights_only=False)
        opt_sd = saved["optimizer_state_dict"]
        saved_lr = opt_sd["param_groups"][0]["lr"]
        assert saved_lr == pytest.approx(post_train_lr), (
            f"saved optimizer LR {saved_lr} != post-step LR {post_train_lr}; "
            "B-1 regression: scheduler.step() did NOT run before save_checkpoint."
        )


# =============================================================================
# B-8 — FacenetBatchSampler seed propagation
# =============================================================================


class TestSamplerSeedPropagation:
    def test_trainer_seed_propagates_to_sampler_config(self, tiny_train_setup):
        """B-8 regression: trainer_config.seed must be injected into the
        sampler config so the sampler's per-instance RNG is deterministic
        across runs. Without this, the FacenetBatchSampler seeds itself
        from `random.random()` and the per-epoch identity order shifts
        whenever any code path in the builder touches the global RNG."""
        import yaml as _yaml

        from tools.trainer_builder import TrainerBuilder

        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        sampler_cfg = cfg_dir / "sampler.yaml"
        sampler_cfg.write_text(
            _yaml.safe_dump({"faces_per_identity": 2, "num_identities_per_batch": 2})
        )
        # Patch the trainer YAML to drop the explicit batch_size since the
        # sampler dictates batching.
        trainer_yaml_path = cfg_dir / "trainer.yaml"
        data = _yaml.safe_load(trainer_yaml_path.read_text())
        data["dataloader"]["train"]["batch_size"] = None
        data["dataloader"]["train"]["shuffle"] = None
        data["seed"] = 42
        trainer_yaml_path.write_text(_yaml.safe_dump(data))

        env_dict = {**tiny_train_setup, "SAMPLER": "facenet", "SAMPLER_CONFIG": str(sampler_cfg)}
        from config.env_config import ENVConfig

        env = ENVConfig(
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
            sampler=env_dict["SAMPLER"],
            sampler_config=env_dict["SAMPLER_CONFIG"],
        )
        builder = TrainerBuilder(env)
        # After build, the sampler config should carry the injected seed.
        assert builder.sampler_config is not None
        assert builder.sampler_config._params.get("seed") == 42


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

    def test_onnx_export_merges_lora_into_base(self, tiny_train_setup):
        """When LoRA is enabled, the exported ONNX must NOT contain any
        lora_A / lora_B initializers — `_maybe_export_onnx` deepcopies +
        `merge_and_unload`s the PEFT model so the resulting graph is the
        plain base, with LoRA contributions baked into base weights.
        Without that, every saved .onnx carries dead PEFT side-paths and
        is slower at inference."""
        import onnx

        from tools.trainer_builder import TrainerBuilder

        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        _enable_lora(cfg_dir)
        # Need ONNX export wired on too, otherwise _maybe_export_onnx skips.
        _enable_onnx_export(cfg_dir)

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        trainer.save_checkpoint(0.0, 0.0, "ckpt.pth")

        onnx_path = Path(tiny_train_setup["_ckpt_dir"]) / "ckpt.onnx"
        assert onnx_path.exists()

        model = onnx.load(str(onnx_path))
        leaked = [init.name for init in model.graph.initializer if "lora_" in init.name]
        assert not leaked, f"LoRA tensors leaked into ONNX initializers: {leaked}"
        node_op_types = {node.op_type for node in model.graph.node}
        # Sanity: a non-trivial graph (i.e. the export actually ran).
        assert "Gemm" in node_op_types or "MatMul" in node_op_types

    def test_resume_preserves_lora_adapter_weights(self, tiny_train_setup, tmp_path):
        """Resume from a LoRA checkpoint must restore the trained LoRA
        weights (not reinit fresh adapters). Pre-fix this silently failed:
        load_checkpoint ran on the bare backbone, the wrapped-source keys
        like `base_model.model.*.lora_A.*` didn't match, strict=False
        dropped them, then `_maybe_apply_lora` reinitialized.

        We force LoRA's `lora_B` to a known non-zero value before saving so
        the post-resume forward differs from a freshly-wrapped LoRA model
        — that lets us assert state actually transferred."""
        from peft.peft_model import PeftModel

        from tools.trainer_builder import TrainerBuilder

        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        _enable_lora(cfg_dir)

        # First run: build, then poke `lora_B` to a non-zero value and save.
        env = _build_env(tiny_train_setup)
        trainer1 = TrainerBuilder(env).build_trainer()
        assert isinstance(trainer1.backbone, PeftModel)
        with torch.no_grad():
            for n, p in trainer1.backbone.named_parameters():
                if ".lora_B." in n:
                    p.fill_(0.1)
        trainer1.save_checkpoint(0.0, 0.0, "ckpt.pth")
        ckpt_path = Path(tiny_train_setup["_ckpt_dir"]) / "ckpt.pth"
        assert ckpt_path.exists()

        # Reference forward from the trained model.
        trainer1.backbone.eval()
        x = torch.randn(2, 3, 160, 160)
        with torch.no_grad():
            expected = trainer1.backbone(x).clone()

        # Second run: point at the saved checkpoint to "resume". Force a
        # fresh checkpoint dir so we don't collide with the first run.
        trainer_yaml_path = cfg_dir / "trainer.yaml"
        data = yaml.safe_load(trainer_yaml_path.read_text())
        data["checkpoint"]["save"]["dir"] = str(tmp_path / "resume_ckpt")
        data["checkpoint"]["load"]["path"] = str(ckpt_path)
        data["checkpoint"]["load"]["backbone"] = True
        data["checkpoint"]["load"]["loss"] = True
        trainer_yaml_path.write_text(yaml.safe_dump(data))

        env2 = _build_env(tiny_train_setup)
        trainer2 = TrainerBuilder(env2).build_trainer()
        assert isinstance(trainer2.backbone, PeftModel)

        # Sanity: lora_B should be 0.1 (the saved value), not 0.0 (fresh init).
        b_tensors = [p for n, p in trainer2.backbone.named_parameters() if ".lora_B." in n]
        assert b_tensors, "expected lora_B tensors after resume"
        for p in b_tensors:
            assert torch.allclose(p, torch.full_like(p, 0.1), atol=1e-6), (
                "LoRA adapter weights were reinitialized on resume — "
                "the wrapped checkpoint detection failed"
            )

        trainer2.backbone.eval()
        with torch.no_grad():
            actual = trainer2.backbone(x)
        assert torch.allclose(expected, actual, atol=1e-6)

    def test_load_checkpoint_warns_on_missing_keys_instead_of_crashing(self, tiny_train_setup, tmp_path):
        """Regression: trainer.load_checkpoint used `checkpoint["X"]` direct
        access. Old/wrapped-pretrained .pths that ship `{}` for optimizer
        / scheduler state crashed with KeyError mid-resume. The fix swaps
        to `.get()` + warning."""
        from tools.trainer_builder import TrainerBuilder

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        trainer.save_checkpoint(0.0, 0.0, "ckpt.pth")
        ckpt_path = Path(tiny_train_setup["_ckpt_dir"]) / "ckpt.pth"

        # Strip optimizer_state_dict and scheduler_state_dict from the saved
        # checkpoint to simulate the wrap_*_pretrained.py format (or a
        # partial save that crashed mid-write).
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        del ckpt["optimizer_state_dict"]
        del ckpt["scheduler_state_dict"]
        torch.save(ckpt, ckpt_path)

        # Build a fresh trainer with both load flags True. Pre-fix this
        # would KeyError out of __init__; post-fix it warns and continues.
        trainer.load_checkpoint(
            ckpt_path,
            load_backbone=True,
            load_loss=False,
            load_scheduler=True,   # missing key — must not crash
            load_optimizer=True,   # missing key — must not crash
        )

    def test_periodic_eval_logging_skips_non_scalar_results(self, tiny_train_setup, tmp_path):
        """Regression: when periodic_eval uses the verification evaluator,
        results contain `roc_curve` and `score_distributions` (dicts).
        `_maybe_run_periodic_eval` used to format every value with `:.6f`,
        crashing with `unsupported format string passed to dict.__format__`
        and bringing down the entire training mid-run."""
        from unittest.mock import MagicMock

        from tools.trainer_builder import TrainerBuilder

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()

        # Stub the periodic evaluator with a verification-style payload.
        fake_evaluator = MagicMock()
        fake_evaluator.evaluate.return_value = {
            "lfw_accuracy_mean": 0.99,
            "eer": 0.05,
            "tar@far=1e-03": 0.85,
            "roc_curve": {"fpr": [0.0, 0.5, 1.0], "tpr": [0.0, 0.95, 1.0]},
            "score_distributions": {"genuine": [0.1, 0.2], "impostor": [0.7, 0.8]},
        }
        trainer.periodic_evaluator = fake_evaluator
        trainer.epoch = 1
        # Run via every_n=1 so the trigger condition fires unconditionally.
        trainer.config._params["periodic_eval"] = {"every_n_epochs": 1}  # type: ignore[attr-defined]
        trainer.config.periodic_eval = {"every_n_epochs": 1}

        # Should NOT raise — non-scalars must be filtered before the format.
        out = trainer._maybe_run_periodic_eval()
        assert out is not None
        # Full dict (scalars + curves) flows back so history JSON keeps it.
        assert "roc_curve" in out and "score_distributions" in out

    def test_peek_is_lora_wrapped_classifies_correctly(self, tiny_train_setup, tmp_path):
        """The wrapped-checkpoint detector must return True for a PEFT-saved
        .pth and False for a bare-backbone .pth, since picking the wrong
        order silently breaks loading either way."""
        from tools.trainer_builder import TrainerBuilder
        from trainer.trainer import Trainer

        # Bare save first.
        env = _build_env(tiny_train_setup)
        trainer_bare = TrainerBuilder(env).build_trainer()
        trainer_bare.save_checkpoint(0.0, 0.0, "bare.pth")
        bare_path = Path(tiny_train_setup["_ckpt_dir"]) / "bare.pth"

        # Wrapped save.
        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        _enable_lora(cfg_dir)
        env2 = _build_env(tiny_train_setup)
        trainer_lora = TrainerBuilder(env2).build_trainer()
        trainer_lora.save_checkpoint(0.0, 0.0, "lora.pth")
        lora_path = Path(tiny_train_setup["_ckpt_dir"]) / "lora.pth"

        assert Trainer._peek_is_lora_wrapped(bare_path) is False
        assert Trainer._peek_is_lora_wrapped(lora_path) is True

    def test_onnx_export_does_not_mutate_training_state(self, tiny_train_setup):
        """The deepcopy + merge_and_unload path must leave the live
        backbone (still being trained) untouched. Otherwise resuming after
        a checkpoint save would silently lose the LoRA structure."""
        from peft.peft_model import PeftModel

        from tools.trainer_builder import TrainerBuilder

        cfg_dir = Path(tiny_train_setup["_cfg_dir"])
        _enable_lora(cfg_dir)
        _enable_onnx_export(cfg_dir)

        env = _build_env(tiny_train_setup)
        trainer = TrainerBuilder(env).build_trainer()
        assert isinstance(trainer.backbone, PeftModel)

        before_lora_param_ids = {
            id(p) for n, p in trainer.backbone.named_parameters() if "lora_" in n
        }
        trainer.save_checkpoint(0.0, 0.0, "ckpt.pth")

        # Same PeftModel instance, same LoRA params, all still trainable.
        assert isinstance(trainer.backbone, PeftModel)
        after_lora_param_ids = {
            id(p) for n, p in trainer.backbone.named_parameters() if "lora_" in n
        }
        assert before_lora_param_ids == after_lora_param_ids
        assert all(
            p.requires_grad
            for n, p in trainer.backbone.named_parameters()
            if "lora_" in n
        )
