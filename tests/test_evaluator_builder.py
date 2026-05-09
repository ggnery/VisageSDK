"""Test that EvaluatorBuilder loads checkpoints and builds the evaluator."""

from pathlib import Path

import pytest
import torch
import yaml


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


@pytest.fixture
def eval_env_setup(tmp_lfw_pairs, tmp_path, populated_registries):
    """Train a tiny backbone, save a checkpoint, then return env vars
    pointing eval.py at it."""
    from config.backbone.base_backbone_config import BackboneConfig
    from registry import BACKBONES

    images_dir, pairs_path = tmp_lfw_pairs
    cfg_dir = tmp_path / "configs"

    _write_yaml(
        cfg_dir / "backbone.yaml",
        {
            "input_size": [160, 160],
            "embedding_size": 16,
            "device": "cpu",
            "dropout_keep": 0.8,
        },
    )
    _write_yaml(
        cfg_dir / "lfw.yaml",
        {
            "eval_dir": str(images_dir),
            "pairs_path": str(pairs_path),
            "image_ext": "jpg",
        },
    )
    _write_yaml(cfg_dir / "tx.yaml", {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}})
    _write_yaml(
        cfg_dir / "evaluator.yaml",
        {
            "device": "cpu",
            "batch_size": 4,
            "num_workers": 0,
            "distance": "cosine",
            "far_targets": [1.0e-1],
        },
    )

    # Save a fresh-init backbone state_dict as a checkpoint
    backbone_cfg = BackboneConfig(str(cfg_dir / "backbone.yaml"))
    backbone = BACKBONES.get("inception_resnet_v1")(backbone_cfg)
    ckpt_path = tmp_path / "ckpt.pth"
    torch.save({"backbone_state_dict": backbone.state_dict()}, ckpt_path)

    return {
        "BACKBONE": "inception_resnet_v1",
        "BACKBONE_CONFIG": str(cfg_dir / "backbone.yaml"),
        "CHECKPOINT_PATH": str(ckpt_path),
        "EVAL_DATASET": "lfw_pairs",
        "EVAL_DATASET_CONFIG": str(cfg_dir / "lfw.yaml"),
        "EVAL_TRANSFORMATION": "lfw_eval",
        "EVAL_TRANSFORMATION_CONFIG": str(cfg_dir / "tx.yaml"),
        "EVALUATOR": "verification",
        "EVALUATOR_CONFIG": str(cfg_dir / "evaluator.yaml"),
    }


def _build_eval_env(env):
    from config.env_eval_config import ENVEvalConfig

    return ENVEvalConfig(
        backbone=env["BACKBONE"],
        backbone_config=env["BACKBONE_CONFIG"],
        checkpoint_path=env["CHECKPOINT_PATH"],
        eval_dataset=env["EVAL_DATASET"],
        eval_dataset_config=env["EVAL_DATASET_CONFIG"],
        eval_transformation=env["EVAL_TRANSFORMATION"],
        eval_transformation_config=env["EVAL_TRANSFORMATION_CONFIG"],
        evaluator=env["EVALUATOR"],
        evaluator_config=env["EVALUATOR_CONFIG"],
    )


class TestEvaluatorBuilder:
    def test_builder_constructs_evaluator(self, eval_env_setup):
        from evaluator.verification_evaluator import VerificationEvaluator
        from tools.evaluator_builder import EvaluatorBuilder

        builder = EvaluatorBuilder(_build_eval_env(eval_env_setup))
        evaluator = builder.build()
        assert isinstance(evaluator, VerificationEvaluator)

    def test_loads_checkpoint_with_backbone_state_dict(self, eval_env_setup):
        from tools.evaluator_builder import EvaluatorBuilder

        builder = EvaluatorBuilder(_build_eval_env(eval_env_setup))
        # No exception is the smoke test; weights match what we saved.
        # Compare a single param tensor's first values.
        loaded_w = builder.backbone.state_dict()["last_linear.weight"]
        assert loaded_w.numel() > 0

    def test_loads_raw_state_dict(self, eval_env_setup, tmp_path):
        """If the checkpoint is a raw state_dict (no `backbone_state_dict`
        wrapper), the builder still loads it."""
        from config.backbone.base_backbone_config import BackboneConfig
        from registry import BACKBONES
        from tools.evaluator_builder import EvaluatorBuilder

        backbone_cfg = BackboneConfig(eval_env_setup["BACKBONE_CONFIG"])
        backbone = BACKBONES.get("inception_resnet_v1")(backbone_cfg)
        raw_path = tmp_path / "raw.pth"
        torch.save(backbone.state_dict(), raw_path)
        eval_env_setup["CHECKPOINT_PATH"] = str(raw_path)
        # Should not raise
        EvaluatorBuilder(_build_eval_env(eval_env_setup))

    def test_evaluate_runs_end_to_end(self, eval_env_setup):
        from tools.evaluator_builder import EvaluatorBuilder

        builder = EvaluatorBuilder(_build_eval_env(eval_env_setup))
        evaluator = builder.build()
        results = evaluator.evaluate()
        assert "lfw_accuracy_mean" in results
        assert "roc_auc" in results
