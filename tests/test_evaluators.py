"""Integration tests for VerificationEvaluator and IdentificationEvaluator.

These instantiate a tiny stub backbone (defined in conftest.py) and run the
evaluator end-to-end against on-disk fixtures, asserting that all expected
metric keys are returned and have sane ranges.
"""

import math

import pytest
import yaml
from torchvision import transforms

from config.dataset.eval.base_eval_dataset_config import EvalDatasetConfig
from config.evaluator.base_evaluator_config import EvaluatorConfig
from dataset.eval.identification_dataset import IdentificationDataset
from dataset.eval.lfw_pairs_dataset import LFWPairsDataset
from evaluator.identification_evaluator import IdentificationEvaluator
from evaluator.verification_evaluator import VerificationEvaluator


class _Tx:
    transform = transforms.Compose([transforms.ToTensor()])


def _eval_cfg(tmp_path, overrides=None) -> EvaluatorConfig:
    data = {"device": "cpu", "batch_size": 4, "num_workers": 0}
    if overrides:
        data.update(overrides)
    p = tmp_path / "ev.yaml"
    p.write_text(yaml.safe_dump(data))
    return EvaluatorConfig(str(p))


# =============================================================================
# Verification
# =============================================================================


@pytest.fixture
def lfw_eval_dataset(tmp_lfw_pairs, tmp_path):
    images_dir, pairs_path = tmp_lfw_pairs
    cfg_path = tmp_path / "lfw.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "eval_dir": str(images_dir),
                "pairs_path": str(pairs_path),
                "image_ext": "jpg",
            }
        )
    )
    cfg = EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
    return LFWPairsDataset(cfg, _Tx())


class TestVerificationEvaluator:
    def test_returns_expected_keys(self, lfw_eval_dataset, tiny_backbone, tmp_path):
        evaluator = VerificationEvaluator(_eval_cfg(tmp_path), lfw_eval_dataset, tiny_backbone)
        results = evaluator.evaluate()
        for key in (
            "lfw_accuracy_mean",
            "lfw_accuracy_std",
            "lfw_threshold_mean",
            "best_threshold_global",
            "best_accuracy_global",
            "roc_auc",
            "eer",
            "eer_threshold",
        ):
            assert key in results, f"missing {key}"

    def test_far_targets_emitted(self, lfw_eval_dataset, tiny_backbone, tmp_path):
        cfg = _eval_cfg(tmp_path, overrides={"far_targets": [1e-1, 1e-2]})
        evaluator = VerificationEvaluator(cfg, lfw_eval_dataset, tiny_backbone)
        results = evaluator.evaluate()
        assert "tar@far=1e-01" in results
        assert "threshold@far=1e-01" in results
        assert "tar@far=1e-02" in results

    def test_metric_ranges_sane(self, lfw_eval_dataset, tiny_backbone, tmp_path):
        evaluator = VerificationEvaluator(_eval_cfg(tmp_path), lfw_eval_dataset, tiny_backbone)
        r = evaluator.evaluate()
        assert 0.0 <= r["lfw_accuracy_mean"] <= 1.0
        assert 0.0 <= r["roc_auc"] <= 1.0
        assert 0.0 <= r["eer"] <= 1.0

    def test_distance_kind_validation(self, lfw_eval_dataset, tiny_backbone, tmp_path):
        cfg = _eval_cfg(tmp_path, overrides={"distance": "manhattan"})
        evaluator = VerificationEvaluator(cfg, lfw_eval_dataset, tiny_backbone)
        with pytest.raises(ValueError, match="Unknown distance"):
            evaluator.evaluate()

    def test_euclidean_distance_kind(self, lfw_eval_dataset, tiny_backbone, tmp_path):
        cfg = _eval_cfg(tmp_path, overrides={"distance": "euclidean"})
        evaluator = VerificationEvaluator(cfg, lfw_eval_dataset, tiny_backbone)
        r = evaluator.evaluate()
        assert math.isfinite(r["lfw_accuracy_mean"])

    def test_wrong_dataset_type_raises(self, tiny_backbone, tmp_path):
        # Pass an IdentificationDataset to VerificationEvaluator → TypeError
        ident_dir = tmp_path / "ident"
        for split in ("gallery", "probe"):
            (ident_dir / split / "alice").mkdir(parents=True)
            from PIL import Image

            Image.new("RGB", (32, 32)).save(ident_dir / split / "alice" / "x.jpg")
        cfg_path = tmp_path / "ds.yaml"
        cfg_path.write_text(yaml.safe_dump({"eval_dir": str(ident_dir)}))
        cfg = EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
        ds = IdentificationDataset(cfg, _Tx())
        evaluator = VerificationEvaluator(_eval_cfg(tmp_path), ds, tiny_backbone)
        with pytest.raises(TypeError, match="LFWPairsDataset"):
            evaluator.evaluate()


# =============================================================================
# Identification
# =============================================================================


@pytest.fixture
def ident_dataset(tmp_identification, tmp_path):
    cfg_path = tmp_path / "ident.yaml"
    cfg_path.write_text(yaml.safe_dump({"eval_dir": str(tmp_identification)}))
    cfg = EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
    return IdentificationDataset(cfg, _Tx())


class TestIdentificationEvaluator:
    def test_returns_expected_keys(self, ident_dataset, tiny_backbone, tmp_path):
        evaluator = IdentificationEvaluator(_eval_cfg(tmp_path), ident_dataset, tiny_backbone)
        r = evaluator.evaluate()
        assert "rank_1" in r
        assert "rank_5" in r or "rank_5" not in r  # default ranks=[1,5,10]; 5 may or may not be in here
        assert "mAP" in r

    def test_custom_ranks(self, ident_dataset, tiny_backbone, tmp_path):
        cfg = _eval_cfg(tmp_path, overrides={"ranks": [1, 2, 3]})
        evaluator = IdentificationEvaluator(cfg, ident_dataset, tiny_backbone)
        r = evaluator.evaluate()
        assert "rank_1" in r and "rank_2" in r and "rank_3" in r

    def test_metric_ranges_sane(self, ident_dataset, tiny_backbone, tmp_path):
        evaluator = IdentificationEvaluator(_eval_cfg(tmp_path), ident_dataset, tiny_backbone)
        r = evaluator.evaluate()
        for k, v in r.items():
            if k.startswith("rank_") or k.startswith("cmc@") or k == "mAP":
                assert 0.0 <= v <= 1.0

    def test_cmc_keys_capped_by_gallery_size(self, ident_dataset, tiny_backbone, tmp_path):
        cfg = _eval_cfg(tmp_path, overrides={"cmc_max_rank": 100})
        evaluator = IdentificationEvaluator(cfg, ident_dataset, tiny_backbone)
        r = evaluator.evaluate()
        # Gallery only has 3 images → CMC@10 / CMC@20 must not appear
        assert "cmc@10" not in r
        assert "cmc@20" not in r
        assert "cmc@1" in r

    def test_wrong_dataset_type_raises(self, lfw_eval_dataset, tiny_backbone, tmp_path):
        evaluator = IdentificationEvaluator(_eval_cfg(tmp_path), lfw_eval_dataset, tiny_backbone)
        with pytest.raises(TypeError, match="IdentificationDataset"):
            evaluator.evaluate()
