"""Tests for IdentificationDataset."""

import pytest
import yaml

from config.eval_dataset_config import EvalDatasetConfig
from dataset.eval.identification_dataset import IdentificationDataset
from tests.conftest import _PassthroughTransformation


@pytest.fixture
def ident_config(tmp_identification, tmp_path):
    cfg_path = tmp_path / "ident.yaml"
    cfg_path.write_text(yaml.safe_dump({"eval_dir": str(tmp_identification)}))
    return EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})


class TestIdentificationDataset:
    def test_role_distribution(self, ident_config):
        ds = IdentificationDataset(ident_config, _PassthroughTransformation())
        n_gallery = sum(1 for r in ds.roles if r == "gallery")
        n_probe = sum(1 for r in ds.roles if r == "probe")
        # 3 classes × 1 gallery + 3 classes × 2 probes
        assert n_gallery == 3
        assert n_probe == 6

    def test_gallery_appears_before_probe(self, ident_config):
        ds = IdentificationDataset(ident_config, _PassthroughTransformation())
        last_gallery_idx = max(i for i, r in enumerate(ds.roles) if r == "gallery")
        first_probe_idx = min(i for i, r in enumerate(ds.roles) if r == "probe")
        assert last_gallery_idx < first_probe_idx

    def test_total_data_matches_roles(self, ident_config):
        ds = IdentificationDataset(ident_config, _PassthroughTransformation())
        assert len(ds.data) == len(ds.roles)

    def test_classes_present_in_both(self, ident_config):
        ds = IdentificationDataset(ident_config, _PassthroughTransformation())
        gallery_classes = {ds.data[i][0] for i, r in enumerate(ds.roles) if r == "gallery"}
        probe_classes = {ds.data[i][0] for i, r in enumerate(ds.roles) if r == "probe"}
        assert gallery_classes == probe_classes == {"alice", "bob", "carol"}

    def test_missing_gallery_raises(self, tmp_path):
        # Only probe — no gallery
        (tmp_path / "probe" / "alice").mkdir(parents=True)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({"eval_dir": str(tmp_path)}))
        cfg = EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
        with pytest.raises(FileNotFoundError, match="Gallery"):
            IdentificationDataset(cfg, _PassthroughTransformation())

    def test_missing_probe_raises(self, tmp_path):
        (tmp_path / "gallery" / "alice").mkdir(parents=True)
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump({"eval_dir": str(tmp_path)}))
        cfg = EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
        with pytest.raises(FileNotFoundError, match="Probe"):
            IdentificationDataset(cfg, _PassthroughTransformation())
