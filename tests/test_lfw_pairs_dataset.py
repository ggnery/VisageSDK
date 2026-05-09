"""Tests for LFWPairsDataset."""

import pytest
import yaml
from torchvision import transforms

from config.dataset.eval.base_eval_dataset_config import EvalDatasetConfig
from dataset.eval.lfw_pairs_dataset import LFWPairsDataset


class StubTransformation:
    transform = transforms.Compose([transforms.ToTensor()])


@pytest.fixture
def lfw_config(tmp_lfw_pairs, tmp_path):
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
    return EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})


class TestLFWPairsDataset:
    def test_header_parsed(self, lfw_config):
        ds = LFWPairsDataset(lfw_config, StubTransformation())
        assert ds.n_folds == 2
        assert ds.n_pairs_per_fold == 2

    def test_pair_count(self, lfw_config):
        ds = LFWPairsDataset(lfw_config, StubTransformation())
        # 2 folds × (2 same + 2 diff) = 8 pairs
        assert len(ds.pairs) == 8

    def test_same_diff_label_distribution(self, lfw_config):
        ds = LFWPairsDataset(lfw_config, StubTransformation())
        same = sum(1 for p in ds.pairs if p[2] == 1)
        diff = sum(1 for p in ds.pairs if p[2] == 0)
        assert same == 4
        assert diff == 4

    def test_fold_ids(self, lfw_config):
        ds = LFWPairsDataset(lfw_config, StubTransformation())
        folds = sorted({p[3] for p in ds.pairs})
        assert folds == [0, 1]

    def test_unique_image_indices_within_data(self, lfw_config):
        ds = LFWPairsDataset(lfw_config, StubTransformation())
        # All pair indices must be valid into self.data
        for ia, ib, _, _ in ds.pairs:
            assert 0 <= ia < len(ds.data)
            assert 0 <= ib < len(ds.data)

    def test_each_unique_image_recorded_once(self, lfw_config):
        ds = LFWPairsDataset(lfw_config, StubTransformation())
        paths = [path for _, path in ds.data]
        assert len(paths) == len(set(paths)), "Each image must appear at most once in data"

    def test_getitem_returns_index_and_tensor(self, lfw_config):
        import torch

        ds = LFWPairsDataset(lfw_config, StubTransformation())
        idx, image = ds[0]
        assert idx == 0
        assert isinstance(image, torch.Tensor)

    def test_missing_pairs_file_raises(self, tmp_path):
        bad_path = tmp_path / "bad.yaml"
        bad_path.write_text(
            yaml.safe_dump(
                {
                    "eval_dir": str(tmp_path / "imgs"),
                    "pairs_path": str(tmp_path / "missing_pairs.txt"),
                    "image_ext": "jpg",
                }
            )
        )
        cfg = EvalDatasetConfig(str(bad_path), backbone_info={"input_size": [32, 32]})
        with pytest.raises(FileNotFoundError):
            LFWPairsDataset(cfg, StubTransformation())

    def test_referenced_image_must_exist(self, tmp_lfw_pairs, tmp_path):
        """If pairs.txt mentions an image that's not on disk, raise."""
        images_dir, _ = tmp_lfw_pairs
        # Build a pairs.txt that references a missing image
        bad_pairs = tmp_path / "bad_pairs.txt"
        bad_pairs.write_text("1 1\nalice 1 99\nalice 1 carol 1\n")
        cfg_path = tmp_path / "cfg.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "eval_dir": str(images_dir),
                    "pairs_path": str(bad_pairs),
                    "image_ext": "jpg",
                }
            )
        )
        cfg = EvalDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
        with pytest.raises(FileNotFoundError):
            LFWPairsDataset(cfg, StubTransformation())
