"""Tests for ImageFolderDataset including B3 (sorted determinism) regression."""

from pathlib import Path

import pytest
import yaml

from config.dataset.train_val.base_train_val_dataset_config import TrainValDatasetConfig
from dataset.train_val.image_folder_dataset import ImageFolderDataset, _scan_dir


class StubTransformation:
    """Minimal stub: just exposes a passthrough `transform` (lambda)."""
    def __init__(self):
        from torchvision import transforms
        self.transform = transforms.Compose([transforms.ToTensor()])


@pytest.fixture
def dataset_config(tmp_imagefolder, tmp_path):
    cfg_path = tmp_path / "ds.yaml"
    cfg_path.write_text(yaml.safe_dump({
        "train_dir": str(tmp_imagefolder / "train"),
        "val_dir": str(tmp_imagefolder / "val"),
        "num_classes": 3,
    }))
    return TrainValDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})


class TestSplitArgument:
    def test_split_train_reads_train_dir(self, dataset_config, tmp_imagefolder):
        ds = ImageFolderDataset(dataset_config, StubTransformation(), split="train")
        for label, path in ds.data:
            assert Path(path).is_relative_to(tmp_imagefolder / "train")

    def test_split_val_reads_val_dir(self, dataset_config, tmp_imagefolder):
        ds = ImageFolderDataset(dataset_config, StubTransformation(), split="val")
        for label, path in ds.data:
            assert Path(path).is_relative_to(tmp_imagefolder / "val")

    def test_invalid_split_raises(self, dataset_config):
        with pytest.raises(ValueError, match="split must be one of"):
            ImageFolderDataset(dataset_config, StubTransformation(), split="test")


class TestDeterminism:
    def test_sorted_classes_b3_regression(self, tmp_imagefolder):
        """B3 regression: scanning twice must yield the exact same order."""
        order1 = _scan_dir(tmp_imagefolder / "train")
        order2 = _scan_dir(tmp_imagefolder / "train")
        assert order1 == order2

    def test_classes_appear_in_alphabetical_order(self, tmp_imagefolder):
        pairs = _scan_dir(tmp_imagefolder / "train")
        seen = []
        for label, _ in pairs:
            if label not in seen:
                seen.append(label)
        assert seen == sorted(seen)
        assert seen == ["alice", "bob", "carol"]

    def test_label_to_idx_stable(self, dataset_config):
        """Same data → same label_to_idx mapping every time."""
        ds1 = ImageFolderDataset(dataset_config, StubTransformation(), split="train")
        ds2 = ImageFolderDataset(dataset_config, StubTransformation(), split="train")
        assert ds1.label_to_idx == ds2.label_to_idx


class TestDataContents:
    def test_total_count_matches(self, dataset_config):
        ds = ImageFolderDataset(dataset_config, StubTransformation(), split="train")
        # 3 classes × 4 images
        assert len(ds) == 12

    def test_label_map_groups_by_class(self, dataset_config):
        ds = ImageFolderDataset(dataset_config, StubTransformation(), split="train")
        # All 3 classes should have entries; each with 4 indices
        assert len(ds.label_map) == 3
        for indices in ds.label_map.values():
            assert len(indices) == 4

    def test_getitem_returns_label_and_tensor(self, dataset_config):
        import torch
        ds = ImageFolderDataset(dataset_config, StubTransformation(), split="train")
        label, image = ds[0]
        assert isinstance(label, int)
        assert isinstance(image, torch.Tensor)

    def test_missing_dir_raises(self, tmp_path):
        from torchvision import transforms

        bad_cfg_path = tmp_path / "bad.yaml"
        bad_cfg_path.write_text(yaml.safe_dump({
            "train_dir": str(tmp_path / "does_not_exist"),
            "val_dir": str(tmp_path / "also_missing"),
            "num_classes": 0,
        }))
        bad_cfg = TrainValDatasetConfig(str(bad_cfg_path), backbone_info={"input_size": [32, 32]})

        class TX:
            transform = transforms.Compose([transforms.ToTensor()])

        with pytest.raises(FileNotFoundError):
            ImageFolderDataset(bad_cfg, TX(), split="train")
