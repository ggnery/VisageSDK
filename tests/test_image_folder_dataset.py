"""Tests for ImageFolderDataset including B3 (sorted determinism) regression."""

from pathlib import Path

import pytest
import yaml

from config.train_val_dataset_config import TrainValDatasetConfig
from dataset.train_val.image_folder_dataset import ImageFolderDataset, _scan_dir
from tests.conftest import _PassthroughTransformation


@pytest.fixture
def dataset_config(tmp_imagefolder, tmp_path):
    cfg_path = tmp_path / "ds.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "train_dir": str(tmp_imagefolder / "train"),
                "val_dir": str(tmp_imagefolder / "val"),
                "num_classes": 3,
            }
        )
    )
    return TrainValDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})


class TestSplitArgument:
    def test_split_train_reads_train_dir(self, dataset_config, tmp_imagefolder):
        ds = ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="train")
        for _, path in ds.data:
            assert Path(path).is_relative_to(tmp_imagefolder / "train")

    def test_split_val_reads_val_dir(self, dataset_config, tmp_imagefolder):
        ds = ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="val")
        for _, path in ds.data:
            assert Path(path).is_relative_to(tmp_imagefolder / "val")

    def test_invalid_split_raises(self, dataset_config):
        with pytest.raises(ValueError, match="split must be one of"):
            ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="test")


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
        ds1 = ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="train")
        ds2 = ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="train")
        assert ds1.label_to_idx == ds2.label_to_idx


class TestDataContents:
    def test_total_count_matches(self, dataset_config):
        ds = ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="train")
        # 3 classes × 4 images
        assert len(ds) == 12

    def test_label_map_groups_by_class(self, dataset_config):
        ds = ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="train")
        # All 3 classes should have entries; each with 4 indices
        assert len(ds.label_map) == 3
        for indices in ds.label_map.values():
            assert len(indices) == 4

    def test_getitem_returns_label_and_tensor(self, dataset_config):
        import torch

        ds = ImageFolderDataset(dataset_config, _PassthroughTransformation(), split="train")
        label, image = ds[0]
        assert isinstance(label, int)
        assert isinstance(image, torch.Tensor)

    def test_missing_dir_raises(self, tmp_path):
        from torchvision import transforms

        bad_cfg_path = tmp_path / "bad.yaml"
        bad_cfg_path.write_text(
            yaml.safe_dump(
                {
                    "train_dir": str(tmp_path / "does_not_exist"),
                    "val_dir": str(tmp_path / "also_missing"),
                    # `num_classes` must be a positive int per the
                    # TrainValDatasetConfig validation; we're testing the
                    # FileNotFoundError path, not the num_classes guard.
                    "num_classes": 1,
                }
            )
        )
        bad_cfg = TrainValDatasetConfig(str(bad_cfg_path), backbone_info={"input_size": [32, 32]})

        class TX:
            transform = transforms.Compose([transforms.ToTensor()])

        with pytest.raises(FileNotFoundError):
            ImageFolderDataset(bad_cfg, TX(), split="train")


class TestLabelToIdxUnion:
    """B-2 regression: label_to_idx must be built from the UNION of
    train+val so that the same class name always maps to the same
    integer ID in both splits. Otherwise the CE / margin head trained on
    train mispredicts on val whenever the per-split class sets differ."""

    def _build_asymmetric_layout(self, tmp_path):
        """train has {alice, bob, carol}; val has {bob, carol, dave}."""
        from PIL import Image

        def _img(p):
            p.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16)).save(p)

        for cls in ("alice", "bob", "carol"):
            _img(tmp_path / "train" / cls / "x.jpg")
        for cls in ("bob", "carol", "dave"):
            _img(tmp_path / "val" / cls / "x.jpg")
        cfg_path = tmp_path / "ds.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "train_dir": str(tmp_path / "train"),
                    "val_dir": str(tmp_path / "val"),
                    "num_classes": 4,
                }
            )
        )
        return TrainValDatasetConfig(str(cfg_path), backbone_info={"input_size": [16, 16]})

    def test_train_and_val_share_label_indices(self, tmp_path):
        cfg = self._build_asymmetric_layout(tmp_path)
        train_ds = ImageFolderDataset(cfg, _PassthroughTransformation(), split="train")
        val_ds = ImageFolderDataset(cfg, _PassthroughTransformation(), split="val")
        # Same union of 4 classes on both sides.
        assert train_ds.label_to_idx == val_ds.label_to_idx
        # Sanity: alphabetical → alice=0, bob=1, carol=2, dave=3.
        assert train_ds.label_to_idx == {"alice": 0, "bob": 1, "carol": 2, "dave": 3}

    def test_num_classes_smaller_than_union_raises(self, tmp_path):
        cfg = self._build_asymmetric_layout(tmp_path)
        # Override num_classes to a too-small value
        cfg._params["num_classes"] = 2
        with pytest.raises(ValueError, match="num_classes"):
            ImageFolderDataset(cfg, _PassthroughTransformation(), split="train")


class TestNumClassesValidation:
    """B-9 regression: dataset YAML with `num_classes: null` used to
    silently propagate None into `nn.Linear(embedding_size, None)` and
    crash with an opaque TypeError deep in PyTorch. Validate up-front."""

    def test_null_num_classes_raises(self, tmp_path):
        cfg_path = tmp_path / "ds.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "train_dir": str(tmp_path / "t"),
                    "val_dir": str(tmp_path / "v"),
                    "num_classes": None,
                }
            )
        )
        with pytest.raises(ValueError, match="num_classes"):
            TrainValDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})

    def test_zero_num_classes_raises(self, tmp_path):
        cfg_path = tmp_path / "ds.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "train_dir": str(tmp_path / "t"),
                    "val_dir": str(tmp_path / "v"),
                    "num_classes": 0,
                }
            )
        )
        with pytest.raises(ValueError, match="num_classes"):
            TrainValDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
