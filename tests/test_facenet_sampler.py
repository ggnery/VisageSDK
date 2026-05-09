"""Tests for FacenetBatchSampler batch construction."""

import pytest
import yaml
from torchvision import transforms

from batch_sampler.facenet_batch_sampler import FacenetBatchSampler
from config.batch_sampler.base_batch_sampler_config import BatchSamplerConfig
from config.dataset.train_val.base_train_val_dataset_config import TrainValDatasetConfig
from dataset.train_val.image_folder_dataset import ImageFolderDataset


class _Tx:
    transform = transforms.Compose([transforms.ToTensor()])


@pytest.fixture
def small_train_dataset(tmp_imagefolder, tmp_path):
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
    cfg = TrainValDatasetConfig(str(cfg_path), backbone_info={"input_size": [32, 32]})
    return ImageFolderDataset(cfg, _Tx(), split="train")


@pytest.fixture
def make_sampler(tmp_path):
    counter = {"i": 0}

    def _make(faces_per_identity, num_identities_per_batch):
        counter["i"] += 1
        path = tmp_path / f"sampler_{counter['i']}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "faces_per_identity": faces_per_identity,
                    "num_identities_per_batch": num_identities_per_batch,
                }
            )
        )
        return BatchSamplerConfig(str(path))

    return _make


class TestFacenetBatchSampler:
    def test_batch_size_matches_product(self, small_train_dataset, make_sampler):
        cfg = make_sampler(faces_per_identity=2, num_identities_per_batch=2)
        sampler = FacenetBatchSampler(cfg, small_train_dataset)
        assert sampler.batch_size == 4
        # Each batch yielded has exactly batch_size indices
        for batch in sampler:
            assert len(batch) == 4

    def test_filters_out_identities_with_few_samples(self, small_train_dataset, make_sampler):
        # Each class has 4 images; require 5 → no valid identities
        cfg = make_sampler(faces_per_identity=5, num_identities_per_batch=1)
        sampler = FacenetBatchSampler(cfg, small_train_dataset)
        assert sampler.num_valid_identities == 0
        assert list(iter(sampler)) == []

    def test_all_indices_are_valid(self, small_train_dataset, make_sampler):
        cfg = make_sampler(faces_per_identity=2, num_identities_per_batch=2)
        sampler = FacenetBatchSampler(cfg, small_train_dataset)
        ds_len = len(small_train_dataset)
        for batch in sampler:
            for idx in batch:
                assert 0 <= idx < ds_len

    def test_each_batch_is_balanced_across_identities(self, small_train_dataset, make_sampler):
        cfg = make_sampler(faces_per_identity=2, num_identities_per_batch=3)
        sampler = FacenetBatchSampler(cfg, small_train_dataset)
        for batch in sampler:
            # Each batch should draw `faces_per_identity` per identity
            labels = [small_train_dataset.data[i][0] for i in batch]
            counts = {label: labels.count(label) for label in set(labels)}
            for cnt in counts.values():
                assert cnt == 2

    def test_len_matches_iteration_for_full_pass(self, small_train_dataset, make_sampler):
        cfg = make_sampler(faces_per_identity=2, num_identities_per_batch=1)
        sampler = FacenetBatchSampler(cfg, small_train_dataset)
        n_iter = sum(1 for _ in sampler)
        # __len__ may slightly differ from n_iter when last batch is padded;
        # they should be at most off by 1.
        assert abs(len(sampler) - n_iter) <= 1
