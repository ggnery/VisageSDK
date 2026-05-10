"""Tests for loss forward passes and their loss_stats outputs."""

import pytest
import torch
import yaml

from config.loss.base_loss_config import LossConfig
from loss.center_loss import CenterLoss
from loss.cross_entropy_loss import CrossEntropyLoss
from loss.margin_cosine_product_loss import MarginCosineProductLoss
from loss.triplet_loss import TripletLoss


@pytest.fixture
def loss_config_factory(tmp_path):
    """Build a LossConfig from inline YAML data."""
    counter = {"i": 0}

    def _make(yaml_data: dict, num_classes: int = 3, embedding_size: int = 8) -> LossConfig:
        counter["i"] += 1
        path = tmp_path / f"loss_{counter['i']}.yaml"
        path.write_text(yaml.safe_dump(yaml_data))
        return LossConfig(
            str(path),
            backbone_info={"embedding_size": embedding_size},
            dataset_info={"num_classes": num_classes},
        )

    return _make


# =============================================================================
# CrossEntropyLoss
# =============================================================================


class TestCrossEntropyLoss:
    def test_forward_returns_loss_and_stats(self, loss_config_factory):
        cfg = loss_config_factory({"device": "cpu", "label_smoothing": 0.1, "use_bias": True})
        loss = CrossEntropyLoss(cfg)
        emb = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 0])
        value, stats = loss(emb, labels)
        assert isinstance(value, torch.Tensor) and value.dim() == 0
        assert "cls_accuracy" in stats
        assert 0.0 <= stats["cls_accuracy"] <= 1.0
        # I3: redundant loss key was removed
        assert "cross_entropy_loss" not in stats

    def test_loss_is_finite(self, loss_config_factory):
        cfg = loss_config_factory({"device": "cpu", "label_smoothing": 0.0, "use_bias": False})
        loss = CrossEntropyLoss(cfg)
        emb = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 0])
        value, _ = loss(emb, labels)
        assert torch.isfinite(value)


# =============================================================================
# MarginCosineProductLoss
# =============================================================================


class TestMarginCosineProductLoss:
    def test_forward_runs(self, loss_config_factory):
        cfg = loss_config_factory({"device": "cpu", "s": 30.0, "m": 0.4})
        loss = MarginCosineProductLoss(cfg)
        emb = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 0])
        value, stats = loss(emb, labels)
        assert torch.isfinite(value)
        assert "cls_accuracy" in stats
        # I3: no redundant loss key
        assert "margin_cosine_loss" not in stats


# =============================================================================
# CenterLoss
# =============================================================================


class TestCenterLoss:
    def test_forward_returns_components(self, loss_config_factory):
        cfg = loss_config_factory({"device": "cpu", "alpha": 0.5, "use_bias": True})
        loss = CenterLoss(cfg)
        emb = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 0])
        value, stats = loss(emb, labels)
        assert torch.isfinite(value)
        # Decomposition is preserved (these aren't redundant: they show the
        # contribution of each term).
        assert "cross_entropy_loss" in stats
        assert "center_loss" in stats
        assert "cls_accuracy" in stats
        # I3: redundant `loss` key was removed
        assert "loss" not in stats


# =============================================================================
# TripletLoss
# =============================================================================


class TestTripletLoss:
    def test_forward_with_valid_triplets(self, loss_config_factory):
        cfg = loss_config_factory({"device": "cpu", "margin": 0.2})
        loss = TripletLoss(cfg)
        # Construct embeddings where each identity has 2 samples
        torch.manual_seed(0)
        n_id, samples_per_id = 4, 2
        centroids = torch.randn(n_id, 8)
        emb = centroids.repeat_interleave(samples_per_id, dim=0) + 0.05 * torch.randn(
            n_id * samples_per_id, 8
        )
        labels = torch.arange(n_id).repeat_interleave(samples_per_id)
        value, stats = loss(emb, labels)
        assert torch.isfinite(value)
        # I3: redundant `triplet_loss` key removed
        assert "triplet_loss" not in stats
        # Mining stats present
        assert "avg_pos_distance" in stats
        assert "avg_neg_distance" in stats
        assert "active_triplets" in stats
        assert "total_triplets" in stats

    def test_returns_zero_loss_when_no_triplets_can_be_formed(self, loss_config_factory):
        """If every label is unique we cannot form anchor-positive pairs."""
        cfg = loss_config_factory({"device": "cpu", "margin": 0.2})
        loss = TripletLoss(cfg)
        emb = torch.randn(4, 8)
        labels = torch.tensor([0, 1, 2, 3])  # all distinct
        value, stats = loss(emb, labels)
        assert value.item() == pytest.approx(0.0)

    def test_zero_triplet_loss_is_connected_to_graph(self, loss_config_factory):
        """Regression: pre-fix `torch.tensor(0.0, requires_grad=True)` was a
        leaf tensor disconnected from the embeddings, so `.backward()` was a
        silent no-op and the trainer kept stepping the optimizer with stale
        gradients. The fix ties the zero loss to `embeddings * 0` so the
        autograd graph reaches every model parameter. Verify by checking
        that backward through the zero-triplet path produces grad tensors
        (all zero is fine — the contract is that grads exist)."""
        cfg = loss_config_factory({"device": "cpu", "margin": 0.2})
        loss = TripletLoss(cfg)
        # Mimic an upstream model parameter so we can check grad propagation.
        upstream = torch.randn(4, 8, requires_grad=True)
        emb = upstream * 2.0  # arbitrary trainable transform
        labels = torch.tensor([0, 1, 2, 3])  # all distinct → no triplets
        value, _ = loss(emb, labels)
        assert value.requires_grad, "zero loss must keep the graph connected"
        value.backward()
        assert upstream.grad is not None, (
            "backward() through the zero-triplet branch produced no gradient — "
            "the loss is detached from the model's parameters."
        )

    def test_active_triplets_le_total(self, loss_config_factory):
        cfg = loss_config_factory({"device": "cpu", "margin": 0.5})
        loss = TripletLoss(cfg)
        torch.manual_seed(1)
        emb = torch.randn(8, 8)
        labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        _, stats = loss(emb, labels)
        assert stats["active_triplets"] <= stats["total_triplets"]
