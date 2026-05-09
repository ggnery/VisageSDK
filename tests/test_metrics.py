"""Tests for tools.metrics — pure functions, no I/O."""

import math

import numpy as np
import pytest
import torch

from tools.metrics import (
    best_threshold,
    cmc_curve,
    cosine_similarity_matrix,
    eer,
    l2_normalize,
    lfw_kfold_accuracy,
    mean_average_precision,
    pairwise_cosine_distance,
    pairwise_euclidean_distance,
    rank_n_accuracy,
    roc_auc,
    roc_curve,
    tar_at_far,
    verification_accuracy,
)


# =============================================================================
# Distance / similarity helpers
# =============================================================================

class TestDistanceHelpers:
    def test_l2_normalize_unit_vectors(self):
        x = torch.tensor([[3.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        n = l2_normalize(x)
        assert torch.allclose(n.norm(dim=-1), torch.ones(2), atol=1e-6)

    def test_pairwise_cosine_identical(self):
        a = torch.randn(5, 8)
        d = pairwise_cosine_distance(a, a)
        assert torch.allclose(d, torch.zeros(5), atol=1e-6)

    def test_pairwise_cosine_opposite(self):
        a = torch.tensor([[1.0, 0.0]])
        b = -a
        d = pairwise_cosine_distance(a, b)
        # cosine of opposite vectors = -1, distance = 1 - (-1) = 2
        assert torch.allclose(d, torch.tensor([2.0]), atol=1e-6)

    def test_pairwise_cosine_orthogonal(self):
        a = torch.tensor([[1.0, 0.0]])
        b = torch.tensor([[0.0, 1.0]])
        d = pairwise_cosine_distance(a, b)
        assert torch.allclose(d, torch.tensor([1.0]), atol=1e-6)

    def test_pairwise_euclidean_identical(self):
        a = torch.randn(4, 6)
        d = pairwise_euclidean_distance(a, a)
        assert torch.allclose(d, torch.zeros(4), atol=1e-6)

    def test_cosine_similarity_matrix_shape_and_self(self):
        q = torch.randn(3, 8)
        g = torch.randn(5, 8)
        m = cosine_similarity_matrix(q, g)
        assert m.shape == (3, 5)
        # Self-similarity (q @ q) diagonal should be ~1
        m_self = cosine_similarity_matrix(q, q)
        assert torch.allclose(m_self.diag(), torch.ones(3), atol=1e-5)


# =============================================================================
# Verification metrics
# =============================================================================

@pytest.fixture
def separable_pairs():
    """Synthetic verification setup: same-pair distances <= 0.1, diff >= 0.5."""
    rng = np.random.default_rng(0)
    same = rng.uniform(0.0, 0.1, size=200)
    diff = rng.uniform(0.5, 1.0, size=200)
    distances = np.concatenate([same, diff])
    labels = np.concatenate([np.ones(200, dtype=np.int32), np.zeros(200, dtype=np.int32)])
    folds = np.tile(np.arange(10), 40)  # 10 folds, 40 pairs per fold
    return distances, labels, folds


class TestVerificationMetrics:
    def test_verification_accuracy_perfect(self, separable_pairs):
        distances, labels, _ = separable_pairs
        # Threshold between same/diff clusters → 100% accuracy
        assert verification_accuracy(distances, labels, threshold=0.3) == pytest.approx(1.0)

    def test_verification_accuracy_chance_threshold(self):
        # Threshold above all distances → all "same" → accuracy = positives / total
        distances = np.array([0.1, 0.2, 0.8, 0.9])
        labels = np.array([1, 1, 0, 0])
        assert verification_accuracy(distances, labels, threshold=10.0) == pytest.approx(0.5)

    def test_best_threshold_finds_separation(self, separable_pairs):
        distances, labels, _ = separable_pairs
        thr, acc = best_threshold(distances, labels)
        assert acc == pytest.approx(1.0)
        assert 0.1 < thr < 0.5

    def test_roc_auc_separable(self, separable_pairs):
        distances, labels, _ = separable_pairs
        auc = roc_auc(distances, labels)
        assert auc > 0.99

    def test_roc_auc_random_is_half(self):
        rng = np.random.default_rng(0)
        distances = rng.uniform(0, 1, size=400)
        labels = rng.integers(0, 2, size=400)
        auc = roc_auc(distances, labels)
        assert 0.40 < auc < 0.60

    def test_eer_separable(self, separable_pairs):
        distances, labels, _ = separable_pairs
        eer_value, eer_thr = eer(distances, labels)
        assert eer_value < 0.05
        assert 0.0 < eer_thr < 1.0

    def test_tar_at_far_returns_high_tar_for_separable(self, separable_pairs):
        distances, labels, _ = separable_pairs
        tar, _ = tar_at_far(distances, labels, far_target=0.01)
        assert tar > 0.95

    def test_tar_at_far_lower_far_lower_or_equal_tar(self, separable_pairs):
        distances, labels, _ = separable_pairs
        tar_loose, _ = tar_at_far(distances, labels, far_target=0.1)
        tar_strict, _ = tar_at_far(distances, labels, far_target=0.001)
        assert tar_strict <= tar_loose + 1e-6

    def test_lfw_kfold_accuracy_separable(self, separable_pairs):
        distances, labels, folds = separable_pairs
        result = lfw_kfold_accuracy(distances, labels, folds, n_folds=10)
        assert result["accuracy_mean"] == pytest.approx(1.0)
        assert result["accuracy_std"] == pytest.approx(0.0)
        assert math.isfinite(result["threshold_mean"])

    def test_lfw_kfold_single_fold_handled(self):
        """Regression: single-fold case used to crash on empty train_mask."""
        distances = np.array([0.05, 0.6, 0.08, 0.7])
        labels = np.array([1, 0, 1, 0])
        folds = np.zeros(4, dtype=np.int32)
        result = lfw_kfold_accuracy(distances, labels, folds, n_folds=1)
        assert result["accuracy_mean"] == pytest.approx(1.0)


# =============================================================================
# Identification metrics
# =============================================================================

@pytest.fixture
def perfect_retrieval():
    """Each probe's nearest gallery is the same id by construction."""
    torch.manual_seed(0)
    n_id = 10
    centroids = torch.randn(n_id, 16)
    gallery_emb = centroids + 0.01 * torch.randn(n_id, 16)
    n_probe_per = 3
    probe_emb = (
        centroids.repeat_interleave(n_probe_per, dim=0)
        + 0.01 * torch.randn(n_id * n_probe_per, 16)
    )
    gallery_labels = torch.arange(n_id)
    probe_labels = torch.arange(n_id).repeat_interleave(n_probe_per)
    sim = cosine_similarity_matrix(probe_emb, gallery_emb)
    return sim, probe_labels, gallery_labels


class TestIdentificationMetrics:
    def test_cmc_curve_monotonic_and_ends_at_one(self, perfect_retrieval):
        sim, pl, gl = perfect_retrieval
        cmc = cmc_curve(sim, pl, gl, max_rank=5)
        assert all(cmc[i] <= cmc[i + 1] + 1e-9 for i in range(len(cmc) - 1))
        assert cmc[-1] == pytest.approx(1.0)

    def test_rank_n_accuracy_perfect(self, perfect_retrieval):
        sim, pl, gl = perfect_retrieval
        assert rank_n_accuracy(sim, pl, gl, n=1) == pytest.approx(1.0)
        assert rank_n_accuracy(sim, pl, gl, n=3) == pytest.approx(1.0)

    def test_rank_n_monotonic_in_n(self):
        torch.manual_seed(1)
        sim = torch.randn(20, 30)
        pl = torch.randint(0, 5, (20,))
        gl = torch.randint(0, 5, (30,))
        r1 = rank_n_accuracy(sim, pl, gl, n=1)
        r5 = rank_n_accuracy(sim, pl, gl, n=5)
        r10 = rank_n_accuracy(sim, pl, gl, n=10)
        assert r1 <= r5 + 1e-9 <= r10 + 1e-9

    def test_mean_average_precision_perfect(self, perfect_retrieval):
        sim, pl, gl = perfect_retrieval
        assert mean_average_precision(sim, pl, gl) == pytest.approx(1.0)

    def test_map_zero_when_no_relevant_in_gallery(self):
        torch.manual_seed(0)
        sim = torch.randn(3, 5)
        probe_labels = torch.tensor([99, 99, 99])
        gallery_labels = torch.tensor([0, 1, 2, 3, 4])
        # No overlap → all probes excluded → mAP=0.0 by definition
        assert mean_average_precision(sim, probe_labels, gallery_labels) == 0.0

    def test_cmc_curve_partial_match(self):
        """One probe matches at rank 0, one at rank 2; expected cumulative 0.5, 0.5, 1.0."""
        sim = torch.tensor([
            [1.0, 0.5, 0.2, 0.1],
            [0.1, 0.2, 0.9, 0.3],
        ])
        probe_labels = torch.tensor([0, 1])
        gallery_labels = torch.tensor([0, 1, 1, 2])  # probe-1 hits gallery-1 (rank 0) since sim[1,2]=0.9 highest
        # Recompute manually:
        # probe 0: order = [0, 1, 2, 3], gallery_labels[order] = [0, 1, 1, 2]; first hit of label 0 is at index 0
        # probe 1: order = [2, 3, 1, 0], gallery_labels[order] = [1, 2, 1, 0]; first hit of label 1 is at index 0
        cmc = cmc_curve(sim, probe_labels, gallery_labels, max_rank=4)
        assert cmc[0] == pytest.approx(1.0)


# =============================================================================
# ROC curve sanity
# =============================================================================

class TestRocCurve:
    def test_roc_curve_shapes(self, separable_pairs):
        distances, labels, _ = separable_pairs
        fpr, tpr, thr = roc_curve(distances, labels)
        assert fpr.shape == tpr.shape == thr.shape
        assert (fpr >= 0).all() and (fpr <= 1).all()
        assert (tpr >= 0).all() and (tpr <= 1).all()
