"""Metric functions for verification (pair-based) and identification (gallery/probe).

Embeddings are assumed unnormalized; cosine helpers L2-normalize internally.
"""

import numpy as np
import torch
import torch.nn.functional as F

# =============================================================================
# Distance / similarity helpers
# =============================================================================


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def pairwise_cosine_distance(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Cosine distance (1 - cos_sim) between paired rows. Shapes (N, D), (N, D) -> (N,)."""
    a = l2_normalize(emb_a)
    b = l2_normalize(emb_b)
    return 1.0 - (a * b).sum(dim=-1)


def pairwise_euclidean_distance(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """Euclidean distance between paired rows. (N, D), (N, D) -> (N,)."""
    return torch.norm(emb_a - emb_b, p=2, dim=-1)


def cosine_similarity_matrix(query: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix between two sets of embeddings. (Nq, D), (Ng, D) -> (Nq, Ng)."""
    q = l2_normalize(query)
    g = l2_normalize(gallery)
    return q @ g.T


# =============================================================================
# Verification metrics (pair-based)
# =============================================================================


def verification_accuracy(distances: np.ndarray, labels: np.ndarray, threshold: float) -> float:
    """Binary accuracy treating pairs with distance <= threshold as 'same'.

    Args:
        distances: (N,) pair distances
        labels: (N,) 1 for same-pair, 0 for different
        threshold: distance threshold at or below which pairs are predicted same
    """
    predictions = (distances <= threshold).astype(np.int32)
    return float((predictions == labels).mean())


def best_threshold(
    distances: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Sweep thresholds and return the (threshold, accuracy) maximizing accuracy."""
    if thresholds is None:
        lo, hi = float(distances.min()), float(distances.max())
        thresholds = np.linspace(lo, hi, 400)
    best_acc = -1.0
    best_thr = float(thresholds[0])
    for t in thresholds:
        acc = verification_accuracy(distances, labels, float(t))
        if acc > best_acc:
            best_acc = acc
            best_thr = float(t)
    return best_thr, best_acc


def roc_curve(
    distances: np.ndarray, labels: np.ndarray, n_thresholds: int = 400
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for the ROC curve.

    The score being thresholded is *similarity* (-distance), so a higher
    threshold means stricter "same" prediction. Returned thresholds are in
    distance space for easy reuse with `verification_accuracy`.
    """
    lo, hi = float(distances.min()), float(distances.max())
    thresholds = np.linspace(lo, hi, n_thresholds)
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = max(int(pos_mask.sum()), 1)
    n_neg = max(int(neg_mask.sum()), 1)
    tpr = np.zeros_like(thresholds)
    fpr = np.zeros_like(thresholds)
    for i, t in enumerate(thresholds):
        # `<=` (not `<`) so the max-distance pair is included at t=hi —
        # otherwise (TPR, FPR) never reaches (1.0, 1.0) and AUC underestimates.
        pred_same = distances <= t
        tpr[i] = float(np.logical_and(pred_same, pos_mask).sum()) / n_pos
        fpr[i] = float(np.logical_and(pred_same, neg_mask).sum()) / n_neg
    return fpr, tpr, thresholds


def roc_auc(distances: np.ndarray, labels: np.ndarray) -> float:
    """Trapezoidal AUC of the ROC curve."""
    fpr, tpr, _ = roc_curve(distances, labels)
    # lexsort breaks FPR ties by TPR so the trapezoidal integration stays
    # monotonic across plateau regions of FPR.
    order = np.lexsort((tpr, fpr))
    return float(np.trapezoid(tpr[order], fpr[order]))


def eer(distances: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate. Returns (eer_value, threshold_at_eer)."""
    fpr, tpr, thresholds = roc_curve(distances, labels)
    fnr = 1.0 - tpr
    diff = np.abs(fpr - fnr)
    idx = int(np.argmin(diff))
    return float((fpr[idx] + fnr[idx]) / 2.0), float(thresholds[idx])


def tar_at_far(distances: np.ndarray, labels: np.ndarray, far_target: float) -> tuple[float, float]:
    """True Accept Rate at a target FAR. Returns (tar, threshold).

    The smallest swept threshold yields FAR=0 by construction, so at least
    one candidate always satisfies fpr<=far_target for any positive target.
    On TPR ties, prefer the LARGEST threshold (most permissive — operationally
    more useful at the same TAR).
    """
    fpr, tpr, thresholds = roc_curve(distances, labels, n_thresholds=2000)
    valid = fpr <= far_target
    idx_candidates = np.where(valid)[0]
    candidate_tprs = tpr[idx_candidates]
    max_tpr = candidate_tprs.max()
    # Tie-break by largest threshold.
    tied = idx_candidates[candidate_tprs == max_tpr]
    best = int(tied[np.argmax(thresholds[tied])])
    return float(tpr[best]), float(thresholds[best])


def lfw_kfold_accuracy(
    distances: np.ndarray,
    labels: np.ndarray,
    fold_indices: np.ndarray,
    n_folds: int = 10,
) -> dict[str, float]:
    """Standard LFW protocol: per fold pick best threshold on training folds,
    measure accuracy on held-out fold. Returns mean and std accuracy.

    Args:
        distances: (N,) pair distances
        labels:    (N,) 0/1 same indicator
        fold_indices: (N,) integer fold id per pair
        n_folds: total folds
    """
    accuracies: list[float] = []
    thresholds: list[float] = []
    for fold in range(n_folds):
        train_mask = fold_indices != fold
        test_mask = fold_indices == fold
        if not test_mask.any():
            continue
        if not train_mask.any():
            # Single fold: tune and evaluate on the same set (no held-out split possible).
            thr, acc = best_threshold(distances[test_mask], labels[test_mask])
        else:
            thr, _ = best_threshold(distances[train_mask], labels[train_mask])
            acc = verification_accuracy(distances[test_mask], labels[test_mask], thr)
        accuracies.append(acc)
        thresholds.append(thr)
    accs = np.array(accuracies)
    thrs = np.array(thresholds)
    # ddof=1 matches the LFW community's reported 10-fold protocol.
    return {
        "accuracy_mean": float(accs.mean()) if len(accs) else 0.0,
        "accuracy_std": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
        "threshold_mean": float(thrs.mean()) if len(thrs) else 0.0,
        "threshold_std": float(thrs.std(ddof=1)) if len(thrs) > 1 else 0.0,
    }


# =============================================================================
# Identification metrics (gallery/probe retrieval)
# =============================================================================


def cmc_curve(
    similarity: torch.Tensor,
    probe_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    max_rank: int | None = None,
) -> np.ndarray:
    """Cumulative Match Characteristic.

    Returns array of shape (max_rank,) where cmc[k] = fraction of probes
    whose true identity appears within the top-(k+1) retrieved gallery items.

    If a probe has no matching gallery identity, it is excluded from the metric.
    Self-matches (probe id == gallery id when datasets overlap) should be
    handled by the caller (e.g. mask-out before passing similarity).
    """
    sim = similarity.detach().cpu()
    pl = probe_labels.detach().cpu().numpy()
    gl = gallery_labels.detach().cpu().numpy()
    n_gallery = int(gl.shape[0])
    max_rank = int(max_rank) if max_rank is not None else n_gallery

    # For each probe, sort gallery indices by descending similarity. `stable=True`
    # keeps tie-breaking deterministic across runs.
    order = torch.argsort(sim, dim=1, descending=True, stable=True).numpy()
    matches = np.zeros(max_rank, dtype=np.float64)
    n_valid = 0
    for i, true_label in enumerate(pl):
        if not np.any(gl == true_label):
            continue
        n_valid += 1
        ranked = gl[order[i]]
        hit = np.where(ranked == true_label)[0]
        if len(hit) == 0:
            continue
        first_hit = hit[0]
        if first_hit < max_rank:
            matches[first_hit:] += 1.0
    if n_valid == 0:
        return matches
    return matches / n_valid


def rank_n_accuracy(
    similarity: torch.Tensor,
    probe_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    n: int = 1,
) -> float:
    """Fraction of probes whose true id appears in the top-N retrieved."""
    cmc = cmc_curve(similarity, probe_labels, gallery_labels, max_rank=n)
    return float(cmc[n - 1])


def mean_average_precision(
    similarity: torch.Tensor,
    probe_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
) -> float:
    """Mean Average Precision over probes.

    For each probe, AP averages precision at every position where a relevant
    (same-id) gallery item is retrieved. mAP averages AP over probes.
    Probes whose identity has no gallery match are excluded.
    """
    sim = similarity.detach().cpu()
    pl = probe_labels.detach().cpu().numpy()
    gl = gallery_labels.detach().cpu().numpy()
    order = torch.argsort(sim, dim=1, descending=True, stable=True).numpy()

    aps: list[float] = []
    for i, true_label in enumerate(pl):
        if not np.any(gl == true_label):
            continue
        ranked = gl[order[i]]
        relevant = (ranked == true_label).astype(np.float64)
        n_relevant = int(relevant.sum())
        if n_relevant == 0:
            continue
        hits = np.cumsum(relevant)
        positions = np.arange(1, len(relevant) + 1)
        precisions = hits / positions
        ap = float((precisions * relevant).sum() / n_relevant)
        aps.append(ap)
    if not aps:
        return 0.0
    return float(np.mean(aps))
