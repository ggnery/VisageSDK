from typing import Any

import numpy as np
import torch

from dataset.eval.lfw_pairs_dataset import LFWPairsDataset
from evaluator.base_evaluator import BaseEvaluator
from tools.metrics import (
    best_threshold,
    eer,
    lfw_kfold_accuracy,
    pairwise_cosine_distance,
    pairwise_euclidean_distance,
    roc_auc,
    roc_curve,
    tar_at_far,
)


class VerificationEvaluator(BaseEvaluator):
    """Pair-based metrics: LFW 10-fold accuracy, TAR@FAR, ROC-AUC, EER."""

    def evaluate(self) -> dict[str, Any]:
        if not isinstance(self.dataset, LFWPairsDataset):
            raise TypeError(
                f"VerificationEvaluator requires LFWPairsDataset, got {type(self.dataset).__name__}"
            )

        embeddings = self.encode()

        idx_a = torch.tensor([p[0] for p in self.dataset.pairs])
        idx_b = torch.tensor([p[1] for p in self.dataset.pairs])
        labels = np.array([p[2] for p in self.dataset.pairs], dtype=np.int32)
        folds = np.array([p[3] for p in self.dataset.pairs], dtype=np.int32)

        emb_a = embeddings[idx_a]
        emb_b = embeddings[idx_b]

        distance_kind = getattr(self.config, "distance", "cosine")
        if distance_kind == "cosine":
            distances = pairwise_cosine_distance(emb_a, emb_b).numpy()
        elif distance_kind == "euclidean":
            distances = pairwise_euclidean_distance(emb_a, emb_b).numpy()
        else:
            raise ValueError(f"Unknown distance kind: {distance_kind}")

        results: dict[str, Any] = {}

        kfold = lfw_kfold_accuracy(distances, labels, folds, n_folds=self.dataset.n_folds)
        results["lfw_accuracy_mean"] = kfold["accuracy_mean"]
        results["lfw_accuracy_std"] = kfold["accuracy_std"]
        results["lfw_threshold_mean"] = kfold["threshold_mean"]

        thr, acc = best_threshold(distances, labels)
        results["best_threshold_global"] = thr
        results["best_accuracy_global"] = acc

        results["roc_auc"] = roc_auc(distances, labels)
        eer_value, eer_thr = eer(distances, labels)
        results["eer"] = eer_value
        results["eer_threshold"] = eer_thr

        far_targets: list[float] = list(getattr(self.config, "far_targets", [1e-3, 1e-4, 1e-5]))
        for far in far_targets:
            tar, thr_far = tar_at_far(distances, labels, far)
            results[f"tar@far={far:.0e}"] = tar
            results[f"threshold@far={far:.0e}"] = thr_far

        # ROC points for plotting; redundant with the internal curves computed
        # above but cheap to recompute on typical pair counts.
        fpr_arr, tpr_arr, thr_arr = roc_curve(distances, labels)
        results["roc_curve"] = {
            "fpr": fpr_arr.tolist(),
            "tpr": tpr_arr.tolist(),
            "thresholds": thr_arr.tolist(),
        }

        # Genuine vs impostor distributions for the GUI score histogram.
        results["score_distributions"] = {
            "genuine": distances[labels == 1].astype(float).tolist(),
            "impostor": distances[labels == 0].astype(float).tolist(),
            "distance_kind": distance_kind,
        }

        return results
