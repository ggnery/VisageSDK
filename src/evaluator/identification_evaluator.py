from typing import Dict, List

import torch

from dataset.eval.identification_dataset import IdentificationDataset
from evaluator.base_evaluator import BaseEvaluator
from tools.metrics import (
    cmc_curve,
    cosine_similarity_matrix,
    mean_average_precision,
    rank_n_accuracy,
)


class IdentificationEvaluator(BaseEvaluator):
    """Gallery/probe metrics: rank-N accuracy, mAP, CMC@K."""

    def evaluate(self) -> Dict[str, float]:
        if not isinstance(self.dataset, IdentificationDataset):
            raise TypeError(
                f"IdentificationEvaluator requires IdentificationDataset, got {type(self.dataset).__name__}"
            )

        embeddings = self.encode()

        labels = [self.dataset.data[i][0] for i in range(len(self.dataset))]
        roles = self.dataset.roles
        gallery_idx = [i for i, r in enumerate(roles) if r == "gallery"]
        probe_idx = [i for i, r in enumerate(roles) if r == "probe"]

        if not gallery_idx or not probe_idx:
            raise ValueError("Identification dataset must contain both gallery and probe images")

        label_to_id: Dict[str, int] = {}
        for label in labels:
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
        encoded = torch.tensor([label_to_id[label] for label in labels])

        gallery_emb = embeddings[gallery_idx]
        probe_emb = embeddings[probe_idx]
        gallery_labels = encoded[gallery_idx]
        probe_labels = encoded[probe_idx]

        similarity = cosine_similarity_matrix(probe_emb, gallery_emb)

        results: Dict[str, float] = {}

        ranks: List[int] = list(getattr(self.config, "ranks", [1, 5, 10]))
        for n in ranks:
            results[f"rank_{n}"] = rank_n_accuracy(similarity, probe_labels, gallery_labels, n=n)

        results["mAP"] = mean_average_precision(similarity, probe_labels, gallery_labels)

        max_rank = min(int(getattr(self.config, "cmc_max_rank", 20)), gallery_emb.shape[0])
        cmc = cmc_curve(similarity, probe_labels, gallery_labels, max_rank=max_rank)
        for k in (1, 5, 10, 20):
            if k <= len(cmc):
                results[f"cmc@{k}"] = float(cmc[k - 1])

        return results
