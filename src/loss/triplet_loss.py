from collections import defaultdict
from typing import override

import torch
import torch.nn.functional as F

from config.loss_config import LossConfig
from loss.base_loss import BaseLoss


class TripletLoss(BaseLoss):
    """FaceNet triplet loss with online semi-hard mining."""

    def __init__(self, loss_config: LossConfig):
        super().__init__(loss_config)
        self.margin = loss_config.margin

    @override
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor) -> tuple[torch.Tensor, dict]:
        embeddings = F.normalize(embeddings, p=2, dim=1)
        triplets, mining_info = self.mine_triplets(embeddings, y_true)

        if len(triplets) == 0:
            # No valid triplets (all-singleton class set). Tie zero loss to
            # `embeddings` so .backward() still traverses the graph (a leaf
            # zero tensor would be a silent no-op).
            zero_loss = (embeddings * 0.0).sum()
            # Populate the same keys the non-empty branch emits so downstream
            # logging in trainer (sum-aggregating numeric values) doesn't KeyError.
            mining_info.update(
                {
                    "avg_pos_distance": 0.0,
                    "avg_neg_distance": 0.0,
                    "active_triplets": 0,
                    "total_triplets": 0,
                }
            )
            return zero_loss, mining_info

        # Extract anchor, positive, negative indices
        anchor_indices, positive_indices, negative_indices = zip(*triplets, strict=True)

        # Get embeddings for anchors, positives, negatives
        anchor_embeddings = embeddings[list(anchor_indices)]
        positive_embeddings = embeddings[list(positive_indices)]
        negative_embeddings = embeddings[list(negative_indices)]

        # Compute distances
        pos_distances = torch.sum((anchor_embeddings - positive_embeddings) ** 2, dim=1)
        neg_distances = torch.sum((anchor_embeddings - negative_embeddings) ** 2, dim=1)

        # Triplet loss
        losses = F.relu(pos_distances - neg_distances + self.margin)
        loss = torch.mean(losses)

        # Mining stats only — the loss value itself is already logged as loss/train.
        mining_info.update(
            {
                "avg_pos_distance": torch.mean(pos_distances).item(),
                "avg_neg_distance": torch.mean(neg_distances).item(),
                "active_triplets": torch.sum(losses > 0).item(),
                "total_triplets": len(triplets),
            }
        )

        return loss, mining_info

    def mine_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor) -> tuple[list, dict]:
        """All anchor-positive pairs paired with a semi-hard negative each."""
        distances = torch.cdist(embeddings, embeddings, p=2) ** 2

        label_to_indices = defaultdict(list)
        for i, label in enumerate(labels):
            label_to_indices[label.item()].append(i)

        triplets = []
        mining_stats = {
            "total_pairs": 0,
            "valid_pairs": 0,
            "semi_hard_negatives": 0,
            "hard_negatives": 0,
            "easy_negatives": 0,
        }

        for _, indices in label_to_indices.items():
            if len(indices) < 2:
                continue

            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    anchor_idx = indices[i]
                    positive_idx = indices[j]

                    mining_stats["total_pairs"] += 1

                    ap_distance = distances[anchor_idx, positive_idx]
                    negative_idx = self.find_semi_hard_negative(anchor_idx, ap_distance, distances, labels)

                    if negative_idx is not None:
                        triplets.append((anchor_idx, positive_idx, negative_idx))
                        mining_stats["valid_pairs"] += 1

                        an_distance = distances[anchor_idx, negative_idx]
                        if an_distance > ap_distance and an_distance < ap_distance + self.margin:
                            mining_stats["semi_hard_negatives"] += 1
                        elif an_distance <= ap_distance:
                            mining_stats["hard_negatives"] += 1
                        else:
                            mining_stats["easy_negatives"] += 1

        return triplets, mining_stats

    def find_semi_hard_negative(
        self, anchor_idx: int, ap_distance: torch.Tensor, distances: torch.Tensor, labels: torch.Tensor
    ) -> int | None:
        """Return an index n with d(a,p) < d(a,n) < d(a,p) + margin, or None."""
        anchor_label = labels[anchor_idx]
        anchor_distances = distances[anchor_idx]

        different_identity_mask = labels != anchor_label
        candidate_indices = torch.where(different_identity_mask)[0]

        if len(candidate_indices) == 0:
            return None

        candidate_distances = anchor_distances[candidate_indices]
        semi_hard_mask = (candidate_distances > ap_distance) & (
            candidate_distances < ap_distance + self.margin
        )

        semi_hard_candidates = candidate_indices[semi_hard_mask]

        if len(semi_hard_candidates) > 0:
            return int(semi_hard_candidates[torch.randint(0, len(semi_hard_candidates), (1,))].item())

        # No semi-hard negative: drop this anchor-positive pair rather than
        # injecting a pure hard negative — the latter destabilizes early
        # training and contradicts the semi-hard mining protocol.
        return None
