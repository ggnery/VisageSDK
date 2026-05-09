from abc import ABC, abstractmethod
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone.base_backbone import BaseBackbone
from config.evaluator.base_evaluator_config import EvaluatorConfig
from dataset.eval.base_eval_dataset import BaseEvalDataset


class BaseEvaluator(ABC):
    """Encodes all images in the eval dataset, then runs metric computation."""

    def __init__(self, config: EvaluatorConfig, dataset: BaseEvalDataset, backbone: BaseBackbone):
        self.config = config
        self.dataset = dataset
        self.backbone = backbone
        self.device = torch.device(config.device)

    def encode(self) -> torch.Tensor:
        """Encode every image in the dataset; returns (N, embedding_size)."""
        loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=getattr(self.config, "num_workers", 0),
            pin_memory=False,
        )
        self.backbone.eval()
        embeddings = torch.empty(len(self.dataset), self.backbone.embedding_size)
        with torch.no_grad():
            for indices, images in tqdm(loader, desc=f"Encoding {type(self.dataset).__name__}"):
                images = images.to(self.device)
                emb = self.backbone(images)
                embeddings[indices] = emb.detach().cpu()
        return embeddings

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Run evaluation. Returns a dict of metric_name -> value.

        Most entries are scalar (float / int). Evaluators MAY include nested
        structures (e.g. `{"roc_curve": {"fpr": [...], "tpr": [...]}}`) for
        downstream plotting; eval.py and the GUI render scalars and
        non-scalars separately.
        """
        raise NotImplementedError
