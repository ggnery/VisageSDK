from registry import EVAL_DATASETS

from .base_eval_dataset import BaseEvalDataset
from .identification_dataset import IdentificationDataset
from .lfw_pairs_dataset import LFWPairsDataset

EVAL_DATASETS.register("lfw_pairs", LFWPairsDataset)
EVAL_DATASETS.register("identification", IdentificationDataset)

__all__ = ["BaseEvalDataset", "LFWPairsDataset", "IdentificationDataset"]
