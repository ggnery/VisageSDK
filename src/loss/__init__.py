from registry import LOSSES

from .base_loss import BaseLoss
from .center_loss import CenterLoss
from .cross_entropy_loss import CrossEntropyLoss
from .margin_cosine_product_loss import MarginCosineProductLoss
from .triplet_loss import TripletLoss

LOSSES.register("triplet", TripletLoss)
LOSSES.register("center", CenterLoss)
LOSSES.register("cross_entropy", CrossEntropyLoss)
LOSSES.register("margin_cosine", MarginCosineProductLoss)

__all__ = [
    "BaseLoss",
    "TripletLoss",
    "CenterLoss",
    "CrossEntropyLoss",
    "MarginCosineProductLoss",
]
