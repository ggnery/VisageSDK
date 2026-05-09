from registry import EARLY_STOPPERS

from .base_early_stopper import BaseEarlyStopper
from .adaptative_early_stopper import AdaptativeEarlyStopper

EARLY_STOPPERS.register("adaptative", AdaptativeEarlyStopper)

__all__ = ["BaseEarlyStopper", "AdaptativeEarlyStopper"]
