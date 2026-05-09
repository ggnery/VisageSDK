from registry import EARLY_STOPPERS

from .adaptative_early_stopper import AdaptativeEarlyStopper
from .base_early_stopper import BaseEarlyStopper

EARLY_STOPPERS.register("adaptative", AdaptativeEarlyStopper)

__all__ = ["BaseEarlyStopper", "AdaptativeEarlyStopper"]
