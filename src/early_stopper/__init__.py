from registry import EARLY_STOPPERS

from .adaptive_early_stopper import AdaptiveEarlyStopper
from .base_early_stopper import BaseEarlyStopper

EARLY_STOPPERS.register("adaptive", AdaptiveEarlyStopper)

__all__ = ["BaseEarlyStopper", "AdaptiveEarlyStopper"]
