from abc import ABC, abstractmethod

from config.early_stopper.base_early_stopper_config import EarlyStopperConfig


class BaseEarlyStopper(ABC):
    def __init__(self, config: EarlyStopperConfig):
        self.config = config

    @abstractmethod
    def early_stop(self, val_loss: float) -> bool:
        """Return True iff training should stop."""
        raise NotImplementedError()
