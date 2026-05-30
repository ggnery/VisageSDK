from abc import ABC, abstractmethod

from config.early_stopper_config import EarlyStopperConfig


class BaseEarlyStopper(ABC):
    def __init__(self, config: EarlyStopperConfig):
        self.config = config

    @abstractmethod
    def early_stop(self, score: float) -> bool:
        """Return True iff training should stop.

        `score` is the monitored metric (the trainer passes val_loss). Subclasses
        decide whether smaller or larger is better via their own `mode` config.
        """
        raise NotImplementedError()
