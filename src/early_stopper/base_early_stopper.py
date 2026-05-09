from abc import ABC

from config.early_stopper.base_early_stopper_config import EarlyStopperConfig


class BaseEarlyStopper(ABC):
    def __init__(self, config: EarlyStopperConfig):
        self.config = config

    def early_stop(self, val_loss: float) -> bool:
        """Override to implement custom early stopping logic.

        Args:
            val_loss: current validation loss

        Returns:
            True if training should stop, False otherwise.
        """
        raise NotImplementedError()
