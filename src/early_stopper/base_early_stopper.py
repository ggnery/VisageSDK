from abc import ABC

from config.early_stopper.base_early_stopper_config import BaseEarlyStopperConfig
from trainer.training_context import EpochContext

class BaseEarlyStopper(ABC):
    def __init__(self, config: BaseEarlyStopperConfig):
        self.config = config
        
    def early_stop(self, epoch_ctx: EpochContext) -> bool:
        """Override this method to implement custom early stopping logic

        Args:
            val_loss (float): current validation loss

        Returns:
            bool: True if early stopping should be triggered, False otherwise
        """
        raise NotImplementedError()