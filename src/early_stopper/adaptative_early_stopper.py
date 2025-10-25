import logging
from config.early_stopper.adaptative_early_stopper_config import AdaptativeEarlyStopperConfig
from early_stopper.base_early_stopper import BaseEarlyStopper
from typing import override

from trainer.training_context import EpochContext

class AdaptativeEarlyStopper(BaseEarlyStopper):
    def __init__(self, config: AdaptativeEarlyStopperConfig):
        super().__init__(config)
        
        self.base_patience = config.base_patience
        self.delta = config.delta
        self.patience_increase_ratio = config.patience_increase_ratio
        self.wait_count = 0
        self.best_score = None
        self.dynamic_patience = config.base_patience
        self.logger = logging.getLogger(__name__)
    
    @override
    def early_stop(self, epoch_ctx: EpochContext) -> bool:
        if self.best_score is None or epoch_ctx.val_loss < self.best_score - self.delta:
            self.best_score = epoch_ctx.val_loss
            self.wait_count = 0
            self.dynamic_patience = self.base_patience  # reset to base
        else:
            self.wait_count += 1
            # Adjust patience if improvement is near
            if self.wait_count >= (self.base_patience * self.patience_increase_ratio):
                self.dynamic_patience += 1
            if self.wait_count >= self.dynamic_patience:
                self.logger.info("Stopping early due to lack of improvement.")
                return True  # Signal to stop training
        return False