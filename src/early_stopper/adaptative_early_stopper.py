import logging
from typing import override

from config.early_stopper.base_early_stopper_config import EarlyStopperConfig
from early_stopper.base_early_stopper import BaseEarlyStopper


class AdaptativeEarlyStopper(BaseEarlyStopper):
    def __init__(self, config: EarlyStopperConfig):
        super().__init__(config)

        self.base_patience = config.base_patience
        self.delta = config.delta
        self.patience_increase_ratio = config.patience_increase_ratio
        self.wait_count = 0
        self.best_score = None
        self.dynamic_patience = config.base_patience
        self.logger = logging.getLogger(__name__)

    @override
    def early_stop(self, val_loss: float) -> bool:
        if self.best_score is None or val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.wait_count = 0
            self.dynamic_patience = self.base_patience
        else:
            self.wait_count += 1
            if self.wait_count >= (self.base_patience * self.patience_increase_ratio):
                self.dynamic_patience += 1
            if self.wait_count >= self.dynamic_patience:
                self.logger.info("Stopping early due to lack of improvement.")
                return True
        return False
