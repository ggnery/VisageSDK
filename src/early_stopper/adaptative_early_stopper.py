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
            # Grow `dynamic_patience` ONCE when `wait_count` crosses the
            # increase threshold, instead of every epoch. The pre-fix code
            # incremented every epoch, so with `patience_increase_ratio < 1`
            # (the YAML default is 0.8) `dynamic_patience` and `wait_count`
            # grew in lockstep and the stop condition `wait_count >=
            # dynamic_patience` was never reached — the stopper silently
            # never triggered.
            threshold = int(self.base_patience * self.patience_increase_ratio)
            if self.wait_count == threshold and self.dynamic_patience == self.base_patience:
                # One-shot extension: grant a "second chance" window of length
                # `base_patience - threshold` so total grace ≈ 2 × base_patience
                # when ratio < 1, or grace = base_patience when ratio >= 1.
                extra = max(1, self.base_patience - threshold)
                self.dynamic_patience = self.base_patience + extra
            if self.wait_count >= self.dynamic_patience:
                self.logger.info("Stopping early due to lack of improvement.")
                return True
        return False
