import logging
from typing import override

from config.early_stopper_config import EarlyStopperConfig
from early_stopper.base_early_stopper import BaseEarlyStopper


class AdaptiveEarlyStopper(BaseEarlyStopper):
    def __init__(self, config: EarlyStopperConfig):
        super().__init__(config)

        self.base_patience = config.base_patience
        self.delta = config.delta
        self.patience_increase_ratio = config.patience_increase_ratio
        self.mode = str(getattr(config, "mode", "min"))
        if self.mode not in {"min", "max"}:
            raise ValueError(f"early_stopper.mode must be 'min' or 'max', got {self.mode!r}")
        self.wait_count = 0
        self.best_score = None
        self.dynamic_patience = config.base_patience
        self.logger = logging.getLogger(__name__)

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.delta
        return score > self.best_score + self.delta

    @override
    def early_stop(self, score: float) -> bool:
        if self._is_improvement(score):
            self.best_score = score
            self.wait_count = 0
            self.dynamic_patience = self.base_patience
        else:
            self.wait_count += 1
            # Grow dynamic_patience once when wait_count crosses the threshold
            # (one-shot "second chance" — incrementing every epoch never stops).
            threshold = int(self.base_patience * self.patience_increase_ratio)
            if self.wait_count == threshold and self.dynamic_patience == self.base_patience:
                extra = max(1, self.base_patience - threshold)
                self.dynamic_patience = self.base_patience + extra
            if self.wait_count >= self.dynamic_patience:
                self.logger.info("Stopping early due to lack of improvement.")
                return True
        return False
