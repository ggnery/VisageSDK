
from typing_extensions import override

from config.loss.base_loss_config import BaseLossConfig


class MarginCosineProductLossConfig(BaseLossConfig):
    s: float
    m: float
    
    @override
    def build_config(self) -> None:
        self.s = self.config["s"]
        self.m = self.config["m"]