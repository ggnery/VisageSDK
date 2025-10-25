from typing_extensions import override

from config.loss.base_loss_config import BaseLossConfig


class CenterLossConfig(BaseLossConfig):
    alpha: bool
    
    @override
    def build_config(self) -> None:
        self.use_bias = self.config["use_bias"]
        self.alpha = self.config["alpha"]