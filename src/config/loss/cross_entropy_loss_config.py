from typing_extensions import override

from config.loss.base_loss_config import BaseLossConfig


class CrossEntropyLossConfig(BaseLossConfig):
    use_bias: bool
    label_smoothing: float
    
    @override
    def build_config(self) -> None:
        self.label_smoothing = self.config["label_smoothing"]
        self.use_bias = self.config["use_bias"]