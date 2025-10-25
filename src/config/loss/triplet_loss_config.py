from typing_extensions import override

from config.loss.base_loss_config import BaseLossConfig


class TripletLossConfig(BaseLossConfig):
    @override
    def build_config(self) -> None:
        self.margin = self.config["margin"]