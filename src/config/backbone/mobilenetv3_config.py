from typing_extensions import override
from config.backbone.base_backbone_config import BaseBackboneConfig


class MobileNetV3Config(BaseBackboneConfig):
    model_size: str
    width_mult: float
    reduced_tail: bool
    dilated: bool
    dropout: float
    
    @override
    def build_config(self) -> None:
        self.model_size = self.config["model_size"]
        self.width_mult = self.config["width_mult"]
        self.reduced_tail = self.config["reduced_tail"]
        self.dilated = self.config["dilated"]