from typing_extensions import override
from config.backbone.base_backbone_config import BaseBackboneConfig

class InceptionResNetV1Config(BaseBackboneConfig):
    dropout_keep: float
    
    @override
    def build_config(self) -> None:
        self.dropout_keep = self.config["dropout_keep"]
        