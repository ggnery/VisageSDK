from typing_extensions import override
from config.backbone.base_backbone_config import BaseBackboneConfig

class InceptionResNetV2Config(BaseBackboneConfig):
    k: int
    l: int
    m: int
    n: int
    dropout_keep: float
        
    @override
    def build_config(self) -> None:
        self.k= self.config["k"]
        self.l= self.config["l"] 
        self.m= self.config["m"]
        self.n= self.config["n"]
        self.dropout_keep = self.config["dropout_keep"]