from typing_extensions import override
from config.backbone.base_backbone_config import BaseBackboneConfig

class InceptionV4Config(BaseBackboneConfig):
    k: int
    l: int
    m: int
    n: int
        
    @override
    def build_config(self) -> None:
        self.k= self.config["k"]
        self.l= self.config["l"] 
        self.m= self.config["m"]
        self.n= self.config["n"]
        