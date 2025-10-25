from typing import List
from config.base_config import BaseConfig

class BaseBackboneConfig(BaseConfig):
    input_size: List
    embedding_size: int
    device: str
    
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)    
        
        self.input_size = self.config["input_size"]
        self.embedding_size = self.config["embedding_size"]
        self.device = self.config["device"]
        
        self.build_config()