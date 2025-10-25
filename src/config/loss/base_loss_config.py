from typing import Dict
from config.base_config import BaseConfig


class BaseLossConfig(BaseConfig):
    device: str
    embedding_size: int
    num_classes: int
    
    def __init__(self, config_path: str, backbone_additional_info: Dict, dataset_additional_info: Dict) -> None:
        super().__init__(config_path)
        
        self.device = self.config["device"]
        self.embedding_size = backbone_additional_info["embedding_size"]
        self.num_classes = dataset_additional_info["num_classes"]
        
        self.build_config()