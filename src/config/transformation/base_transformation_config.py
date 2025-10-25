from typing import Dict
from config.base_config import BaseConfig

class BaseTransformationConfig(BaseConfig):
    def __init__(self, config_path: str, backbone_additional_info: Dict) -> None:
        super().__init__(config_path)
        
        self.input_size = backbone_additional_info["input_size"]
        
        self.build_config()