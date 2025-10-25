from pathlib import Path
from typing import Dict, List

from config.base_config import BaseConfig

class BaseTrainValDatasetConfig(BaseConfig):
    train_dir: Path
    val_dir: Path
    num_classes: int
    input_size: List[int]
    
    def __init__(self, config_path: str, backbone_additional_info: Dict) -> None:
        super().__init__(config_path)
        
        self.train_dir = self.config["train_dir"]
        self.val_dir = self.config["val_dir"]
        self.num_classes = self.config["num_classes"]
        self.input_size = backbone_additional_info["input_size"]
        
        self.build_config()
    