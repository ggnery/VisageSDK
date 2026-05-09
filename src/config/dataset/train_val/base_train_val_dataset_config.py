from typing import Dict, List
from config.base_config import BaseConfig


class TrainValDatasetConfig(BaseConfig):
    """Train/val dataset config.

    Required YAML keys: train_dir, val_dir, num_classes.
    Injected: input_size (from backbone).
    """
    input_size: List[int]

    def __init__(self, config_path: str, backbone_info: Dict) -> None:
        super().__init__(config_path)
        self.input_size = backbone_info["input_size"]
