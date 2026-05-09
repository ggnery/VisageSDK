from typing import Dict
from config.base_config import BaseConfig


class LossConfig(BaseConfig):
    """Loss config — YAML params + injected backbone/dataset info.

    Required YAML keys: device (loss-specific keys read via attr lookup).
    Injected: embedding_size, num_classes.
    """
    embedding_size: int
    num_classes: int

    def __init__(self, config_path: str, backbone_info: Dict, dataset_info: Dict) -> None:
        super().__init__(config_path)
        self.embedding_size = backbone_info["embedding_size"]
        self.num_classes = dataset_info["num_classes"]


BaseLossConfig = LossConfig
