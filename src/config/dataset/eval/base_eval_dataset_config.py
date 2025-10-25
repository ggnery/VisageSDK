from typing import Dict, List
from config.base_config import BaseConfig

class BaseEvalDatasetConfig(BaseConfig):
    eval_dir: str
    input_size: List[int]

    def __init__(self, config_path: str, backbone_additional_info: Dict) -> None:
        super().__init__(config_path)

        self.eval_dir = self.config["eval_dir"]
        self.input_size = backbone_additional_info["input_size"]
        
        self.build_config()
