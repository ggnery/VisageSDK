from typing import List, override
from config.transformation.base_transformation_config import BaseTransformationConfig

class LFWEvalTransformationConfig(BaseTransformationConfig):
    eval_std_normalize: List
    eval_mean_normalize: List

    @override
    def build_config(self) -> None:
        self.eval_std_normalize = self.config["normalize"]["std"]
        self.eval_mean_normalize = self.config["normalize"]["mean"]