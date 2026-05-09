from typing import override
from config.transformation.base_transformation_config import TransformationConfig
from transformation.base_transformation import BaseTransformation

from torchvision import transforms


class LFWEvalTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig):
        norm = cfg.normalize
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
