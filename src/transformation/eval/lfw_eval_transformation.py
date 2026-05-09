from typing import override

from torchvision import transforms

from config.transformation.base_transformation_config import TransformationConfig
from transformation.base_transformation import BaseTransformation


class LFWEvalTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig):
        norm = cfg.normalize
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=norm["mean"], std=norm["std"]),
        ]
