from typing import List, override

from torchvision import transforms
from config.transformation.base_transformation_config import TransformationConfig
from transformation.base_transformation import BaseTransformation


class CasiaWebFaceTrainTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> List:
        train = cfg.train
        return [
            transforms.RandomHorizontalFlip(p=train["random_horizontal_flip"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=train["normalize"]["mean"], std=train["normalize"]["std"]),
        ]


class CasiaWebFaceValTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> List:
        val = cfg.val
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=val["normalize"]["mean"], std=val["normalize"]["std"]),
        ]
