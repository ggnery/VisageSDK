from typing import List, override

from torchvision import transforms
from config.transformation.base_transformation_config import TransformationConfig
from transformation.base_transformation import BaseTransformation


class VGGFace2TrainTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> List:
        train = cfg.train
        return [
            transforms.RandomHorizontalFlip(p=train["random_horizontal_flip"]),
            transforms.RandomRotation(degrees=train["random_rotation"]),
            transforms.ColorJitter(
                brightness=train["color_jitter"]["brightness"],
                contrast=train["color_jitter"]["contrast"],
                saturation=train["color_jitter"]["saturation"],
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=train["normalize"]["mean"], std=train["normalize"]["std"]),
        ]


class VGGFace2ValTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> List:
        val = cfg.val
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=val["normalize"]["mean"], std=val["normalize"]["std"]),
        ]
