from typing import override

from torchvision import transforms

from config.transformation.base_transformation_config import TransformationConfig
from transformation.base_transformation import BaseTransformation


class CasiaWebFaceTrainTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> list:
        train = cfg.train
        # Apply every augmentation the YAML actually declares, instead of
        # silently ignoring `random_rotation` / `color_jitter` keys (which
        # the old version did, leaving users to wonder why their listed
        # augmentations weren't influencing training).
        layers: list = [
            transforms.RandomHorizontalFlip(p=train["random_horizontal_flip"]),
        ]
        if "random_rotation" in train and train["random_rotation"]:
            layers.append(transforms.RandomRotation(degrees=train["random_rotation"]))
        if "color_jitter" in train and train["color_jitter"]:
            jitter = train["color_jitter"]
            layers.append(
                transforms.ColorJitter(
                    brightness=jitter.get("brightness", 0.0),
                    contrast=jitter.get("contrast", 0.0),
                    saturation=jitter.get("saturation", 0.0),
                )
            )
        layers.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=train["normalize"]["mean"], std=train["normalize"]["std"]),
            ]
        )
        return layers


class CasiaWebFaceValTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> list:
        val = cfg.val
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=val["normalize"]["mean"], std=val["normalize"]["std"]),
        ]
