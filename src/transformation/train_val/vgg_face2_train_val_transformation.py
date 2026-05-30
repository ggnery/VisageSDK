from typing import override

from torchvision import transforms

from config.transformation_config import TransformationConfig
from transformation.base_transformation import BaseTransformation


class VGGFace2TrainTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> list:
        train = cfg.train
        layers: list = []
        # RandomResizedCrop (optional) does a per-sample crop+resize, providing
        # scale + translation augmentation beyond what the static Resize prepended
        # by BaseTransformation gives. Useful for datasets with low intra-class
        # diversity where stronger spatial augmentation cuts overfitting.
        if "random_resized_crop" in train and train["random_resized_crop"]:
            rrc = train["random_resized_crop"]
            scale = tuple(rrc.get("scale", [0.8, 1.0]))
            ratio = tuple(rrc.get("ratio", [0.75, 1.3333]))
            layers.append(
                transforms.RandomResizedCrop(cfg.input_size, scale=scale, ratio=ratio)
            )
        layers.append(transforms.RandomHorizontalFlip(p=train["random_horizontal_flip"]))
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


class VGGFace2ValTransformation(BaseTransformation):
    @override
    def build_transformation(self, cfg: TransformationConfig) -> list:
        val = cfg.val
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=val["normalize"]["mean"], std=val["normalize"]["std"]),
        ]
