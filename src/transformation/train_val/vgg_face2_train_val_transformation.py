from typing import List, override

from torchvision import transforms
from config.transformation.train_val.vgg_face2_train_val_transformation_config import VGGFace2TrainValTransformationConfig
from transformation.base_transformation import BaseTransformation

class VGGFace2TrainTransformation(BaseTransformation):
    @override
    def build_transformation(self, transformation_config: VGGFace2TrainValTransformationConfig) -> List:
        return [
            transforms.RandomHorizontalFlip(p=transformation_config.train_random_horizontal_flip),
            transforms.RandomRotation(degrees=transformation_config.train_random_rotation),
            transforms.ColorJitter(
                brightness=transformation_config.train_brightness, 
                contrast=transformation_config.train_contrast, 
                saturation=transformation_config.train_saturation
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=transformation_config.train_mean_normalize, std=transformation_config.train_std_normalize)
        ]

class VGGFace2ValTransformation(BaseTransformation):
    @override
    def build_transformation(self, transformation_config: VGGFace2TrainValTransformationConfig) -> transforms.Compose:
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=transformation_config.val_mean_normalize, std=transformation_config.val_std_normalize)
        ]