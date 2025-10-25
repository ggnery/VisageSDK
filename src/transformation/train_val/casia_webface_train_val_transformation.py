from typing import List, override

from torchvision import transforms
from config.transformation.train_val.casia_webface_train_val_transformation_config import CasiaWebFaceTrainValTransformationConfig
from transformation.base_transformation import BaseTransformation

class CasiaWebFaceTrainTransformation(BaseTransformation):
    @override
    def build_transformation(self, transformation_config: CasiaWebFaceTrainValTransformationConfig) -> List:
        return [
            transforms.RandomHorizontalFlip(p=transformation_config.train_random_horizontal_flip),
            transforms.ToTensor(),
            transforms.Normalize(mean=transformation_config.train_mean_normalize, std=transformation_config.train_std_normalize)
        ]

class CasiaWebFaceValTransformation(BaseTransformation):
    @override
    def build_transformation(self, transformation_config: CasiaWebFaceTrainValTransformationConfig) -> transforms.Compose:
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=transformation_config.val_mean_normalize, std=transformation_config.val_std_normalize)
        ]