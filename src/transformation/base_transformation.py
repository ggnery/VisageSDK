from abc import ABC
from typing import List

from torchvision import transforms

from config.transformation.base_transformation_config import TransformationConfig


class BaseTransformation(ABC):
    transform: transforms.Compose

    def __init__(self, transformation_config: TransformationConfig):
        self.transform = transforms.Compose(
            [transforms.Resize(transformation_config.input_size)]
            + self.build_transformation(transformation_config)
        )

    def build_transformation(self, transformation_config: TransformationConfig) -> List:
        """Override to build a custom transformation pipeline.

        Args:
            transformation_config: config object exposing YAML keys as attrs.
        Returns:
            list of torchvision transforms appended after the resize.
        """
        return []
