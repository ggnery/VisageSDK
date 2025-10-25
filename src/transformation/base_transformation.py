from abc import ABC
from typing import List

from torchvision import transforms
from config.transformation.base_transformation_config import BaseTransformationConfig

class BaseTransformation(ABC):
    transform: transforms.Compose
    
    def __init__(self, transformation_config: BaseTransformationConfig):
        self.transform = transforms.Compose([transforms.Resize(transformation_config.input_size)] + self.build_transformation(transformation_config))
        
    def build_transformation(self, transformation_config: BaseTransformationConfig) -> List:
        """Programer should override this method to build a custom transformation

        Args:
            config (BaseTransformationConfig): config

        Returns:
            List: list of transformations
        """
        return []