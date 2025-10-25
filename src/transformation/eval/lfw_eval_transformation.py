from typing import override
from config.transformation.eval.lfw_eval_transformation import LFWEvalTransformationConfig
from transformation.base_transformation import BaseTransformation

from torchvision import transforms

class LFWEvalTransformation(BaseTransformation):
    
    @override
    def build_transformation(self, transformation_config: LFWEvalTransformationConfig):
        return [
            transforms.ToTensor(),
            transforms.Normalize(mean=transformation_config.eval_mean_normalize, std=transformation_config.eval_std_normalize)
        ]