from registry import TRANSFORMATIONS

from .base_transformation import BaseTransformation
from .train_val.vgg_face2_train_val_transformation import (
    VGGFace2TrainTransformation,
    VGGFace2ValTransformation,
)
from .train_val.casia_webface_train_val_transformation import (
    CasiaWebFaceTrainTransformation,
    CasiaWebFaceValTransformation,
)
from .eval.lfw_eval_transformation import LFWEvalTransformation

TRANSFORMATIONS.register("vgg_face2_train", VGGFace2TrainTransformation)
TRANSFORMATIONS.register("vgg_face2_val", VGGFace2ValTransformation)
TRANSFORMATIONS.register("casia_webface_train", CasiaWebFaceTrainTransformation)
TRANSFORMATIONS.register("casia_webface_val", CasiaWebFaceValTransformation)
TRANSFORMATIONS.register("lfw_eval", LFWEvalTransformation)

__all__ = ["BaseTransformation"]
