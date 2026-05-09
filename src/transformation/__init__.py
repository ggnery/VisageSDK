from registry import (
    TRAIN_TRANSFORMATIONS,
    VAL_TRANSFORMATIONS,
    EVAL_TRANSFORMATIONS,
)

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

TRAIN_TRANSFORMATIONS.register("vgg_face2", VGGFace2TrainTransformation)
VAL_TRANSFORMATIONS.register("vgg_face2", VGGFace2ValTransformation)
TRAIN_TRANSFORMATIONS.register("casia_webface", CasiaWebFaceTrainTransformation)
VAL_TRANSFORMATIONS.register("casia_webface", CasiaWebFaceValTransformation)
EVAL_TRANSFORMATIONS.register("lfw", LFWEvalTransformation)

__all__ = ["BaseTransformation"]
