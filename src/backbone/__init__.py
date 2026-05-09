from registry import BACKBONES

from .base_backbone import BaseBackbone
from .inception_resnet_v1 import InceptionResNetV1
from .inception_resnet_v2 import InceptionResNetV2
from .inception_v4 import InceptionV4
from .mobilenetv3 import MobileNetV3

BACKBONES.register("inception_resnet_v1", InceptionResNetV1)
BACKBONES.register("inception_resnet_v2", InceptionResNetV2)
BACKBONES.register("inception_v4", InceptionV4)
BACKBONES.register("mobilenetv3", MobileNetV3)

__all__ = [
    "BaseBackbone",
    "InceptionResNetV1",
    "InceptionResNetV2",
    "InceptionV4",
    "MobileNetV3",
]
