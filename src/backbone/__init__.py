from registry import BACKBONES

from .base_backbone import BaseBackbone
from .dinov3 import DinoV3Backbone
from .inception_resnet_v1 import InceptionResNetV1
from .inception_resnet_v2 import InceptionResNetV2
from .inception_v4 import InceptionV4
from .megadescriptor import MegaDescriptorBackbone
from .mobilenetv3 import MobileNetV3
from .vit import LVFaceVisionTransformer

BACKBONES.register("inception_resnet_v1", InceptionResNetV1)
BACKBONES.register("inception_resnet_v2", InceptionResNetV2)
BACKBONES.register("inception_v4", InceptionV4)
BACKBONES.register("mobilenetv3", MobileNetV3)
BACKBONES.register("lvface_vit_b", LVFaceVisionTransformer)
BACKBONES.register("dinov3", DinoV3Backbone)
BACKBONES.register("megadescriptor", MegaDescriptorBackbone)

__all__ = [
    "BaseBackbone",
    "DinoV3Backbone",
    "InceptionResNetV1",
    "InceptionResNetV2",
    "InceptionV4",
    "MegaDescriptorBackbone",
    "MobileNetV3",
    "LVFaceVisionTransformer",
]
