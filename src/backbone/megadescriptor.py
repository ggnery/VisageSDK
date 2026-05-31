"""MegaDescriptor backbone via timm.

MegaDescriptor (Cermak et al., 2024) is a Swin Transformer trained on
~1M animal re-identification images across hundreds of species. Distributed
by BVRA on the HuggingFace hub as timm-loadable checkpoints. License: MIT.

Available variants (load by HF id):
    BVRA/MegaDescriptor-T-224  → swin_tiny_patch4_window7_224    (28.3M, 768d)
    BVRA/MegaDescriptor-S-224  → swin_small_patch4_window7_224   (49.6M, 768d)
    BVRA/MegaDescriptor-B-224  → swin_base_patch4_window7_224    (87.8M, 1024d)
    BVRA/MegaDescriptor-L-224  → swin_large_patch4_window7_224   (196M,  1536d)
    BVRA/MegaDescriptor-L-384  → swin_large_patch4_window12_384  (228.8M, 1536d)

Pretraining domain (wildlife re-id) is much closer to cow-face re-id than
LVFace's human-face prior or InceptionResNetV1's VGGFace2 prior — expect
the strongest zero/few-shot transfer of any backbone in the framework.

The timm import is deferred to __init__ so module import (e.g. by registry
side-effect) doesn't require the dependency. `num_classes=0` tells timm to
drop the classifier and return the pooled feature vector directly, so the
forward path is just `feature_head(timm_model(x))`.
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn as nn

from backbone.base_backbone import BaseBackbone
from config.backbone_config import BackboneConfig


class MegaDescriptorBackbone(BaseBackbone):
    """MegaDescriptor (timm Swin) backbone.

    YAML keys (BackboneConfig):
        input_size: [H, W]      typically [224, 224] for T/S/B/L-224, [384, 384] for L-384.
                                Must match the Swin window grid — bumping a 224-trained
                                variant to 384 requires positional rescaling and is not
                                supported here.
        embedding_size: int     output dim. If equal to the timm model's `num_features`
                                (768 for Tiny/Small, 1024 for Base, 1536 for Large),
                                the feature head is nn.Identity(); otherwise a
                                Linear + BatchNorm1d bridges native_dim → embedding_size.
        device: str
        model_name: str         timm id, e.g. "hf-hub:BVRA/MegaDescriptor-T-224"

    LoRA targeting (see configs/trainer/megadescriptor_cosface_lora.yaml):
        timm Swin blocks expose attention as `attn.{qkv, proj}` and MLP as
        `mlp.{fc1, fc2}`. Use `attn.proj` (not bare `proj`) so PEFT doesn't
        wrap the patch-merging `downsample.reduction` Linears too.
    """

    def __init__(self, backbone_config: BackboneConfig) -> None:
        super().__init__(backbone_config)
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "MegaDescriptorBackbone requires `timm`. Install with "
                "`uv add timm` or `pip install timm>=1.0`."
            ) from e

        self.model_name = str(backbone_config.model_name)

        # num_classes=0 strips the wildlife-id classifier head and makes
        # `model(x)` return the global-pooled (B, num_features) feature vector.
        self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)

        native_dim = int(getattr(self.model, "num_features", 0))
        if native_dim <= 0:
            raise ValueError(
                f"Couldn't read num_features from {self.model_name!r} — timm model "
                "doesn't expose a feature dim. Pick a different MegaDescriptor variant."
            )
        self.native_dim = native_dim

        # Identity when native_dim matches embedding_size so PEFT
        # `modules_to_save: ["feature"]` is a no-op in that case.
        if self.embedding_size == native_dim:
            self.feature: nn.Module = nn.Identity()
        else:
            self.feature = nn.Sequential(
                nn.Linear(native_dim, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size, eps=1e-5),
            )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.model(x)  # (B, native_dim) — timm pre-pools when num_classes=0
        return self.feature(pooled)
