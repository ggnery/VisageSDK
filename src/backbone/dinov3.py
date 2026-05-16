"""DINOv3 vision transformer backbone via HuggingFace transformers.

Loads any DINOv3 variant by HF id (`facebook/dinov3-vit{s,b,l,h}16-...`),
projects the CLS or mean-pooled patch token to `embedding_size`. Designed
for LoRA fine-tuning: PEFT adapters wrap the underlying `self.model.*`
linears; the framework-side `feature` head is `modules_to_save`.

The transformers import is deferred to __init__ so this module can be
imported (e.g. by registry side-effect) without the dependency installed —
construction is what actually requires it.
"""

from __future__ import annotations

from typing import override

import torch
import torch.nn as nn

from backbone.base_backbone import BaseBackbone
from config.backbone_config import BackboneConfig


class DinoV3Backbone(BaseBackbone):
    """DINOv3 backbone.

    YAML keys (BackboneConfig):
        input_size: [H, W]      typically [224, 224]; DINOv3 accepts any
                                multiple of patch_size=16 (positional embeds
                                are interpolated for non-default sizes).
        embedding_size: int     output dim. If it matches the native hidden
                                size, no projection head is added; otherwise
                                a Linear + BatchNorm1d head bridges the gap.
        device: str
        model_name: str         HF id, e.g.
                                "facebook/dinov3-vitb16-pretrain-lvd1689m"
        token: "cls" | "mean"   default "cls". "mean" averages patch tokens
                                (skips CLS + register tokens).

    LoRA targeting (configs/trainer/dinov3_lora_finetune.yaml):
        DINOv3-S/B/L use a plain MLP (`fc1`/`fc2`); DINOv3-H uses SwiGLU
        (`w1`/`w2`/`w3`). Verify the exact names for your variant with:
            from backbone import DinoV3Backbone
            bb = DinoV3Backbone(cfg)
            for n, m in bb.named_modules():
                if isinstance(m, nn.Linear): print(n)
    """

    def __init__(self, backbone_config: BackboneConfig) -> None:
        super().__init__(backbone_config)
        try:
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "DinoV3Backbone requires `transformers`. Install with "
                "`uv add transformers` or `pip install transformers>=4.50`."
            ) from e

        self.model_name = str(backbone_config.model_name)
        self.token_strategy = str(getattr(backbone_config, "token", "cls"))
        if self.token_strategy not in {"cls", "mean"}:
            raise ValueError(
                f"backbone.token must be 'cls' or 'mean', got {self.token_strategy!r}"
            )

        # HF download/cache on first use (HF_HOME or ~/.cache/huggingface/).
        self.model = AutoModel.from_pretrained(self.model_name)

        # Read native hidden size from the loaded HF config so the user
        # doesn't have to hardcode it per variant.
        hf_cfg = self.model.config
        native_dim = int(getattr(hf_cfg, "hidden_size", 0))
        if native_dim <= 0:
            raise ValueError(
                f"Couldn't read hidden_size from {self.model_name!r}'s config — "
                "DinoV3Backbone needs a transformer-style config exposing hidden_size."
            )
        self.native_dim = native_dim
        # DINOv3 introduced register tokens (Darcet et al., 2024); HF exposes
        # the count on the config. Fall back to 0 for variants that don't use them.
        self.num_register_tokens = int(getattr(hf_cfg, "num_register_tokens", 0))

        # Bridge native_dim → embedding_size when they differ. Identity when
        # they match so PEFT `modules_to_save: ["feature"]` is still a no-op
        # in that case (and the user can drop it from the YAML).
        if self.embedding_size == native_dim:
            self.feature: nn.Module = nn.Identity()
        else:
            self.feature = nn.Sequential(
                nn.Linear(native_dim, self.embedding_size, bias=False),
                nn.BatchNorm1d(self.embedding_size, eps=1e-5),
            )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=x)
        hidden = outputs.last_hidden_state  # (B, 1 + n_register + n_patches, dim)
        if self.token_strategy == "cls":
            pooled = hidden[:, 0]
        else:
            # Skip CLS (index 0) and any register tokens before averaging patches.
            pooled = hidden[:, 1 + self.num_register_tokens :].mean(dim=1)
        return self.feature(pooled)
