"""LVFace-compatible Vision Transformer backbone.

Architecturally identical to LVFace's `VisionTransformer`
(github.com/bytedance/LVFace, in turn forked from InsightFace), so the
official `LVFace-B_Glint360K.pt` state_dict loads with strict=True.

Differences vs the upstream code, all surface-level:

- Subclasses our `BaseBackbone` so the trainer pipeline picks it up.
- Reimplements `DropPath` / `to_2tuple` inline to avoid a `timm` runtime
  dependency for what is effectively 15 lines of code.
- Drops the upstream `torch.cuda.amp.autocast(...)` calls inside
  `Attention.forward`; the trainer already orchestrates AMP via its own
  context, and pinning it in the layer breaks bf16 / cpu paths.
- The MAE-style random masking branch is omitted from `forward`; it
  matters only for masked-pretraining and we use this backbone for
  inference / re-id fine-tuning. The `mask_token` Parameter is kept so
  the upstream state_dict still loads with strict=True.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import torch
import torch.nn as nn

from backbone.base_backbone import BaseBackbone
from config.backbone.base_backbone_config import BackboneConfig


def _to_2tuple(x: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(x, tuple):
        return x
    return (x, x)


class _DropPath(nn.Module):
    """Per-sample stochastic depth — drops the residual branch with prob `p`."""

    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        # Broadcast a per-sample mask over the remaining dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * (mask / keep)


class _Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU6,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        # Match upstream key names exactly: qkv, proj.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        # qkv: (B, N, 3 * C) → (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.ReLU6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = _Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.mlp = _Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer, drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    """Strided Conv2d patch embedding — kernel=stride=patch_size."""

    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        ih, iw = _to_2tuple(img_size)
        ph, pw = _to_2tuple(patch_size)
        # Floor division mirrors LVFace exactly: image gets cropped if it
        # doesn't tile evenly. With img=112, patch=9 → 12*12 = 144 patches
        # (the trailing 4 pixels in each dim are silently dropped by stride).
        self.num_patches = (ih // ph) * (iw // pw)
        self.img_size = (ih, iw)
        self.patch_size = (ph, pw)
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(ph, pw), stride=(ph, pw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if (h, w) != self.img_size:
            raise ValueError(
                f"Expected image size {self.img_size}, got ({h}, {w}). "
                "Resize via the transformation YAML to match backbone.input_size."
            )
        # (B, C, H, W) → (B, embed_dim, gh, gw) → (B, gh*gw, embed_dim)
        return self.proj(x).flatten(2).transpose(1, 2)


class LVFaceVisionTransformer(BaseBackbone):
    """ViT backbone matching LVFace's published checkpoints.

    Constructor reads from BackboneConfig YAML keys:
        input_size: [H, W]              (e.g. [112, 112])
        embedding_size: int             (output dim, e.g. 512)
        patch_size: int                 (default 9)
        depth: int                      (default 24 = LVFace-B)
        num_heads: int                  (default 8)
        mlp_ratio: float                (default 4.0)
        qkv_bias: bool                  (default False)
        drop_rate: float                (default 0.0)
        attn_drop_rate: float           (default 0.0)
        drop_path_rate: float           (default 0.0)
    """

    def __init__(self, backbone_config: BackboneConfig) -> None:
        super().__init__(backbone_config)

        # `input_size` arrives as a list (from YAML) of length 2; collapse it
        # into a (h, w) tuple before passing into PatchEmbed, which is typed
        # to accept either `int` or a 2-tuple.
        if isinstance(self.input_size, int):
            ih = iw = self.input_size
        else:
            ih, iw = int(self.input_size[0]), int(self.input_size[1])
        patch_size = int(getattr(backbone_config, "patch_size", 9))
        depth = int(getattr(backbone_config, "depth", 24))
        num_heads = int(getattr(backbone_config, "num_heads", 8))
        mlp_ratio = float(getattr(backbone_config, "mlp_ratio", 4.0))
        qkv_bias = bool(getattr(backbone_config, "qkv_bias", False))
        drop_rate = float(getattr(backbone_config, "drop_rate", 0.0))
        attn_drop_rate = float(getattr(backbone_config, "attn_drop_rate", 0.0))
        drop_path_rate = float(getattr(backbone_config, "drop_path_rate", 0.0))

        embed_dim = self.embedding_size
        self.patch_embed = _PatchEmbed(
            img_size=(ih, iw), patch_size=patch_size, in_channels=3, embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        # Match upstream parameter names (pos_embed, mask_token) so the
        # official state_dict loads with strict=True.
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Linearly increasing per-layer drop_path, like the upstream rule.
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # Feature head: flatten all tokens → (B, num_patches * embed_dim) and
        # project down to embed_dim with two BatchNorm1d sandwiches. Same as
        # LVFace; using bias=False on both Linears matches the published
        # checkpoint exactly (no `feature.0.bias` / `feature.2.bias` keys).
        self.feature = nn.Sequential(
            nn.Linear(embed_dim * num_patches, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, eps=2e-5),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, eps=2e-5),
        )

        # Match upstream init exactly: pos_embed is truncated normal,
        # mask_token is regular normal. Distinction matters only for
        # from-scratch training (loading the LVFace checkpoint overwrites
        # both), but mismatched defaults make exact reproductions of the
        # upstream's training trajectory impossible.
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Flatten all token embeddings, then run the feature head.
        x = x.reshape(b, self.num_patches * self.embedding_size)
        x = self.feature(x)
        return x
