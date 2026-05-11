"""Unit tests for the LVFace-compatible ViT backbone.

Architecture is parametrized via YAML; we run against a tiny variant
(depth=2, heads=2, embed=16, img=18) so each test stays under a second
even though the full LVFace-B has 113M params.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from backbone.vit import LVFaceVisionTransformer
from config.backbone.base_backbone_config import BackboneConfig


@pytest.fixture
def tiny_vit_config(tmp_path) -> BackboneConfig:
    """A small ViT config that still exercises every architectural axis:
    patch embedding, multiple transformer blocks, attention, MLP, the
    LayerNorm + flatten + feature head."""
    cfg_path = tmp_path / "vit.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "input_size": [18, 18],   # 18 / 9 = 2 → 4 tokens
                "embedding_size": 16,
                "device": "cpu",
                "patch_size": 9,
                "depth": 2,
                "num_heads": 2,
                "mlp_ratio": 4.0,
                "qkv_bias": False,
                "drop_rate": 0.0,
                "attn_drop_rate": 0.0,
                "drop_path_rate": 0.0,
                "mask_ratio": 0.0,
            }
        )
    )
    return BackboneConfig(str(cfg_path))


@pytest.fixture
def tiny_vit(tiny_vit_config) -> LVFaceVisionTransformer:
    return LVFaceVisionTransformer(tiny_vit_config)


# =============================================================================
# Construction + forward shape
# =============================================================================


class TestConstruction:
    def test_constructor_reads_all_keys(self, tiny_vit, tiny_vit_config):
        # Required BaseBackbone interface
        assert tiny_vit.embedding_size == 16
        assert tiny_vit.input_size == [18, 18]
        # ViT-specific structure mirrors the YAML
        assert len(tiny_vit.blocks) == 2
        assert tiny_vit.num_patches == 4  # (18/9) ** 2
        assert tiny_vit.patch_embed.num_patches == 4

    def test_forward_emits_embedding(self, tiny_vit):
        tiny_vit.eval()
        x = torch.randn(3, 3, 18, 18)
        with torch.no_grad():
            out = tiny_vit(x)
        assert out.shape == (3, 16)

    def test_forward_rejects_wrong_input_size(self, tiny_vit):
        """The PatchEmbed assertion is the framework's only safeguard
        against a Resize step accidentally being skipped — preserve it."""
        tiny_vit.eval()
        with pytest.raises(ValueError, match="Expected image size"):
            tiny_vit(torch.randn(1, 3, 32, 32))


# =============================================================================
# State-dict invariants for loading the published LVFace weights
# =============================================================================


class TestStateDictKeys:
    EXPECTED_TOP_LEVEL_PREFIXES = {
        "pos_embed",
        "mask_token",
        "patch_embed",
        "blocks",
        "norm",
        "feature",
    }

    def test_has_lvface_compatible_top_level_keys(self, tiny_vit):
        """The LVFace state_dict has these exact prefixes; any deviation
        breaks `strict=True` loads of the published checkpoint."""
        prefixes = {k.split(".", 1)[0] for k in tiny_vit.state_dict()}
        # Tiny config uses ln norm + present mask_token; check the contract.
        assert self.EXPECTED_TOP_LEVEL_PREFIXES.issubset(prefixes), (
            f"missing prefixes: {self.EXPECTED_TOP_LEVEL_PREFIXES - prefixes}"
        )

    def test_block_keys_match_lvface(self, tiny_vit):
        """Each Block exposes norm1/norm2 + attn.qkv/attn.proj + mlp.fc1/mlp.fc2.
        These names are referenced verbatim by the LoRA target_modules YAML."""
        block_keys = {
            k.split("blocks.0.", 1)[1]
            for k in tiny_vit.state_dict()
            if k.startswith("blocks.0.")
        }
        for required in (
            "norm1.weight", "norm1.bias",
            "norm2.weight", "norm2.bias",
            "attn.qkv.weight",
            "attn.proj.weight", "attn.proj.bias",
            "mlp.fc1.weight", "mlp.fc1.bias",
            "mlp.fc2.weight", "mlp.fc2.bias",
        ):
            assert required in block_keys, f"missing block key {required!r}"

    def test_feature_head_has_no_linear_bias(self, tiny_vit):
        """Both Linears in the feature head are bias=False (matches LVFace);
        a `feature.0.bias` or `feature.2.bias` key would break strict loads."""
        sd = tiny_vit.state_dict()
        assert "feature.0.weight" in sd
        assert "feature.0.bias" not in sd
        assert "feature.2.weight" in sd
        assert "feature.2.bias" not in sd


# =============================================================================
# Round-trip vs LVFace's published checkpoint
# =============================================================================


# Skip the full-size load if the official weights aren't available on this
# machine — the test is meaningful only when the published file is around.
LVFACE_CHECKPOINT = Path("./models/base/LVFace-B_Glint360K.pt")


class TestLVFaceCheckpointLoad:
    @pytest.mark.skipif(
        not LVFACE_CHECKPOINT.exists(),
        reason=f"{LVFACE_CHECKPOINT} not available locally",
    )
    def test_strict_load_against_lvface_b(self, tmp_path):
        """The whole point of this backbone: a strict load of the official
        LVFace-B weights must succeed with no key mismatches."""
        cfg_path = tmp_path / "lvface_b.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "input_size": [112, 112],
                    "embedding_size": 512,
                    "device": "cpu",
                    "patch_size": 9,
                    "depth": 24,
                    "num_heads": 8,
                    "mlp_ratio": 4.0,
                    "drop_path_rate": 0.0,
                    "mask_ratio": 0.0,
                }
            )
        )
        cfg = BackboneConfig(str(cfg_path))
        model = LVFaceVisionTransformer(cfg)
        sd = torch.load(LVFACE_CHECKPOINT, map_location="cpu", weights_only=False)
        # strict=True → raises on any missing/unexpected key
        model.load_state_dict(sd, strict=True)


# =============================================================================
# LoRA target compatibility
# =============================================================================


class TestLoRATargets:
    def test_named_modules_match_lora_yaml_targets(self, tiny_vit):
        """The lvface_lora_finetune.yaml targets `qkv`, `attn.proj`, `fc1`,
        `fc2` via PEFT's endswith match. Verify each name resolves to a
        Linear in every block — otherwise PEFT silently inserts zero
        adapters."""
        modules_by_name = dict(tiny_vit.named_modules())
        for block_idx in range(len(tiny_vit.blocks)):
            for leaf in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
                full = f"blocks.{block_idx}.{leaf}"
                assert full in modules_by_name, f"{full} missing"
                assert isinstance(modules_by_name[full], torch.nn.Linear)

    def test_apply_lora_wraps_every_block(self, tiny_vit):
        """End-to-end: applying LoRA on the tiny variant produces trainable
        params under every block (sanity check that endswith match hits)."""
        from tools.lora import apply_lora

        wrapped = apply_lora(
            tiny_vit, rank=2, alpha=4.0,
            target_modules=["qkv", "attn.proj", "fc1", "fc2"],
        )
        trainable_block_idxs = {
            n.split(".blocks.")[1].split(".")[0]
            for n, p in wrapped.named_parameters()
            if p.requires_grad and ".blocks." in n
        }
        # Every block index should appear as the owner of a LoRA tensor
        assert trainable_block_idxs == {"0", "1"}

    def test_target_attn_proj_does_not_match_patch_embed_proj(self, tiny_vit):
        """Regression: `target_modules=["proj"]` would catch BOTH `attn.proj`
        AND `patch_embed.proj` (Conv2d) via PEFT's `key.endswith(".<target>")`
        rule. The YAML uses the fully-qualified `attn.proj` to keep LoRA
        off the patch embedding — verify that contract holds."""
        from tools.lora import apply_lora

        wrapped = apply_lora(
            tiny_vit, rank=2, alpha=4.0,
            target_modules=["qkv", "attn.proj", "fc1", "fc2"],
        )
        leaked = [
            n for n, p in wrapped.named_parameters()
            if p.requires_grad and "patch_embed" in n
        ]
        assert not leaked, f"LoRA leaked into patch_embed: {leaked}"


# =============================================================================
# AMP / mixed precision compatibility
# =============================================================================


class TestAmpCompat:
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="AMP requires CUDA")
    def test_runs_under_autocast_fp16(self, tiny_vit_config):
        """Trainer wraps every forward in torch.amp.autocast. Make sure the
        ViT cooperates (no implicit fp32 ops that would OOM or fail)."""
        model = LVFaceVisionTransformer(tiny_vit_config).cuda().eval()
        x = torch.randn(2, 3, 18, 18, device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(x)
        assert out.shape == (2, 16)
        assert out.dtype in (torch.float16, torch.float32)
