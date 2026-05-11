"""Tests for transformation YAML parsing and defensive behavior."""

import yaml

from config.transformation.base_transformation_config import TransformationConfig
from transformation.train_val.casia_webface_train_val_transformation import (
    CasiaWebFaceTrainTransformation,
)
from transformation.train_val.vgg_face2_train_val_transformation import (
    VGGFace2TrainTransformation,
)


def _make_cfg(tmp_path, train_block: dict) -> TransformationConfig:
    data = {
        "train": train_block,
        "val": {"normalize": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]}},
    }
    p = tmp_path / "tx.yaml"
    p.write_text(yaml.safe_dump(data))
    return TransformationConfig(str(p), backbone_info={"input_size": [112, 112]})


class TestVGGFace2TrainTransformationDefensive:
    """B-4 regression: `VGGFace2TrainTransformation` used to direct-index
    YAML keys (`train["color_jitter"]["brightness"]`), crashing with
    KeyError when a user trimmed `color_jitter` or `random_rotation` out
    of the YAML. The sibling `CasiaWebFaceTrainTransformation` was
    already defensive — this test pins both to the same contract."""

    def _full_block(self) -> dict:
        return {
            "normalize": {"std": [0.5, 0.5, 0.5], "mean": [0.5, 0.5, 0.5]},
            "random_horizontal_flip": 0.5,
            "random_rotation": 10,
            "color_jitter": {"brightness": 0.1, "contrast": 0.1, "saturation": 0.1},
        }

    def test_full_block_builds(self, tmp_path):
        cfg = _make_cfg(tmp_path, self._full_block())
        tx = VGGFace2TrainTransformation(cfg)
        # 4 user-defined layers + Resize prepended by BaseTransformation +
        # ToTensor + Normalize. Just sanity-check the pipeline materializes.
        assert tx.transform is not None

    def test_missing_color_jitter_is_tolerated(self, tmp_path):
        block = self._full_block()
        del block["color_jitter"]
        cfg = _make_cfg(tmp_path, block)
        # Pre-fix: KeyError. Post-fix: builds cleanly without ColorJitter.
        tx = VGGFace2TrainTransformation(cfg)
        assert tx.transform is not None

    def test_missing_random_rotation_is_tolerated(self, tmp_path):
        block = self._full_block()
        del block["random_rotation"]
        cfg = _make_cfg(tmp_path, block)
        tx = VGGFace2TrainTransformation(cfg)
        assert tx.transform is not None

    def test_both_optional_keys_missing(self, tmp_path):
        block = self._full_block()
        del block["color_jitter"]
        del block["random_rotation"]
        cfg = _make_cfg(tmp_path, block)
        tx = VGGFace2TrainTransformation(cfg)
        assert tx.transform is not None

    def test_casia_and_vgg_share_defensive_behavior(self, tmp_path):
        """Both train transforms must accept identical minimal YAML."""
        block = {
            "normalize": {"std": [0.5, 0.5, 0.5], "mean": [0.5, 0.5, 0.5]},
            "random_horizontal_flip": 0.5,
        }
        cfg = _make_cfg(tmp_path, block)
        CasiaWebFaceTrainTransformation(cfg)
        VGGFace2TrainTransformation(cfg)


class TestVGGFace2EndToEndApply:
    """Make sure the post-fix pipeline still produces a tensor when fed
    a PIL image — guards against accidentally breaking the augment chain."""

    def test_forward_apply(self, tmp_path):
        from PIL import Image

        cfg = _make_cfg(
            tmp_path,
            {
                "normalize": {"std": [0.5, 0.5, 0.5], "mean": [0.5, 0.5, 0.5]},
                "random_horizontal_flip": 0.0,
            },
        )
        tx = VGGFace2TrainTransformation(cfg)
        img = Image.new("RGB", (160, 160), color=(128, 128, 128))
        out = tx.transform(img)
        # Resize → ToTensor → Normalize; shape is (3, 112, 112) from cfg.input_size.
        import torch

        assert isinstance(out, torch.Tensor)
        assert out.shape == (3, 112, 112)
