"""Tests for individual blocks inside src/backbone/inception_v4.py."""

import pytest
import torch

from backbone.inception_v4 import InceptionA


class TestInceptionABlock:
    """B-6 regression: `InceptionA.branch_3` previously hardcoded
    `Conv2d(384, ...)` while accepting `in_channels` as a constructor
    argument. The block thus only worked when `in_channels == 384`
    (which happened to be the only call site in InceptionV4). Anyone
    sub-classing or re-using it with a different in_channels would
    crash deep in the forward pass with a confusing channel-mismatch
    error. The fix wires `in_channels` through to branch_3 too."""

    def test_default_384_channels_still_works(self):
        block = InceptionA(384)
        x = torch.randn(1, 384, 35, 35)
        out = block(x)
        # Each branch emits 96 channels → cat over dim=1 → 384.
        assert out.shape == (1, 384, 35, 35)

    @pytest.mark.parametrize("in_channels", [128, 256, 512])
    def test_alternate_in_channels_work(self, in_channels: int):
        """Pre-fix: any in_channels != 384 crashed in branch_3 with
        `expected input X channels, but got Y`. Post-fix: all sizes
        work as long as the input matches `in_channels`."""
        block = InceptionA(in_channels)
        x = torch.randn(1, in_channels, 35, 35)
        out = block(x)
        # Output channels remain 4 * 96 = 384 regardless of in_channels.
        assert out.shape == (1, 384, 35, 35)
