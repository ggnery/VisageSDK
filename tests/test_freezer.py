"""Tests for tools.freezer."""

import pytest
import torch.nn as nn

from tools.freezer import (
    freeze_by_patterns,
    freeze_summary,
    log_freeze_state,
    unfreeze_by_patterns,
)


@pytest.fixture
def tiny_module():
    """Module with named parameters for pattern matching."""
    return nn.Sequential(
        nn.Linear(8, 4),
        nn.Linear(4, 2),
        nn.Linear(2, 1),
    )


class TestFreezeByPatterns:
    def test_freeze_with_patterns_matches_subset(self, tiny_module):
        frozen = freeze_by_patterns(tiny_module, patterns=["0.weight", "0.bias"])
        assert sorted(frozen) == ["0.bias", "0.weight"]
        # Other params remain trainable
        for name, p in tiny_module.named_parameters():
            if name in frozen:
                assert not p.requires_grad
            else:
                assert p.requires_grad

    def test_freeze_with_glob(self, tiny_module):
        frozen = freeze_by_patterns(tiny_module, patterns=["0.*"])
        assert "0.weight" in frozen and "0.bias" in frozen
        assert "1.weight" not in frozen

    def test_freeze_with_except_inverts(self, tiny_module):
        frozen = freeze_by_patterns(tiny_module, except_patterns=["2.*"])
        # Everything except `2.*` should be frozen
        assert "0.weight" in frozen
        assert "0.bias" in frozen
        assert "1.weight" in frozen
        assert "1.bias" in frozen
        assert "2.weight" not in frozen
        assert "2.bias" not in frozen

    def test_both_patterns_and_except_raises(self, tiny_module):
        with pytest.raises(ValueError, match="exactly one"):
            freeze_by_patterns(tiny_module, patterns=["a"], except_patterns=["b"])

    def test_neither_raises(self, tiny_module):
        with pytest.raises(ValueError, match="exactly one"):
            freeze_by_patterns(tiny_module)


class TestUnfreezeByPatterns:
    def test_unfreeze_only_targets_frozen_matching(self, tiny_module):
        # Start fully frozen
        freeze_by_patterns(tiny_module, patterns=["*"])
        unfrozen = unfreeze_by_patterns(tiny_module, ["1.*"])
        assert sorted(unfrozen) == ["1.bias", "1.weight"]
        # Layer 0 stays frozen
        assert not tiny_module[0].weight.requires_grad
        # Layer 1 unfrozen
        assert tiny_module[1].weight.requires_grad

    def test_unfreeze_no_match_returns_empty(self, tiny_module):
        freeze_by_patterns(tiny_module, patterns=["*"])
        unfrozen = unfreeze_by_patterns(tiny_module, ["nonexistent"])
        assert unfrozen == []


class TestFreezeSummary:
    def test_counts_match(self, tiny_module):
        trainable, total = freeze_summary(tiny_module)
        assert trainable == total  # all trainable initially
        freeze_by_patterns(tiny_module, patterns=["0.*"])
        trainable_after, total_after = freeze_summary(tiny_module)
        assert total_after == total
        assert trainable_after < trainable

    def test_log_does_not_raise(self, tiny_module):
        # Smoke test — must not throw
        log_freeze_state(tiny_module)
