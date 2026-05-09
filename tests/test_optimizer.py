"""Tests for tools.optimizer.build_optimizer including param_groups assignment."""

import pytest
import torch
import torch.nn as nn

from backbone.base_backbone import BaseBackbone
from config.trainer.trainer_config import TrainerConfig
from loss.base_loss import BaseLoss
from tools.optimizer import build_optimizer


class StubBackbone(BaseBackbone):
    """Real BaseBackbone subclass with parameters but no config dependency."""

    def __init__(self):
        nn.Module.__init__(self)
        self.features = nn.Sequential(nn.Linear(8, 4), nn.Linear(4, 2))
        self.last_linear = nn.Linear(2, 4)


class StubLoss(BaseLoss):
    """Real BaseLoss subclass with a single trainable parameter."""

    def __init__(self):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.randn(4))


class StubTrainerConfig(TrainerConfig):
    """Real TrainerConfig subclass with the optimizer fields set directly."""

    def __init__(self, optimizer_type="SGD", optimizer_params=None, optimizer_param_groups=None):
        # Skip TrainerConfig.__init__ — we only need optimizer-related fields here.
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params or {"lr": 0.01}
        self.optimizer_param_groups = optimizer_param_groups


class TestBuildOptimizer:
    def test_default_two_groups(self):
        model, loss = StubBackbone(), StubLoss()
        cfg = StubTrainerConfig()
        opt = build_optimizer(model, loss, cfg)
        assert len(opt.param_groups) == 2
        assert opt.defaults["lr"] == 0.01

    def test_unsupported_type_raises(self):
        model, loss = StubBackbone(), StubLoss()
        cfg = StubTrainerConfig(optimizer_type="DoesNotExist")
        with pytest.raises(ValueError, match="not implemented"):
            build_optimizer(model, loss, cfg)

    def test_supported_types_instantiate(self):
        from torch.optim import SGD, Adam, AdamW, RMSprop

        model, loss = StubBackbone(), StubLoss()
        for name, expected in [("SGD", SGD), ("Adam", Adam), ("AdamW", AdamW), ("RMSprop", RMSprop)]:
            cfg = StubTrainerConfig(optimizer_type=name, optimizer_params={"lr": 0.01})
            opt = build_optimizer(model, loss, cfg)
            assert isinstance(opt, expected)


class TestParamGroups:
    def test_pattern_assignment(self):
        model, loss = StubBackbone(), StubLoss()
        groups = [
            {"pattern": "backbone.features.0.*", "lr": 1e-5},
            {"pattern": "backbone.last_linear*", "lr": 1e-3},
            {"pattern": "loss.*", "lr": 5e-3},
        ]
        cfg = StubTrainerConfig(
            optimizer_params={"lr": 1e-2},
            optimizer_param_groups=groups,
        )
        opt = build_optimizer(model, loss, cfg)
        # 3 patterns + default group = 4 groups (assuming all match something)
        # Group lrs should reflect the spec
        lrs = [g["lr"] for g in opt.param_groups]
        assert 1e-5 in lrs
        assert 1e-3 in lrs
        assert 5e-3 in lrs
        # Default group catches features.1.* with the optimizer default
        assert 1e-2 in lrs

    def test_unmatched_params_go_to_default(self):
        model, loss = StubBackbone(), StubLoss()
        groups = [{"pattern": "backbone.last_linear*", "lr": 1e-3}]
        cfg = StubTrainerConfig(
            optimizer_params={"lr": 1e-2},
            optimizer_param_groups=groups,
        )
        opt = build_optimizer(model, loss, cfg)
        # Two groups: matched + default
        assert len(opt.param_groups) == 2
        # Default group has the optimizer base lr
        default = [g for g in opt.param_groups if g["lr"] == 1e-2]
        assert len(default) == 1
        # And it has more params than the matched group
        matched = [g for g in opt.param_groups if g["lr"] == 1e-3]
        assert sum(p.numel() for p in default[0]["params"]) > sum(p.numel() for p in matched[0]["params"])

    def test_first_match_wins(self):
        model, loss = StubBackbone(), StubLoss()
        groups = [
            {"pattern": "backbone.features.*", "lr": 1e-4},
            {"pattern": "backbone.features.0.*", "lr": 1e-5},  # never reached
        ]
        cfg = StubTrainerConfig(optimizer_param_groups=groups)
        opt = build_optimizer(model, loss, cfg)
        # No group should have lr=1e-5 because the more general pattern caught everything first
        assert 1e-5 not in [g["lr"] for g in opt.param_groups]

    def test_empty_groups_are_dropped(self):
        model, loss = StubBackbone(), StubLoss()
        groups = [{"pattern": "totally.unmatched.*", "lr": 1e-9}]
        cfg = StubTrainerConfig(optimizer_param_groups=groups)
        opt = build_optimizer(model, loss, cfg)
        # Only the default group should remain
        assert len(opt.param_groups) == 1
        assert 1e-9 not in [g["lr"] for g in opt.param_groups]

    def test_missing_pattern_key_raises(self):
        model, loss = StubBackbone(), StubLoss()
        groups = [{"lr": 1e-3}]  # forgot `pattern`
        cfg = StubTrainerConfig(optimizer_param_groups=groups)
        with pytest.raises(ValueError, match="pattern"):
            build_optimizer(model, loss, cfg)

    def test_frozen_params_remain_in_optimizer(self):
        """Frozen params should still be added; optimizer just skips them on step."""
        model, loss = StubBackbone(), StubLoss()
        for p in model.features[0].parameters():
            p.requires_grad = False
        cfg = StubTrainerConfig()
        opt = build_optimizer(model, loss, cfg)
        # Identity check (param tensors compare elementwise with `==`, not by id)
        all_param_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        for p in model.features[0].parameters():
            assert id(p) in all_param_ids
