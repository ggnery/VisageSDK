"""Tests for tools.scheduler — including B4 regression (StairLR per-group)."""

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR, ReduceLROnPlateau, StepLR

from tools.scheduler import build_scheduler


class StubConfig:
    def __init__(self, lr_schedule_type, lr_schedule_params):
        self.lr_schedule_type = lr_schedule_type
        self.lr_schedule_params = lr_schedule_params


def _two_group_optimizer():
    """Optimizer with two param groups at different base LRs."""
    p1 = nn.Parameter(torch.zeros(2))
    p2 = nn.Parameter(torch.zeros(2))
    return SGD([
        {"params": [p1], "lr": 0.1},
        {"params": [p2], "lr": 1e-4},
    ], lr=0.1)


def _single_group_optimizer(lr=0.1):
    p = nn.Parameter(torch.zeros(2))
    return SGD([p], lr=lr)


class TestStairLR:
    def test_single_group_targets_absolute(self):
        opt = _single_group_optimizer(lr=0.1)
        cfg = StubConfig("StairLR", {1: 0.01, 3: 0.001})
        sched = build_scheduler(opt, cfg)
        assert isinstance(sched, LambdaLR)
        # Epoch 0: still 0.1
        assert opt.param_groups[0]["lr"] == pytest.approx(0.1)
        sched.step()  # epoch 1
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
        sched.step()  # epoch 2 (still 0.01, before milestone 3)
        assert opt.param_groups[0]["lr"] == pytest.approx(0.01)
        sched.step()  # epoch 3
        assert opt.param_groups[0]["lr"] == pytest.approx(0.001)

    def test_b4_regression_per_group_absolute_targets(self):
        """B4: with multi-group optimizer, every group must reach the absolute
        target at each milestone — not be scaled multiplicatively from group 0.
        """
        opt = _two_group_optimizer()
        cfg = StubConfig("StairLR", {1: 0.01})
        sched = build_scheduler(opt, cfg)

        # Step into epoch 1
        sched.step()

        # Both groups should hit lr=0.01 at the milestone, not lr_g1 * 0.1
        # (which would have been 1e-5 under the old multiplicative behavior).
        for g in opt.param_groups:
            assert g["lr"] == pytest.approx(0.01)


class TestStepLR:
    def test_step_lr_built(self):
        opt = _single_group_optimizer(lr=0.1)
        cfg = StubConfig("StepLR", {"step_size": 2, "gamma": 0.5})
        sched = build_scheduler(opt, cfg)
        assert isinstance(sched, StepLR)


class TestMultiStepLR:
    def test_multi_step_lr_built(self):
        opt = _single_group_optimizer(lr=0.1)
        cfg = StubConfig("MultiStepLR", {"milestones": [3, 6], "gamma": 0.1})
        sched = build_scheduler(opt, cfg)
        assert isinstance(sched, MultiStepLR)
        assert sched.milestones == {3: 1, 6: 1}
        assert sched.gamma == 0.1


class TestReduceLROnPlateau:
    def test_reduce_lr_built(self):
        opt = _single_group_optimizer(lr=0.1)
        cfg = StubConfig("ReduceLROnPlateau",
                         {"mode": "min", "factor": 0.5, "patience": 2})
        sched = build_scheduler(opt, cfg)
        assert isinstance(sched, ReduceLROnPlateau)


class TestUnknownScheduler:
    def test_unknown_raises(self):
        opt = _single_group_optimizer()
        cfg = StubConfig("DoesNotExist", {})
        with pytest.raises(ValueError, match="not implemented"):
            build_scheduler(opt, cfg)
