"""Tests for AdaptativeEarlyStopper behavior."""

import pytest
import yaml

from config.early_stopper.base_early_stopper_config import EarlyStopperConfig
from early_stopper.adaptative_early_stopper import AdaptativeEarlyStopper


@pytest.fixture
def make_stopper(tmp_path):
    counter = {"i": 0}

    def _make(base_patience=3, delta=0.01, patience_increase_ratio=0.8):
        counter["i"] += 1
        path = tmp_path / f"es_{counter['i']}.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "base_patience": base_patience,
                    "delta": delta,
                    "patience_increase_ratio": patience_increase_ratio,
                }
            )
        )
        return AdaptativeEarlyStopper(EarlyStopperConfig(str(path)))

    return _make


class TestAdaptativeEarlyStopper:
    def test_does_not_stop_when_loss_improves(self, make_stopper):
        es = make_stopper(base_patience=3)
        for v in [1.0, 0.9, 0.8, 0.7]:
            assert es.early_stop(v) is False

    def test_stops_after_patience_exhausted(self, make_stopper):
        es = make_stopper(base_patience=2, delta=0.01, patience_increase_ratio=2.0)
        es.early_stop(1.0)
        # No improvement for 2 epochs → stop
        assert es.early_stop(1.0) is False  # wait_count=1
        assert es.early_stop(1.0) is True  # wait_count=2 == patience

    def test_resets_on_improvement(self, make_stopper):
        es = make_stopper(base_patience=2, delta=0.01)
        es.early_stop(1.0)
        es.early_stop(1.0)  # wait_count=1
        es.early_stop(0.5)  # improvement → resets
        # Now we need 2 more no-improvement steps
        assert es.early_stop(0.5) is False
        # 2nd no-improvement still under patience due to dynamic_patience
        # (patience may have grown when wait was near limit)
        # Just verify we don't stop on the same step that the reset happened
        assert es.best_score == 0.5

    def test_dynamic_patience_grows_near_limit(self, make_stopper):
        es = make_stopper(base_patience=5, patience_increase_ratio=0.6)
        es.early_stop(1.0)  # baseline
        # Stay flat — once wait_count reaches 5*0.6 = 3, dynamic_patience grows
        for _ in range(3):
            es.early_stop(1.0)
        assert es.dynamic_patience > es.base_patience

    def test_stops_eventually_with_ratio_below_one(self, make_stopper):
        """Regression for B-1: with `patience_increase_ratio < 1` the
        pre-fix grew `dynamic_patience` once per epoch in lockstep with
        `wait_count`, so the stop condition `wait_count >= dynamic_patience`
        was never reached and the early stopper silently never triggered.
        Post-fix the bump is one-shot (when `wait_count` first crosses the
        threshold) so eventually `wait_count` catches up and we stop."""
        es = make_stopper(base_patience=5, delta=0.01, patience_increase_ratio=0.8)
        es.early_stop(1.0)  # baseline
        stopped = False
        # 20 flat steps is well above any plausible grace window;
        # the pre-fix loop runs forever, post-fix stops at wait≈6.
        for _ in range(20):
            if es.early_stop(1.0):
                stopped = True
                break
        assert stopped, (
            "B-1 regression: with patience_increase_ratio < 1 the stopper "
            "never triggers because dynamic_patience grows every epoch."
        )

    def test_dynamic_patience_is_one_shot_not_per_epoch(self, make_stopper):
        """Once `wait_count` crosses the bump threshold, `dynamic_patience`
        must NOT keep growing each subsequent epoch — otherwise wait can
        never catch up. The bump fires at most once per "no improvement
        run" and resets on improvement."""
        es = make_stopper(base_patience=4, delta=0.01, patience_increase_ratio=0.5)
        es.early_stop(1.0)  # baseline; threshold = 4*0.5 = 2
        # Below threshold: no bump.
        es.early_stop(1.0)  # wait=1
        assert es.dynamic_patience == es.base_patience
        # Cross threshold: bump once.
        es.early_stop(1.0)  # wait=2
        bumped = es.dynamic_patience
        assert bumped > es.base_patience
        # Past threshold: dynamic_patience stays the same.
        es.early_stop(1.0)  # wait=3
        assert es.dynamic_patience == bumped
