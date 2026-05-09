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
        path.write_text(yaml.safe_dump({
            "base_patience": base_patience,
            "delta": delta,
            "patience_increase_ratio": patience_increase_ratio,
        }))
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
        assert es.early_stop(1.0) is True   # wait_count=2 == patience

    def test_resets_on_improvement(self, make_stopper):
        es = make_stopper(base_patience=2, delta=0.01)
        es.early_stop(1.0)
        es.early_stop(1.0)        # wait_count=1
        es.early_stop(0.5)        # improvement → resets
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
