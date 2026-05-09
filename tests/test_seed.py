"""Tests for tools.seed."""

import random
import numpy as np
import torch

from tools.seed import make_dataloader_generator, seed_worker, set_seed


class TestSetSeed:
    def test_none_is_noop(self):
        before = torch.rand(1).item()
        set_seed(None)
        after = torch.rand(1).item()
        # No assertion on equality — RNG advanced normally; we just verify no crash
        assert isinstance(after, float)

    def test_same_seed_produces_same_torch_random(self):
        set_seed(42)
        a = torch.rand(5)
        set_seed(42)
        b = torch.rand(5)
        assert torch.allclose(a, b)

    def test_same_seed_produces_same_numpy(self):
        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_same_seed_produces_same_python_random(self):
        set_seed(7)
        a = [random.random() for _ in range(5)]
        set_seed(7)
        b = [random.random() for _ in range(5)]
        assert a == b


class TestDataloaderGenerator:
    def test_none_returns_none(self):
        assert make_dataloader_generator(None) is None

    def test_seeded_is_reproducible(self):
        g1 = make_dataloader_generator(42)
        g2 = make_dataloader_generator(42)
        assert torch.equal(torch.randperm(10, generator=g1), torch.randperm(10, generator=g2))

    def test_different_seeds_differ(self):
        g1 = make_dataloader_generator(1)
        g2 = make_dataloader_generator(2)
        assert not torch.equal(torch.randperm(10, generator=g1), torch.randperm(10, generator=g2))


class TestSeedWorker:
    def test_seed_worker_does_not_raise(self):
        # In real DataLoader workers torch.initial_seed() is set; we just check the call works.
        seed_worker(0)
