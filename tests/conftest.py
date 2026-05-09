"""Shared pytest fixtures.

This file is loaded automatically by pytest. It puts `src/` on sys.path so
test modules can import `registry`, `tools.metrics`, etc. directly, and
provides reusable on-disk fixtures (image folders, LFW pairs, identification
gallery/probe layouts).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402


def _make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=color).save(path)


@pytest.fixture
def tmp_imagefolder(tmp_path) -> Path:
    """ImageFolder layout with 3 classes × 4 images per split."""
    classes = ("alice", "bob", "carol")
    for split in ("train", "val"):
        for cls in classes:
            for i in range(4):
                _make_image(
                    tmp_path / split / cls / f"img_{i}.jpg",
                    color=((i * 40) % 256, (hash(cls) * 17) % 256, 100),
                )
    return tmp_path


@pytest.fixture
def tmp_lfw_pairs(tmp_path) -> tuple[Path, Path]:
    """Tiny LFW-format dataset: 3 names × 3 images, 2 folds × 2 same/2 diff."""
    images_dir = tmp_path / "lfw"
    names = ("alice", "bob", "carol")
    for name in names:
        for i in range(1, 4):
            _make_image(
                images_dir / name / f"{name}_{i:04d}.jpg",
                color=((hash(name) * 7) % 256, i * 60, 80),
            )
    pairs_path = tmp_path / "pairs.txt"
    pairs_path.write_text(
        "2 2\n"
        # fold 0
        "alice 1 2\n"
        "bob 1 3\n"
        "alice 1 bob 2\n"
        "carol 1 alice 3\n"
        # fold 1
        "carol 2 3\n"
        "alice 2 3\n"
        "bob 1 carol 1\n"
        "alice 1 carol 2\n"
    )
    return images_dir, pairs_path


@pytest.fixture
def tmp_identification(tmp_path) -> Path:
    """Gallery (1 image / class) + probe (2 images / class) for 3 classes."""
    classes = ("alice", "bob", "carol")
    for cls in classes:
        _make_image(tmp_path / "gallery" / cls / "g.jpg", color=((hash(cls) * 11) % 256, 200, 50))
        for i in range(2):
            _make_image(
                tmp_path / "probe" / cls / f"p{i}.jpg",
                color=((hash(cls) * 11) % 256, 200 - i * 10, 50 + i * 5),
            )
    return tmp_path


@pytest.fixture
def populated_registries():
    """Force side-effect imports so all registries are populated.

    Returns a tuple of registry handles for convenient access.
    """
    import backbone  # noqa: F401
    import batch_sampler  # noqa: F401
    import dataset.eval  # noqa: F401
    import dataset.train_val  # noqa: F401
    import early_stopper  # noqa: F401
    import evaluator  # noqa: F401
    import loss  # noqa: F401
    import transformation  # noqa: F401
    from registry import (
        BACKBONES,
        DATASETS,
        EARLY_STOPPERS,
        EVAL_DATASETS,
        EVALUATORS,
        LOSSES,
        SAMPLERS,
        TRANSFORMATIONS,
    )

    class Bag:
        pass

    bag = Bag()
    bag.BACKBONES = BACKBONES
    bag.LOSSES = LOSSES
    bag.DATASETS = DATASETS
    bag.EVAL_DATASETS = EVAL_DATASETS
    bag.SAMPLERS = SAMPLERS
    bag.EARLY_STOPPERS = EARLY_STOPPERS
    bag.TRANSFORMATIONS = TRANSFORMATIONS
    bag.EVALUATORS = EVALUATORS
    return bag


class _TinyBackbone(torch.nn.Module):
    """Minimal embedding model used to drive evaluator / trainer tests
    without paying for a real CNN init/forward."""

    def __init__(self, embedding_size: int = 16):
        super().__init__()
        self.embedding_size = embedding_size
        self.device = torch.device("cpu")
        self.layer = torch.nn.Linear(32 * 32 * 3, embedding_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x.flatten(1))


@pytest.fixture
def tiny_backbone():
    """Pre-instantiated tiny backbone for evaluator tests."""
    return _TinyBackbone(embedding_size=16)
