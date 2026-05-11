"""Shared pytest fixtures: on-disk dataset trees + a tiny backbone."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torchvision import transforms

from transformation.base_transformation import BaseTransformation


class _PassthroughTransformation(BaseTransformation):
    """BaseTransformation that skips the resize and only emits ToTensor."""

    def __init__(self) -> None:
        self.transform = transforms.Compose([transforms.ToTensor()])

    def build_transformation(self, transformation_config) -> list:
        # Unreachable (we override __init__); kept to satisfy the ABC.
        return [transforms.ToTensor()]


@pytest.fixture
def passthrough_transformation() -> _PassthroughTransformation:
    return _PassthroughTransformation()


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


_COMPONENT_PACKAGES = (
    "backbone",
    "batch_sampler",
    "dataset.eval",
    "dataset.train_val",
    "early_stopper",
    "evaluator",
    "loss",
    "transformation",
)


@pytest.fixture
def populated_registries():
    """Force side-effect imports so every registry is populated."""
    for pkg in _COMPONENT_PACKAGES:
        importlib.import_module(pkg)
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

    return SimpleNamespace(
        BACKBONES=BACKBONES,
        LOSSES=LOSSES,
        DATASETS=DATASETS,
        EVAL_DATASETS=EVAL_DATASETS,
        SAMPLERS=SAMPLERS,
        EARLY_STOPPERS=EARLY_STOPPERS,
        TRANSFORMATIONS=TRANSFORMATIONS,
        EVALUATORS=EVALUATORS,
    )


class _TinyBackbone(torch.nn.Module):
    """Minimal embedding model for evaluator/trainer tests."""

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
