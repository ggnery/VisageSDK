"""Lazy-import wrapper to avoid circular imports with backbone.base_backbone."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trainer import Trainer  # noqa: F401

__all__ = ["Trainer"]


def __getattr__(name: str):
    if name == "Trainer":
        from .trainer import Trainer

        return Trainer
    raise AttributeError(f"module {__name__} has no attribute {name}")
