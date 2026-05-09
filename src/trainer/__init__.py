__all__ = ["Trainer"]


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__} has no attribute {name}")
