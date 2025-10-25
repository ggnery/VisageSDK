from .training_context import BatchContext, TrainingContext, EpochContext

__all__ = ["Trainer", "BatchContext", "TrainingContext", "EpochContext"]

def __getattr__(name):
    if name == "Trainer":
        # Lazy import to avoid circular dependency with backbone.base_backbone
        from .trainer import Trainer
        return Trainer
    raise AttributeError(f"module {__name__} has no attribute {name}")