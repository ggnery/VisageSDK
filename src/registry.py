from typing import Dict, Type


class Registry:
    def __init__(self, kind: str):
        self._kind = kind
        self._items: Dict[str, Type] = {}

    def register(self, name: str, cls: Type) -> Type:
        if name in self._items:
            raise ValueError(f"{self._kind} '{name}' already registered as {self._items[name].__name__}")
        self._items[name] = cls
        return cls

    def get(self, name: str) -> Type:
        if name not in self._items:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(f"{self._kind} '{name}' not found. Available: {available}")
        return self._items[name]

    def names(self):
        return list(self._items)


BACKBONES = Registry("backbone")
LOSSES = Registry("loss")
DATASETS = Registry("dataset")
EVAL_DATASETS = Registry("eval_dataset")
SAMPLERS = Registry("sampler")
EARLY_STOPPERS = Registry("early_stopper")
TRANSFORMATIONS = Registry("transformation")
EVALUATORS = Registry("evaluator")
