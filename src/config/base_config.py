from typing import Any, Dict
import yaml


class BaseConfig:
    """Loads a YAML and exposes its keys as attributes.

    Subclasses inject cross-component values explicitly via __init__
    (e.g. embedding_size, num_classes). YAML keys are accessed via
    attribute lookup with no boilerplate."""

    _params: Dict[str, Any]

    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            self._params = yaml.safe_load(file) or {}

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        params = self.__dict__.get("_params", {})
        if name in params:
            return params[name]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def get_config_string(self) -> str:
        sep = "=" * 25
        lines = [sep, f"{type(self).__name__} CONFIGURATION", sep]
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            lines.append(f"{k}: {v}")
        for k, v in self._params.items():
            if k not in self.__dict__:
                lines.append(f"{k}: {v}")
        lines.append(sep)
        lines.append("")
        return "\n".join(lines) + "\n"
