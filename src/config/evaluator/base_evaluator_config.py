from config.base_config import BaseConfig


class EvaluatorConfig(BaseConfig):
    """Evaluator config — all params via YAML.

    Common keys:
        device: cuda | cpu
        batch_size: how many images to encode at once
        distance: cosine | euclidean (verification only)
        far_targets: list of FAR points for TAR@FAR (verification)
        ranks: list of N values for rank-N accuracy (identification)
    """

    device: str
    batch_size: int

    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        self.device = self._params["device"]
        self.batch_size = self._params["batch_size"]
