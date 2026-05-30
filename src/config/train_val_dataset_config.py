from config.base_config import BaseConfig


class TrainValDatasetConfig(BaseConfig):
    """Train/val dataset config.

    Required YAML keys: train_dir, val_dir, num_classes (positive int).
    Injected: input_size (from backbone).
    """

    input_size: list[int]

    def __init__(self, config_path: str, backbone_info: dict) -> None:
        super().__init__(config_path)
        self.input_size = backbone_info["input_size"]
        # Fail fast on the most common YAML mistake: `num_classes: null`
        # (the placeholder shipped in vgg_face2.yaml) silently propagates
        # through LossConfig → nn.Linear and crashes with a cryptic
        # `TypeError: 'NoneType' object cannot be interpreted as an integer`
        # deep in PyTorch. Validate here so the error points at the YAML.
        num_classes = self._params.get("num_classes")
        if num_classes is None:
            raise ValueError(
                f"`num_classes` is required in dataset config {config_path}. "
                "Set it to the number of identity classes in your dataset."
            )
        try:
            num_classes_int = int(num_classes)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"`num_classes` in {config_path} must be a positive integer, got {num_classes!r}"
            ) from e
        if num_classes_int <= 0:
            raise ValueError(
                f"`num_classes` in {config_path} must be a positive integer, got {num_classes_int}"
            )
        self._params["num_classes"] = num_classes_int
