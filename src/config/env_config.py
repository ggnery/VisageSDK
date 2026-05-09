import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class ENVConfig:
    """Holds environment variables that select components by registry name.

    For each component: NAME selects the implementation (registry key);
    CONFIG points to a YAML file. Optional components (sampler, early stopper)
    are skipped entirely when their NAME var is not set.
    """

    backbone: str
    backbone_config: str

    train_val_dataset: str
    train_val_dataset_config: str

    train_transformation: str
    val_transformation: str
    train_val_transformation_config: str

    loss: str
    loss_config: str

    trainer_config: str

    sampler: str | None = None
    sampler_config: str | None = None

    early_stopper: str | None = None
    early_stopper_config: str | None = None

    @classmethod
    def from_env(cls) -> "ENVConfig":
        load_dotenv()

        def get_required(name: str) -> str:
            v = os.getenv(name)
            if not v:
                raise ValueError(f"Missing required environment variable: {name}")
            return v

        return cls(
            backbone=get_required("BACKBONE"),
            backbone_config=get_required("BACKBONE_CONFIG"),
            train_val_dataset=get_required("TRAIN_VAL_DATASET"),
            train_val_dataset_config=get_required("TRAIN_VAL_DATASET_CONFIG"),
            train_transformation=get_required("TRAIN_TRANSFORMATION"),
            val_transformation=get_required("VAL_TRANSFORMATION"),
            train_val_transformation_config=get_required("TRAIN_VAL_TRANSFORMATION_CONFIG"),
            loss=get_required("LOSS"),
            loss_config=get_required("LOSS_CONFIG"),
            trainer_config=get_required("TRAINER_CONFIG"),
            sampler=os.getenv("SAMPLER") or None,
            sampler_config=os.getenv("SAMPLER_CONFIG") or None,
            early_stopper=os.getenv("EARLY_STOPPER") or None,
            early_stopper_config=os.getenv("EARLY_STOPPER_CONFIG") or None,
        )
