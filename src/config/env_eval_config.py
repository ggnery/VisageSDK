import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class ENVEvalConfig:
    """Environment variables driving a standalone evaluation run."""

    backbone: str
    backbone_config: str
    checkpoint_path: str

    eval_dataset: str
    eval_dataset_config: str

    eval_transformation: str
    eval_transformation_config: str

    evaluator: str
    evaluator_config: str

    @classmethod
    def from_env(cls) -> "ENVEvalConfig":
        load_dotenv()

        def required(name: str) -> str:
            v = os.getenv(name)
            if not v:
                raise ValueError(f"Missing required environment variable: {name}")
            return v

        return cls(
            backbone=required("BACKBONE"),
            backbone_config=required("BACKBONE_CONFIG"),
            checkpoint_path=required("CHECKPOINT_PATH"),
            eval_dataset=required("EVAL_DATASET"),
            eval_dataset_config=required("EVAL_DATASET_CONFIG"),
            eval_transformation=required("EVAL_TRANSFORMATION"),
            eval_transformation_config=required("EVAL_TRANSFORMATION_CONFIG"),
            evaluator=required("EVALUATOR"),
            evaluator_config=required("EVALUATOR_CONFIG"),
        )
