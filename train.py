import sys
from pathlib import Path

src_path = Path(__file__).parent
sys.path.insert(0, str(src_path / "src"))

from config import ENVConfig
from tools.trainer_builder import TrainerBuilder


def main(env_config: ENVConfig):
    builder = TrainerBuilder(env_config)
    trainer = builder.build_trainer()

    trainer.train()
    print(20 * "=")
    print("Train Finished!")
    print(20 * "=")


if __name__ == "__main__":
    env_config = ENVConfig.from_env()
    main(env_config)
