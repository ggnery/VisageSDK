from config.base_config import BaseConfig

class BaseEarlyStopperConfig(BaseConfig):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        
        self.build_config()
    