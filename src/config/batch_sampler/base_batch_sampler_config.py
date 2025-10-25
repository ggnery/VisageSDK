from config.base_config import BaseConfig

class BaseBatchSamplerConfig(BaseConfig):
    def __init__(self, config_path):
        super().__init__(config_path)
        
        self.build_config()