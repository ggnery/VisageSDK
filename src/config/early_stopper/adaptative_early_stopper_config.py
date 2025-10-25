from config.early_stopper.base_early_stopper_config import BaseEarlyStopperConfig

class AdaptativeEarlyStopperConfig(BaseEarlyStopperConfig): 
    base_patience: int
    patience_increase_ratio: float
    delta: float
    
    def build_config(self) -> None:
        self.base_patience = self.config["base_patience"]
        self.patience_increase_ratio = self.config["patience_increase_ratio"]
        self.delta = self.config["delta"]