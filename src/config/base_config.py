from abc import ABC
from typing import Any
import yaml
class BaseConfig(ABC):
    config: Any
    
    def __init__(self, config_path: str) -> None:
        with open(config_path, "r") as file:
            self.config = yaml.safe_load(file)
       
        super().__init__()
    
    def build_config(self) -> None:
        """Override this method to add custom attributes to your dataset config
        """
        pass
    
    def get_config_string(self) -> str:
        result = ""
        result += 25*"=" + "\n"
        result += f"{self.__class__.__name__} CONFIGURATION\n"
        result += 25*"=" + "\n"
        for attr_name, attr_value in self.__dict__.items():
            if attr_name != "config":
                result += (f"{attr_name}: {attr_value}\n")
        result += 25*"=" + "\n\n"
    
        return result
        
    