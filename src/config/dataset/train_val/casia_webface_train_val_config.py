from typing import List
from typing_extensions import override
from .base_train_val_dataset_config import BaseTrainValDatasetConfig

class CasiaWebFaceTrainValConfig(BaseTrainValDatasetConfig):  
    @override
    def build_config(self) -> None:
        pass # There is no aditional config for CasiaWebFaceConfig