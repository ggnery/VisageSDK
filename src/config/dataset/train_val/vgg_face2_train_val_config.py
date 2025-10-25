from typing_extensions import override
from .base_train_val_dataset_config import BaseTrainValDatasetConfig

class VGGFace2TrainValConfig(BaseTrainValDatasetConfig):  
    @override
    def build_config(self) -> None:
        pass # There is no aditional config for VGGFace2Dataset