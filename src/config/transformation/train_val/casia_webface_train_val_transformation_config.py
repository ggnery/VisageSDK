from typing import List, override
from config.transformation.base_transformation_config import BaseTransformationConfig

class CasiaWebFaceTrainValTransformationConfig(BaseTransformationConfig):
    train_std_normalize: List
    train_mean_normalize: List
    train_random_horizontal_flip: float
    
    val_std_normalize: List
    val_mean_normalize: List
    
    @override
    def build_config(self) -> None:
        self.build_train_transformations()
        self.build_val_transformations()
        
    def build_train_transformations(self):
        train_transformations = self.config["train"]
        
        self.train_std_normalize = train_transformations["normalize"]["std"]
        self.train_mean_normalize = train_transformations["normalize"]["mean"]
        
        self.train_random_horizontal_flip = train_transformations["random_horizontal_flip"]

    
    def build_val_transformations(self):
        val_transformations = self.config["val"]
        self.val_std_normalize = val_transformations["normalize"]["std"]
        self.val_mean_normalize = val_transformations["normalize"]["mean"]