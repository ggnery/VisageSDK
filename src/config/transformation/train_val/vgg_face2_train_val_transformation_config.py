from typing import List, override
from config.transformation.base_transformation_config import BaseTransformationConfig

class VGGFace2TrainValTransformationConfig(BaseTransformationConfig):
    train_std_normalize: List
    train_mean_normalize: List
    train_random_horizontal_flip: float
    train_random_rotation: float
    train_brightness: float
    train_contrast: float
    train_saturation: float
    
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
        self.train_random_rotation = train_transformations["random_rotation"]
        
        self.train_brightness =  train_transformations["color_jitter"]["brightness"]
        self.train_contrast =  train_transformations["color_jitter"]["contrast"]
        self.train_saturation =  train_transformations["color_jitter"]["saturation"]
    
    def build_val_transformations(self):
        val_transformations = self.config["val"]
        self.val_std_normalize = val_transformations["normalize"]["std"]
        self.val_mean_normalize = val_transformations["normalize"]["mean"]