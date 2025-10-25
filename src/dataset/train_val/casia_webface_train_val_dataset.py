from typing import List, Tuple
from typing_extensions import override
from config.dataset.train_val.casia_webface_train_val_config import CasiaWebFaceTrainValConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset
from pathlib import Path

class CasiaWebFaceTrain(BaseTrainValDataset):
    @override
    def read_data(self, dataset_config: CasiaWebFaceTrainValConfig) -> List[Tuple[str, str]]:
        train_dir = Path(dataset_config.train_dir)
        data_pairs = []
        
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        
        # Iterate through each class directory (person)
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Iterate through all images in the class directory
                for img_file in class_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img_path = str(img_file.absolute())
                        data_pairs.append((class_name, img_path))
        
        return data_pairs
        
class CasiaWebFaceVal(BaseTrainValDataset):
    @override
    def read_data(self, dataset_config: CasiaWebFaceTrainValConfig) -> List[Tuple[str, str]]:
        val_dir = Path(dataset_config.val_dir)
        data_pairs = []
        
        if not val_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {val_dir}")
        
        # Iterate through each class directory (person)
        for class_dir in val_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Iterate through all images in the class directory
                for img_file in class_dir.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        img_path = str(img_file.absolute())
                        data_pairs.append((class_name, img_path))
        
        return data_pairs