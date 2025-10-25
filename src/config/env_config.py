import os
from dataclasses import dataclass
from dotenv import load_dotenv

@dataclass
class ENVConfig:
    """Configuration class to hold all environment variables"""
    backbone_config_path: str
    backbone_config_class: str
    backbone_class: str
    
    train_val_transformation_config_path: str
    train_val_transformation_config_class: str
    train_transformation_class:str
    val_transformation_class:str
    
    train_val_dataset_config_path: str
    train_val_dataset_config_class: str
    train_dataset_class: str
    val_dataset_class: str
    
    loss_config_path: str
    loss_config_class: str
    loss_class: str
    
    use_sampler: str
    batch_sampler_config_path: str
    batch_sampler_config_class: str
    batch_sampler_class: str    
    
    use_early_stopper: str
    early_stopper_config_path: str
    early_stopper_config_class: str
    early_stopper_class: str
    
    trainer_config_path: str
    trainer_config_class: str
    
    @classmethod
    def from_env(cls):
        """Create Config instance from environment variables"""
        load_dotenv()
        
        # Create config with all environment variables
        config = cls(
            backbone_config_path=os.getenv("BACKBONE_CONFIG_PATH"),
            backbone_config_class=os.getenv("BACKBONE_CONFIG_CLASS"),
            backbone_class=os.getenv("BACKBONE_CLASS"),
            
            train_val_transformation_config_path=os.getenv("TRAIN_VAL_TRANSFORMATION_CONFIG_PATH"),
            train_val_transformation_config_class=os.getenv("TRAIN_VAL_TRANSFORMATION_CONFIG_CLASS"),
            train_transformation_class=os.getenv("TRAIN_TRANSFORMATION_CLASS"),
            val_transformation_class=os.getenv("VAL_TRANSFORMATION_CLASS"),
            
            train_val_dataset_config_path=os.getenv("TRAIN_VAL_DATASET_CONFIG_PATH"),
            train_val_dataset_config_class=os.getenv("TRAIN_VAL_DATASET_CONFIG_CLASS"),
            train_dataset_class=os.getenv("TRAIN_DATASET_CLASS"),
            val_dataset_class=os.getenv("VAL_DATASET_CLASS"),
            
            loss_config_path=os.getenv("LOSS_CONFIG_PATH"),
            loss_config_class=os.getenv("LOSS_CONFIG_CLASS"),
            loss_class=os.getenv("LOSS_CLASS"),
            
            use_sampler=os.getenv("USE_SAMPLER"),
            batch_sampler_config_path=os.getenv("BATCH_SAMPLER_CONFIG_PATH"),
            batch_sampler_config_class=os.getenv("BATCH_SAMPLER_CONFIG_CLASS"),
            batch_sampler_class=os.getenv("BATCH_SAMPLER_CLASS"),
            
            use_early_stopper=os.getenv("USE_EARLY_STOPPER"),
            early_stopper_config_path=os.getenv("EARLY_STOPPER_CONFIG_PATH"),
            early_stopper_config_class=os.getenv("EARLY_STOPPER_CONFIG_CLASS"),
            early_stopper_class=os.getenv("EARLY_STOPPER_CLASS"),
            
            trainer_config_path=os.getenv("TRAINER_CONFIG_PATH"),
            trainer_config_class=os.getenv("TRAINER_CONFIG_CLASS"),           
        )
        
        # Validate all fields are set
        missing = [field for field, value in config.__dict__.items() if value is None]
        if missing:
            raise ValueError(f"Missing environment variables: {[f.upper() for f in missing]}")
        
        return config