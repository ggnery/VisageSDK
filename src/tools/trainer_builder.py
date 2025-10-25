from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from backbone.base_backbone import BaseBackbone
from batch_sampler.base_batch_sampler import BaseBatchSampler
from config.backbone.base_backbone_config import BaseBackboneConfig
from config.batch_sampler.base_batch_sampler_config import BaseBatchSamplerConfig
from config.dataset.train_val.base_train_val_dataset_config import BaseTrainValDatasetConfig
from config.early_stopper.base_early_stopper_config import BaseEarlyStopperConfig
from config.env_config import ENVConfig
from datetime import datetime
from pathlib import Path
import importlib

from config.loss.base_loss_config import BaseLossConfig
from dataset.train_val.base_train_val_dataset import BaseTrainValDataset
from early_stopper.base_early_stopper import BaseEarlyStopper
from loss.base_loss import BaseLoss
from tools.optimizer import build_optimizer
from tools.scheduler import build_scheduler
from trainer.trainer import Trainer

def import_class(class_path):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

class TrainerBuilder:
    config_str: str = ""
    backbone_config: BaseBackboneConfig
    train_val_dataset_config: BaseTrainValDatasetConfig
    loss_config: BaseLossConfig
    early_stopper_config: BaseEarlyStopperConfig = None
    batch_sampler_config: BaseBatchSamplerConfig = None
    
    backbone: BaseBackbone
    train_dataset: BaseTrainValDataset
    val_dataset: BaseTrainValDataset
    loss: BaseLoss
    early_stopper: BaseEarlyStopper = None
    batch_sampler: BaseBatchSampler = None
    scheduler: LRScheduler
    optimizer: Optimizer
    
    def __init__(self, env_config: ENVConfig):      
        self.build_configs(env_config)
        self.build_instances(env_config)
        
    def build_configs(self, env_config: ENVConfig):
        # Build backbone config
        self.backbone_config = import_class(env_config.backbone_config_class)(env_config.backbone_config_path)
        self.config_str += self.backbone_config.get_config_string()
        backbone_additional_info = {
            "input_size": self.backbone_config.input_size,
            "embedding_size": self.backbone_config.embedding_size
        }
        
        self.train_val_transformation_config = import_class(env_config.train_val_transformation_config_class)(env_config.train_val_transformation_config_path, backbone_additional_info)
        self.config_str += self.train_val_transformation_config.get_config_string()
        
        #build dataset config
        self.train_val_dataset_config = import_class(env_config.train_val_dataset_config_class)(env_config.train_val_dataset_config_path, backbone_additional_info)
        self.config_str += self.train_val_dataset_config.get_config_string()
        dataset_additional_info = {
            "num_classes": self.train_val_dataset_config.num_classes
        }
        
        # build loss config
        self.loss_config = import_class(env_config.loss_config_class)(env_config.loss_config_path, backbone_additional_info, dataset_additional_info)
        self.config_str += self.loss_config.get_config_string()
        
        # build early stopper config
        if env_config.use_early_stopper == "True":
            self.early_stopper_config = import_class(env_config.early_stopper_config_class)(env_config.early_stopper_config_path)
            self.config_str += self.early_stopper_config.get_config_string()

        # build sampler config
        if env_config.use_sampler == "True":
            self.batch_sampler_config = import_class(env_config.batch_sampler_config_class)(env_config.batch_sampler_config_path)
            self.config_str += self.batch_sampler_config.get_config_string()

        # build trainer config
        self.trainer_config = import_class(env_config.trainer_config_class)(env_config.trainer_config_path)   
        self.config_str += self.trainer_config.get_config_string()
        
        # persist configs
        config_save_dir = Path(self.trainer_config.checkpoint_save_dir)
        config_save_dir.mkdir(parents=True, exist_ok=True)
        with open(config_save_dir / f"train_config_{datetime.now()}.txt", 'w') as f:
            f.write(self.config_str)
         
        print(self.config_str)
    
    def build_instances(self, env_config: ENVConfig):
        self.backbone = import_class(env_config.backbone_class)(self.backbone_config).to(self.backbone_config.device)
        self.train_transformation = import_class(env_config.train_transformation_class)(self.train_val_transformation_config)
        self.val_transformation = import_class(env_config.val_transformation_class)(self.train_val_transformation_config)
        self.train_dataset = import_class(env_config.train_dataset_class)(self.train_val_dataset_config, self.train_transformation)
        self.val_dataset = import_class(env_config.val_dataset_class)(self.train_val_dataset_config, self.val_transformation)
        self.loss = import_class(env_config.loss_class)(self.loss_config).to(self.loss_config.device) 
        
        if self.early_stopper_config != None:
            self.early_stopper = import_class(env_config.early_stopper_class)(self.early_stopper_config)

        if self.batch_sampler_config != None:
            self.batch_sampler = import_class(env_config.batch_sampler_class)(self.batch_sampler_config, self.train_dataset)
        
        self.optimizer = build_optimizer(self.backbone, self.loss, self.trainer_config)
        self.scheduler = build_scheduler(self.optimizer, self.trainer_config)

    
    def build_trainer(self) -> Trainer:
        return Trainer(
            self.trainer_config,
            self.train_dataset, 
            self.val_dataset, 
            self.backbone, 
            self.loss,
            self.optimizer,
            self.scheduler,
            self.batch_sampler,
            self.early_stopper
        )
        
        