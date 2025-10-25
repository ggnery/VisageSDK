import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path / "src"))

from backbone.inception_resnet_v2 import InceptionResNetV2 
from config.backbone.inception_resnet_v2_config import InceptionResNetV2Config
from config.dataset.train_val.vgg_face2_train_val_config import VGGFace2TrainValConfig
from dataset.train_val.vgg_face2_train_val_dataset import VGGFace2Train, VGGFace2Val
from config.trainer.trainer_config import TrainerConfig
from config.loss.triplet_loss_config import TripletLossConfig
from loss.triplet_loss import TripletLoss
from trainer import Trainer
from tools.optimizer import build_optimizer
from tools.scheduler import build_scheduler
from batch_sampler.facenet_batch_sampler import FacenetBatchSampler
from config.batch_sampler.facenet_batch_sampler_config import FacenetBatchSamplerConfig
from transformation.train_val.vgg_face2_train_val_transformation import VGGFace2TrainTransformation, VGGFace2ValTransformation
from config.transformation.train_val.vgg_face2_train_val_transformation_config import VGGFace2TrainValTransformationConfig

device = "cuda"

def main():
    dataset_config_path = src_path / "configs" / "dataset" / "vgg_face2.yaml"
    facenet_trainer_config_path = src_path / "configs" / "trainer" / "facenet_trainer.yaml" 
    inception_resnet_v2_config_path = src_path / "configs" / "backbone" / "inception_resnet_v2.yaml" 
    triplet_loss_config_path = src_path / "configs" / "loss" / "triplet_loss.yaml"
    facenet_sampler_config = src_path / "configs" / "batch_sampler" / "facenet_batch_sampler.yaml"
    transformation_config_path = src_path / "configs" / "transformation" / "vgg_face2_transformation.yaml"
    
    inception_resnet_v2_config = InceptionResNetV2Config(inception_resnet_v2_config_path)
    inception_resnet_v2_config_additional_info = {
            "input_size": inception_resnet_v2_config.input_size,
            "embedding_size": inception_resnet_v2_config.embedding_size
        }
    
    vgg_face2_dataset_config = VGGFace2TrainValConfig(dataset_config_path, inception_resnet_v2_config_additional_info)
    vgg_face2_dataset_config_additional_info = {
            "num_classes": vgg_face2_dataset_config.num_classes
        }
    
    triplet_loss_config = TripletLossConfig(triplet_loss_config_path, 
                                            inception_resnet_v2_config_additional_info, 
                                            vgg_face2_dataset_config_additional_info)

    facenet_trainer_config = TrainerConfig(facenet_trainer_config_path)
    
    batch_sampler_config = FacenetBatchSamplerConfig(facenet_sampler_config)
    
    vgg_face2_transformation_config = VGGFace2TrainValTransformationConfig(transformation_config_path, inception_resnet_v2_config_additional_info)
    vgg_face2_train_transformation = VGGFace2TrainTransformation(vgg_face2_transformation_config)
    vgg_face2_val_transformation = VGGFace2ValTransformation(vgg_face2_transformation_config)

    inception_resnet_v2 = InceptionResNetV2(inception_resnet_v2_config).to(inception_resnet_v2_config.device)
    
    triplet_loss = TripletLoss(triplet_loss_config).to(triplet_loss_config.device)
    vgg_face2_train_dataset = VGGFace2Train(vgg_face2_dataset_config, vgg_face2_train_transformation)
    vgg_face2_val_dataset = VGGFace2Val(vgg_face2_dataset_config, vgg_face2_val_transformation)
    sampler = FacenetBatchSampler(batch_sampler_config, vgg_face2_train_dataset)
    
    optimizer = build_optimizer(inception_resnet_v2, triplet_loss, facenet_trainer_config)
    scheduler = build_scheduler(optimizer, facenet_trainer_config)
    
    trainer = Trainer(facenet_trainer_config, 
            vgg_face2_train_dataset, 
            vgg_face2_val_dataset, 
            inception_resnet_v2,
            triplet_loss,
            optimizer,
            scheduler,
            sampler)
    
    trainer.train()
    
if __name__ == "__main__":
    main()