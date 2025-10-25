import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path / "src"))

from config.backbone.inception_resnet_v2_config import InceptionResNetV2Config
from config.dataset.eval.lfw_eval_config import LFWEvalDatasetConfig
from config.transformation.eval.lfw_eval_transformation import LFWEvalTransformationConfig
from transformation.eval.lfw_eval_transformation import LFWEvalTransformation

def main():
    inception_resnet_v2_config_path = src_path / "configs" / "backbone" / "inception_resnet_v2.yaml" 
    lfw_eval_dataset_config_path = src_path / "configs" / "dataset" / "eval" / "lfw_eval.yaml"
    lfw_eval_transformation_config_path = src_path / "configs" / "transformation" / "eval" / "lfw_eval_transformation.yaml"

    inception_resnet_v2_config = InceptionResNetV2Config(inception_resnet_v2_config_path)
    inception_resnet_v2_config_additional_info = {
            "input_size": inception_resnet_v2_config.input_size,
            "embedding_size": inception_resnet_v2_config.embedding_size
        }
    
    lfw_eval_dataset_config = LFWEvalDatasetConfig(lfw_eval_dataset_config_path, inception_resnet_v2_config_additional_info)
    print(lfw_eval_dataset_config.get_config_string())
    
    lfw_eval_transformation_config = LFWEvalTransformationConfig(lfw_eval_transformation_config_path, inception_resnet_v2_config_additional_info)
    print(lfw_eval_transformation_config.get_config_string())
    
    lfw_eval_transformation = LFWEvalTransformation(lfw_eval_transformation_config)
    
if __name__ == "__main__":
    main()