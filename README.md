# EmbeddingFramework

A modular, configuration-driven deep learning framework designed for embedding learning tasks such as face recognition, person re-identification, and metric learning. The framework emphasizes modularity, extensibility, and reproducibility without requiring modifications to core scripts.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [How to Train a Model](#how-to-train-a-model)
- [Adding Custom Components](#adding-custom-components)
- [Framework Architecture](#framework-architecture)
- [Constraints and Best Practices](#constraints-and-best-practices)
- [Examples](#examples)


## Architecture Overview

The EmbeddingFramework follows a modular architecture with seven core components:

1. **Backbone**: Neural network architectures for feature extraction
2. **Loss Functions**: Training objectives for embedding learning
3. **Datasets**: Data loading and preprocessing pipelines
4. **Batch Samplers**: Custom sampling strategies for training
5. **Early Stoppers**: Customizable early stopping strategies for training
6. **Trainer**: Training loop orchestration
7. **Configuration System**: YAML-based configuration management

All components inherit from abstract base classes, ensuring consistent interfaces while allowing complete customization.

## Quick Start

### Installation

```bash
git clone <repository-url>
cd EmbeddingFramework
pip install -r requirements.txt
```

### Prerequisites

- Python 3.12
- CUDA-capable GPU (recommended)

## How to Train a Model

The framework uses environment variables to configure experiments, allowing you to train models without modifying any base scripts.

### 1. Set Environment Variables

Make a copy of `.env.example` and rename it to `.env`:

Example of `.env`:
```bash
# Backbone Configuration
export BACKBONE_CONFIG_PATH="./configs/backbone/inception_resnet_v2.yaml"
export BACKBONE_CONFIG_CLASS="config.backbone.inception_resnet_v2_config.InceptionResnetV2Config"
export BACKBONE_CLASS="backbone.inception_resnet_v2.InceptionResNetV2"

# Loss Configuration
export LOSS_CONFIG_PATH="./configs/loss/triplet_loss.yaml"
export LOSS_CONFIG_CLASS="config.loss.triplet_loss_config.TripletLossConfig"
export LOSS_CLASS="loss.triplet_loss.TripletLoss"

# Dataset Configuration
export DATASET_CONFIG_PATH="./configs/dataset/vgg_face2.yaml"
export DATASET_CONFIG_CLASS="config.dataset.vgg_face2_config.VGGFace2Config"
export TRAIN_DATASET_CLASS="dataset.vgg_face2_dataset.VGGFace2Train"
export VAL_DATASET_CLASS="dataset.vgg_face2_dataset.VGGFace2Val"

# Batch Sampler Configuration (Optional - set USE_SAMPLER=True to enable)
export USE_SAMPLER=True
export BATCH_SAMPLER_CONFIG_PATH="./configs/batch_sampler/facenet_batch_sampler.yaml"
export BATCH_SAMPLER_CONFIG_CLASS="config.batch_sampler.facenet_batch_sampler_config.FacenetBatchSamplerConfig"
export BATCH_SAMPLER_CLASS="batch_sampler.facenet_base_sampler.FacenetBatchSampler"

# Early Stopper Configuration (Optional - set USE_EARLY_STOPPER=True to enable)
export USE_EARLY_STOPPER=True
export EARLY_STOPPER_CONFIG_PATH="./configs/early_stopper/adaptative_early_stopper.yaml"
export EARLY_STOPPER_CONFIG_CLASS="config.early_stopper.adaptative_early_stopper_config.AdaptativeEarlyStopperConfig"
export EARLY_STOPPER_CLASS="early_stopper.adaptative_early_stopper.AdaptativeEarlyStopper"

# Trainer Configuration
export TRAINER_CONFIG_PATH="./configs/trainer/facenet_trainer.yaml"
export TRAINER_CONFIG_CLASS="config.trainer.trainer_config.TrainerConfig"
```

### 2. Run Training

```bash
python train.py
```

The training script will:
- Load and instantiate all components based on environment variables
- Create data loaders with your specified sampling strategy
- Initialize optimizers and schedulers
- Run the training loop with automatic checkpointing
- Save training history and configuration snapshots

### 3. Monitor Training

Training progress is logged with:
- Real-time loss updates
- Learning rate scheduling information
- Automatic checkpoint saving
- Detailed training statistics in JSON format

### 4. Configuration Files

The framework includes example configurations:

#### Backbone Configuration (`configs/backbone/inception_resnet_v2.yaml`)
```yaml
input_size: [299, 299]
embedding_size: 512
dropout_keep: 0.4 
device: cuda
```

#### Loss Configuration (`configs/loss/triplet_loss.yaml`)
```yaml
margin: 0.2
device: cuda
```

#### Trainer Configuration (`configs/trainer/facenet_trainer.yaml`)
```yaml
optimizer: 
  type: RMSprop             
  params:
    lr: 0.1
    weight_decay : 0.0001
    alpha: 0.9            
    eps: 1.0         
    momentum: 0.9            
    centered: false

lr_schedule:
  type: LambdaLR
  params:
    2: 0.01
    3: 0.001
    4: 0.0001

dataloader:
  train:
    batch_size: None # If sampler is present this must be None
    shuffle: None # If sampler is present this must be None
    num_workers: 8
  val:
    batch_size: 150
    shuffle: true
    num_workers: 8

num_epochs: 4 
device: cuda     

checkpoint:
  save_dir: ./checkpoints/facenet
  load_path: null
  frequency: 5
```

## Adding Custom Components

The framework's strength lies in its extensibility. You can add custom components without modifying any base scripts.

### Custom Backbone

#### 1. Create Your Backbone Class

```python
# src/backbone/my_custom_backbone.py
from typing import Dict
import torch
import torch.nn as nn
from backbone.base_backbone import BaseBackbone
from config.backbone.my_custom_backbone_config import MyCustomBackboneConfig

class MyCustomBackbone(BaseBackbone):
    def __init__(self, config: MyCustomBackboneConfig):
        super().__init__(config)
        
        # Initialize your architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, self.embedding_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = features.view(features.size(0), -1)
        embeddings = self.classifier(features)
        
        # Return only embeddings
        return embeddings
    
    def get_backbone_info(self) -> Dict:
        # Return required backbone info for loss computation
        return {
            "embedding_matrix": self.classifier.weight,
            "embedding_bias": self.classifier.bias
        }
```

#### 2. Create Configuration Class

```python
# src/config/backbone/my_custom_backbone_config.py
from config.backbone.base_backbone_config import BaseBackboneConfig

class MyCustomBackboneConfig(BaseBackboneConfig):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load custom parameters from YAML
        self.custom_param = self.config['custom_param']
        self.embedding_size = self.config['embedding_size']
        self.device = self.config['device']
        
        self.build_config()
```

#### 3. Create YAML Configuration

```yaml
# configs/backbone/my_custom_backbone.yaml
embedding_size: 256
custom_param: 0.5
device: cuda
```

#### 4. Update Environment Variables

```bash
export BACKBONE_CONFIG_PATH="./configs/backbone/my_custom_backbone.yaml"
export BACKBONE_CONFIG_CLASS="config.backbone.my_custom_backbone_config.MyCustomBackboneConfig"
export BACKBONE_CLASS="backbone.my_custom_backbone.MyCustomBackbone"
```

### Custom Loss Function

#### 1. Create Your Loss Class

```python
# src/loss/my_custom_loss.py
from typing import Dict, Tuple
import torch
import torch.nn as nn
from loss.base_loss import BaseLoss
from config.loss.my_custom_loss_config import MyCustomLossConfig

class MyCustomLoss(BaseLoss):
    def __init__(self, config: MyCustomLossConfig):
        super().__init__(config)
        self.margin = config.margin
    
    def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor, backbone_info: Dict) -> Tuple[torch.Tensor, Dict]:
        # Simple example: MSE loss with margin
        batch_size = embeddings.size(0)
        
        # Access backbone info if needed for loss computation
        embedding_matrix = backbone_info.get("embedding_matrix", None)
        embedding_bias = backbone_info.get("embedding_bias", None)
        
        # Your custom loss calculation here
        loss = nn.MSELoss()(embeddings, torch.zeros_like(embeddings)) + self.margin
        
        loss_stats = {
            'batch_size': batch_size,
            'loss_value': loss.item(),
            'used_backbone_info': embedding_matrix is not None
        }
        
        return loss, loss_stats
```

#### 2. Create Configuration Class

```python
# src/config/loss/my_custom_loss_config.py
from config.loss.base_loss_config import BaseLossConfig

class MyCustomLossConfig(BaseLossConfig):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load parameters from YAML
        self.margin = self.config['margin']
        self.device = self.config['device']
        
        self.build_config()
```

#### 3. Create YAML Configuration

```yaml
# configs/loss/my_custom_loss.yaml
margin: 1.0
device: cuda
```

#### 4. Update Environment Variables

```bash
export LOSS_CONFIG_PATH="./configs/loss/my_custom_loss.yaml"
export LOSS_CONFIG_CLASS="config.loss.my_custom_loss_config.MyCustomLossConfig"
export LOSS_CLASS="loss.my_custom_loss.MyCustomLoss"
```

### Custom Dataset

#### 1. Create Dataset Classes

```python
# src/dataset/my_custom_dataset.py
from typing import List, Tuple
from pathlib import Path
import torchvision.transforms as transforms
from dataset.base_dataset import BaseDataset
from config.dataset.my_custom_dataset_config import MyCustomDatasetConfig

class MyCustomDatasetTrain(BaseDataset):
    def read_data(self, dataset_config: MyCustomDatasetConfig) -> List[Tuple[str, str]]:
        # Read training data from directory structure: data_dir/class_name/image.jpg
        data_dir = Path(dataset_config.data_dir)
        data_pairs = []
        
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_file in class_dir.glob("*.jpg"):  # Simple: just JPG files
                    data_pairs.append((class_name, str(img_file)))
        
        return data_pairs
    
    def build_transformation(self, dataset_config: MyCustomDatasetConfig) -> List:
        return [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]

class MyCustomDatasetVal(BaseDataset):
    def read_data(self, dataset_config: MyCustomDatasetConfig) -> List[Tuple[str, str]]:
        # Same logic as training for simplicity
        return MyCustomDatasetTrain.read_data(self, dataset_config)
    
    def build_transformation(self, dataset_config: MyCustomDatasetConfig) -> List:
        return [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
```

#### 2. Create Configuration Class

```python
# src/config/dataset/my_custom_dataset_config.py
from config.dataset.base_dataset_config import BaseDatasetConfig

class MyCustomDatasetConfig(BaseDatasetConfig):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load parameters from YAML
        self.data_dir = self.config['data_dir']
        
        self.build_config()
```

#### 3. Create YAML Configuration

```yaml
# configs/dataset/my_custom_dataset.yaml
data_dir: "./data/my_dataset"
```

#### 4. Update Environment Variables

```bash
export DATASET_CONFIG_PATH="./configs/dataset/my_custom_dataset.yaml"
export DATASET_CONFIG_CLASS="config.dataset.my_custom_dataset_config.MyCustomDatasetConfig"
export TRAIN_DATASET_CLASS="dataset.my_custom_dataset.MyCustomDatasetTrain"
export VAL_DATASET_CLASS="dataset.my_custom_dataset.MyCustomDatasetVal"
```

### Custom Batch Sampler

#### 1. Create Batch Sampler Class

```python
# src/batch_sampler/my_custom_sampler.py
from typing import Iterator, List
import random
from batch_sampler.base_batch_sampler import BaseBatchSampler
from config.batch_sampler.my_custom_sampler_config import MyCustomSamplerConfig

class MyCustomSampler(BaseBatchSampler):
    def __init__(self, config: MyCustomSamplerConfig, dataset):
        super().__init__(dataset)
        self.batch_size = config.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        # Simple random sampling
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        
        for i in range(0, len(indices), self.batch_size):
            batch = indices[i:i + self.batch_size]
            if len(batch) == self.batch_size:  # Only yield full batches
                yield batch
    
    def __len__(self) -> int:
        return len(self.dataset) // self.batch_size
```

#### 2. Create Configuration Class

```python
# src/config/batch_sampler/my_custom_sampler_config.py
from config.batch_sampler.base_batch_sampler_config import BaseBatchSamplerConfig

class MyCustomSamplerConfig(BaseBatchSamplerConfig):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load parameters from YAML
        self.batch_size = self.config['batch_size']
        
        self.build_config()
```

#### 3. Create YAML Configuration

```yaml
# configs/batch_sampler/my_custom_sampler.yaml
batch_size: 32
```

#### 4. Update Environment Variables

```bash
export USE_SAMPLER=True
export BATCH_SAMPLER_CONFIG_PATH="./configs/batch_sampler/my_custom_sampler.yaml"
export BATCH_SAMPLER_CONFIG_CLASS="config.batch_sampler.my_custom_sampler_config.MyCustomSamplerConfig"
export BATCH_SAMPLER_CLASS="batch_sampler.my_custom_sampler.MyCustomSampler"
```

### Custom Early Stopper

#### 1. Create Early Stopper Class

```python
# src/early_stopper/my_custom_early_stopper.py
from typing import override
import logging
from early_stopper.base_early_stopper import BaseEarlyStopper
from config.early_stopper.my_custom_early_stopper_config import MyCustomEarlyStopperConfig

class MyCustomEarlyStopper(BaseEarlyStopper):
    def __init__(self, config: MyCustomEarlyStopperConfig):
        super().__init__(config)
        
        self.patience = config.patience
        self.min_delta = config.min_delta
        self.wait_count = 0
        self.best_loss = float('inf')
        self.logger = logging.getLogger(__name__)
    
    @override
    def early_stop(self, val_loss: float) -> bool:
        # Check if validation loss improved by at least min_delta
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait_count = 0
            self.logger.info(f"Validation loss improved to {val_loss:.4f}")
        else:
            self.wait_count += 1
            self.logger.info(f"No improvement. Wait count: {self.wait_count}/{self.patience}")
            
            if self.wait_count >= self.patience:
                self.logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return True
        
        return False
```

#### 2. Create Configuration Class

```python
# src/config/early_stopper/my_custom_early_stopper_config.py
from config.early_stopper.base_early_stopper_config import BaseEarlyStopperConfig

class MyCustomEarlyStopperConfig(BaseEarlyStopperConfig):
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Load parameters from YAML
        self.patience = self.config['patience']
        self.min_delta = self.config['min_delta']
        
        self.build_config()
```

#### 3. Create YAML Configuration

```yaml
# configs/early_stopper/my_custom_early_stopper.yaml
patience: 10        # Number of epochs to wait before stopping
min_delta: 0.001    # Minimum improvement required to reset patience
```

#### 4. Update Environment Variables

```bash
export USE_EARLY_STOPPER=True
export EARLY_STOPPER_CONFIG_PATH="./configs/early_stopper/my_custom_early_stopper.yaml"
export EARLY_STOPPER_CONFIG_CLASS="config.early_stopper.my_custom_early_stopper_config.MyCustomEarlyStopperConfig"
export EARLY_STOPPER_CLASS="early_stopper.my_custom_early_stopper.MyCustomEarlyStopper"
```

### Custom Configuration

All configuration classes inherit from `BaseConfig` which provides:

- Automatic YAML loading
- Configuration string generation for logging
- Extensible `build_config()` method for custom initialization

### Custom Optimizers and Schedulers

The framework includes tools for optimizer and scheduler management. To add custom ones:

#### 1. Extend Optimizer Builder

```python
# Modify src/tools/optimizer.py
def build_optimizer(model: BaseBackbone, config: TrainerConfig) -> Optimizer:
    optimizer_type = config.optmizer_type
    optimizer_params = config.optmizer_params
    
    match optimizer_type:
        case "RMSprop":
            return RMSprop(model.parameters(), **optimizer_params)
        case "Adam":
            return Adam(model.parameters(), **optimizer_params)
        # Add your custom optimizer
        case "MyCustomOptimizer":
            return MyCustomOptimizer(model.parameters(), **optimizer_params)
        case _:
            raise Exception(f"Optimizer {optimizer_type} not implemented")
```

#### 2. Extend Scheduler Builder

```python
# Modify src/tools/scheduler.py
def build_scheduler(optimizer: Optimizer, config: TrainerConfig) -> LRScheduler:
    scheduler_type = config.lr_schedule_type
    scheduler_params = config.lr_schedule_params
    
    match scheduler_type:
        case "LambdaLR":
            return build_lambda_lr(optimizer, scheduler_params)
        # Add your custom scheduler
        case "MyCustomScheduler":
            return build_custom_scheduler(optimizer, scheduler_params)
        case _:
            raise Exception("Scheduler not implemented")
```

## Framework Architecture

### Design Principles

1. **Modularity**: Each component is independent and interchangeable
2. **Configuration-Driven**: All experiments defined through YAML configs and environment variables
3. **Extensibility**: Add new components without modifying base code
4. **Reproducibility**: Automatic config logging and checkpoint management
5. **Type Safety**: Strong typing throughout the framework

### Core Components

#### Base Classes Hierarchy

All components inherit from abstract base classes that define consistent interfaces:

```
BaseConfig
├── BaseBackboneConfig
├── BaseLossConfig
├── BaseDatasetConfig
├── BaseBatchSamplerConfig
├── BaseEarlyStopperConfig
└── TrainerConfig

BaseBackbone (nn.Module)
└── Custom backbone implementations

BaseLoss (nn.Module)
└── Custom loss implementations

BaseDataset (Dataset)
└── Custom dataset implementations

BaseBatchSampler (BatchSampler)
└── Custom sampler implementations

BaseEarlyStopper (ABC)
└── Custom early stopper implementations
```

#### Configuration System

The configuration system follows a hierarchical approach:

1. **Environment Variables** (`.env` or shell exports) define which components to use
2. **YAML Files** store component-specific parameters
3. **Config Classes** load, validate, and structure configuration data
4. **Automatic Logging** saves complete configuration snapshots with each training run


### Key Design Patterns

#### 1. Strategy Pattern
Each component type (backbone, loss, dataset, etc.) uses the strategy pattern where:
- Abstract base classes define interfaces
- Concrete implementations provide specific behavior
- Environment variables select which strategy to use

#### 2. Dependency Injection
Components receive their dependencies through constructor injection:
- Config objects are injected into components
- Datasets receive both dataset and backbone configs
- Batch samplers receive config and dataset instances

#### 3. Factory Pattern
The framework uses factory patterns for:
- Dynamic class loading via `import_class()`
- Optimizer creation through `build_optimizer()`
- Scheduler creation through `build_scheduler()`

#### 4. Template Method Pattern
Base classes define the overall structure while allowing customization:
- `BaseDataset` handles common dataset operations
- `BaseConfig` manages YAML loading and logging
- `Trainer` orchestrates the training process

### Extension Points

The framework provides several extension points for customization:

1. **Component Interface Compliance**: Implement abstract methods from base classes
2. **Configuration Extension**: Add custom parameters through YAML and config classes  
3. **Optimizer/Scheduler Integration**: Add new optimizers and schedulers through tool functions
4. **Statistics and Logging**: Custom metrics through component return values

## Constraints and Best Practices

### Framework Constraints

#### 1. **No Modification of Base Scripts**
The core principle of the EmbeddingFramework is that users should **never need to modify** any base scripts or core files:

**Files that must NOT be modified:**
- `train.py` - Main training script
- `src/trainer/trainer.py` - Core training loop
- Any `base_*.py` files in the src directory
- Core utility files in `src/tools/`

**How to customize instead:**
- Create new component implementations that inherit from base classes
- Write custom configuration files in YAML format
- Set environment variables to specify which components to use
- Extend optimizer/scheduler tools through the existing switch statements

#### 2. **Interface Compliance Requirements**

All custom components must strictly implement the required abstract methods:

**BaseBackbone Requirements:**
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Must return embeddings_tensor (B x embedding_size)
    pass

def get_backbone_info(self) -> Dict:
    # Must return dict with keys: "embedding_matrix", "embedding_bias"
    pass
```

**BaseLoss Requirements:**
```python
def forward(self, embeddings: torch.Tensor, y_true: torch.Tensor, backbone_info: Dict) -> Tuple[torch.Tensor, Dict]:
    # Must return (loss_tensor, loss_stats_dict)
    # backbone_info contains embedding matrix and bias from backbone
    pass
```

**BaseDataset Requirements:**
```python
def read_data(self, dataset_config: BaseDatasetConfig) -> List[Tuple[str, str]]:
    # Must return list of (class_label, image_path) tuples
    pass

def build_transformation(self, dataset_config: BaseDatasetConfig) -> List:
    # Must return list of torchvision transforms
    pass
```

**BaseBatchSampler Requirements:**
```python
def __iter__(self) -> Iterator[List[int]]:
    # Must yield lists of dataset indices
    pass

def __len__(self) -> int:
    # Must return total number of batches
    pass
```

**BaseEarlyStopper Requirements:**
```python
def early_stop(self, val_loss: float) -> bool:
    # Must return True to trigger early stopping, False to continue
    # Called after each validation epoch with current validation loss
    pass
```

#### 3. **Configuration Structure Requirements**

All configuration classes must:
- Inherit from the appropriate base configuration class
- Load parameters from YAML files in the constructor
- Call `self.build_config()` after parameter loading
- Follow the naming convention: `[Component]Config`

#### 4. **Environment Variable Requirements**

The framework expects exactly these environment variables (all required):
```bash
BACKBONE_CONFIG_PATH, BACKBONE_CONFIG_CLASS, BACKBONE_CLASS
LOSS_CONFIG_PATH, LOSS_CONFIG_CLASS, LOSS_CLASS  
DATASET_CONFIG_PATH, DATASET_CONFIG_CLASS, TRAIN_DATASET_CLASS, VAL_DATASET_CLASS
USE_SAMPLER  # Set to "True" or "False"
# If USE_SAMPLER=True, these are also required:
BATCH_SAMPLER_CONFIG_PATH, BATCH_SAMPLER_CONFIG_CLASS, BATCH_SAMPLER_CLASS
USE_EARLY_STOPPER  # Set to "True" or "False"
# If USE_EARLY_STOPPER=True, these are also required:
EARLY_STOPPER_CONFIG_PATH, EARLY_STOPPER_CONFIG_CLASS, EARLY_STOPPER_CLASS
TRAINER_CONFIG_PATH, TRAINER_CONFIG_CLASS
```

### Best Practices

#### Development Workflow

1. **Study Existing Implementations First**
   - Examine `src/backbone/inception_resnet_v2.py` for backbone patterns
   - Review `src/loss/triplet_loss.py` for loss function structure
   - Analyze `src/dataset/vgg_face2_dataset.py` for dataset implementation
   - Study `src/batch_sampler/facenet_base_sampler.py` for sampling strategies

2. **Follow Naming Conventions**
   ```
   Component Class: MyCustomBackbone
   Config Class: MyCustomBackboneConfig  
   Config File: my_custom_backbone.yaml
   ```

3. **Create Comprehensive Tests**
   - Test component initialization with various config parameters
   - Verify output shapes and types match expected interfaces
   - Test edge cases and error conditions

4. **Document Configuration Parameters**
   ```yaml
   # my_component.yaml
   embedding_size: 512  # Size of output embeddings
   custom_param: 0.5    # Custom parameter for my component
   device: cuda         # Training device
   ```

## Examples

### FaceNet Training Example

The repository includes a complete FaceNet implementation for face recognition using the VGGFace2 dataset.

#### Complete Setup Script

Create a shell script `setup_facenet.sh`:

```bash
#!/bin/bash

# FaceNet Environment Configuration
export BACKBONE_CONFIG_PATH="./configs/backbone/inception_resnet_v2.yaml"
export BACKBONE_CONFIG_CLASS="config.backbone.inception_resnet_v2_config.InceptionResnetV2Config"
export BACKBONE_CLASS="backbone.inception_resnet_v2.InceptionResNetV2"

export LOSS_CONFIG_PATH="./configs/loss/triplet_loss.yaml"
export LOSS_CONFIG_CLASS="config.loss.triplet_loss_config.TripletLossConfig"
export LOSS_CLASS="loss.triplet_loss.TripletLoss"

export DATASET_CONFIG_PATH="./configs/dataset/vgg_face2.yaml"
export DATASET_CONFIG_CLASS="config.dataset.vgg_face2_config.VGGFace2Config"
export TRAIN_DATASET_CLASS="dataset.vgg_face2_dataset.VGGFace2Train"
export VAL_DATASET_CLASS="dataset.vgg_face2_dataset.VGGFace2Val"

export USE_SAMPLER=True
export BATCH_SAMPLER_CONFIG_PATH="./configs/batch_sampler/facenet_batch_sampler.yaml"
export BATCH_SAMPLER_CONFIG_CLASS="config.batch_sampler.facenet_batch_sampler_config.FacenetBatchSamplerConfig"
export BATCH_SAMPLER_CLASS="batch_sampler.facenet_base_sampler.FacenetBatchSampler"

export TRAINER_CONFIG_PATH="./configs/trainer/facenet_trainer.yaml"
export TRAINER_CONFIG_CLASS="config.trainer.trainer_config.TrainerConfig"

# Optional: Enable early stopping
export USE_EARLY_STOPPER=True
export EARLY_STOPPER_CONFIG_PATH="./configs/early_stopper/adaptative_early_stopper.yaml"
export EARLY_STOPPER_CONFIG_CLASS="config.early_stopper.adaptative_early_stopper_config.AdaptativeEarlyStopperConfig"
export EARLY_STOPPER_CLASS="early_stopper.adaptative_early_stopper.AdaptativeEarlyStopper"

# Run training
python train.py
```

#### Expected Training Output

```
=========================
InceptionResnetV2Config CONFIGURATION
=========================
embedding_size: 512
dropout_keep: 0.4
device: cuda
input_size: [299, 299]
=========================

=========================
TripletLossConfig CONFIGURATION
=========================
margin: 0.2
device: cuda
=========================

Found 1250 identities with >= 4 samples

Train epoch 0: 100%|██████████| 312/312 [15:42<00:00, 1.32it/s, loss=0.891]
Val epoch 0: 100%|██████████| 167/167 [01:28<00:00, 1.89it/s]
Epoch 0/4 - LR: 0.100000 - Train Loss: 1.2456 - Val Loss: 0.9234
Saved checkpoint: ./checkpoints/facenet/InceptionResNetV2_TripletLoss_VGGFace2_epoch_0.pth

Train epoch 1: 100%|██████████| 312/312 [15:40<00:00, 1.33it/s, loss=0.234]
Val epoch 1: 100%|██████████| 167/167 [01:27<00:00, 1.91it/s]
Epoch 1/4 - LR: 0.100000 - Train Loss: 0.8923 - Val Loss: 0.7456
```

## Contributing

When contributing new components to the framework:

1. **Follow Interface Contracts**: Implement all required abstract methods
2. **Include Configuration**: Provide both config class and YAML examples
3. **Add Documentation**: Include docstrings and usage examples
4. **Test Thoroughly**: Verify components work with different configurations
5. **Create a merge request**: Create a merge request to add your new component