# Source Code Package

Core implementation of the RL image augmentation system.

## Structure

- `models/` - Neural network architectures (VGG classifier, DQN agent)
- `environment/` - RL environment and image transformation functions
- `data/` - Data handling utilities and augmented image buffer
- `utils/` - General utility functions for training and evaluation

## Key Components

### Models
- **VGG Classifier**: Pre-trained CIFAR-10 image classifier
- **DQN Agent**: Deep Q-Network for learning optimal augmentation policies

### Environment
- **ImageAugmentationEnv**: RL environment for image augmentation tasks
- **Transforms**: Library of image transformation functions and action mappings

### Data
- **AugmentedImageBuffer**: Efficient storage and retrieval of augmented images

## Usage

This package provides the core functionality used by training scripts and evaluation tools. Components are typically imported and used by higher-level scripts rather than run directly.

```python
from src.models.vgg import VGG
from src.models.agent import DQNAgent
from src.environment.environment import ImageAugmentationEnv
```
