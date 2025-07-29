# src/__init__.py
"""
Main source code package for the RL Image Augmentation project.
"""

__version__ = "1.0.0"

# Import commonly used classes for easier access
from .models.vgg import VGG
from .models.agent import DQNAgent
from .environment.environment import ImageAugmentationEnv
from .environment.transforms import get_num_actions, get_action_transform, get_action_name
from .data.augmented_image_buffer import AugmentedImageBuffer

__all__ = [
    'VGG',
    'DQNAgent', 
    'ImageAugmentationEnv',
    'get_num_actions',
    'get_action_transform',
    'get_action_name',
    'AugmentedImageBuffer'
]