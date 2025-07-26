import torchvision.transforms.functional as TF
import torch
import random


def apply_gaussian_noise(img, std=0.02):
    """Add slight gaussian noise to the image."""
    noise = torch.randn_like(img) * std
    return torch.clamp(img + noise, 0, 1)


def apply_random_crop_pad(img, padding=2):
    """Apply random crop with padding - good for CIFAR-10."""
    # Add padding and then crop back to original size
    padded = TF.pad(img, padding, padding_mode='reflect')
    # Random crop
    _, h, w = img.shape
    top = random.randint(0, 2 * padding)
    left = random.randint(0, 2 * padding)
    return TF.crop(padded, top, left, h, w)


def apply_color_jitter(img, brightness=0.1, contrast=0.1, saturation=0.1):
    """Apply mild color jittering."""
    img = TF.adjust_brightness(img, 1 + random.uniform(-brightness, brightness))
    img = TF.adjust_contrast(img, 1 + random.uniform(-contrast, contrast))
    img = TF.adjust_saturation(img, 1 + random.uniform(-saturation, saturation))
    return img


# Improved action map with more suitable transformations for CIFAR-10
_ACTIONS_MAP = {
    # Brightness adjustments (milder)
    0: (lambda img: TF.adjust_brightness(img, brightness_factor=1.1), "Brightness +10%"),
    1: (lambda img: TF.adjust_brightness(img, brightness_factor=0.9), "Brightness -10%"),
    
    # Contrast adjustments (milder)
    2: (lambda img: TF.adjust_contrast(img, contrast_factor=1.1), "Contrast +10%"),
    3: (lambda img: TF.adjust_contrast(img, contrast_factor=0.9), "Contrast -10%"),
    
    # Saturation adjustments (new)
    4: (lambda img: TF.adjust_saturation(img, saturation_factor=1.2), "Saturation +20%"),
    5: (lambda img: TF.adjust_saturation(img, saturation_factor=0.8), "Saturation -20%"),
    
    # Small rotations (reduced angles)
    6: (lambda img: TF.rotate(img, angle=3), "Rotate +3 degrees"),
    7: (lambda img: TF.rotate(img, angle=-3), "Rotate -3 degrees"),
    
    # Random crop with padding (good for CIFAR-10)
    8: (lambda img: apply_random_crop_pad(img, padding=2), "Random Crop (pad=2)"),
    9: (lambda img: apply_random_crop_pad(img, padding=3), "Random Crop (pad=3)"),
    
    # Gaussian noise (very mild)
    10: (lambda img: apply_gaussian_noise(img, std=0.01), "Gaussian Noise (mild)"),
    
    # Color jitter
    11: (lambda img: apply_color_jitter(img), "Color Jitter"),
    
    # Horizontal flip (kept but will be penalized in reward)
    12: (lambda img: TF.hflip(img), "Horizontal Flip"),
    
    # Sharpness adjustment (new)
    13: (lambda img: TF.adjust_sharpness(img, sharpness_factor=1.3), "Sharpen +30%"),
    14: (lambda img: TF.adjust_sharpness(img, sharpness_factor=0.7), "Soften -30%"),
    
    # Identity (no-op)
    15: (lambda img: img, "No-op (Identity)"),
}


def get_action_transform(action_id):
    """Get the transformation function for a given action ID."""
    if action_id not in _ACTIONS_MAP:
        raise ValueError(f"Action with ID {action_id} not defined in _ACTIONS_MAP.")
    return _ACTIONS_MAP[action_id][0]


def get_num_actions():
    """Get the total number of available actions."""
    return len(_ACTIONS_MAP)


def get_all_transforms():
    """Get all available transformations with their names."""
    return [_ACTIONS_MAP[i] for i in sorted(_ACTIONS_MAP.keys())]


def get_action_name(action_id):
    """Get the name of an action."""
    if action_id not in _ACTIONS_MAP:
        raise ValueError(f"Action with ID {action_id} not defined in _ACTIONS_MAP.")
    return _ACTIONS_MAP[action_id][1]


def get_conservative_actions():
    """Get list of conservative (less aggressive) action IDs."""
    # These actions are less likely to harm the image
    return [0, 1, 2, 3, 4, 5, 10, 11, 15]


def get_aggressive_actions():
    """Get list of aggressive action IDs that should be used carefully."""
    # These actions can significantly change the image
    return [6, 7, 8, 9, 12, 13, 14]


# Action categories for reward calculation
ACTION_CATEGORIES = {
    'brightness': [0, 1],
    'contrast': [2, 3],
    'saturation': [4, 5],
    'rotation': [6, 7],
    'spatial': [8, 9, 12],  # spatial transformations
    'noise': [10],
    'color': [11],
    'sharpness': [13, 14],
    'identity': [15]
}