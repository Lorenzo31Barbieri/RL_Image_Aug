# transforms.py

import torchvision.transforms.functional as TF
import torch

# IMPORTANT: Each value in _ACTIONS_MAP must now be a TUPLE: (function_lambda, "Description_String")
_ACTIONS_MAP = {
    0: (lambda img: TF.adjust_brightness(img, brightness_factor=1.2), "Brightness +20%"),
    1: (lambda img: TF.adjust_brightness(img, brightness_factor=0.8), "Brightness -20%"),
    2: (lambda img: TF.adjust_contrast(img, contrast_factor=1.2), "Contrast +20%"),
    3: (lambda img: TF.adjust_contrast(img, contrast_factor=0.8), "Contrast -20%"),
    4: (lambda img: TF.rotate(img, angle=5), "Rotate +5 degrees"),
    5: (lambda img: TF.rotate(img, angle=-5), "Rotate -5 degrees"),
    6: (lambda img: TF.hflip(img), "Horizontal Flip"),
    7: (lambda img: img, "No-op (Identity)"),
}

def get_action_transform(action_id):
    """
    Returns a PyTorch transformation function (which takes a tensor)
    based on the action ID.
    """
    if action_id not in _ACTIONS_MAP:
        raise ValueError(f"Action with ID {action_id} not defined in _ACTIONS_MAP.")
    # Returns only the function (the first element of the tuple)
    return _ACTIONS_MAP[action_id][0] 

def get_num_actions():
    """
    Returns the total number of available actions.
    """
    return len(_ACTIONS_MAP)

def get_all_transforms(image_size=None):
    """
    Returns a list of all available transformation functions AND their names.
    Useful for initializing the environment and for display.
    """
    # This now returns a list of tuples: (function, name_string)
    return [ _ACTIONS_MAP[i] for i in sorted(_ACTIONS_MAP.keys()) ]