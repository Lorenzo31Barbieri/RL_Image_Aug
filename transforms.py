# transforms.py (Updated)

import torchvision.transforms.functional as TF
import torch


_ACTIONS_MAP = {
    0: (lambda img: TF.adjust_brightness(img, brightness_factor=1.2), "Brightness +20%"),
    1: (lambda img: TF.adjust_brightness(img, brightness_factor=0.8), "Brightness -20%"),
    2: (lambda img: TF.adjust_contrast(img, contrast_factor=1.2), "Contrast +20%"),
    3: (lambda img: TF.adjust_contrast(img, contrast_factor=0.8), "Contrast -20%"),
    # Per CIFAR10, rotazioni più ampie potrebbero essere troppo aggressive su immagini piccole
    # Considera di ridurre l'angolo se le performance calano drasticamente
    4: (lambda img: TF.rotate(img, angle=5), "Rotate +5 degrees"),
    5: (lambda img: TF.rotate(img, angle=-5), "Rotate -5 degrees"),
    6: (lambda img: TF.hflip(img), "Horizontal Flip"),
    7: (lambda img: img, "No-op (Identity)"),
}

def get_action_transform(action_id):
    if action_id not in _ACTIONS_MAP:
        raise ValueError(f"Action with ID {action_id} not defined in _ACTIONS_MAP.")
    return _ACTIONS_MAP[action_id][0] 

def get_num_actions():
    return len(_ACTIONS_MAP)

def get_all_transforms(image_size=None): # image_size non è più usato direttamente qui
    return [ _ACTIONS_MAP[i] for i in sorted(_ACTIONS_MAP.keys()) ]