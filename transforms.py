import torchvision.transforms.functional as TF
import torch

# Questo dizionario definisce tutte le tue azioni discrete.
# È buona pratica renderlo globale o un attributo di una classe se il file diventa più complesso.
_ACTIONS_MAP = {
    0: lambda img: TF.adjust_brightness(img, brightness_factor=1.2),  # Luminosità +20%
    1: lambda img: TF.adjust_brightness(img, brightness_factor=0.8),  # Luminosità -20%
    2: lambda img: TF.adjust_contrast(img, contrast_factor=1.2),     # Contrasto +20%
    3: lambda img: TF.adjust_contrast(img, contrast_factor=0.8),     # Contrasto -20%
    4: lambda img: TF.rotate(img, angle=5),                          # Ruota +5 gradi
    5: lambda img: TF.rotate(img, angle=-5),                         # Ruota -5 gradi
    6: lambda img: TF.hflip(img),                                    # Flip orizzontale
    7: lambda img: img,                                              # No-op (non fare nulla)
    # Aggiungi qui tutte le altre azioni che vuoi esplorare
}

def get_action_transform(action_id):
    """
    Ritorna una funzione di trasformazione PyTorch (che prende un tensore o PIL Image)
    in base all'ID dell'azione.
    """
    if action_id not in _ACTIONS_MAP:
        raise ValueError(f"Azione con ID {action_id} non definita in _ACTIONS_MAP.")
    return _ACTIONS_MAP[action_id]

def get_num_actions():
    """
    Ritorna il numero totale di azioni disponibili.
    """
    return len(_ACTIONS_MAP)