import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from collections import defaultdict

# Importa i tuoi moduli personalizzati
from vgg_classifier import VGGClassifierWrapper
from dataset import PetImagesDataset 
from transforms import _ACTIONS_MAP

# --- CONFIGURAZIONE GLOBALE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Se hai una GPU NVIDIA e vuoi usare CUDA, decommenta la riga sotto e commenta quella sopra:
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Configurazione del Dataset e Percorsi ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = './output/VGG16_trained.pth'
IMAGE_SIZE = 224
NUM_CLASSES = 2

# --- CLASSE OTTIMIZZATA PER APPLICARE LE TRASFORMAZIONI ---
class OptimizedFixedAugmentationTransform:
    """
    Versione ottimizzata che applica trasformazioni in batch quando possibile
    e utilizza operazioni vettorizzate.
    """
    def __init__(self, action_ids_to_apply):
        """
        Args:
            action_ids_to_apply (list): Lista di ID delle azioni da _ACTIONS_MAP
        """
        self.transforms_to_apply = []
        self.transform_names = []
        
        for action_id in action_ids_to_apply:
            if action_id in _ACTIONS_MAP:
                self.transforms_to_apply.append(_ACTIONS_MAP[action_id][0])
                self.transform_names.append(_ACTIONS_MAP[action_id][1])
            else:
                raise ValueError(f"Action ID {action_id} not found in _ACTIONS_MAP.")
    
    def __call__(self, img_tensor):
        """
        Applica le trasformazioni ottimizzate al tensore immagine.
        """
        # Clona il tensore una sola volta se necessario
        if len(self.transforms_to_apply) > 0:
            result = img_tensor.clone() if img_tensor.requires_grad else img_tensor
            
            for transform_func in self.transforms_to_apply:
                result = transform_func(result)
            
            return result
        return img_tensor

# --- CLASSE PER BATCH PROCESSING ---
class BatchAugmentationTransform:
    """
    Applica trasformazioni a un intero batch di immagini per migliori performance.
    """
    def __init__(self, action_ids_to_apply):
        self.action_ids = action_ids_to_apply
        self.transforms_to_apply = []
        self.transform_names = []
        
        for action_id in action_ids_to_apply:
            if action_id in _ACTIONS_MAP:
                self.transforms_to_apply.append(_ACTIONS_MAP[action_id][0])
                self.transform_names.append(_ACTIONS_MAP[action_id][1])
            else:
                raise ValueError(f"Action ID {action_id} not found in _ACTIONS_MAP.")
    
    def __call__(self, batch_tensor):
        """
        Applica trasformazioni a un batch di tensori.
        Args:
            batch_tensor: Tensor di shape (batch_size, channels, height, width)
        Returns:
            Tensor trasformato della stessa shape
        """
        if len(self.transforms_to_apply) == 0:
            return batch_tensor
            
        result = batch_tensor
        for transform_func in self.transforms_to_apply:
            # Applica la trasformazione a ogni immagine nel batch
            batch_list = []
            for i in range(result.size(0)):
                transformed_img = transform_func(result[i])
                batch_list.append(transformed_img.unsqueeze(0))
            result = torch.cat(batch_list, dim=0)
        
        return result

# --- DEFINIZIONE DELLA FIXED AUGMENTATION ---
FIXED_AUGMENTATION_ACTION_IDS = [0, 3, 6]  # Brightness +20%, Contrast -20%, Horizontal Flip

# --- Pipeline di pre-elaborazione ottimizzate ---
# Trasformazioni base per preprocessing
BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Trasformazioni con augmentation fissa
FIXED_AUGMENTED_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    OptimizedFixedAugmentationTransform(FIXED_AUGMENTATION_ACTION_IDS),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- Funzioni di Valutazione Ottimizzate ---

def evaluate_model_optimized(model, dataloader, device, title="Evaluation", use_amp=True):
    """
    Versione ottimizzata della valutazione con diverse ottimizzazioni:
    - Automatic Mixed Precision (AMP) opzionale
    - Batch processing ottimizzato
    - Riduzione delle operazioni CPU-GPU
    """
    model.eval()
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    # Usa AMP se disponibile e richiesto
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    print(f"\n--- {title} ---")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=title)):
            # Trasferimento dati ottimizzato
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass con AMP opzionale
            if use_amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            
            # Ottimizzazione: usa argmax invece di max per predictions
            predicted = torch.argmax(outputs, dim=1)
            
            # Accumula i risultati (trasferimento batch-wise invece che elemento-wise)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    end_time = time.time()
    
    # Calcolo metriche
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    print(f"Evaluation Time: {end_time - start_time:.2f} seconds")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print("-" * 50)
    
    return accuracy, f1, avg_loss, conf_matrix

def evaluate_with_batch_augmentation(model, dataset, device, batch_size=64, title="Batch Augmentation Evaluation"):
    """
    Valutazione con augmentation applicata a livello di batch per migliori performance.
    """
    model.eval()
    
    # Crea dataloader senza augmentation
    base_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Crea un dataset temporaneo con trasformazioni base
    temp_dataset = PetImagesDataset(root_dir=DATA_ROOT_DIR, transform=base_transform)
    dataloader = DataLoader(temp_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # Inizializza il batch augmentation
    batch_augmentation = BatchAugmentationTransform(FIXED_AUGMENTATION_ACTION_IDS)
    
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    print(f"\n--- {title} ---")
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=title):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Applica augmentation al batch
            # Nota: questo richiede di denormalizzare e rinormalizzare
            # Per semplicità, manteniamo l'approccio precedente ma ottimizzato
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            predicted = torch.argmax(outputs, dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    end_time = time.time()
    
    # Calcolo metriche
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    
    print(f"Evaluation Time: {end_time - start_time:.2f} seconds")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print("-" * 50)
    
    return accuracy, f1, avg_loss, conf_matrix

def load_model_optimized(model_path, device, num_classes):
    """
    Caricamento ottimizzato del modello con gestione errori migliorata.
    """
    print("Loading pre-trained VGG16 classifier...")
    model = VGGClassifierWrapper(num_classes=num_classes)
    
    try:
        # Carica su CPU prima, poi trasferisci al dispositivo target
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Gestione compatibilità chiavi
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.') or k.startswith('classifier.'):
                new_state_dict['vgg16.' + k] = v
            else:
                new_state_dict[k] = v
        
        model.load_state_dict(new_state_dict, strict=False)
        model = model.to(device)
        
        print(f"Successfully loaded classifier weights from {model_path}")
        
        # Ottimizzazione: congela i pesi e metti in modalità eval
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
            
        # Ottimizzazione per inference
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            
        return model
        
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {model_path}")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def create_optimized_dataloaders(data_root, batch_size=64, num_workers=0):
    """
    Crea dataloader ottimizzati con configurazioni per migliori performance.
    """
    # Dataset standard
    standard_dataset = PetImagesDataset(root_dir=data_root, transform=BASE_TRANSFORM)
    standard_dataloader = DataLoader(
        standard_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Dataset con augmentation
    augmented_dataset = PetImagesDataset(root_dir=data_root, transform=FIXED_AUGMENTED_TRANSFORM)
    augmented_dataloader = DataLoader(
        augmented_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=None,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return standard_dataloader, augmented_dataloader

def main():
    """
    Funzione principale ottimizzata per la valutazione.
    """
    print("=== OPTIMIZED FIXED AUGMENTATION EVALUATION ===")
    print(f"Device: {DEVICE}")
    print(f"Fixed Augmentation Actions: {[_ACTIONS_MAP[i][1] for i in FIXED_AUGMENTATION_ACTION_IDS]}")
    
    total_start_time = time.time()
    
    # 1. Caricamento modello ottimizzato
    try:
        classifier_model = load_model_optimized(
            PRE_TRAINED_CLASSIFIER_PATH, DEVICE, NUM_CLASSES
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # 2. Creazione dataloader ottimizzati
    print("\nCreating optimized dataloaders...")
    standard_dataloader, augmented_dataloader = create_optimized_dataloaders(
        DATA_ROOT_DIR, batch_size=128, num_workers=0  # Batch size più grande per efficiency
    )
    
    print(f"Standard dataset size: {len(standard_dataloader.dataset)}")
    print(f"Augmented dataset size: {len(augmented_dataloader.dataset)}")
    
    # 3. Valutazione baseline
    base_accuracy, base_f1, base_loss, _ = evaluate_model_optimized(
        classifier_model, standard_dataloader, DEVICE, 
        title="Baseline Classifier (Standard Preprocessing)",
        use_amp=True  # Imposta a True se hai GPU NVIDIA con supporto AMP
    )
    
    # 4. Valutazione con fixed augmentation
    aug_accuracy, aug_f1, aug_loss, _ = evaluate_model_optimized(
        classifier_model, augmented_dataloader, DEVICE,
        title=f"Fixed Augmentation Classifier",
        use_amp=True
    )
    
    # 5. Risultati finali
    total_end_time = time.time()
    
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    print(f"Baseline Accuracy:          {base_accuracy:.4f}")
    print(f"Fixed Augmentation Accuracy: {aug_accuracy:.4f}")
    print(f"Accuracy Difference:        {(aug_accuracy - base_accuracy):+.4f}")
    print(f"Baseline F1-Score:          {base_f1:.4f}")
    print(f"Fixed Augmentation F1-Score: {aug_f1:.4f}")
    print(f"F1-Score Difference:        {(aug_f1 - base_f1):+.4f}")
    print(f"Baseline Loss:              {base_loss:.4f}")
    print(f"Fixed Augmentation Loss:    {aug_loss:.4f}")
    print(f"Loss Difference:            {(aug_loss - base_loss):+.4f}")
    print(f"\nTotal Execution Time: {total_end_time - total_start_time:.2f} seconds")
    
    # Interpretazione risultati
    if aug_accuracy > base_accuracy:
        improvement = (aug_accuracy - base_accuracy) * 100
        print(f"\n✅ Fixed augmentation IMPROVED accuracy by {improvement:.2f}%")
    elif aug_accuracy < base_accuracy:
        degradation = (base_accuracy - aug_accuracy) * 100
        print(f"\n❌ Fixed augmentation DEGRADED accuracy by {degradation:.2f}%")
    else:
        print(f"\n➡️ Fixed augmentation had NO significant impact on accuracy")

if __name__ == '__main__':
    main()