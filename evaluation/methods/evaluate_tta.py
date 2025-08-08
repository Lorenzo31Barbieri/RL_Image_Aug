"""
Modulo per la valutazione con Test-Time Augmentation (TTA).
Supporta sia ttach library che implementazione manuale.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

from evaluation.core.evaluation_core import (
    time_evaluation_context,
    print_evaluation_summary,
    validate_evaluation_inputs
)

# Controllo disponibilità ttach
try:
    import ttach as tta
    TTA_AVAILABLE = True
except ImportError:
    TTA_AVAILABLE = False


class ManualTTAWrapper:
    """
    Implementazione manuale di TTA per quando ttach non è disponibile.
    """
    
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.transforms = self._create_transforms()
    
    def _create_transforms(self) -> List:
        """Crea lista di trasformazioni TTA manuali."""
        transforms_list = []
        
        # Immagine originale
        transforms_list.append(lambda x: x)
        
        # Flip orizzontale
        transforms_list.append(lambda x: torch.flip(x, dims=[3]))
        
        # Variazioni di luminosità
        transforms_list.append(lambda x: torch.clamp(x * 1.1, 0, 1))
        transforms_list.append(lambda x: torch.clamp(x * 0.9, 0, 1))
        
        # Variazioni di contrasto
        transforms_list.append(lambda x: torch.clamp((x - 0.5) * 1.1 + 0.5, 0, 1))
        transforms_list.append(lambda x: torch.clamp((x - 0.5) * 0.9 + 0.5, 0, 1))
        
        return transforms_list
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Esegue predizione TTA.
        
        Args:
            x: Tensore di input
        
        Returns:
            Predizione media su tutte le augmentation
        """
        predictions = []
        
        with torch.no_grad():
            for transform in self.transforms:
                transformed_x = transform(x)
                pred = self.model(transformed_x)
                predictions.append(pred)
        
        # Media delle predizioni (sui logits)
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction
    
    def get_num_augmentations(self) -> int:
        """Restituisce il numero di augmentation utilizzate."""
        return len(self.transforms)


def evaluate_tta(classifier_model: torch.nn.Module,
                test_dataset: torch.utils.data.Dataset,
                device: torch.device,
                num_samples: int = 1000,
                use_ttach: bool = True,
                verbose: bool = True) -> Dict[str, Any]:
    """
    Valuta il classificatore SOLO con Test-Time Augmentation.
    
    Args:
        classifier_model: Modello classificatore pre-trained
        test_dataset: Dataset di test con preprocessing standard
        device: Device per computazione
        num_samples: Numero di campioni da valutare
        use_ttach: Se usare la libreria ttach (se disponibile)
        verbose: Se stampare informazioni dettagliate
    
    Returns:
        Dict con risultati della valutazione TTA:
        - accuracy: Accuratezza con TTA
        - avg_confidence: Confidenza media con TTA
        - f1_score: F1-score con TTA
        - num_augmentations: Numero di augmentation utilizzate
        - method_used: 'ttach' o 'manual'
        - inference_time: Tempo totale di inferenza
        - time_per_sample: Tempo per campione
        - method: 'tta'
        - total_samples_evaluated: Numero campioni valutati
        - predictions: Lista predizioni (per confusion matrix)
        - labels: Lista label vere (per confusion matrix)
        - confidences: Lista confidenze
    """
    validate_evaluation_inputs(classifier_model, device=device)
    
    if num_samples > len(test_dataset):
        num_samples = len(test_dataset)
        if verbose:
            print(f" Adjusted num_samples to dataset size: {num_samples}")
    
    if verbose:
        print(f" Starting TTA evaluation...")
        print(f" Dataset size: {len(test_dataset)} samples")
        print(f" Evaluating on: {num_samples} samples")
        print(f" Device: {device}")
    
    # Determina quale implementazione TTA usare
    if use_ttach and TTA_AVAILABLE:
        method_used = 'ttach'
        if verbose:
            print(" Using ttach library for TTA")
    else:
        method_used = 'manual'
        if verbose:
            print(" Using manual TTA implementation")
            if use_ttach and not TTA_AVAILABLE:
                print(" ttach not available, falling back to manual")
    
    with time_evaluation_context("TTA"):
        # Seleziona campioni casuali
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        # Configura TTA
        if method_used == 'ttach':
            tta_model = _setup_ttach_model(classifier_model)
            num_augmentations = len(tta_model.transforms.aug_transforms) + 1  # +1 per identità
        else:
            tta_model = ManualTTAWrapper(classifier_model)
            num_augmentations = tta_model.get_num_augmentations()
        
        if verbose:
            print(f" TTA configured with {num_augmentations} augmentations")
        
        # Esegui valutazione SOLO TTA
        results = _evaluate_tta_on_samples(
            classifier_model=classifier_model,
            tta_model=tta_model,
            test_dataset=test_dataset,
            indices=indices,
            device=device,
            method_used=method_used,
            verbose=verbose
        )
        
        # Aggiungi metadati specifici per TTA
        results.update({
            'method': 'tta',
            'method_used': method_used,
            'num_augmentations': num_augmentations,
            'total_samples_evaluated': num_samples,
            'dataset_size': len(test_dataset)
        })
        
        if verbose:
            print_evaluation_summary(results, "Test-Time Augmentation")
    
    return results


def _setup_ttach_model(classifier_model: torch.nn.Module):
    """Configura il modello TTA usando ttach."""
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1]),  # Variazioni luminosità/contrasto
    ])
    
    tta_model = tta.ClassificationTTAWrapper(classifier_model, tta_transforms)
    return tta_model


def _evaluate_tta_on_samples(classifier_model: torch.nn.Module,
                           tta_model,
                           test_dataset: torch.utils.data.Dataset,
                           indices: np.ndarray,
                           device: torch.device,
                           method_used: str,
                           verbose: bool) -> Dict[str, Any]:
    """Esegue la valutazione TTA sui campioni selezionati."""
    
    tta_correct = []
    tta_confidences = []
    tta_predictions = []
    tta_labels = []
    
    start_time = time.time()
    
    progress_desc = f"TTA evaluation ({method_used})"
    
    for idx in tqdm(indices, desc=progress_desc, disable=not verbose):
        image, label = test_dataset[idx]
        
        # Assicurati che l'immagine sia un tensore
        if not isinstance(image, torch.Tensor):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        image = image.unsqueeze(0).to(device)  # Aggiungi dimensione batch
        
        with torch.no_grad():
            # Predizione TTA
            if method_used == 'ttach':
                tta_output = tta_model(image)
            else:  # manual
                tta_output = tta_model.predict(image)
            
            tta_prob = torch.nn.functional.softmax(tta_output, dim=1)
            tta_confidence, tta_pred = torch.max(tta_prob, 1)
            tta_is_correct = (tta_pred.item() == label)
            
            # Registra risultati TTA
            tta_correct.append(tta_is_correct)
            tta_confidences.append(tta_confidence.item())
            tta_predictions.append(tta_pred.item())
            tta_labels.append(label)
    
    total_time = time.time() - start_time
    num_samples = len(indices)
    
    # Calcola metriche TTA
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(tta_labels, tta_predictions)
    f1 = f1_score(tta_labels, tta_predictions, average='weighted')
    avg_confidence = np.mean(tta_confidences)
    conf_matrix = confusion_matrix(tta_labels, tta_predictions)
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'inference_time': total_time,
        'time_per_sample': total_time / num_samples,
        'predictions': tta_predictions,
        'labels': tta_labels,
        'confidences': tta_confidences,
        'total_samples': num_samples
    }