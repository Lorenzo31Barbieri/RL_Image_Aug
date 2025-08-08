"""
Modulo per la valutazione di fixed augmentation.
Applica una sequenza fissa di trasformazioni.
"""

import torch
from typing import Dict, Any, List
from tqdm import tqdm

from evaluation.core.evaluation_core import (
    time_evaluation_context,
    calculate_basic_metrics,
    print_evaluation_summary,
    validate_evaluation_inputs
)
from evaluation.core.data_utils import (
    get_cifar10_test_dataset,
    create_fixed_augmentation_transform,
    create_standard_preprocessing
)


def evaluate_fixed_augmentation(classifier_model: torch.nn.Module,
                               test_dataset: torch.utils.data.Dataset,
                               augmentation_ids: List[int],
                               device: torch.device,
                               num_samples: int = None,
                               batch_size: int = 64,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Valuta il classificatore SOLO con fixed augmentation.
    
    Args:
        classifier_model: Modello classificatore pre-trained
        test_dataset: Dataset di test (senza trasformazioni)
        augmentation_ids: Lista di ID delle trasformazioni da applicare
        device: Device per computazione
        num_samples: Number of samples to evaluate (None = all)
        batch_size: Dimensione del batch
        verbose: Se stampare informazioni dettagliate
    
    Returns:
        Dict con risultati della valutazione fixed augmentation
    """
    validate_evaluation_inputs(classifier_model, device=device)
    
    # Create subset if num_samples specified
    if num_samples is not None:
        from evaluation.core.data_utils import create_sample_subset
        test_dataset = create_sample_subset(test_dataset, num_samples, random_seed=42)
        
        if verbose:
            print(f" Created subset with {num_samples} samples")
    
    if verbose:
        print(f" Starting fixed augmentation evaluation...")
        print(f" Dataset size: {len(test_dataset)} samples")
        print(f" Augmentation IDs: {augmentation_ids}")
        print(f" Batch size: {batch_size}")
    
    # Ottieni nomi delle trasformazioni
    augmentation_names = []
    try:
        from src.environment.transforms import _ACTIONS_MAP
        augmentation_names = [_ACTIONS_MAP[aid][1] for aid in _ACTIONS_MAP if aid in augmentation_ids]
    except ImportError:
        augmentation_names = [f"Transform_{aid}" for aid in augmentation_ids]
    
    with time_evaluation_context("FIXED AUGMENTATION"):
        # Valutazione SOLO con Fixed Augmentation
        augmented_transform = create_fixed_augmentation_transform(
            action_ids=augmentation_ids,
            normalize=True
        )
        
        results = _evaluate_with_transform(
            classifier_model=classifier_model,
            dataset=test_dataset,
            transform=augmented_transform,
            device=device,
            batch_size=batch_size,
            description=f"Fixed Aug ({', '.join(augmentation_names[:3])}{'...' if len(augmentation_names) > 3 else ''})",
            return_details=True  # Per confusion matrix
        )
        
        # Aggiungi metadati specifici
        results.update({
            'augmentation_ids': augmentation_ids,
            'augmentation_names': augmentation_names,
            'method': 'fixed_augmentation',
            'total_samples': len(test_dataset),
            'samples_evaluated': len(test_dataset)
        })
        
        if verbose:
            print_evaluation_summary(results, "Fixed Augmentation")
    
    return results


def _evaluate_with_transform(classifier_model: torch.nn.Module,
                           dataset: torch.utils.data.Dataset,
                           transform: torch.nn.Module,
                           device: torch.device,
                           batch_size: int,
                           description: str,
                           return_details: bool = False) -> Dict[str, Any]:
    """
    Valuta il classificatore applicando una specifica trasformazione.
    
    Args:
        classifier_model: Modello classificatore
        dataset: Dataset originale
        transform: Trasformazione da applicare
        device: Device di computazione
        batch_size: Dimensione del batch
        description: Descrizione per progress bar
        return_details: Se restituire predizioni dettagliate
    
    Returns:
        Dict con risultati della valutazione
    """
    import time
    from torch.utils.data import DataLoader
    
    # Crea dataset con trasformazione
    dataset_with_transform = torch.utils.data.Dataset.__new__(type(dataset))
    dataset_with_transform.__dict__.update(dataset.__dict__)
    dataset_with_transform.transform = transform
    
    # Crea DataLoader
    dataloader = DataLoader(
        dataset_with_transform,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    classifier_model.eval()
    all_predictions = []
    all_labels = []
    all_confidences = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=description):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = classifier_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    inference_time = time.time() - start_time
    total_samples = len(dataset)
    
    # Calcola metriche
    metrics = calculate_basic_metrics(all_predictions, all_labels, all_confidences)
    
    # Aggiungi metriche temporali e di loss
    metrics.update({
        'avg_loss': total_loss / total_samples,
        'inference_time': inference_time,
        'time_per_sample': inference_time / total_samples
    })
    
    # Aggiungi dettagli se richiesti
    if return_details:
        metrics.update({
            'predictions': all_predictions,
            'labels': all_labels,
            'confidences': all_confidences
        })
    
    return metrics
