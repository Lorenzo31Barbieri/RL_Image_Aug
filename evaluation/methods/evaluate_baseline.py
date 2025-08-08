"""
Modulo per la valutazione del classificatore baseline.
Espone una funzione principale per valutare un classificatore senza augmentation.
"""

import torch
from typing import Dict, Any

# Import dei moduli core
from evaluation.core.evaluation_core import (
    time_evaluation_context,
    evaluate_model_predictions,
    print_evaluation_summary,
    validate_evaluation_inputs
)


def evaluate_baseline(classifier_model: torch.nn.Module,
                     test_loader: torch.utils.data.DataLoader = None,
                     test_dataset: torch.utils.data.Dataset = None,
                     device: torch.device = None,
                     num_samples: int = None,
                     batch_size: int = 64,
                     verbose: bool = True,
                     return_details: bool = True) -> Dict[str, Any]:
    """
    Valuta le performance del classificatore baseline senza augmentation.
    """
    # Create subset loader if num_samples specified
    if num_samples is not None and test_dataset is not None:
        from evaluation.core.data_utils import create_sample_subset
        from torch.utils.data import DataLoader
        
        subset = create_sample_subset(test_dataset, num_samples)  # Removed random_seed=42
        test_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        if verbose:
            print(f" Created subset with {num_samples} samples")
    
    validate_evaluation_inputs(classifier_model, test_loader, device)
    
    if verbose:
        print(f" Starting baseline evaluation...")
        print(f" Dataset size: {len(test_loader.dataset)} samples")
        print(f" Batch size: {test_loader.batch_size}")
        print(f" Device: {device}")
    
    with time_evaluation_context("BASELINE"):
        # Usa la funzione core per valutazione
        results = evaluate_model_predictions(
            model=classifier_model,
            dataloader=test_loader,
            device=device,
            return_details=return_details
        )
        
        # Aggiungi metadati specifici per baseline
        results.update({
            'method': 'baseline',
            'augmentation_applied': False,
            'model_type': type(classifier_model).__name__,
            'samples_evaluated': len(test_loader.dataset)
        })
        
        if verbose:
            print_evaluation_summary(results, "Baseline Classifier")
    
    return results


def evaluate_baseline_with_confidence_analysis(classifier_model: torch.nn.Module,
                                              test_loader: torch.utils.data.DataLoader,
                                              device: torch.device,
                                              confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Valuta il baseline con analisi dettagliata della confidenza.
    
    Args:
        classifier_model: Modello classificatore
        test_loader: DataLoader per test
        device: Device di computazione
        confidence_threshold: Soglia per considerare predizioni "sicure"
    
    Returns:
        Dict con risultati estesi inclusa analisi di confidenza
    """
    # Esegui valutazione standard
    results = evaluate_baseline(
        classifier_model=classifier_model,
        test_loader=test_loader,
        device=device,
        verbose=False,
        return_details=True
    )
    
    # Analisi confidenza
    confidences = results['confidences']
    predictions = results['predictions']
    labels = results['labels']
    
    # Separa predizioni per livello di confidenza
    high_conf_indices = [i for i, conf in enumerate(confidences) if conf >= confidence_threshold]
    low_conf_indices = [i for i, conf in enumerate(confidences) if conf < confidence_threshold]
    
    # Calcola accuratezza per livello di confidenza
    if high_conf_indices:
        high_conf_correct = sum(1 for i in high_conf_indices if predictions[i] == labels[i])
        high_conf_accuracy = high_conf_correct / len(high_conf_indices)
    else:
        high_conf_accuracy = 0.0
    
    if low_conf_indices:
        low_conf_correct = sum(1 for i in low_conf_indices if predictions[i] == labels[i])
        low_conf_accuracy = low_conf_correct / len(low_conf_indices)
    else:
        low_conf_accuracy = 0.0
    
    # Aggiungi analisi confidenza ai risultati
    confidence_analysis = {
        'confidence_threshold': confidence_threshold,
        'high_confidence_samples': len(high_conf_indices),
        'low_confidence_samples': len(low_conf_indices),
        'high_confidence_accuracy': high_conf_accuracy,
        'low_confidence_accuracy': low_conf_accuracy,
        'high_confidence_ratio': len(high_conf_indices) / len(confidences),
        'avg_high_confidence': sum(confidences[i] for i in high_conf_indices) / len(high_conf_indices) if high_conf_indices else 0,
        'avg_low_confidence': sum(confidences[i] for i in low_conf_indices) / len(low_conf_indices) if low_conf_indices else 0
    }
    
    results['confidence_analysis'] = confidence_analysis
    
    print(f"\n CONFIDENCE ANALYSIS:")
    print(f"  Threshold: {confidence_threshold}")
    print(f"  High confidence samples: {len(high_conf_indices)} ({confidence_analysis['high_confidence_ratio']:.1%})")
    print(f"  High confidence accuracy: {high_conf_accuracy:.4f}")
    print(f"  Low confidence samples: {len(low_conf_indices)} ({1-confidence_analysis['high_confidence_ratio']:.1%})")
    print(f"  Low confidence accuracy: {low_conf_accuracy:.4f}")
    
    return results