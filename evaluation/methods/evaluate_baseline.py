"""
Modulo per la valutazione del classificatore baseline.
Espone una funzione principale per valutare un classificatore senza augmentation.
"""

import torch
from typing import Dict, Any
from tqdm import tqdm

# Import dei moduli core
from evaluation.core.evaluation_core import (
    time_evaluation_context,
    evaluate_model_predictions,
    print_evaluation_summary,
    validate_evaluation_inputs
)


def evaluate_baseline(classifier_model: torch.nn.Module,
                     test_loader: torch.utils.data.DataLoader,
                     device: torch.device,
                     verbose: bool = True,
                     return_details: bool = True) -> Dict[str, Any]:
    """
    Valuta le performance del classificatore baseline senza augmentation.
    
    Args:
        classifier_model: Modello classificatore pre-trained
        test_loader: DataLoader per i dati di test
        device: Device per computazione
        verbose: Se stampare informazioni dettagliate
        return_details: Se restituire predizioni e label individuali
    
    Returns:
        Dict con risultati della valutazione:
        - accuracy: Accuratezza complessiva
        - avg_confidence: Confidenza media
        - f1_score: F1-score pesato
        - avg_loss: Loss medio
        - inference_time: Tempo totale di inferenza
        - time_per_sample: Tempo per campione
        - predictions: Lista predizioni (se return_details=True)
        - labels: Lista label vere (se return_details=True)
        - confidences: Lista confidenze (se return_details=True)
        - confusion_matrix: Matrice di confusione
        - total_samples: Numero totale di campioni
    
    Raises:
        ValueError: Se gli input non sono validi
    """
    # Validazione input
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
            'model_type': type(classifier_model).__name__
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


def quick_baseline_check(classifier_model: torch.nn.Module,
                        test_loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        max_batches: int = 10) -> Dict[str, Any]:
    """
    Esegue una valutazione rapida del baseline su un subset di dati.
    Utile per test rapidi o debug.
    
    Args:
        classifier_model: Modello classificatore
        test_loader: DataLoader per test
        device: Device di computazione
        max_batches: Numero massimo di batch da processare
    
    Returns:
        Dict con risultati della valutazione rapida
    """
    validate_evaluation_inputs(classifier_model, test_loader, device)
    
    print(f" Quick baseline check (max {max_batches} batches)...")
    
    classifier_model.eval()
    correct = 0
    total = 0
    all_confidences = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc="Quick check")):
            if batch_idx >= max_batches:
                break
                
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = classifier_model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_confidences.extend(confidences.cpu().numpy())
    
    accuracy = correct / total if total > 0 else 0
    avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
    
    results = {
        'quick_check': True,
        'batches_processed': min(max_batches, len(test_loader)),
        'samples_processed': total,
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'method': 'baseline_quick'
    }
    
    print(f" Quick check complete:")
    print(f"  Processed: {total} samples ({results['batches_processed']} batches)")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Avg Confidence: {avg_confidence:.4f}")
    
    return results


# Funzione wrapper per compatibilità con script esistenti
def run_baseline_evaluation(classifier_path: str = './checkpoint/ckpt.pth',
                          data_root: str = './data',
                          batch_size: int = 64,
                          device: torch.device = None) -> Dict[str, Any]:
    """
    Funzione wrapper per eseguire valutazione baseline completa.
    Carica automaticamente modello e dati.
    
    Args:
        classifier_path: Percorso del classificatore
        data_root: Directory root per i dati
        batch_size: Dimensione del batch
        device: Device (auto-detect se None)
    
    Returns:
        Dict con risultati della valutazione
    """
    # Import dinamici per evitare dipendenze circolari
    from core.model_loader import load_classifier
    from core.data_utils import get_cifar10_test_loader
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f" Running complete baseline evaluation...")
    print(f" Classifier: {classifier_path}")
    print(f" Data root: {data_root}")
    print(f" Device: {device}")
    
    # Carica modello
    classifier = load_classifier(classifier_path, device)
    
    # Carica dati
    test_loader = get_cifar10_test_loader(
        data_root=data_root,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Esegui valutazione
    results = evaluate_baseline(
        classifier_model=classifier,
        test_loader=test_loader,
        device=device,
        verbose=True,
        return_details=True
    )
    
    return results


if __name__ == '__main__':
    """
    Script principale per test del modulo.
    """
    print("Testing baseline evaluation module...")
    
    # Configurazione
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Test con caricamento automatico
        results = run_baseline_evaluation(
            classifier_path='./checkpoint/ckpt.pth',
            data_root='./data',
            batch_size=64,
            device=device
        )
        
        print(f"\n Baseline evaluation completed successfully!")
        print(f" Final accuracy: {results['accuracy']:.4f}")
        print(f" Average confidence: {results['avg_confidence']:.4f}")
        
    except Exception as e:
        print(f" Error during evaluation: {e}")
        print("Make sure the classifier and data are available at the specified paths.")