import torch
import numpy as np
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from collections import defaultdict

@contextmanager
def time_evaluation_context(evaluation_name: str):
    """Context manager per misurare il tempo di valutazione."""
    print(f"\n=== {evaluation_name.upper()} EVALUATION ===")
    start_time = time.time()
    try:
        yield start_time
    finally:
        elapsed_time = time.time() - start_time
        print(f"Evaluation completed in {elapsed_time:.2f} seconds")


def calculate_basic_metrics(predictions: List[int], 
                          labels: List[int], 
                          confidences: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Calcola metriche di base per classificazione.
    
    Args:
        predictions: Lista delle predizioni
        labels: Lista delle label vere
        confidences: Lista delle confidenze (opzionale)
    
    Returns:
        Dict con accuracy, f1_score, confusion_matrix e avg_confidence
    """
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    conf_matrix = confusion_matrix(labels, predictions)
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_matrix,
        'total_samples': len(labels)
    }
    
    if confidences is not None:
        results['avg_confidence'] = np.mean(confidences)
        results['confidence_std'] = np.std(confidences)
    
    return results


def calculate_improvement_metrics(initial_correct: List[bool],
                                final_correct: List[bool],
                                initial_confidence: List[float],
                                final_confidence: List[float]) -> Dict[str, Any]:
    """
    Calcola metriche di miglioramento confrontando stato iniziale e finale.
    
    Args:
        initial_correct: Lista di booleani per correttezza iniziale
        final_correct: Lista di booleani per correttezza finale
        initial_confidence: Lista delle confidenze iniziali
        final_confidence: Lista delle confidenze finali
    
    Returns:
        Dict con metriche di miglioramento
    """
    total_samples = len(initial_correct)
    
    # Calcola cambiamenti di correttezza
    improvements = sum(1 for i, f in zip(initial_correct, final_correct) if not i and f)
    degradations = sum(1 for i, f in zip(initial_correct, final_correct) if i and not f)
    no_change = sum(1 for i, f in zip(initial_correct, final_correct) if i == f)
    
    # Calcola accuratezze
    initial_accuracy = sum(initial_correct) / total_samples
    final_accuracy = sum(final_correct) / total_samples
    accuracy_improvement = final_accuracy - initial_accuracy
    
    # Calcola cambiamenti di confidenza
    confidence_changes = [f - i for i, f in zip(initial_confidence, final_confidence)]
    avg_confidence_change = np.mean(confidence_changes)
    
    return {
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'improvements': improvements,
        'degradations': degradations,
        'no_change': no_change,
        'improvement_rate': improvements / total_samples,
        'degradation_rate': degradations / total_samples,
        'net_improvement_rate': (improvements - degradations) / total_samples,
        'initial_avg_confidence': np.mean(initial_confidence),
        'final_avg_confidence': np.mean(final_confidence),
        'avg_confidence_change': avg_confidence_change,
        'confidence_changes': confidence_changes
    }


def evaluate_model_predictions(model: torch.nn.Module,
                             dataloader: torch.utils.data.DataLoader,
                             device: torch.device,
                             return_details: bool = False) -> Dict[str, Any]:
    """
    Valuta un modello su un dataloader e restituisce predizioni e metriche.
    
    Args:
        model: Modello PyTorch da valutare
        dataloader: DataLoader per i dati di test
        device: Device per computazione
        return_details: Se restituire dettagli come predizioni individuali
    
    Returns:
        Dict con risultati della valutazione
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predictions = torch.max(probabilities, 1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    inference_time = time.time() - start_time
    total_samples = len(dataloader.dataset)
    
    # Calcola metriche di base
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


def analyze_class_performance(predictions: List[int], 
                            labels: List[int], 
                            num_classes: int = 10) -> Dict[str, Any]:
    """
    Analizza le performance per classe.
    
    Args:
        predictions: Lista delle predizioni
        labels: Lista delle label vere
        num_classes: Numero di classi
    
    Returns:
        Dict con analisi per classe
    """
    from collections import defaultdict
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_predictions = defaultdict(list)
    
    for pred, label in zip(predictions, labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1
        class_predictions[label].append(pred)
    
    class_accuracies = {}
    for class_id in range(num_classes):
        if class_total[class_id] > 0:
            class_accuracies[class_id] = class_correct[class_id] / class_total[class_id]
        else:
            class_accuracies[class_id] = 0.0
    
    return {
        'class_accuracies': class_accuracies,
        'class_correct': dict(class_correct),
        'class_total': dict(class_total),
        'class_predictions': dict(class_predictions)
    }


def print_evaluation_summary(results: Dict[str, Any], 
                           method_name: str = "Method",
                           detailed: bool = False) -> None:
    """
    Stampa un riassunto dei risultati di valutazione.
    
    Args:
        results: Dizionario con i risultati
        method_name: Nome del metodo valutato
        detailed: Se stampare informazioni dettagliate
    """
    print(f"\n{'='*50}")
    print(f"{method_name.upper()} EVALUATION RESULTS")
    print(f"{'='*50}")
    
    # Metriche principali
    if 'accuracy' in results:
        print(f"Accuracy: {results['accuracy']:.4f}")
    
    if 'avg_confidence' in results:
        print(f"Average Confidence: {results['avg_confidence']:.4f}")
    
    if 'f1_score' in results:
        print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Metriche temporali
    if 'inference_time' in results:
        print(f"Total Time: {results['inference_time']:.2f}s")
    
    if 'time_per_sample' in results:
        print(f"Time per Sample: {results['time_per_sample']*1000:.1f}ms")
    
    # Metriche di miglioramento (se presenti)
    if 'accuracy_improvement' in results:
        print(f"Accuracy Improvement: {results['accuracy_improvement']:+.4f}")
    
    if 'improvements' in results and 'degradations' in results:
        print(f"Improvements: {results['improvements']}")
        print(f"Degradations: {results['degradations']}")
        if 'improvement_rate' in results:
            print(f"Net Success Rate: {results.get('net_improvement_rate', 0):.1%}")
    
    if detailed and 'confusion_matrix' in results:
        print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    
    print(f"{'='*50}")


def validate_evaluation_inputs(model: Optional[torch.nn.Module] = None,
                             dataloader: Optional[torch.utils.data.DataLoader] = None,
                             device: Optional[torch.device] = None) -> None:
    """
    Valida gli input comuni per le funzioni di valutazione.
    
    Args:
        model: Modello da validare (opzionale)
        dataloader: DataLoader da validare (opzionale)  
        device: Device da validare (opzionale)
    
    Raises:
        ValueError: Se gli input non sono validi
    """
    if model is not None:
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module")
        if not next(model.parameters()).is_cuda and device and device.type == 'cuda':
            print("Warning: Model is not on CUDA but device is CUDA")
    
    if dataloader is not None:
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ValueError("dataloader must be a PyTorch DataLoader")
        if len(dataloader.dataset) == 0:
            raise ValueError("DataLoader dataset is empty")
    
    if device is not None:
        if not isinstance(device, torch.device):
            raise ValueError("device must be a torch.device")


def save_evaluation_results(results: Dict[str, Any], 
                          filepath: str,
                          method_name: str = "evaluation") -> None:
    """
    Salva i risultati di valutazione su file.
    
    Args:
        results: Dizionario con i risultati
        filepath: Percorso del file di output
        method_name: Nome del metodo (per metadati)
    """
    import json
    import pickle
    from datetime import datetime
    
    # Prepara metadati
    metadata = {
        'method_name': method_name,
        'timestamp': datetime.now().isoformat(),
        'total_samples': results.get('total_samples', 'unknown')
    }
    
    # Combina risultati e metadati
    output_data = {
        'metadata': metadata,
        'results': results
    }
    
    # Salva in base all'estensione
    if filepath.endswith('.json'):
        # Rimuovi elementi non serializzabili per JSON
        json_safe_results = {k: v for k, v in results.items() 
                           if not isinstance(v, (np.ndarray, torch.Tensor))}
        output_data['results'] = json_safe_results
        
        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
    else:
        # Default a pickle per preservare tutti i dati
        with open(filepath, 'wb') as f:
            pickle.dump(output_data, f)
    
    print(f"Results saved to {filepath}")