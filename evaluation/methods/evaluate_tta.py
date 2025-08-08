"""
Modulo per la valutazione con Test-Time Augmentation (TTA).
Supporta sia ttach library che implementazione manuale.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional
from tqdm import tqdm
import time

# Import dei moduli core
from evaluation.core.evaluation_core import (
    time_evaluation_context,
    calculate_improvement_metrics,
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
    Valuta il classificatore con Test-Time Augmentation.
    
    Args:
        classifier_model: Modello classificatore pre-trained
        test_dataset: Dataset di test con preprocessing standard
        device: Device per computazione
        num_samples: Numero di campioni da valutare
        use_ttach: Se usare la libreria ttach (se disponibile)
        verbose: Se stampare informazioni dettagliate
    
    Returns:
        Dict con risultati della valutazione TTA:
        - baseline_accuracy: Accuratezza senza TTA (su campioni selezionati)
        - tta_accuracy: Accuratezza con TTA
        - accuracy_improvement: Miglioramento dell'accuratezza
        - improvements: Numero di campioni migliorati
        - degradations: Numero di campioni peggiorati
        - improvement_rate: Tasso di miglioramento
        - degradation_rate: Tasso di peggioramento
        - avg_confidence_improvement: Miglioramento medio della confidenza
        - confidence_improvements: Lista dei cambiamenti di confidenza
        - num_augmentations: Numero di augmentation utilizzate
        - inference_time: Tempo totale di inferenza
        - time_per_sample: Tempo per campione
        - method_used: 'ttach' o 'manual'
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
        
        # Esegui valutazione
        results = _evaluate_tta_on_samples(
            classifier_model=classifier_model,
            tta_model=tta_model,
            test_dataset=test_dataset,
            indices=indices,
            device=device,
            method_used=method_used,
            verbose=verbose
        )
        
        # Aggiungi metadati
        results.update({
            'method': 'tta',
            'method_used': method_used,
            'num_augmentations': num_augmentations,
            'total_samples_evaluated': num_samples,
            'dataset_size': len(test_dataset)
        })
        
        if verbose:
            _print_tta_summary(results)
    
    return results


def _setup_ttach_model(classifier_model: torch.nn.Module):
    """Configura il modello TTA usando ttach."""
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1]),  # Variazioni luminosità/contrasto
        # Nota: evitare rotazioni per CIFAR-10 che potrebbero essere troppo aggressive
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
    
    baseline_correct = []
    tta_correct = []
    baseline_confidences = []
    tta_confidences = []
    improvements = 0
    degradations = 0
    
    start_time = time.time()
    
    progress_desc = f"TTA evaluation ({method_used})"
    
    for idx in tqdm(indices, desc=progress_desc, disable=not verbose):
        image, label = test_dataset[idx]
        
        # Assicurati che l'immagine sia un tensore
        if not isinstance(image, torch.Tensor):
            # Se il dataset non ha trasformazioni, applica ToTensor
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        image = image.unsqueeze(0).to(device)  # Aggiungi dimensione batch
        
        with torch.no_grad():
            # Predizione baseline
            baseline_output = classifier_model(image)
            baseline_prob = torch.nn.functional.softmax(baseline_output, dim=1)
            baseline_confidence, baseline_pred = torch.max(baseline_prob, 1)
            baseline_is_correct = (baseline_pred.item() == label)
            
            # Predizione TTA
            if method_used == 'ttach':
                tta_output = tta_model(image)
            else:  # manual
                tta_output = tta_model.predict(image)
            
            tta_prob = torch.nn.functional.softmax(tta_output, dim=1)
            tta_confidence, tta_pred = torch.max(tta_prob, 1)
            tta_is_correct = (tta_pred.item() == label)
            
            # Registra risultati
            baseline_correct.append(baseline_is_correct)
            tta_correct.append(tta_is_correct)
            baseline_confidences.append(baseline_confidence.item())
            tta_confidences.append(tta_confidence.item())
            
            # Conta miglioramenti/peggioramenti
            if not baseline_is_correct and tta_is_correct:
                improvements += 1
            elif baseline_is_correct and not tta_is_correct:
                degradations += 1
    
    total_time = time.time() - start_time
    num_samples = len(indices)
    
    # Calcola metriche
    baseline_accuracy = sum(baseline_correct) / num_samples
    tta_accuracy = sum(tta_correct) / num_samples
    accuracy_improvement = tta_accuracy - baseline_accuracy
    
    confidence_improvements = [tta - base for tta, base in zip(tta_confidences, baseline_confidences)]
    avg_confidence_improvement = np.mean(confidence_improvements)
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'tta_accuracy': tta_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'baseline_avg_confidence': np.mean(baseline_confidences),
        'tta_avg_confidence': np.mean(tta_confidences),
        'avg_confidence_improvement': avg_confidence_improvement,
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_samples,
        'degradation_rate': degradations / num_samples,
        'net_improvement_rate': (improvements - degradations) / num_samples,
        'confidence_improvements': confidence_improvements,
        'inference_time': total_time,
        'time_per_sample': total_time / num_samples
    }


def _print_tta_summary(results: Dict[str, Any]) -> None:
    """Stampa riassunto dettagliato dei risultati TTA."""
    
    print(f"\n{'='*60}")
    print("TEST-TIME AUGMENTATION EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f" METHOD: {results['method_used'].upper()}")
    print(f" Augmentations used: {results['num_augmentations']}")
    print(f" Samples evaluated: {results['total_samples_evaluated']}")
    
    print(f"\n ACCURACY COMPARISON:")
    print(f"  Baseline: {results['baseline_accuracy']:.4f}")
    print(f"  TTA: {results['tta_accuracy']:.4f}")
    
    print(f"  Improvement: {results['accuracy_improvement']:+.4f}")
    
    print(f"\n CONFIDENCE ANALYSIS:")
    print(f"  Baseline confidence: {results['baseline_avg_confidence']:.4f}")
    print(f"  TTA confidence: {results['tta_avg_confidence']:.4f}")
    print(f"  Change: {results['avg_confidence_improvement']:+.4f}")
    
    print(f"\n IMPROVEMENT BREAKDOWN:")
    print(f"  Improved samples: {results['improvements']} ({results['improvement_rate']:.1%})")
    print(f"  Degraded samples: {results['degradations']} ({results['degradation_rate']:.1%})")
    print(f"  Net success rate: {results['net_improvement_rate']:+.1%}")
    
    print(f"\n PERFORMANCE:")
    print(f"  Total time: {results['inference_time']:.2f}s")
    print(f"  Time per sample: {results['time_per_sample']*1000:.1f}ms")
    
    baseline_time_estimate = results['time_per_sample'] / results['num_augmentations']
    slowdown = results['time_per_sample'] / baseline_time_estimate if baseline_time_estimate > 0 else results['num_augmentations']
    print(f"  Estimated slowdown: {slowdown:.1f}x")
    
    print(f"{'='*60}")


def evaluate_tta_detailed_analysis(classifier_model: torch.nn.Module,
                                 test_dataset: torch.utils.data.Dataset,
                                 device: torch.device,
                                 num_samples: int = 500,
                                 confidence_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Esegue un'analisi dettagliata di TTA includendo analisi per confidenza.
    
    Args:
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        device: Device di computazione
        num_samples: Numero di campioni da analizzare
        confidence_threshold: Soglia per analisi confidenza
    
    Returns:
        Dict con analisi dettagliata dei risultati TTA
    """
    print(f" Starting detailed TTA analysis...")
    
    # Esegui valutazione TTA standard
    results = evaluate_tta(
        classifier_model=classifier_model,
        test_dataset=test_dataset,
        device=device,
        num_samples=num_samples,
        verbose=False
    )
    
    # Analisi per livelli di confidenza
    confidence_improvements = results['confidence_improvements']
    
    # Separa campioni per confidenza baseline
    high_conf_improvements = []
    low_conf_improvements = []
    
    # Re-evaluta per ottenere confidenze baseline dettagliate
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    baseline_confidences = []
    
    for idx in indices:
        image, label = test_dataset[idx]
        if not isinstance(image, torch.Tensor):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        image = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            baseline_output = classifier_model(image)
            baseline_prob = torch.nn.functional.softmax(baseline_output, dim=1)
            baseline_confidence, _ = torch.max(baseline_prob, 1)
            baseline_confidences.append(baseline_confidence.item())
    
    # Categorizza miglioramenti per confidenza
    for i, (base_conf, conf_imp) in enumerate(zip(baseline_confidences, confidence_improvements)):
        if base_conf >= confidence_threshold:
            high_conf_improvements.append(conf_imp)
        else:
            low_conf_improvements.append(conf_imp)
    
    # Calcola statistiche per confidenza
    detailed_analysis = {
        'confidence_threshold': confidence_threshold,
        'high_confidence_samples': len(high_conf_improvements),
        'low_confidence_samples': len(low_conf_improvements),
        'high_conf_avg_improvement': np.mean(high_conf_improvements) if high_conf_improvements else 0,
        'low_conf_avg_improvement': np.mean(low_conf_improvements) if low_conf_improvements else 0,
        'high_conf_improvement_std': np.std(high_conf_improvements) if high_conf_improvements else 0,
        'low_conf_improvement_std': np.std(low_conf_improvements) if low_conf_improvements else 0
    }
    
    results['detailed_analysis'] = detailed_analysis
    
    print(f"\n DETAILED CONFIDENCE ANALYSIS:")
    print(f"  High confidence samples ({confidence_threshold}+): {len(high_conf_improvements)}")
    print(f"  Avg improvement (high conf): {detailed_analysis['high_conf_avg_improvement']:+.4f}")
    print(f"  Low confidence samples (<{confidence_threshold}): {len(low_conf_improvements)}")
    print(f"  Avg improvement (low conf): {detailed_analysis['low_conf_avg_improvement']:+.4f}")
    
    return results


def compare_tta_configurations(classifier_model: torch.nn.Module,
                             test_dataset: torch.utils.data.Dataset,
                             device: torch.device,
                             num_samples: int = 500) -> Dict[str, Any]:
    """
    Confronta diverse configurazioni TTA.
    
    Args:
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        device: Device di computazione
        num_samples: Numero di campioni per confronto
    
    Returns:
        Dict con confronto tra configurazioni TTA
    """
    print(f" Comparing TTA configurations...")
    
    results = {'configurations': []}
    
    # Configurazione 1: Solo ttach (se disponibile)
    if TTA_AVAILABLE:
        print(" Testing ttach configuration...")
        ttach_results = evaluate_tta(
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            device=device,
            num_samples=num_samples,
            use_ttach=True,
            verbose=False
        )
        ttach_results['config_name'] = 'ttach'
        results['configurations'].append(ttach_results)
    
    # Configurazione 2: Implementazione manuale
    print(" Testing manual TTA configuration...")
    manual_results = evaluate_tta(
        classifier_model=classifier_model,
        test_dataset=test_dataset,
        device=device,
        num_samples=num_samples,
        use_ttach=False,
        verbose=False
    )
    manual_results['config_name'] = 'manual'
    results['configurations'].append(manual_results)
    
    # Trova la migliore configurazione
    if results['configurations']:
        best_config = max(results['configurations'], key=lambda x: x['accuracy_improvement'])
        results['best_config'] = best_config
        
        # Stampa confronto
        print(f"\n{'='*60}")
        print("TTA CONFIGURATIONS COMPARISON")
        print(f"{'='*60}")
        
        print(f"{'Config':<10} {'Accuracy':<10} {'Improvement':<12} {'Time/Sample':<12} {'Augs':<5}")
        print("-" * 60)
        
        for config in results['configurations']:
            print(f"{config['config_name']:<10} "
                  f"{config['tta_accuracy']:<10.4f} "
                  f"{config['accuracy_improvement']:<12.4f} "
                  f"{config['time_per_sample']*1000:<12.1f} "
                  f"{config['num_augmentations']:<5}")
        
        print(f"\n BEST CONFIG: {best_config['config_name']}")
        print(f"  Accuracy improvement: {best_config['accuracy_improvement']:+.4f}")
        print(f"  Augmentations: {best_config['num_augmentations']}")
    
    return results


def evaluate_tta_efficiency(classifier_model: torch.nn.Module,
                          test_dataset: torch.utils.data.Dataset,
                          device: torch.device,
                          num_samples: int = 200) -> Dict[str, Any]:
    """
    Valuta l'efficienza di TTA in termini di costo/beneficio.
    
    Args:
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        device: Device di computazione
        num_samples: Numero di campioni per test efficienza
    
    Returns:
        Dict con analisi di efficienza
    """
    print(f" Evaluating TTA efficiency...")
    
    # Misura tempo baseline
    baseline_times = []
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    for idx in indices[:10]:  # Usa solo 10 campioni per stima baseline
        image, _ = test_dataset[idx]
        if not isinstance(image, torch.Tensor):
            import torchvision.transforms as transforms
            to_tensor = transforms.ToTensor()
            image = to_tensor(image)
        
        image = image.unsqueeze(0).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            _ = classifier_model(image)
        baseline_times.append(time.time() - start_time)
    
    avg_baseline_time = np.mean(baseline_times)
    
    # Esegui valutazione TTA
    tta_results = evaluate_tta(
        classifier_model=classifier_model,
        test_dataset=test_dataset,
        device=device,
        num_samples=num_samples,
        verbose=False
    )
    
    # Calcola metriche di efficienza
    slowdown_factor = tta_results['time_per_sample'] / avg_baseline_time if avg_baseline_time > 0 else tta_results['num_augmentations']
    
    # Efficienza = miglioramento accuratezza / costo computazionale extra
    extra_compute_cost = slowdown_factor - 1
    efficiency_score = tta_results['accuracy_improvement'] / extra_compute_cost if extra_compute_cost > 0 else 0
    
    efficiency_results = {
        'avg_baseline_time': avg_baseline_time,
        'tta_time_per_sample': tta_results['time_per_sample'],
        'slowdown_factor': slowdown_factor,
        'extra_compute_cost': extra_compute_cost,
        'efficiency_score': efficiency_score,
        'accuracy_improvement': tta_results['accuracy_improvement'],
        'num_augmentations': tta_results['num_augmentations']
    }
    

    
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    print(f"  Baseline time: {avg_baseline_time*1000:.1f}ms")
    print(f"  TTA time: {tta_results['time_per_sample']*1000:.1f}ms")
    print(f"  Slowdown: {slowdown_factor:.1f}x")
    print(f"  Accuracy gain: {tta_results['accuracy_improvement']:+.4f}")
    print(f"  Efficiency score: {efficiency_score:.6f}")
    
    return efficiency_results


# Funzione wrapper per compatibilità
def run_tta_evaluation(classifier_path: str = './checkpoint/ckpt.pth',
                      data_root: str = './data',
                      num_samples: int = 1000,
                      use_ttach: bool = True,
                      device: torch.device = None) -> Dict[str, Any]:
    """
    Funzione wrapper per eseguire valutazione TTA completa.
    
    Args:
        classifier_path: Percorso del classificatore
        data_root: Directory root per i dati
        num_samples: Numero di campioni da valutare
        use_ttach: Se usare ttach library
        device: Device (auto-detect se None)
    
    Returns:
        Dict con risultati della valutazione TTA
    """
    from core.model_loader import load_classifier
    from core.data_utils import get_cifar10_test_dataset
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f" Running complete TTA evaluation...")
    print(f" Classifier: {classifier_path}")
    print(f" Data root: {data_root}")
    print(f" Samples: {num_samples}")
    print(f" Device: {device}")
    
    # Carica modello
    classifier = load_classifier(classifier_path, device)
    
    # Carica dataset
    test_dataset = get_cifar10_test_dataset(data_root=data_root)
    
    # Esegui valutazione
    results = evaluate_tta(
        classifier_model=classifier,
        test_dataset=test_dataset,
        device=device,
        num_samples=num_samples,
        use_ttach=use_ttach,
        verbose=True
    )
    
    return results


if __name__ == '__main__':
    """
    Script principale per test del modulo.
    """
    print("Testing TTA evaluation module...")
    print(f"ttach available: {TTA_AVAILABLE}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Test valutazione standard
        results = run_tta_evaluation(
            classifier_path='./checkpoint/ckpt.pth',
            data_root='./data',
            num_samples=500,
            use_ttach=True,
            device=device
        )
        
        print(f"\n TTA evaluation completed!")
        print(f" Accuracy improvement: {results['accuracy_improvement']:+.4f}")
        print(f" Confidence change: {results['avg_confidence_improvement']:+.4f}")
        print(f" Method used: {results['method_used']}")
        
        # Test confronto configurazioni se possibile
        from core.model_loader import load_classifier
        from core.data_utils import get_cifar10_test_dataset
        
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        test_dataset = get_cifar10_test_dataset('./data')
        
        print(f"\n Testing configuration comparison...")
        comparison_results = compare_tta_configurations(
            classifier_model=classifier,
            test_dataset=test_dataset,
            device=device,
            num_samples=200
        )
        
        print(f"\n Testing efficiency analysis...")
        efficiency_results = evaluate_tta_efficiency(
            classifier_model=classifier,
            test_dataset=test_dataset,
            device=device,
            num_samples=100
        )
        
    except Exception as e:
        print(f" Error during evaluation: {e}")
        print("Make sure the classifier and data are available at the specified paths.")