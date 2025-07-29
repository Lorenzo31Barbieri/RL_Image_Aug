"""
Modulo per la valutazione con fixed augmentation.
Applica una sequenza fissa di trasformazioni e confronta con baseline.
"""

import torch
from typing import Dict, Any, List
from tqdm import tqdm

# Import dei moduli core
from evaluation.core.evaluation_core import (
    time_evaluation_context,
    calculate_basic_metrics,
    calculate_improvement_metrics,
    print_evaluation_summary,
    validate_evaluation_inputs
)
from evaluation.core.data_utils import (
    get_cifar10_test_dataset,
    create_fixed_augmentation_transform,
    create_standard_preprocessing
)
from evaluation.core.model_loader import load_classifier


def evaluate_fixed_augmentation(classifier_model: torch.nn.Module,
                               test_dataset: torch.utils.data.Dataset,
                               augmentation_ids: List[int],
                               device: torch.device,
                               batch_size: int = 64,
                               verbose: bool = True) -> Dict[str, Any]:
    """
    Valuta il classificatore con fixed augmentation e lo confronta con baseline.
    
    Args:
        classifier_model: Modello classificatore pre-trained
        test_dataset: Dataset di test (senza trasformazioni)
        augmentation_ids: Lista di ID delle trasformazioni da applicare
        device: Device per computazione
        batch_size: Dimensione del batch
        verbose: Se stampare informazioni dettagliate
    
    Returns:
        Dict con risultati della valutazione:
        - baseline_accuracy: Accuratezza baseline
        - augmented_accuracy: Accuratezza con augmentation
        - accuracy_improvement: Miglioramento dell'accuratezza
        - baseline_confidence: Confidenza media baseline
        - augmented_confidence: Confidenza media con augmentation
        - confidence_improvement: Miglioramento della confidenza
        - augmentation_ids: ID delle trasformazioni applicate
        - augmentation_names: Nomi delle trasformazioni
        - inference_time: Tempo di inferenza
        - time_per_sample: Tempo per campione
    """
    validate_evaluation_inputs(classifier_model, device=device)
    
    if verbose:
        print(f"🎯 Starting fixed augmentation evaluation...")
        print(f"📊 Dataset size: {len(test_dataset)} samples")
        print(f"🔧 Augmentation IDs: {augmentation_ids}")
        print(f"📦 Batch size: {batch_size}")
    
    # Ottieni nomi delle trasformazioni
    augmentation_names = []
    try:
        from src.environment.transforms import _ACTIONS_MAP
        augmentation_names = [_ACTIONS_MAP[aid][1] for aid in augmentation_ids if aid in _ACTIONS_MAP]
    except ImportError:
        augmentation_names = [f"Transform_{aid}" for aid in augmentation_ids]
    
    with time_evaluation_context("FIXED AUGMENTATION"):
        # 1. Valutazione Baseline (preprocessing standard)
        baseline_results = _evaluate_with_transform(
            classifier_model=classifier_model,
            dataset=test_dataset,
            transform=create_standard_preprocessing(),
            device=device,
            batch_size=batch_size,
            description="Baseline"
        )
        
        # 2. Valutazione con Fixed Augmentation
        augmented_transform = create_fixed_augmentation_transform(
            action_ids=augmentation_ids,
            normalize=True
        )
        
        augmented_results = _evaluate_with_transform(
            classifier_model=classifier_model,
            dataset=test_dataset,
            transform=augmented_transform,
            device=device,
            batch_size=batch_size,
            description=f"Fixed Aug ({', '.join(augmentation_names[:3])}{'...' if len(augmentation_names) > 3 else ''})"
        )
        
        # 3. Calcola miglioramenti
        accuracy_improvement = augmented_results['accuracy'] - baseline_results['accuracy']
        confidence_improvement = augmented_results['avg_confidence'] - baseline_results['avg_confidence']
        
        # 4. Compila risultati finali
        results = {
            # Risultati baseline
            'baseline_accuracy': baseline_results['accuracy'],
            'baseline_confidence': baseline_results['avg_confidence'],
            'baseline_f1_score': baseline_results['f1_score'],
            
            # Risultati augmented
            'augmented_accuracy': augmented_results['accuracy'],
            'augmented_confidence': augmented_results['avg_confidence'],
            'augmented_f1_score': augmented_results['f1_score'],
            
            # Miglioramenti
            'accuracy_improvement': accuracy_improvement,
            'confidence_improvement': confidence_improvement,
            
            # Metadati
            'augmentation_ids': augmentation_ids,
            'augmentation_names': augmentation_names,
            'method': 'fixed_augmentation',
            'total_samples': len(test_dataset),
            
            # Metriche temporali
            'baseline_time': baseline_results['inference_time'],
            'augmented_time': augmented_results['inference_time'],
            'inference_time': baseline_results['inference_time'] + augmented_results['inference_time'],
            'time_per_sample': (baseline_results['time_per_sample'] + augmented_results['time_per_sample']) / 2
        }
        
        if verbose:
            _print_fixed_aug_summary(results)
    
    return results


def _evaluate_with_transform(classifier_model: torch.nn.Module,
                           dataset: torch.utils.data.Dataset,
                           transform: torch.nn.Module,
                           device: torch.device,
                           batch_size: int,
                           description: str) -> Dict[str, Any]:
    """
    Valuta il classificatore applicando una specifica trasformazione.
    
    Args:
        classifier_model: Modello classificatore
        dataset: Dataset originale
        transform: Trasformazione da applicare
        device: Device di computazione
        batch_size: Dimensione del batch
        description: Descrizione per progress bar
    
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
    
    return metrics


def _print_fixed_aug_summary(results: Dict[str, Any]) -> None:
    """Stampa un riassunto dettagliato dei risultati fixed augmentation."""
    
    print(f"\n{'='*60}")
    print("FIXED AUGMENTATION EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"🔧 AUGMENTATIONS APPLIED:")
    for i, (aug_id, aug_name) in enumerate(zip(results['augmentation_ids'], results['augmentation_names'])):
        print(f"  {i+1}. ID {aug_id}: {aug_name}")
    
    print(f"\n📊 ACCURACY COMPARISON:")
    print(f"  Baseline: {results['baseline_accuracy']:.4f}")
    print(f"  Augmented: {results['augmented_accuracy']:.4f}")
    
    improvement_sign = "📈" if results['accuracy_improvement'] > 0 else "📉" if results['accuracy_improvement'] < 0 else "➡️"
    print(f"  {improvement_sign} Improvement: {results['accuracy_improvement']:+.4f}")
    
    print(f"\n🔍 CONFIDENCE COMPARISON:")
    print(f"  Baseline: {results['baseline_confidence']:.4f}")
    print(f"  Augmented: {results['augmented_confidence']:.4f}")
    print(f"  Change: {results['confidence_improvement']:+.4f}")
    
    print(f"\n⏱️ TIMING:")
    print(f"  Baseline time: {results['baseline_time']:.2f}s")
    print(f"  Augmented time: {results['augmented_time']:.2f}s")
    print(f"  Time per sample: {results['time_per_sample']*1000:.1f}ms")
    
    # Raccomandazione
    if results['accuracy_improvement'] > 0.01:
        recommendation = "✅ Significant improvement - Recommended"
    elif results['accuracy_improvement'] > 0.005:
        recommendation = "⚠️ Modest improvement - Consider cost/benefit"
    elif results['accuracy_improvement'] > 0:
        recommendation = "📊 Minimal improvement - Limited benefit"
    else:
        recommendation = "❌ No improvement or degradation - Not recommended"
    
    print(f"\n💡 RECOMMENDATION: {recommendation}")
    print(f"{'='*60}")


def compare_multiple_augmentations(classifier_model: torch.nn.Module,
                                 test_dataset: torch.utils.data.Dataset,
                                 augmentation_configs: List[Dict[str, Any]],
                                 device: torch.device,
                                 batch_size: int = 64) -> Dict[str, Any]:
    """
    Confronta multiple configurazioni di fixed augmentation.
    
    Args:
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        augmentation_configs: Lista di dict con 'ids' e 'name' per ogni config
        device: Device di computazione
        batch_size: Dimensione del batch
    
    Returns:
        Dict con confronto tra le configurazioni
    """
    print(f"🔍 Comparing {len(augmentation_configs)} augmentation configurations...")
    
    results = {'comparisons': []}
    
    for i, config in enumerate(augmentation_configs):
        aug_ids = config['ids']
        config_name = config.get('name', f"Config_{i+1}")
        
        print(f"\n📋 Evaluating {config_name}: {aug_ids}")
        
        config_results = evaluate_fixed_augmentation(
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            augmentation_ids=aug_ids,
            device=device,
            batch_size=batch_size,
            verbose=False
        )
        
        config_results['config_name'] = config_name
        results['comparisons'].append(config_results)
    
    # Trova la migliore configurazione
    best_config = max(results['comparisons'], key=lambda x: x['accuracy_improvement'])
    results['best_config'] = best_config
    
    # Stampa confronto
    print(f"\n{'='*70}")
    print("AUGMENTATION CONFIGURATIONS COMPARISON")
    print(f"{'='*70}")
    
    print(f"{'Config':<15} {'Baseline':<10} {'Augmented':<10} {'Improvement':<12} {'Conf Change':<12}")
    print("-" * 70)
    
    for comp in results['comparisons']:
        print(f"{comp['config_name']:<15} "
              f"{comp['baseline_accuracy']:<10.4f} "
              f"{comp['augmented_accuracy']:<10.4f} "
              f"{comp['accuracy_improvement']:<12.4f} "
              f"{comp['confidence_improvement']:<12.4f}")
    
    print(f"\n🏆 BEST CONFIG: {best_config['config_name']}")
    print(f"  Accuracy improvement: {best_config['accuracy_improvement']:+.4f}")
    print(f"  Augmentations: {best_config['augmentation_names']}")
    
    return results


def evaluate_augmentation_robustness(classifier_model: torch.nn.Module,
                                   test_dataset: torch.utils.data.Dataset,
                                   augmentation_ids: List[int],
                                   device: torch.device,
                                   num_runs: int = 5,
                                   batch_size: int = 64) -> Dict[str, Any]:
    """
    Valuta la robustezza di una configurazione di augmentation attraverso multiple run.
    
    Args:
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        augmentation_ids: ID delle augmentation da testare
        device: Device di computazione
        num_runs: Numero di run per test di robustezza
        batch_size: Dimensione del batch
    
    Returns:
        Dict con statistiche di robustezza
    """
    print(f"🔬 Testing augmentation robustness over {num_runs} runs...")
    
    accuracies = []
    improvements = []
    confidences = []
    
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}")
        
        results = evaluate_fixed_augmentation(
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            augmentation_ids=augmentation_ids,
            device=device,
            batch_size=batch_size,
            verbose=False
        )
        
        accuracies.append(results['augmented_accuracy'])
        improvements.append(results['accuracy_improvement'])
        confidences.append(results['augmented_confidence'])
    
    import numpy as np
    
    robustness_results = {
        'num_runs': num_runs,
        'augmentation_ids': augmentation_ids,
        'accuracy_mean': np.mean(accuracies),
        'accuracy_std': np.std(accuracies),
        'accuracy_min': np.min(accuracies),
        'accuracy_max': np.max(accuracies),
        'improvement_mean': np.mean(improvements),
        'improvement_std': np.std(improvements),
        'confidence_mean': np.mean(confidences),
        'confidence_std': np.std(confidences),
        'all_accuracies': accuracies,
        'all_improvements': improvements,
        'all_confidences': confidences
    }
    
    print(f"\n📊 ROBUSTNESS RESULTS:")
    print(f"  Mean accuracy: {robustness_results['accuracy_mean']:.4f} ± {robustness_results['accuracy_std']:.4f}")
    print(f"  Mean improvement: {robustness_results['improvement_mean']:.4f} ± {robustness_results['improvement_std']:.4f}")
    print(f"  Range: [{robustness_results['accuracy_min']:.4f}, {robustness_results['accuracy_max']:.4f}]")
    
    # Valuta stabilità
    cv = robustness_results['accuracy_std'] / robustness_results['accuracy_mean'] if robustness_results['accuracy_mean'] > 0 else float('inf')
    
    if cv < 0.01:
        stability = "🟢 Very Stable"
    elif cv < 0.02:
        stability = "🟡 Stable"
    elif cv < 0.05:
        stability = "🟠 Moderately Stable"
    else:
        stability = "🔴 Unstable"
    
    robustness_results['stability_assessment'] = stability
    robustness_results['coefficient_of_variation'] = cv
    
    print(f"  Stability: {stability} (CV: {cv:.4f})")
    
    return robustness_results


# Funzione wrapper per compatibilità
def run_fixed_augmentation_evaluation(classifier_path: str = './checkpoint/ckpt.pth',
                                    data_root: str = './data',
                                    augmentation_ids: List[int] = [0, 3, 6],
                                    batch_size: int = 64,
                                    device: torch.device = None) -> Dict[str, Any]:
    """
    Funzione wrapper per eseguire valutazione fixed augmentation completa.
    
    Args:
        classifier_path: Percorso del classificatore
        data_root: Directory root per i dati
        augmentation_ids: Lista di ID delle trasformazioni
        batch_size: Dimensione del batch
        device: Device (auto-detect se None)
    
    Returns:
        Dict con risultati della valutazione
    """
    from core.model_loader import load_classifier
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Running complete fixed augmentation evaluation...")
    print(f"📁 Classifier: {classifier_path}")
    print(f"📁 Data root: {data_root}")
    print(f"🔧 Augmentations: {augmentation_ids}")
    
    # Carica modello
    classifier = load_classifier(classifier_path, device)
    
    # Carica dataset (senza trasformazioni, verranno applicate nel processo)
    test_dataset = get_cifar10_test_dataset(data_root=data_root, transform=None)
    
    # Esegui valutazione
    results = evaluate_fixed_augmentation(
        classifier_model=classifier,
        test_dataset=test_dataset,
        augmentation_ids=augmentation_ids,
        device=device,
        batch_size=batch_size,
        verbose=True
    )
    
    return results


if __name__ == '__main__':
    """
    Script principale per test del modulo.
    """
    print("Testing fixed augmentation evaluation module...")
    
    # Configurazione di test
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_augmentations = [0, 3, 6]  # Esempio: Brightness +20%, Contrast -20%, Horizontal Flip
    
    try:
        # Test standard
        results = run_fixed_augmentation_evaluation(
            classifier_path='./checkpoint/ckpt.pth',
            data_root='./data',
            augmentation_ids=test_augmentations,
            batch_size=64,
            device=device
        )
        
        print(f"\n🎉 Fixed augmentation evaluation completed!")
        print(f"📊 Accuracy improvement: {results['accuracy_improvement']:+.4f}")
        print(f"🔍 Confidence change: {results['confidence_improvement']:+.4f}")
        
        # Test confronto multiple configurazioni
        configs = [
            {'ids': [0, 3], 'name': 'Brightness+Contrast'},
            {'ids': [6, 7], 'name': 'Flips'},
            {'ids': [0, 6], 'name': 'Brightness+HFlip'},
            {'ids': [0, 3, 6], 'name': 'Full_Combo'}
        ]
        
        print(f"\n🔍 Testing multiple configurations...")
        comparison_results = compare_multiple_augmentations(
            classifier_model=load_classifier('./checkpoint/ckpt.pth', device),
            test_dataset=get_cifar10_test_dataset('./data', transform=None),
            augmentation_configs=configs,
            device=device
        )
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Make sure the classifier, data, and transform modules are available.")