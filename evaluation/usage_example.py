#!/usr/bin/env python3
"""
Esempio completo di utilizzo del sistema di valutazione modulare.

Questo script dimostra come utilizzare il nuovo sistema modulare per:
1. Valutare singoli metodi
2. Confrontare tutti i metodi
3. Creare analisi personalizzate

Uso:
    python usage_example.py --mode single --method baseline
    python usage_example.py --mode comparison --quick
    python usage_example.py --mode custom
"""

import torch
import argparse
import json
from pathlib import Path

# Import del sistema modulare
from core import load_classifier, load_rl_agent, get_cifar10_test_dataset, get_cifar10_test_loader
from methods import (
    evaluate_baseline, 
    evaluate_fixed_augmentation, 
    evaluate_tta, 
    evaluate_rl_agent
)
from comparison import EvaluationComparison, create_default_config


def example_single_method_evaluation():
    """Esempio di valutazione di un singolo metodo."""
    print("🎯 Example: Single Method Evaluation")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica modello e dati
    classifier = load_classifier('./checkpoint/ckpt.pth', device)
    test_loader = get_cifar10_test_loader(batch_size=64)
    
    # Valuta baseline
    print("\n📊 Evaluating baseline classifier...")
    baseline_results = evaluate_baseline(
        classifier_model=classifier,
        test_loader=test_loader,
        device=device,
        verbose=True
    )
    
    print(f"\n✅ Baseline Results:")
    print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  Confidence: {baseline_results['avg_confidence']:.4f}")
    print(f"  Time per sample: {baseline_results['time_per_sample']*1000:.1f}ms")
    
    return baseline_results


def example_tta_evaluation():
    """Esempio di valutazione TTA."""
    print("\n🔬 Example: TTA Evaluation")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica modello e dati
    classifier = load_classifier('./checkpoint/ckpt.pth', device)
    test_dataset = get_cifar10_test_dataset()
    
    # Valuta TTA
    print("\n🧪 Evaluating TTA...")
    tta_results = evaluate_tta(
        classifier_model=classifier,
        test_dataset=test_dataset,
        device=device,
        num_samples=500,
        verbose=True
    )
    
    print(f"\n✅ TTA Results:")
    print(f"  Baseline accuracy: {tta_results['baseline_accuracy']:.4f}")
    print(f"  TTA accuracy: {tta_results['tta_accuracy']:.4f}")
    print(f"  Improvement: {tta_results['accuracy_improvement']:+.4f}")
    print(f"  Augmentations used: {tta_results['num_augmentations']}")
    
    return tta_results


def example_fixed_augmentation_evaluation():
    """Esempio di valutazione fixed augmentation."""
    print("\n🔧 Example: Fixed Augmentation Evaluation")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica modello e dati
    classifier = load_classifier('./checkpoint/ckpt.pth', device)
    test_dataset = get_cifar10_test_dataset()
    
    # Definisci augmentation
    augmentation_ids = [0, 3, 6]  # Brightness, Contrast, HFlip
    
    print(f"\n⚙️ Testing augmentation sequence: {augmentation_ids}")
    
    # Valuta fixed augmentation
    fixed_aug_results = evaluate_fixed_augmentation(
        classifier_model=classifier,
        test_dataset=test_dataset,
        augmentation_ids=augmentation_ids,
        device=device,
        verbose=True
    )
    
    print(f"\n✅ Fixed Augmentation Results:")
    print(f"  Baseline accuracy: {fixed_aug_results['baseline_accuracy']:.4f}")
    print(f"  Augmented accuracy: {fixed_aug_results['augmented_accuracy']:.4f}")
    print(f"  Improvement: {fixed_aug_results['accuracy_improvement']:+.4f}")
    print(f"  Augmentations: {fixed_aug_results['augmentation_names']}")
    
    return fixed_aug_results


def example_rl_evaluation():
    """Esempio di valutazione RL agent."""
    print("\n🤖 Example: RL Agent Evaluation")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Carica modelli
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        agent, model_loaded = load_rl_agent('./models/best_improved_dqn_model.pth', device=device)
        test_dataset = get_cifar10_test_dataset()
        
        if not model_loaded:
            print("⚠️ Warning: Using randomly initialized RL agent")
        
        # Valuta RL agent
        print(f"\n🎮 Evaluating RL agent...")
        rl_results = evaluate_rl_agent(
            agent=agent,
            classifier_model=classifier,
            test_dataset=test_dataset,
            device=device,
            num_episodes=500,
            verbose=True
        )
        
        print(f"\n✅ RL Agent Results:")
        print(f"  Initial accuracy: {rl_results['initial_accuracy']:.4f}")
        print(f"  Final accuracy: {rl_results['final_accuracy']:.4f}")
        print(f"  Improvement: {rl_results['accuracy_improvement']:+.4f}")
        print(f"  Average reward: {rl_results['avg_reward']:.3f}")
        print(f"  Success rate: {rl_results['improvement_rate']:.1%}")
        
        return rl_results
        
    except ImportError:
        print("❌ RL modules not available. Skipping RL evaluation.")
        return None


def example_comprehensive_comparison():
    """Esempio di confronto completo."""
    print("\n🏆 Example: Comprehensive Comparison")
    print("="*60)
    
    # Crea configurazione
    config = create_default_config()
    
    # Personalizza per esempio rapido
    config.update({
        'tta_samples': 300,
        'rl_episodes': 300,
        'batch_size': 32,
        'output_dir': './example_results'
    })
    
    print(f"📋 Using configuration:")
    print(json.dumps(config, indent=2))
    
    # Inizializza sistema di confronto
    comparison = EvaluationComparison(config)
    
    try:
        # Carica modelli e dati
        comparison.load_models()
        comparison.load_data()
        
        # Esegui tutte le valutazioni
        comparison.run_all_evaluations()
        
        # Stampa riassunto comparativo
        comparison.print_comparison_summary()
        
        # Crea grafici
        comparison.create_comparison_plots()
        
        print(f"\n🎉 Comprehensive comparison completed!")
        print(f"📁 Results saved to: {config['output_dir']}")
        
        return comparison.results
        
    except Exception as e:
        print(f"❌ Error during comparison: {e}")
        return None


def example_custom_analysis():
    """Esempio di analisi personalizzata."""
    print("\n🔬 Example: Custom Analysis")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica modello
    classifier = load_classifier('./checkpoint/ckpt.pth', device)
    test_dataset = get_cifar10_test_dataset()
    
    # Confronta diverse configurazioni di fixed augmentation
    augmentation_configs = [
        {'ids': [0], 'name': 'Brightness_Only'},
        {'ids': [3], 'name': 'Contrast_Only'},
        {'ids': [6], 'name': 'HFlip_Only'},
        {'ids': [0, 3], 'name': 'Brightness_Contrast'},
        {'ids': [0, 6], 'name': 'Brightness_HFlip'},
        {'ids': [3, 6], 'name': 'Contrast_HFlip'},
        {'ids': [0, 3, 6], 'name': 'All_Three'}
    ]
    
    print(f"\n🧪 Testing {len(augmentation_configs)} augmentation configurations...")
    
    from evaluation.methods.evaluate_fixed_aug import compare_multiple_augmentations
    
    comparison_results = compare_multiple_augmentations(
        classifier_model=classifier,
        test_dataset=test_dataset,
        augmentation_configs=augmentation_configs,
        device=device,
        batch_size=64
    )
    
    print(f"\n🏆 Best configuration: {comparison_results['best_config']['config_name']}")
    print(f"   Improvement: {comparison_results['best_config']['accuracy_improvement']:+.4f}")
    
    # Analisi TTA dettagliata
    print(f"\n🔍 Running detailed TTA analysis...")
    
    from evaluation.methods.evaluate_tta import evaluate_tta_detailed_analysis
    
    detailed_tta = evaluate_tta_detailed_analysis(
        classifier_model=classifier,
        test_dataset=test_dataset,
        device=device,
        num_samples=300,
        confidence_threshold=0.7
    )
    
    print(f"📊 TTA detailed results:")
    print(f"   High confidence improvement: {detailed_tta['detailed_analysis']['high_conf_avg_improvement']:+.4f}")
    print(f"   Low confidence improvement: {detailed_tta['detailed_analysis']['low_conf_avg_improvement']:+.4f}")
    
    return {
        'fixed_aug_comparison': comparison_results,
        'detailed_tta': detailed_tta
    }


def example_performance_profiling():
    """Esempio di profiling delle performance."""
    print("\n⚡ Example: Performance Profiling")
    print("="*50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica modello
    classifier = load_classifier('./checkpoint/ckpt.pth', device)
    test_dataset = get_cifar10_test_dataset()
    
    # Testa efficienza TTA
    print(f"\n📊 Profiling TTA efficiency...")
    
    from evaluation.methods.evaluate_tta import evaluate_tta_efficiency
    
    tta_efficiency = evaluate_tta_efficiency(
        classifier_model=classifier,
        test_dataset=test_dataset,
        device=device,
        num_samples=100
    )
    
    print(f"⚡ TTA Efficiency Results:")
    print(f"   Efficiency score: {tta_efficiency['efficiency_score']:.6f}")
    print(f"   Rating: {tta_efficiency['efficiency_rating']}")
    
    # Confronta configurazioni TTA
    print(f"\n🔄 Comparing TTA configurations...")
    
    from evaluation.methods.evaluate_tta import compare_tta_configurations
    
    tta_comparison = compare_tta_configurations(
        classifier_model=classifier,
        test_dataset=test_dataset,
        device=device,
        num_samples=200
    )
    
    if 'best_config' in tta_comparison:
        best_config = tta_comparison['best_config']
        print(f"🏆 Best TTA config: {best_config['config_name']}")
        print(f"   Improvement: {best_config['accuracy_improvement']:+.4f}")
        print(f"   Time per sample: {best_config['time_per_sample']*1000:.1f}ms")
    
    return {
        'tta_efficiency': tta_efficiency,
        'tta_comparison': tta_comparison
    }


def save_example_results(results: dict, output_file: str = "example_results.json"):
    """Salva i risultati degli esempi."""
    print(f"\n💾 Saving example results to {output_file}...")
    
    # Converti risultati in formato JSON-serializable
    json_safe_results = {}
    
    for key, value in results.items():
        if isinstance(value, dict):
            json_safe_results[key] = {k: v for k, v in value.items() 
                                    if not isinstance(v, (torch.Tensor, type(None)))}
        else:
            json_safe_results[key] = str(value)
    
    with open(output_file, 'w') as f:
        json.dump(json_safe_results, f, indent=2, default=str)
    
    print(f"✅ Results saved to {output_file}")


def main():
    """Funzione principale con interfaccia command line."""
    parser = argparse.ArgumentParser(
        description="Examples of modular evaluation system usage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python usage_example.py --mode single --method baseline
  python usage_example.py --mode single --method tta
  python usage_example.py --mode comparison --quick
  python usage_example.py --mode custom
  python usage_example.py --mode profiling
        """
    )
    
    parser.add_argument('--mode', type=str, required=True,
                       choices=['single', 'comparison', 'custom', 'profiling'],
                       help='Evaluation mode to run')
    
    parser.add_argument('--method', type=str,
                       choices=['baseline', 'fixed_aug', 'tta', 'rl'],
                       help='Specific method for single mode')
    
    parser.add_argument('--quick', action='store_true',
                       help='Use reduced samples for quick testing')
    
    parser.add_argument('--output', type=str, default='example_results.json',
                       help='Output file for results')
    
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'],
                       help='Force specific device')
    
    args = parser.parse_args()
    
    # Override device se specificato
    if args.device:
        torch.cuda.set_device(0) if args.device == 'cuda' and torch.cuda.is_available() else None
    
    print(f"🚀 Running evaluation examples...")
    print(f"📋 Mode: {args.mode}")
    print(f"💻 Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    results = {}
    
    try:
        if args.mode == 'single':
            if args.method == 'baseline' or args.method is None:
                results['baseline'] = example_single_method_evaluation()
            
            if args.method == 'tta' or args.method is None:
                results['tta'] = example_tta_evaluation()
            
            if args.method == 'fixed_aug' or args.method is None:
                results['fixed_aug'] = example_fixed_augmentation_evaluation()
            
            if args.method == 'rl' or args.method is None:
                results['rl'] = example_rl_evaluation()
        
        elif args.mode == 'comparison':
            results['comprehensive'] = example_comprehensive_comparison()
        
        elif args.mode == 'custom':
            results['custom'] = example_custom_analysis()
        
        elif args.mode == 'profiling':
            results['profiling'] = example_performance_profiling()
        
        # Salva risultati
        if results:
            save_example_results(results, args.output)
        
        print(f"\n🎉 All examples completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ Evaluation interrupted by user")
    
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print(f"💡 Make sure all required models and data are available")
        raise


if __name__ == '__main__':
    main()


# === QUICK START GUIDE ===

def quick_start_example():
    """
    Esempio rapido per iniziare subito.
    Esegui questo per testare il sistema con configurazione minimale.
    """
    print("🚀 QUICK START EXAMPLE")
    print("="*40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # 1. Carica il classificatore
        print("📥 Loading classifier...")
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        
        # 2. Carica dati di test
        print("📊 Loading test data...")
        test_loader = get_cifar10_test_loader(batch_size=32)
        
        # 3. Valuta baseline
        print("🎯 Evaluating baseline...")
        baseline_results = evaluate_baseline(classifier, test_loader, device, verbose=False)
        
        # 4. Confronta con TTA
        print("🔬 Evaluating TTA...")
        test_dataset = get_cifar10_test_dataset()
        tta_results = evaluate_tta(classifier, test_dataset, device, num_samples=100, verbose=False)
        
        # 5. Mostra risultati
        print(f"\n📊 QUICK RESULTS:")
        print(f"   Baseline accuracy: {baseline_results['accuracy']:.4f}")
        print(f"   TTA accuracy: {tta_results['tta_accuracy']:.4f}")
        print(f"   TTA improvement: {tta_results['accuracy_improvement']:+.4f}")
        
        if tta_results['accuracy_improvement'] > 0:
            print("✅ TTA shows improvement!")
        else:
            print("📊 TTA shows limited benefit")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick start failed: {e}")
        print("💡 Check that models and data are available at default paths")
        return False


# Uncomment per eseguire quick start direttamente
# if __name__ == '__main__':
#     quick_start_example()