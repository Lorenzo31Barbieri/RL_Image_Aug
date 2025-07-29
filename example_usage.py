#!/usr/bin/env python3
"""
Esempio di utilizzo del sistema dopo la ristrutturazione.
Questo script mostra come utilizzare il nuovo sistema modulare.
"""

import torch
import sys
from pathlib import Path

# Aggiungi la root del progetto al path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import from new structure
from src.models import VGG, DQNAgent
from src.environment import ImageAugmentationEnv, get_num_actions
from evaluation.core import load_classifier, load_rl_agent, get_cifar10_test_dataset
from evaluation.methods import evaluate_baseline, evaluate_rl_agent, evaluate_tta
from evaluation.comparison import EvaluationComparison
from evaluation.comparison.config import create_default_config, create_quick_test_config, print_config


def test_imports():
    """Test che tutti gli import funzionino correttamente."""
    print("🧪 Testing imports...")
    
    try:
        # Test import models
        print("  ✅ Models imported successfully")
        
        # Test import environment
        print("  ✅ Environment imported successfully")
        
        # Test import evaluation
        print("  ✅ Evaluation system imported successfully")
        
        print("🎉 All imports working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def example_individual_evaluation():
    """Esempio di valutazione individuale."""
    print("\n📊 Example: Individual Method Evaluation")
    print("-" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Carica modello
        print("Loading classifier...")
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        
        # Carica dati
        print("Loading test data...")
        from evaluation.core import get_cifar10_test_loader
        test_loader = get_cifar10_test_loader(batch_size=64)
        
        # Valutazione baseline
        print("Running baseline evaluation...")
        baseline_results = evaluate_baseline(
            classifier_model=classifier,
            test_loader=test_loader,
            device=device,
            verbose=True
        )
        
        print(f"\n✅ Baseline Results:")
        print(f"  Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"  Confidence: {baseline_results['avg_confidence']:.4f}")
        
        return baseline_results
        
    except Exception as e:
        print(f"❌ Error in individual evaluation: {e}")
        return None


def example_rl_evaluation():
    """Esempio di valutazione RL."""
    print("\n🤖 Example: RL Agent Evaluation")
    print("-" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Carica modelli
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        agent, model_loaded = load_rl_agent('./models/best_improved_dqn_model.pth', device=device)
        test_dataset = get_cifar10_test_dataset()
        
        print(f"RL model loaded: {'✅' if model_loaded else '❌ (using random)'}")
        
        # Valutazione RL
        print("Running RL evaluation...")
        rl_results = evaluate_rl_agent(
            agent=agent,
            classifier_model=classifier,
            test_dataset=test_dataset,
            device=device,
            num_episodes=100,  # Reduced for example
            verbose=True
        )
        
        print(f"\n✅ RL Results:")
        print(f"  Accuracy improvement: {rl_results['accuracy_improvement']:+.4f}")
        print(f"  Average reward: {rl_results['avg_reward']:.3f}")
        
        return rl_results
        
    except Exception as e:
        print(f"❌ Error in RL evaluation: {e}")
        return None


def example_comprehensive_comparison():
    """Esempio di confronto completo."""
    print("\n🏆 Example: Comprehensive Comparison")
    print("-" * 50)
    
    try:
        # Crea configurazione
        config = create_quick_test_config()  # Use quick config for example
        print_config(config)
        
        # Inizializza comparison system
        comparison = EvaluationComparison(config)
        
        # Carica modelli e dati
        print("\nLoading models and data...")
        comparison.load_models()
        comparison.load_data()
        
        # Esegui tutte le valutazioni
        print("\nRunning all evaluations...")
        comparison.run_all_evaluations()
        
        # Stampa riassunto
        comparison.print_comparison_summary()
        
        # Crea grafici
        comparison.create_plots()
        
        print(f"\n🎉 Comprehensive comparison completed!")
        print(f"📁 Results saved to: {config['output_dir']}")
        
        return comparison.results
        
    except Exception as e:
        print(f"❌ Error in comprehensive comparison: {e}")
        return None


def example_custom_evaluation():
    """Esempio di valutazione personalizzata."""
    print("\n🔧 Example: Custom Evaluation")
    print("-" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Carica modello
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        test_dataset = get_cifar10_test_dataset()
        
        # TTA con configurazione personalizzata
        print("Running custom TTA evaluation...")
        tta_results = evaluate_tta(
            classifier_model=classifier,
            test_dataset=test_dataset,
            device=device,
            num_samples=200,  # Custom sample size
            use_ttach=True,
            verbose=True
        )
        
        print(f"\n✅ Custom TTA Results:")
        print(f"  Accuracy improvement: {tta_results['accuracy_improvement']:+.4f}")
        print(f"  Method used: {tta_results['method_used']}")
        
        return tta_results
        
    except Exception as e:
        print(f"❌ Error in custom evaluation: {e}")
        return None


def main():
    """Funzione principale con menu interattivo."""
    print("🚀 RL Image Augmentation - New Structure Example")
    print("=" * 60)
    
    # Test imports first
    if not test_imports():
        print("❌ Import test failed. Please check your installation.")
        return
    
    while True:
        print("\n📋 Choose an example to run:")
        print("1. Individual method evaluation (baseline)")
        print("2. RL agent evaluation")
        print("3. Comprehensive comparison")
        print("4. Custom evaluation (TTA)")
        print("5. Exit")
        
        try:
            choice = input("\n❓ Enter your choice (1-5): ").strip()
            
            if choice == '1':
                example_individual_evaluation()
            elif choice == '2':
                example_rl_evaluation()
            elif choice == '3':
                example_comprehensive_comparison()
            elif choice == '4':
                example_custom_evaluation()
            elif choice == '5':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")


def quick_test():
    """Test rapido per verificare che tutto funzioni."""
    print("⚡ Quick Test - Checking system functionality")
    print("-" * 50)
    
    # Test 1: Import
    if not test_imports():
        return False
    
    # Test 2: Model loading
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        print("✅ Classifier loading works")
    except Exception as e:
        print(f"❌ Classifier loading failed: {e}")
        return False
    
    # Test 3: Data loading
    try:
        test_dataset = get_cifar10_test_dataset()
        print(f"✅ Data loading works ({len(test_dataset)} samples)")
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False
    
    # Test 4: Environment
    try:
        from src.environment import ImageAugmentationEnv
        num_actions = get_num_actions()
        print(f"✅ Environment works ({num_actions} actions available)")
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False
    
    print("\n🎉 All tests passed! The system is ready to use.")
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Examples for the restructured project")
    parser.add_argument('--quick-test', action='store_true', 
                       help='Run quick test to verify system works')
    parser.add_argument('--example', choices=['individual', 'rl', 'comprehensive', 'custom'],
                       help='Run specific example')
    
    args = parser.parse_args()
    
    if args.quick_test:
        quick_test()
    elif args.example:
        if args.example == 'individual':
            example_individual_evaluation()
        elif args.example == 'rl':
            example_rl_evaluation()
        elif args.example == 'comprehensive':
            example_comprehensive_comparison()
        elif args.example == 'custom':
            example_custom_evaluation()
    else:
        main()