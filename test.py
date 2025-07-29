"""
Script di test corretto per il sistema di evaluation.
"""

from evaluation.comparison import EvaluationComparison
from evaluation.comparison.config import create_default_config, create_quick_test_config

def main():
    print("🚀 Testing RL Image Augmentation Evaluation System")
    print("=" * 60)
    
    # Crea configurazione (usa quick test per essere più veloce)
    config = create_quick_test_config()
    
    # Mostra configurazione
    print("📋 Configuration:")
    print(f"  Classifier: {config['classifier_path']}")
    print(f"  RL Model: {config['rl_model_path']}")
    print(f"  Data Root: {config['data_root']}")
    print(f"  TTA Samples: {config['tta_samples']}")
    print(f"  RL Episodes: {config['rl_episodes']}")
    print(f"  Output Dir: {config['output_dir']}")
    
    try:
        # Inizializza il sistema di confronto
        print("\n🎯 Initializing evaluation system...")
        comparison = EvaluationComparison(config)
        
        # IMPORTANTE: Carica modelli e dati PRIMA di run_all_evaluations
        print("\n📥 Loading models...")
        comparison.load_models()
        
        print("\n📊 Loading data...")
        comparison.load_data()
        
        # Ora esegui le valutazioni
        print("\n🔄 Running all evaluations...")
        comparison.run_all_evaluations()
        
        # Stampa riassunto comparativo
        print("\n📈 Printing comparison summary...")
        comparison.print_comparison_summary()
        
        # Crea grafici
        print("\n📊 Creating plots...")
        comparison.create_plots()
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📁 Results saved to: {config['output_dir']}")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print("💡 Make sure these files exist:")
        print(f"   - {config['classifier_path']}")
        print(f"   - {config['rl_model_path']} (optional)")
        print(f"   - {config['data_root']} directory")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()


def test_individual_methods():
    """Test dei singoli metodi di valutazione."""
    print("\n🧪 Testing individual evaluation methods...")
    
    import torch
    from evaluation.core import load_classifier, get_cifar10_test_loader, get_cifar10_test_dataset
    from evaluation.methods import evaluate_baseline
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Test baseline evaluation
        print("📊 Testing baseline evaluation...")
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        test_loader = get_cifar10_test_loader(batch_size=32)
        
        baseline_results = evaluate_baseline(
            classifier_model=classifier,
            test_loader=test_loader,
            device=device,
            verbose=True
        )
        
        print(f"✅ Baseline evaluation successful!")
        print(f"   Accuracy: {baseline_results['accuracy']:.4f}")
        print(f"   Confidence: {baseline_results['avg_confidence']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Individual test failed: {e}")
        return False


def minimal_test():
    """Test minimale per verificare che il sistema base funzioni."""
    print("\n⚡ Running minimal test...")
    
    try:
        # Test 1: Import
        from evaluation.core import load_classifier
        print("✅ Core imports working")
        
        # Test 2: Config
        config = create_quick_test_config()
        print("✅ Configuration working")
        
        # Test 3: Comparison object
        comparison = EvaluationComparison(config)
        print("✅ EvaluationComparison object created")
        
        print("🎉 Minimal test passed! System is ready.")
        return True
        
    except Exception as e:
        print(f"❌ Minimal test failed: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the evaluation system")
    parser.add_argument('--minimal', action='store_true',
                       help='Run minimal test only')
    parser.add_argument('--individual', action='store_true', 
                       help='Test individual methods only')
    parser.add_argument('--full', action='store_true',
                       help='Run full comprehensive test')
    
    args = parser.parse_args()
    
    if args.minimal:
        minimal_test()
    elif args.individual:
        test_individual_methods()
    elif args.full:
        main()
    else:
        # Default: run minimal first, then ask
        print("🧪 Running minimal test first...")
        if minimal_test():
            response = input("\n❓ Minimal test passed. Run full evaluation? (y/N): ").lower().strip()
            if response in ['y', 'yes']:
                main()
            else:
                print("👋 Test completed. Run with --full for comprehensive evaluation.")
        else:
            print("❌ Minimal test failed. Please check your setup.")