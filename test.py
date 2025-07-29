import torch
from evaluation.core.data_utils import get_cifar10_test_dataset, get_cifar10_test_loader
from evaluation.core.model_loader import load_classifier
from evaluation.methods import evaluate_baseline, evaluate_tta

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
if __name__ == '__main__':
     quick_start_example()