#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL EVALUATION
===============================

Script definitivo per confrontare tutti i modelli di augmentation:
- Baseline (nessuna augmentation)
- Fixed Augmentation (sequenza fissa)
- Test-Time Augmentation (TTA)
- RL Agent (augmentation adattiva)

Uso: python comprehensive_evaluation.py

Il script è completamente autonomo e fornisce:
- Risultati dettagliati per ogni metodo
- Confronto comparativo automatico
- Grafici di visualizzazione
- Raccomandazioni finali
- Salvataggio risultati
"""

import os
import sys
import torch
import time
import json
from datetime import datetime
from pathlib import Path

# Assicurati che il progetto sia nel path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("🚀 COMPREHENSIVE MODEL EVALUATION")
print("=" * 60)
print(f"📁 Project root: {project_root}")
print(f"💻 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# Configuration
CONFIG = {
    # Model paths
    'classifier_path': './checkpoint/ckpt.pth',
    'rl_model_path': './models/best_improved_dqn_model.pth',
    'data_root': './data',
    
    # Evaluation parameters
    'batch_size': 64,
    'tta_samples': 1000,
    'rl_episodes': 1000,
    'max_steps_per_episode': 3,
    'state_dim': 15,
    
    # Fixed augmentation sequence
    'fixed_aug_ids': [0, 3, 6],  # Brightness +10%, Contrast -10%, HFlip
    
    # Output
    'output_dir': './comprehensive_results',
    'save_results': True,
    'create_plots': True,
    
    # Methods to evaluate
    'evaluate_baseline': True,
    'evaluate_fixed_aug': True,
    'evaluate_tta': True,
    'evaluate_rl': True,
}

def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title.upper()}")
    print(f"{'='*60}")

def print_subsection(title):
    """Print a formatted subsection header."""
    print(f"\n{'-'*40}")
    print(f"🎯 {title}")
    print(f"{'-'*40}")

def check_requirements():
    """Check that all required files and directories exist."""
    print_section("Checking Requirements")
    
    required_files = [
        CONFIG['classifier_path'],
    ]
    
    optional_files = [
        CONFIG['rl_model_path'],
    ]
    
    required_dirs = [
        CONFIG['data_root'],
    ]
    
    all_good = True
    
    # Check required files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"❌ Missing: {file_path}")
            all_good = False
    
    # Check optional files
    for file_path in optional_files:
        if os.path.exists(file_path):
            print(f"✅ Found: {file_path}")
        else:
            print(f"⚠️  Optional file missing: {file_path}")
            if 'rl_model' in file_path:
                print("   RL evaluation will use random agent for comparison")
                
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Found directory: {dir_path}")
        else:
            print(f"📁 Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"✅ Output directory ready: {CONFIG['output_dir']}")
    
    if not all_good:
        print("\n❌ Some required files are missing!")
        print("💡 Make sure you have:")
        print("   - Trained VGG19 classifier in ./checkpoint/ckpt.pth")
        print("   - CIFAR-10 data will be downloaded automatically")
        print("   - RL model is optional (will use random agent if missing)")
        return False
    
    print("\n✅ All requirements satisfied!")
    return True

def load_evaluation_system():
    """Load the evaluation system with error handling."""
    print_section("Loading Evaluation System")
    
    try:
        # Import evaluation modules
        from evaluation.comparison import EvaluationComparison
        print("✅ Evaluation system imported successfully")
        
        # Create comparison object
        comparison = EvaluationComparison(CONFIG)
        print("✅ Evaluation comparison object created")
        
        return comparison
    
    except ImportError as e:
        print(f"❌ Failed to import evaluation system: {e}")
        print("💡 Make sure the evaluation/ directory structure is correct")
        return None
    except Exception as e:
        print(f"❌ Error initializing evaluation system: {e}")
        return None

def run_comprehensive_evaluation():
    """Run the complete evaluation process."""
    start_time = time.time()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        return None
    
    # Load evaluation system
    comparison = load_evaluation_system()
    if comparison is None:
        print("\n❌ Failed to load evaluation system.")
        return None
    
    try:
        # Load models and data
        print_section("Loading Models and Data")
        
        print("📥 Loading models...")
        comparison.load_models()
        
        print("📊 Loading data...")
        comparison.load_data()
        
        # Run all evaluations
        print_section("Running Evaluations")
        print("🔄 This may take several minutes...")
        print(f"📊 Evaluating {CONFIG['tta_samples']} samples for TTA")
        print(f"🤖 Evaluating {CONFIG['rl_episodes']} episodes for RL")
        
        comparison.run_all_evaluations()
        
        # Print results
        print_section("Results Summary")
        comparison.print_comparison_summary()
        
        # Create visualizations
        if CONFIG['create_plots']:
            print_section("Creating Visualizations")
            comparison.create_plots()
            print(f"📊 Plots saved to: {CONFIG['output_dir']}/plots/")
        
        # Calculate total time
        total_time = time.time() - start_time
        
        print_section("Evaluation Complete")
        print(f"🎉 All evaluations completed successfully!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📁 Results saved to: {CONFIG['output_dir']}/")
        
        return comparison.results
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation interrupted by user")
        print("🔄 You can restart the evaluation anytime")
        return None
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Print helpful debugging info
        print("\n🔍 Debugging information:")
        if "not found" in str(e).lower():
            print("💡 File not found - check your model paths")
        elif "cuda" in str(e).lower():
            print("💡 CUDA error - try running with CPU: export CUDA_VISIBLE_DEVICES=''")
        elif "memory" in str(e).lower():
            print("💡 Memory error - try reducing batch_size in CONFIG")
        else:
            print("💡 General error - check the full traceback above")
        
        import traceback
        traceback.print_exc()
        return None

def print_final_summary(results):
    """Print a final summary of all results."""
    if not results:
        return
    
    print_section("Final Summary and Recommendations")
    
    # Extract key metrics
    methods = []
    accuracies = []
    improvements = []
    times = []
    
    if 'baseline' in results:
        baseline_acc = results['baseline']['accuracy']
        methods.append('Baseline')
        accuracies.append(baseline_acc)
        improvements.append(0.0)
        times.append(results['baseline'].get('time_per_sample', 0))
    else:
        baseline_acc = 0
    
    if 'fixed_aug' in results:
        methods.append('Fixed Aug')
        accuracies.append(results['fixed_aug']['augmented_accuracy'])
        improvements.append(results['fixed_aug']['accuracy_improvement'])
        times.append(results['fixed_aug'].get('time_per_sample', 0))
    
    if 'tta' in results:
        methods.append('TTA')
        accuracies.append(results['tta']['tta_accuracy'])
        improvements.append(results['tta']['accuracy_improvement'])
        times.append(results['tta'].get('time_per_sample', 0))
    
    if 'rl' in results:
        methods.append('RL Agent')
        accuracies.append(results['rl']['final_accuracy'])
        improvements.append(results['rl']['accuracy_improvement'])
        times.append(results['rl'].get('time_per_sample', 0))
    
    # Find best method
    if improvements:
        best_idx = max(range(len(improvements)), key=lambda i: improvements[i])
        best_method = methods[best_idx]
        best_improvement = improvements[best_idx]
    else:
        best_method = "None"
        best_improvement = 0
    
    print(f"🏆 BEST PERFORMING METHOD: {best_method}")
    print(f"   Accuracy improvement: {best_improvement:+.4f}")
    print(f"   Final accuracy: {accuracies[best_idx]:.4f}" if improvements else "")
    
    print(f"\n📊 PERFORMANCE RANKING:")
    # Sort by improvement
    ranked = sorted(zip(methods, accuracies, improvements, times), 
                   key=lambda x: x[2], reverse=True)
    
    for i, (method, acc, imp, time_ms) in enumerate(ranked, 1):
        time_str = f"{time_ms*1000:.1f}ms" if time_ms > 0 else "N/A"
        print(f"   {i}. {method:12} | Acc: {acc:.4f} | Imp: {imp:+.4f} | Time: {time_str}")
    
    print(f"\n💡 RECOMMENDATIONS:")
    
    # Decision logic
    if best_improvement > 0.015:  # >1.5% improvement
        print(f"   ✅ {best_method} shows significant improvement - Highly recommended!")
    elif best_improvement > 0.005:  # >0.5% improvement
        print(f"   ⚠️  {best_method} shows moderate improvement - Consider the computational cost")
    elif best_improvement > 0:
        print(f"   📊 {best_method} shows minimal improvement - Limited practical benefit")
    else:
        print(f"   ❌ No method shows meaningful improvement - Consider:")
        print(f"      • Retraining the RL agent with different parameters")
        print(f"      • Using a different set of augmentations")
        print(f"      • The baseline classifier might already be well-optimized")
    
    # Specific recommendations
    print(f"\n🎯 SPECIFIC RECOMMENDATIONS:")
    
    if 'rl' in [r[0] for r in ranked[:2]]:  # RL in top 2
        rl_loaded = results.get('rl', {}).get('model_loaded', False)
        if rl_loaded:
            print("   • RL Agent: Good performance, continue using this approach")
        else:
            print("   • RL Agent: Good performance even with random weights - train a real agent!")
    
    if 'TTA' in [r[0] for r in ranked[:2]]:  # TTA in top 2
        tta_time = next((t for m, a, i, t in ranked if m == 'TTA'), 0)
        if tta_time > 0.01:  # >10ms
            print("   • TTA: Good accuracy but slow - use only when inference time is not critical")
        else:
            print("   • TTA: Good balance of accuracy and speed")
    
    if 'Fixed Aug' in [r[0] for r in ranked[:2]]:
        print("   • Fixed Augmentation: Simple and effective - good baseline to beat")
    
    print(f"\n⚡ EFFICIENCY ANALYSIS:")
    if times and max(times) > 0:
        baseline_time = times[0] if methods[0] == 'Baseline' else min(times)
        for method, acc, imp, time_ms in ranked:
            if time_ms > 0 and baseline_time > 0:
                slowdown = time_ms / baseline_time
                efficiency = imp / (slowdown - 1) if slowdown > 1 else float('inf')
                if efficiency > 0.01:
                    rating = "🟢 Highly Efficient"
                elif efficiency > 0.005:
                    rating = "🟡 Moderately Efficient"
                elif efficiency > 0:
                    rating = "🟠 Low Efficiency"
                else:
                    rating = "🔴 Inefficient"
                print(f"   {method:12}: {slowdown:.1f}x slower | {rating}")

def save_results_summary(results):
    """Save a summary of results to JSON file."""
    if not results or not CONFIG['save_results']:
        return
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'results_summary': {},
        'evaluation_info': {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        }
    }
    
    # Extract key metrics for each method
    for method, result in results.items():
        method_summary = {}
        
        if method == 'baseline':
            method_summary = {
                'accuracy': result.get('accuracy', 0),
                'avg_confidence': result.get('avg_confidence', 0),
                'time_per_sample': result.get('time_per_sample', 0)
            }
        elif method == 'fixed_aug':
            method_summary = {
                'baseline_accuracy': result.get('baseline_accuracy', 0),
                'augmented_accuracy': result.get('augmented_accuracy', 0),
                'accuracy_improvement': result.get('accuracy_improvement', 0),
                'time_per_sample': result.get('time_per_sample', 0)
            }
        elif method == 'tta':
            method_summary = {
                'baseline_accuracy': result.get('baseline_accuracy', 0),
                'tta_accuracy': result.get('tta_accuracy', 0),
                'accuracy_improvement': result.get('accuracy_improvement', 0),
                'num_augmentations': result.get('num_augmentations', 0),
                'time_per_sample': result.get('time_per_sample', 0)
            }
        elif method == 'rl':
            method_summary = {
                'initial_accuracy': result.get('initial_accuracy', 0),
                'final_accuracy': result.get('final_accuracy', 0),
                'accuracy_improvement': result.get('accuracy_improvement', 0),
                'avg_reward': result.get('avg_reward', 0),
                'improvement_rate': result.get('improvement_rate', 0),
                'model_loaded': result.get('model_loaded', False),
                'time_per_sample': result.get('time_per_sample', 0)
            }
        
        summary['results_summary'][method] = method_summary
    
    # Save to file
    summary_file = os.path.join(CONFIG['output_dir'], 'comprehensive_summary.json')
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Summary saved to: {summary_file}")
    except Exception as e:
        print(f"⚠️  Could not save summary: {e}")

def main():
    """Main execution function."""
    print("Starting comprehensive model evaluation...\n")
    
    # Print configuration
    print_subsection("Configuration")
    print(f"📊 TTA Samples: {CONFIG['tta_samples']}")
    print(f"🤖 RL Episodes: {CONFIG['rl_episodes']}")  
    print(f"📦 Batch Size: {CONFIG['batch_size']}")
    print(f"🔧 Fixed Aug IDs: {CONFIG['fixed_aug_ids']}")
    print(f"📁 Output Dir: {CONFIG['output_dir']}")
    
    # Ask for confirmation if evaluation will take long
    total_samples = CONFIG['tta_samples'] + CONFIG['rl_episodes']
    if total_samples > 2000:
        print(f"\n⚠️  This evaluation will process ~{total_samples} samples")
        print("   Estimated time: 10-30 minutes depending on your hardware")
        
        try:
            response = input("\n❓ Continue? (y/N): ").lower().strip()
            if response not in ['y', 'yes']:
                print("👋 Evaluation cancelled")
                return
        except KeyboardInterrupt:
            print("\n👋 Evaluation cancelled")
            return
    
    # Run evaluation
    results = run_comprehensive_evaluation()
    
    if results:
        print_final_summary(results)
        save_results_summary(results)
        
        print_section("Evaluation Completed Successfully")
        print("🎉 All done! Check the results above and in the output directory.")
        print(f"📁 Detailed results: {CONFIG['output_dir']}/")
        
        if CONFIG['create_plots']:
            print(f"📊 Visualizations: {CONFIG['output_dir']}/plots/")
        
    else:
        print_section("Evaluation Failed")
        print("❌ The evaluation did not complete successfully.")
        print("💡 Check the error messages above for guidance.")
        print("🔄 You can run this script again after fixing any issues.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Evaluation interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("💡 This might be a bug - please check the full traceback:")
        import traceback
        traceback.print_exc()