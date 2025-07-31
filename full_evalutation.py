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

AGGIORNAMENTO: Grafici migliorati con tutti i 4 metodi
- Accuracy Comparison (tutti i 4 metodi)
- Accuracy Improvement Comparison (escluso baseline)
- Inference Time Comparison (tutti i 4 metodi)
- Success Rate Comparison (escluso baseline)
- Efficiency Plot (Improvement vs Time)
- Summary & Recommendations
"""

import os
import sys
import torch
import time
import json
import matplotlib.pyplot as plt
import numpy as np
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


class ComprehensiveEvaluator:
    """Enhanced evaluator with improved plotting capabilities."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.plots_dir = os.path.join(config['output_dir'], 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def load_evaluation_system(self):
        """Load the evaluation system with error handling."""
        print_section("Loading Evaluation System")
        
        try:
            # Import evaluation modules
            from evaluation.comparison import EvaluationComparison
            print("✅ Evaluation system imported successfully")
            
            # Create comparison object
            self.comparison = EvaluationComparison(self.config)
            print("✅ Evaluation comparison object created")
            
            return True
        
        except ImportError as e:
            print(f"❌ Failed to import evaluation system: {e}")
            print("💡 Make sure the evaluation/ directory structure is correct")
            return False
        except Exception as e:
            print(f"❌ Error initializing evaluation system: {e}")
            return False
    
    def run_evaluations(self):
        """Run all evaluations."""
        try:
            # Load models and data
            print_section("Loading Models and Data")
            
            print("📥 Loading models...")
            self.comparison.load_models()
            
            print("📊 Loading data...")
            self.comparison.load_data()
            
            # Run all evaluations
            print_section("Running Evaluations")
            print("🔄 This may take several minutes...")
            print(f"📊 Evaluating {self.config['tta_samples']} samples for TTA")
            print(f"🤖 Evaluating {self.config['rl_episodes']} episodes for RL")
            
            self.comparison.run_all_evaluations()
            
            # Store results
            self.results = self.comparison.results
            
            return True
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return False
    
    def create_comprehensive_plots(self):
        """Create comprehensive comparison plots with all 4 methods."""
        if not self.results:
            print("❌ No results available for plotting. Run evaluations first!")
            return
        
        print_section("Creating Comprehensive Plots")
        
        # Extract data for all methods
        methods_data = self._extract_plotting_data()
        
        if not methods_data:
            print("❌ No data available for plotting")
            return
        
        # Create figure with 6 subplots (2x3)
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison (all 4 methods)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_accuracy_comparison(ax1, methods_data)
        
        # 2. Accuracy Improvement Comparison (excluding baseline)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_accuracy_improvement(ax2, methods_data)
        
        # 3. Inference Time Comparison (all 4 methods)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_inference_time(ax3, methods_data)
        
        # 4. Success Rate Comparison (excluding baseline)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_success_rate(ax4, methods_data)
        
        # 5. Efficiency Plot (Improvement vs Time)
        ax5 = plt.subplot(2, 3, 5)
        self._plot_efficiency(ax5, methods_data)
        
        # 6. Summary Text
        ax6 = plt.subplot(2, 3, 6)
        self._plot_summary_text(ax6, methods_data)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.plots_dir, 'comprehensive_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive plot saved to: {plot_path}")
    
    def _extract_plotting_data(self):
        """Extract necessary data for plotting from all methods."""
        data = {}
        
        # Baseline
        if 'baseline' in self.results:
            data['Baseline'] = {
                'accuracy': self.results['baseline']['accuracy'],
                'improvement': 0.0,
                'time_ms': self.results['baseline'].get('time_per_sample', 0) * 1000,
                'success_rate': 0.0,  # Baseline has no success rate
                'color': 'lightblue'
            }
        
        # Fixed Augmentation
        if 'fixed_aug' in self.results:
            data['Fixed Aug'] = {
                'accuracy': self.results['fixed_aug']['augmented_accuracy'],
                'improvement': self.results['fixed_aug']['accuracy_improvement'],
                'time_ms': self.results['fixed_aug'].get('time_per_sample', 0) * 1000,
                'success_rate': self.results['fixed_aug'].get('improvement_rate', 0),
                'color': 'lightgreen'
            }
        
        # TTA
        if 'tta' in self.results:
            data['TTA'] = {
                'accuracy': self.results['tta']['tta_accuracy'],
                'improvement': self.results['tta']['accuracy_improvement'],
                'time_ms': self.results['tta'].get('time_per_sample', 0) * 1000,
                'success_rate': self.results['tta'].get('improvement_rate', 0),
                'color': 'lightcoral'
            }
        
        # RL Agent
        if 'rl' in self.results:
            data['RL Agent'] = {
                'accuracy': self.results['rl']['final_accuracy'],
                'improvement': self.results['rl']['accuracy_improvement'],
                'time_ms': self.results['rl'].get('time_per_sample', 0) * 1000,
                'success_rate': self.results['rl'].get('improvement_rate', 0),
                'color': 'gold'
            }
        
        return data
    
    def _plot_accuracy_comparison(self, ax, data):
        """Plot 1: Accuracy comparison of all methods."""
        methods = list(data.keys())
        accuracies = [data[m]['accuracy'] for m in methods]
        colors = [data[m]['color'] for m in methods]
        
        bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', alpha=0.8)
        
        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison: All Methods', fontweight='bold', fontsize=12)
        ax.set_ylim(0, max(accuracies) * 1.1 if accuracies else 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate labels if needed
        if len(methods) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_accuracy_improvement(self, ax, data):
        """Plot 2: Accuracy improvement comparison (excluding baseline)."""
        # Exclude baseline from improvement comparison
        methods = [m for m in data.keys() if m != 'Baseline']
        improvements = [data[m]['improvement'] for m in methods]
        colors = [data[m]['color'] for m in methods]
        
        bars = ax.bar(methods, improvements, color=colors, edgecolor='black', alpha=0.8)
        
        # Add values on bars
        for bar, imp in zip(bars, improvements):
            y_pos = bar.get_height() + 0.001 if imp >= 0 else bar.get_height() - 0.001
            va = 'bottom' if imp >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                    f'{imp:+.4f}', ha='center', va=va, fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Accuracy Improvement')
        ax.set_title('Accuracy Improvement Comparison', fontweight='bold', fontsize=12)
        ax.axhline(0, color='red', linestyle='--', alpha=0.7)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate labels if needed
        if len(methods) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_inference_time(self, ax, data):
        """Plot 3: Inference time comparison of all methods."""
        methods = list(data.keys())
        times = [data[m]['time_ms'] for m in methods]
        colors = [data[m]['color'] for m in methods]
        
        bars = ax.bar(methods, times, color=colors, edgecolor='black', alpha=0.8)
        
        # Add values on bars
        for bar, time_val in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                    f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Time per Sample (ms)')
        ax.set_title('Inference Time Comparison', fontweight='bold', fontsize=12)
        ax.set_ylim(0, max(times) * 1.2 if times else 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate labels if needed
        if len(methods) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_success_rate(self, ax, data):
        """Plot 4: Success rate comparison (excluding baseline)."""
        # Exclude baseline from success rate
        methods = [m for m in data.keys() if m != 'Baseline']
        success_rates = [data[m]['success_rate'] * 100 for m in methods]  # Convert to percentage
        colors = [data[m]['color'] for m in methods]
        
        bars = ax.bar(methods, success_rates, color=colors, edgecolor='black', alpha=0.8)
        
        # Add values on bars
        for bar, rate in zip(bars, success_rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Success Rate Comparison', fontweight='bold', fontsize=12)
        ax.set_ylim(0, max(success_rates) * 1.2 if success_rates else 100)
        ax.grid(axis='y', alpha=0.3)
        
        # Rotate labels if needed
        if len(methods) > 3:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _plot_efficiency(self, ax, data):
        """Plot 5: Efficiency plot (Improvement vs Time)."""
        # Exclude baseline
        methods = [m for m in data.keys() if m != 'Baseline']
        
        if not methods:
            ax.text(0.5, 0.5, 'No data for efficiency plot', ha='center', va='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Efficiency Plot: Improvement vs Time', fontweight='bold', fontsize=12)
            return
        
        times = [data[m]['time_ms'] for m in methods]
        improvements = [data[m]['improvement'] for m in methods]
        colors = [data[m]['color'] for m in methods]
        
        # Calculate baseline time for slowdown
        baseline_time = data.get('Baseline', {}).get('time_ms', min(times) if times else 1)
        if baseline_time <= 0:
            baseline_time = 1
        
        slowdowns = [t / baseline_time for t in times]
        
        scatter = ax.scatter(slowdowns, improvements, s=200, c=colors, alpha=0.8, edgecolors='black')
        
        # Add labels
        for i, method in enumerate(methods):
            ax.annotate(method, (slowdowns[i], improvements[i]), 
                       xytext=(5, 5), textcoords='offset points', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Slowdown Factor (×)')
        ax.set_ylabel('Accuracy Improvement')
        ax.set_title('Efficiency Plot: Improvement vs Computational Cost', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.axvline(1, color='red', linestyle='--', alpha=0.5, label='Baseline speed')
    
    def _plot_summary_text(self, ax, data):
        """Plot 6: Summary text with recommendations."""
        # Find best method
        improvements = {m: data[m]['improvement'] for m in data.keys() if m != 'Baseline'}
        
        if improvements:
            best_method = max(improvements.items(), key=lambda x: x[1])
            best_name, best_improvement = best_method
        else:
            best_name, best_improvement = "None", 0
        
        # Calculate statistics
        total_methods = len(data)
        methods_with_improvement = len([m for m in improvements.values() if m > 0])
        
        # Create summary text
        summary_text = f"""🎯 COMPREHENSIVE EVALUATION SUMMARY

📊 Methods Evaluated: {total_methods}
🏆 Best Method: {best_name}
   Improvement: {best_improvement:+.4f}
   Final Accuracy: {data.get(best_name, {}).get('accuracy', 0):.4f}

📈 Performance Overview:
"""
        
        # Add performance of each method
        sorted_methods = sorted(data.items(), 
                               key=lambda x: x[1]['improvement'], reverse=True)
        
        for i, (method, info) in enumerate(sorted_methods, 1):
            if method == 'Baseline':
                summary_text += f"   {i}. {method}: {info['accuracy']:.4f} (baseline)\n"
            else:
                summary_text += f"   {i}. {method}: {info['accuracy']:.4f} ({info['improvement']:+.4f})\n"
        
        # Add recommendations
        summary_text += f"\n💡 Recommendations:\n"
        
        if best_improvement > 0.015:
            summary_text += f"   ✅ {best_name}: Excellent improvement!\n"
            summary_text += f"      Highly recommended for production use.\n"
        elif best_improvement > 0.005:
            summary_text += f"   ⚠️  {best_name}: Good improvement.\n"
            summary_text += f"      Consider computational cost vs benefit.\n"
        elif best_improvement > 0:
            summary_text += f"   📊 {best_name}: Minimal improvement.\n"
            summary_text += f"      Limited practical benefit.\n"
        else:
            summary_text += f"   ❌ No significant improvements found.\n"
            summary_text += f"      Consider different approaches.\n"
        
        # Add efficiency info
        if 'RL Agent' in data and data['RL Agent']['improvement'] > 0:
            rl_time = data['RL Agent']['time_ms']
            baseline_time = data.get('Baseline', {}).get('time_ms', 1)
            slowdown = rl_time / baseline_time if baseline_time > 0 else 1
            summary_text += f"\n⚡ RL Agent Efficiency:\n"
            summary_text += f"   Speed: {slowdown:.1f}× slower than baseline\n"
            
            rl_loaded = self.results.get('rl', {}).get('model_loaded', False)
            if not rl_loaded:
                summary_text += f"   ⚠️  Using random weights - train for better results!\n"
        
        # Display the text
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Summary & Recommendations', fontweight='bold', fontsize=12)
    
    def print_results_summary(self):
        """Print detailed results summary."""
        if not self.results:
            return
        
        print_section("Results Summary")
        self.comparison.print_comparison_summary()
    
    def save_results(self):
        """Save results to files."""
        if not self.results or not CONFIG['save_results']:
            return
        
        # Use the existing save functionality
        self.comparison.save_results()
        
        print(f"💾 Results saved to: {CONFIG['output_dir']}/")


def run_comprehensive_evaluation():
    """Run the complete evaluation process."""
    start_time = time.time()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed. Please fix the issues above.")
        return None
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(CONFIG)
    
    # Load evaluation system
    if not evaluator.load_evaluation_system():
        print("\n❌ Failed to load evaluation system.")
        return None
    
    try:
        # Run evaluations
        if not evaluator.run_evaluations():
            print("\n❌ Evaluation failed.")
            return None
        
        # Print results
        evaluator.print_results_summary()
        
        # Create visualizations
        if CONFIG['create_plots']:
            evaluator.create_comprehensive_plots()
        
        # Save results
        if CONFIG['save_results']:
            evaluator.save_results()
        
        # Calculate total time
        total_time = time.time() - start_time
        
        print_section("Evaluation Complete")
        print(f"🎉 All evaluations completed successfully!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📁 Results saved to: {CONFIG['output_dir']}/")
        print(f"📊 Plots saved to: {CONFIG['output_dir']}/plots/")
        
        return evaluator.results
        
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
        
        print_section("Evaluation Completed Successfully")
        print("🎉 All done! Check the results above and in the output directory.")
        print(f"📁 Detailed results: {CONFIG['output_dir']}/")
        print(f"📊 Comprehensive plots: {CONFIG['output_dir']}/plots/comprehensive_comparison.png")
        
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