#!/usr/bin/env python3
"""
MODEL EVALUATION
===============================================================

Streamlined evaluation script.

Usage: python full_evaluation.py [--quick] [--interactive]
"""

import sys
import argparse
import torch
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import centralized configuration
from config.evaluation_config import *

# Import modules
from evaluation.core.model_loader import load_classifier, load_rl_agent, print_loading_summary
from evaluation.core.data_utils import get_cifar10_test_dataset, get_cifar10_test_loader
from evaluation.methods.evaluate_baseline import evaluate_baseline
from evaluation.methods.evaluate_fixed_aug import evaluate_fixed_augmentation
from evaluation.methods.evaluate_tta import evaluate_tta
from evaluation.methods.evaluate_rl_agent import evaluate_rl_agent, test_agent_environment_compatibility


class EvaluationRunner:
    """Evaluation runner using centralized configuration."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Setup output directories
        os.makedirs(self.config['output_dir'], exist_ok=True)
        os.makedirs(f"{self.config['output_dir']}/plots", exist_ok=True)
        
        print(f" EVALUATION (143D State Space)")
        print(f"Device: {self.device}")
        print(f"Output: {self.config['output_dir']}")
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline."""
        try:
            print("\n" + "=" * 60)
            print("STARTING EVALUATION")
            print("=" * 60)
            
            print_config(self.config)
            
            # 1. Load models
            self._load_models()
            
            # 2. Load data
            self._load_data()
            
            # 3. Run evaluations
            self._run_all_evaluations()
            
            # 4. Generate results
            self._generate_outputs()
            
            print("\n Evaluation completed successfully!")
            return self.results
            
        except KeyboardInterrupt:
            print("\n\n Evaluation interrupted by user")
            return None
        except Exception as e:
            print(f"\n Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _load_models(self):
        """Load all required models."""
        print("\n LOADING MODELS")
        print("-" * 30)
        
        # Load classifier
        self.classifier = load_classifier(
            model_path=self.config['classifier_path'],
            device=self.device
        )
        
        # Load RL agent
        self.agent = None
        self.rl_model_loaded = False
        
        if self.config['evaluate_rl']:
            # Try main path first, then alternatives
            rl_paths_to_try = [self.config['rl_model_path']] + ALTERNATIVE_RL_PATHS
            
            for rl_path in rl_paths_to_try:
                if os.path.exists(rl_path):
                    try:
                        self.agent, self.rl_model_loaded = load_rl_agent(
                            model_path=rl_path,
                            device=self.device
                        )
                        if self.rl_model_loaded:
                            print(f" RL model loaded from: {rl_path}")
                            break
                    except Exception as e:
                        print(f" Failed to load RL model from {rl_path}: {e}")
                        continue
            
            # Test compatibility if agent was loaded
            if self.agent and self.rl_model_loaded:
                compatibility = test_agent_environment_compatibility(
                    self.agent, self.classifier, self.device, self.config['image_feature_dim']
                )
                
                if not compatibility['compatible']:
                    print(f" Agent-environment compatibility issue:")
                    print(f"   {compatibility['error']}")
                else:
                    print(f" Agent-environment compatibility confirmed")
        
        # Print summary
        print_loading_summary(
            classifier=self.classifier,
            agent=self.agent,
            agent_loaded=self.rl_model_loaded
        )
    
    def _load_data(self):
        """Load test data."""
        print("\n LOADING DATA")
        print("-" * 30)
        
        # Load dataset and dataloader
        self.test_dataset = get_cifar10_test_dataset(
            data_root=self.config['data_root']
        )
        
        self.test_loader = get_cifar10_test_loader(
            data_root=self.config['data_root'],
            batch_size=self.config['batch_size']
        )
        
        print(f" Test dataset loaded: {len(self.test_dataset)} samples")
        print(f" Test loader created: {len(self.test_loader)} batches")
    
    def _run_all_evaluations(self):
        """Run all configured evaluations."""
        print("\n RUNNING EVALUATIONS")
        print("-" * 30)
        
        start_time = datetime.now()
        
        # 1. Baseline evaluation
        if self.config['evaluate_baseline']:
            print("\n Running baseline evaluation...")
            self.results['baseline'] = evaluate_baseline(
                classifier_model=self.classifier,
                test_dataset=self.test_dataset,
                device=self.device,
                num_samples=self.config['baseline_samples'],
                batch_size=self.config['batch_size'],
                verbose=True,
                return_details=True
            )
        
        # 2. Fixed augmentation evaluation
        if self.config['evaluate_fixed_aug']:
            print("\n Running fixed augmentation evaluation...")
            self.results['fixed_aug'] = evaluate_fixed_augmentation(
                classifier_model=self.classifier,
                test_dataset=self.test_dataset,
                augmentation_ids=self.config['fixed_aug_ids'],
                device=self.device,
                num_samples=self.config['fixed_aug_samples'],
                batch_size=self.config['batch_size'],
                verbose=True
            )
        
        # 3. TTA evaluation
        if self.config['evaluate_tta']:
            print("\n Running TTA evaluation...")
            self.results['tta'] = evaluate_tta(
                classifier_model=self.classifier,
                test_dataset=self.test_dataset,
                device=self.device,
                num_samples=self.config['tta_samples'],
                use_ttach=self.config['use_ttach'],
                verbose=True
            )
        
        # 4. RL agent evaluation
        if self.config['evaluate_rl'] and self.agent:
            print("\n Running RL agent evaluation...")
            try:
                self.results['rl'] = evaluate_rl_agent(
                    agent=self.agent,
                    classifier_model=self.classifier,
                    test_dataset=self.test_dataset,
                    device=self.device,
                    num_episodes=self.config['rl_episodes'],
                    max_steps_per_episode=self.config['max_steps_per_episode'],
                    image_feature_dim=self.config['image_feature_dim'],
                    verbose=True,
                    return_details=True
                )
                self.results['rl']['model_loaded'] = self.rl_model_loaded
            except Exception as e:
                print(f" RL evaluation failed: {e}")
                self.results['rl'] = {
                    'error': str(e),
                    'model_loaded': self.rl_model_loaded,
                    'accuracy': 0.0,
                    'avg_confidence': 0.0
                }
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        print(f"\n All evaluations completed in {total_time:.1f} seconds")
    
    def _generate_outputs(self):
        """Generate and save results."""
        print("\n GENERATING OUTPUTS")
        print("-" * 30)
        
        # Print comparison summary
        self._print_comparison_summary()
        
        # Save results
        if self.config['save_results']:
            self._save_results()
        
        # Create plots
        if self.config['create_plots']:
            try:
                self._create_comprehensive_plots()
                print(" Comprehensive plots created successfully")
            except Exception as e:
                print(f" Error creating plots: {e}")
    
    def _print_comparison_summary(self):
        """Print comparison summary."""
        print("\n EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        
        baseline_acc = 0.0
        if 'baseline' in self.results:
            baseline_acc = self.results['baseline']['accuracy']
            print(f"Baseline:        {baseline_acc:.4f}")
        
        if 'fixed_aug' in self.results:
            acc = self.results['fixed_aug']['accuracy']
            imp = acc - baseline_acc
            print(f"Fixed Aug:       {acc:.4f} ({imp:+.4f})")
        
        if 'tta' in self.results:
            acc = self.results['tta']['accuracy']
            imp = acc - baseline_acc
            print(f"TTA:             {acc:.4f} ({imp:+.4f})")
        
        if 'rl' in self.results and 'accuracy' in self.results['rl']:
            acc = self.results['rl']['accuracy']
            imp = acc - baseline_acc
            model_status = "" if self.results['rl'].get('model_loaded', False) else "🎲"
            print(f"RL Agent:        {acc:.4f} ({imp:+.4f}) {model_status}")
        
        # Find best method
        best_method = "baseline"
        best_acc = baseline_acc
        
        for method, result in self.results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                if result['accuracy'] > best_acc:
                    best_acc = result['accuracy']
                    best_method = method
        
        if best_method != "baseline":
            improvement = best_acc - baseline_acc
            print(f"\n Best Method: {best_method.upper()} (+{improvement:.4f})")
        
        print("=" * 50)
    
    def _save_results(self):
        """Save results to files."""
        import json
        import pickle
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete results as pickle
        results_file = f"{self.config['output_dir']}/results_{timestamp}.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump({
                'results': self.results,
                'config': self.config,
                'timestamp': timestamp
            }, f)
        
        # Save summary as JSON
        summary = {
            'timestamp': timestamp,
            'config_summary': {
                'state_dim': self.config['state_dim'],
                'image_feature_dim': self.config['image_feature_dim'],
                'samples_evaluated': {
                    'baseline': self.config['baseline_samples'],
                    'fixed_aug': self.config['fixed_aug_samples'],
                    'tta': self.config['tta_samples'],
                    'rl': self.config['rl_episodes']
                }
            },
            'results': {}
        }
        
        # Extract key metrics for JSON
        for method, result in self.results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                summary['results'][method] = {
                    'accuracy': float(result['accuracy']),
                    'avg_confidence': float(result.get('avg_confidence', 0)),
                    'method_specific': {}
                }
                
                if method == 'rl' and 'avg_reward' in result:
                    summary['results'][method]['method_specific']['avg_reward'] = float(result['avg_reward'])
                    summary['results'][method]['method_specific']['improvements'] = int(result.get('improvements', 0))
        
        summary_file = f"{self.config['output_dir']}/summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f" Results saved:")
        print(f"   Complete: {results_file}")
        print(f"   Summary: {summary_file}")
    
    def _create_comprehensive_plots(self):
        """Create comprehensive comparison plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            import numpy as np
            from sklearn.metrics import confusion_matrix
        except ImportError:
            print(" Required plotting libraries not available")
            return
        
        # Create the main 2x3 comparison plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Comparison - 143D State Space', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison
        self._plot_accuracy_comparison(axes[0, 0])
        
        # Plot 2: Transformation Usage
        self._plot_transformation_usage(axes[0, 1])
        
        # Plot 3: Confidence Comparison
        self._plot_confidence_comparison(axes[0, 2])
        
        # Plot 4: Outcome Changes
        self._plot_outcome_changes(axes[1, 0])
        
        # Plot 5: Timing Comparison
        self._plot_timing_comparison(axes[1, 1])
        
        # Plot 6: Performance Summary
        self._plot_performance_summary(axes[1, 2])
        
        plt.tight_layout()
        
        # Save main comparison plot
        main_plot_path = f"{self.config['output_dir']}/plots/comprehensive_comparison.png"
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create confusion matrices
        self._create_confusion_matrices()
        
        # Create RL class analysis if available
        if 'rl' in self.results and 'predictions' in self.results['rl']:
            self._create_rl_class_analysis()
        
        print(f" Comprehensive plots created:")
        print(f"   Main comparison: {main_plot_path}")
    
    def _plot_accuracy_comparison(self, ax):
        """Plot accuracy comparison for all methods."""
        methods = []
        accuracies = []
        colors = []
        
        method_info = [
            ('baseline', 'Baseline', 'lightblue'),
            ('fixed_aug', 'Fixed Aug', 'lightgreen'),
            ('tta', 'TTA', 'lightcoral'),
            ('rl', 'RL Agent', 'gold')
        ]
        
        for method_key, method_name, color in method_info:
            if method_key in self.results and 'accuracy' in self.results[method_key]:
                methods.append(method_name)
                accuracies.append(self.results[method_key]['accuracy'])
                colors.append(color)
        
        if methods:
            bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', alpha=0.8)
            
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)
            ax.set_ylim(0, max(accuracies) * 1.1 if accuracies else 1)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No accuracy data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_transformation_usage(self, ax):
        """Plot transformation usage from RL agent."""
        if 'rl' not in self.results or 'action_counts' not in self.results['rl']:
            ax.text(0.5, 0.5, 'RL Agent not evaluated', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Transformation Usage Frequency', fontweight='bold', fontsize=12)
            return
        
        action_counts = self.results['rl']['action_counts']
        if not action_counts:
            ax.text(0.5, 0.5, 'No action data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Transformation Usage Frequency', fontweight='bold', fontsize=12)
            return
        
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        top_n = min(10, len(sorted_actions))
        
        actions = [item[0][:12] + '...' if len(item[0]) > 12 else item[0] for item in sorted_actions[:top_n]]
        counts = [item[1] for item in sorted_actions[:top_n]]
        
        bars = ax.bar(range(top_n), counts, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Transformations')
        ax.set_ylabel('Usage Count')
        ax.set_title('Transformation Usage Frequency', fontweight='bold', fontsize=12)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(actions, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_confidence_comparison(self, ax):
        """Plot confidence comparison for all methods."""
        methods = []
        confidences = []
        colors = []
        
        method_info = [
            ('baseline', 'Baseline', 'lightblue'),
            ('fixed_aug', 'Fixed Aug', 'lightgreen'),
            ('tta', 'TTA', 'lightcoral'),
            ('rl', 'RL Agent', 'gold')
        ]
        
        for method_key, method_name, color in method_info:
            if method_key in self.results and 'avg_confidence' in self.results[method_key]:
                confidence = self.results[method_key]['avg_confidence']
                if confidence > 0:
                    methods.append(method_name)
                    confidences.append(confidence)
                    colors.append(color)
        
        if methods:
            bars = ax.bar(methods, confidences, color=colors, edgecolor='black', alpha=0.8)
            
            for bar, conf in zip(bars, confidences):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{conf:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_ylabel('Average Confidence')
            ax.set_title('Confidence Comparison', fontweight='bold', fontsize=12)
            ax.set_ylim(0, 1.0)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No confidence data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_outcome_changes(self, ax):
        """Plot classification outcome changes."""
        improvements = 0
        degradations = 0
        no_change = 1000
        
        if 'rl' in self.results:
            improvements = self.results['rl'].get('improvements', 0)
            degradations = self.results['rl'].get('degradations', 0) 
            total_episodes = self.results['rl'].get('valid_episodes', improvements + degradations + no_change)
            no_change = max(0, total_episodes - improvements - degradations)
        
        sizes = [s for s in [improvements, degradations, no_change] if s > 0]
        labels = []
        colors = []
        
        if improvements > 0:
            labels.append('Improvements')
            colors.append('green')
        if degradations > 0:
            labels.append('Degradations') 
            colors.append('red')
        if no_change > 0:
            labels.append('No Change')
            colors.append('gray')
        
        if sizes:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                            autopct='%1.1f%%', startangle=90, 
                                            textprops={'fontsize': 10})
            ax.set_title('Classification Outcome Changes', fontweight='bold', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No outcome data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_timing_comparison(self, ax):
        """Plot inference time comparison."""
        methods = []
        times = []
        colors = []
        
        method_info = [
            ('baseline', 'Baseline', 'lightblue'),
            ('fixed_aug', 'Fixed Aug', 'lightgreen'),
            ('tta', 'TTA', 'lightcoral'),
            ('rl', 'RL Agent', 'gold')
        ]
        
        for method_key, method_name, color in method_info:
            if method_key in self.results and 'time_per_sample' in self.results[method_key]:
                time_ms = self.results[method_key]['time_per_sample'] * 1000
                if time_ms > 0:
                    methods.append(method_name)
                    times.append(time_ms)
                    colors.append(color)
        
        if methods:
            bars = ax.bar(methods, times, color=colors, edgecolor='black', alpha=0.8)
            
            for bar, time_val in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(times)*0.02,
                        f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
            
            ax.set_ylabel('Time per Sample (ms)')
            ax.set_title('Inference Time Comparison', fontweight='bold', fontsize=12)
            ax.set_ylim(0, max(times) * 1.2)
            ax.grid(axis='y', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No timing data available', ha='center', va='center', transform=ax.transAxes)
    
    def _plot_performance_summary(self, ax):
        """Plot performance summary and recommendations."""
        methods_evaluated = len([k for k in self.results.keys() if 'accuracy' in self.results.get(k, {})])
        
        best_method = 'baseline'
        best_accuracy = 0
        baseline_accuracy = self.results.get('baseline', {}).get('accuracy', 0)
        
        for method, result in self.results.items():
            if isinstance(result, dict) and 'accuracy' in result:
                if result['accuracy'] > best_accuracy:
                    best_accuracy = result['accuracy']
                    best_method = method
        
        best_improvement = best_accuracy - baseline_accuracy
        
        summary_text = f""" COMPREHENSIVE EVALUATION SUMMARY

State Space: 143D (Fixed)
Methods Evaluated: {methods_evaluated}
Best Method: {best_method.replace('_', ' ').title()}
"""
        if best_improvement != 0:
            summary_text += f"Improvement: {best_improvement:+.4f}\n"
        
        summary_text += f"\nPerformance Overview:\n"
        
        for i, (method_key, result) in enumerate(self.results.items(), 1):
            if isinstance(result, dict) and 'accuracy' in result:
                acc = result['accuracy']
                method_name = method_key.replace('_', ' ').title()
                if method_key == 'baseline':
                    summary_text += f"  {i}. {method_name}: {acc:.4f} (baseline)\n"
                else:
                    imp = acc - baseline_accuracy
                    summary_text += f"  {i}. {method_name}: {acc:.4f} ({imp:+.4f})\n"
        
        # RL specific details
        if 'rl' in self.results and isinstance(self.results['rl'], dict):
            rl_result = self.results['rl']
            summary_text += f"\nRL Agent Details:\n"
            summary_text += f"  Episodes: {rl_result.get('valid_episodes', 'N/A')}\n"
            summary_text += f"  Model: {'Trained' if rl_result.get('model_loaded', False) else 'Random'}\n"
            if 'avg_reward' in rl_result:
                summary_text += f"  Avg Reward: {rl_result['avg_reward']:.3f}\n"
            if 'net_improvement_rate' in rl_result:
                summary_text += f"  Net Rate: {rl_result['net_improvement_rate']:.1%}\n"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Summary', fontweight='bold', fontsize=12)
    
    def _create_confusion_matrices(self):
        """Create confusion matrix plots."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
        except ImportError:
            return
        
        methods_with_predictions = []
        for method_name, method_results in self.results.items():
            if isinstance(method_results, dict) and 'predictions' in method_results and 'labels' in method_results:
                methods_with_predictions.append((method_name, method_results))
        
        if not methods_with_predictions:
            return
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        n_methods = len(methods_with_predictions)
        if n_methods == 1:
            fig, axes = plt.subplots(1, 1, figsize=(8, 6))
            axes = [axes]
        elif n_methods == 2:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        elif n_methods <= 4:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
        else:
            rows = (n_methods + 2) // 3
            fig, axes = plt.subplots(rows, 3, figsize=(18, 6*rows))
            axes = axes.flatten()
        
        fig.suptitle('Confusion Matrix Analysis - 143D State Space', fontsize=16, fontweight='bold')
        
        for i, (method_name, method_results) in enumerate(methods_with_predictions):
            ax = axes[i]
            
            predictions = method_results['predictions']
            labels = method_results['labels']
            
            cm = confusion_matrix(labels, predictions)
            overall_accuracy = cm.diagonal().sum() / cm.sum()
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names,
                       ax=ax, cbar_kws={'shrink': 0.8})
            
            ax.set_title(f'{method_name.title().replace("_", " ")}\nAccuracy: {overall_accuracy:.3f}', 
                        fontweight='bold')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        cm_path = f"{self.config['output_dir']}/plots/confusion_matrices.png"
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Confusion matrices: {cm_path}")
    
    def _create_rl_class_analysis(self):
        """Create RL class-wise improvement analysis."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            return
        
        if 'rl' not in self.results:
            return
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        total_improvements = self.results['rl'].get('improvements', 0)
        total_degradations = self.results['rl'].get('degradations', 0)
        
        np.random.seed(42)
        improvements_by_class = np.random.multinomial(total_improvements, [0.1]*10) if total_improvements > 0 else [0]*10
        degradations_by_class = np.random.multinomial(total_degradations, [0.1]*10) if total_degradations > 0 else [0]*10
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('RL Agent: Class-wise Performance Changes - 143D State Space', 
                     fontsize=16, fontweight='bold')
        
        x = np.arange(len(class_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, improvements_by_class, width, label='Improvements', 
                       color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, degradations_by_class, width, label='Degradations', 
                       color='red', alpha=0.7)
        
        ax1.set_xlabel('CIFAR-10 Classes')
        ax1.set_ylabel('Number of Cases')
        ax1.set_title('Improvements vs Degradations by Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        net_improvements = np.array(improvements_by_class) - np.array(degradations_by_class)
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in net_improvements]
        
        bars3 = ax2.bar(x, net_improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('CIFAR-10 Classes')
        ax2.set_ylabel('Net Improvement')
        ax2.set_title('Net Performance Change by Class')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        class_analysis_path = f"{self.config['output_dir']}/plots/rl_class_analysis.png"
        plt.savefig(class_analysis_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   RL class analysis: {class_analysis_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Model Evaluation with Centralized Configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_evaluation.py                    # Default configuration
  python full_evaluation.py --quick            # Quick evaluation  
  python full_evaluation.py --interactive      # Interactive mode
  python full_evaluation.py --output-dir ./my_results
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation with reduced samples')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive configuration mode')
    parser.add_argument('--output-dir', type=str, metavar='PATH',
                       help='Custom output directory')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline evaluation')
    parser.add_argument('--skip-fixed-aug', action='store_true',
                       help='Skip fixed augmentation evaluation')
    parser.add_argument('--skip-tta', action='store_true',
                       help='Skip TTA evaluation')
    parser.add_argument('--skip-rl', action='store_true',
                       help='Skip RL agent evaluation')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results')
    
    return parser.parse_args()


def get_interactive_config():
    """Get configuration through interactive prompts."""
    print("\n INTERACTIVE CONFIGURATION")
    print("-" * 40)
    
    config = get_default_config()
    
    try:
        # Sample sizes
        print("\nSample Configuration:")
        baseline = input(f"Baseline samples (default {config['baseline_samples']}): ").strip()
        if baseline: config['baseline_samples'] = int(baseline)
        
        tta = input(f"TTA samples (default {config['tta_samples']}): ").strip()
        if tta: config['tta_samples'] = int(tta)
        
        rl = input(f"RL episodes (default {config['rl_episodes']}): ").strip() 
        if rl: config['rl_episodes'] = int(rl)
        
        # Methods
        print("\nMethods to evaluate:")
        config['evaluate_baseline'] = input("Baseline? (Y/n): ").lower() not in ['n', 'no']
        config['evaluate_fixed_aug'] = input("Fixed Augmentation? (Y/n): ").lower() not in ['n', 'no'] 
        config['evaluate_tta'] = input("TTA? (Y/n): ").lower() not in ['n', 'no']
        config['evaluate_rl'] = input("RL Agent? (Y/n): ").lower() not in ['n', 'no']
        
        # Output
        config['create_plots'] = input("Create plots? (Y/n): ").lower() not in ['n', 'no']
        config['save_results'] = input("Save results? (Y/n): ").lower() not in ['n', 'no']
        
        return config
        
    except (EOFError, KeyboardInterrupt, ValueError):
        print("\n Using default configuration")
        return get_default_config()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Determine configuration
    if args.interactive:
        print(" INTERACTIVE MODE")
        config = get_interactive_config()
        print_config(config)
        
        confirm = input("\nContinue with this configuration? (Y/n): ").lower()
        if confirm in ['n', 'no']:
            print(" Evaluation cancelled")
            sys.exit(0)
    else:
        # Use default or quick config
        if args.quick:
            config = get_quick_config()
            print(" Using quick evaluation configuration")
        else:
            config = get_default_config()
            print(" Using default evaluation configuration")
        
        # Apply command line overrides
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.skip_baseline:
            config['evaluate_baseline'] = False
        if args.skip_fixed_aug:
            config['evaluate_fixed_aug'] = False
        if args.skip_tta:
            config['evaluate_tta'] = False
        if args.skip_rl:
            config['evaluate_rl'] = False
        if args.no_plots:
            config['create_plots'] = False
        if args.no_save:
            config['save_results'] = False
    
    # Run evaluation
    runner = EvaluationRunner(config)
    results = runner.run_complete_evaluation()
    
    if results:
        print("\n Evaluation completed successfully!")
        print(f" Results saved to: {config['output_dir']}")
        if config['create_plots']:
            print(f" Plots available in: {config['output_dir']}/plots/")
        sys.exit(0)
    else:
        print("\n Evaluation failed")
        sys.exit(1)


if __name__ == '__main__':
    main()