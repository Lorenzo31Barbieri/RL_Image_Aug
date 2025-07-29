"""
Script di orchestrazione per confrontare tutti i metodi di valutazione.
Esegue baseline, fixed augmentation, TTA e RL agent e confronta i risultati.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import os
import json
import pickle
from datetime import datetime
import argparse

# Import dei moduli di valutazione
from evaluation.core.model_loader import load_classifier, load_rl_agent, print_loading_summary
from evaluation.core.data_utils import get_cifar10_test_dataset, get_cifar10_test_loader, print_data_loading_summary
from evaluation.core.evaluation_core import save_evaluation_results

from evaluation.methods.evaluate_baseline import evaluate_baseline
from evaluation.methods.evaluate_fixed_aug import evaluate_fixed_augmentation
from evaluation.methods.evaluate_tta import evaluate_tta
from evaluation.methods.evaluate_rl_agent import evaluate_rl_agent


class EvaluationComparison:
    """
    Classe principale per orchestrare e confrontare tutte le valutazioni.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inizializza il sistema di confronto.
        
        Args:
            config: Configurazione con percorsi e parametri
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Percorsi di output
        self.output_dir = config.get('output_dir', './evaluation_results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        
        # Crea directory se non esistono
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        print(f"🎯 Evaluation Comparison initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"💻 Device: {self.device}")
    
    def load_models(self) -> None:
        """Carica tutti i modelli necessari."""
        print(f"\n{'='*60}")
        print("LOADING MODELS")
        print(f"{'='*60}")
        
        # Carica classificatore
        self.classifier = load_classifier(
            model_path=self.config['classifier_path'],
            device=self.device
        )
        
        # Carica agente RL (se richiesto)
        self.agent = None
        self.rl_model_loaded = False
        
        if self.config.get('evaluate_rl', True):
            try:
                self.agent, self.rl_model_loaded = load_rl_agent(
                    model_path=self.config['rl_model_path'],
                    state_dim=self.config.get('state_dim', 15),
                    device=self.device
                )
            except Exception as e:
                print(f"⚠️ Could not load RL agent: {e}")
                print("RL evaluation will be skipped.")
                self.config['evaluate_rl'] = False
        
        # Stampa riassunto
        print_loading_summary(
            classifier=self.classifier,
            agent=self.agent,
            agent_loaded=self.rl_model_loaded
        )
    
    def load_data(self) -> None:
        """Carica i dati di test."""
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")
        
        # Dataset per valutazioni che richiedono singole immagini
        self.test_dataset = get_cifar10_test_dataset(
            data_root=self.config['data_root']
        )
        
        # DataLoader per valutazione baseline
        self.test_loader, data_info = self.config.get('batch_size', 64)
        from evaluation.core.data_utils import create_evaluation_dataloader
        self.test_loader, data_info = create_evaluation_dataloader(
            data_root=self.config['data_root'],
            batch_size=self.config.get('batch_size', 64)
        )
        
        print_data_loading_summary(self.test_loader, data_info)
    
    def run_baseline_evaluation(self) -> None:
        """Esegue valutazione baseline."""
        print(f"\n🎯 Running baseline evaluation...")
        
        self.results['baseline'] = evaluate_baseline(
            classifier_model=self.classifier,
            test_loader=self.test_loader,
            device=self.device,
            verbose=True,
            return_details=False  # Per performance
        )
        
        print(f"✅ Baseline evaluation completed")
    
    def run_fixed_augmentation_evaluation(self) -> None:
        """Esegue valutazione fixed augmentation."""
        if not self.config.get('evaluate_fixed_aug', True):
            print("⏭️ Skipping fixed augmentation evaluation")
            return
        
        print(f"\n🔧 Running fixed augmentation evaluation...")
        
        augmentation_ids = self.config.get('fixed_aug_ids', [0, 3, 6])
        
        self.results['fixed_aug'] = evaluate_fixed_augmentation(
            classifier_model=self.classifier,
            test_dataset=self.test_dataset,
            augmentation_ids=augmentation_ids,
            device=self.device,
            batch_size=self.config.get('batch_size', 64),
            verbose=True
        )
        
        print(f"✅ Fixed augmentation evaluation completed")
    
    def run_tta_evaluation(self) -> None:
        """Esegue valutazione TTA."""
        if not self.config.get('evaluate_tta', True):
            print("⏭️ Skipping TTA evaluation")
            return
        
        print(f"\n🔬 Running TTA evaluation...")
        
        num_samples = self.config.get('tta_samples', 1000)
        
        self.results['tta'] = evaluate_tta(
            classifier_model=self.classifier,
            test_dataset=self.test_dataset,
            device=self.device,
            num_samples=num_samples,
            use_ttach=self.config.get('use_ttach', True),
            verbose=True
        )
        
        print(f"✅ TTA evaluation completed")
    
    def run_rl_evaluation(self) -> None:
        """Esegue valutazione RL agent."""
        if not self.config.get('evaluate_rl', True) or self.agent is None:
            print("⏭️ Skipping RL evaluation")
            return
        
        print(f"\n🤖 Running RL agent evaluation...")
        
        num_episodes = self.config.get('rl_episodes', 1000)
        
        self.results['rl'] = evaluate_rl_agent(
            agent=self.agent,
            classifier_model=self.classifier,
            test_dataset=self.test_dataset,
            device=self.device,
            num_episodes=num_episodes,
            max_steps_per_episode=self.config.get('max_steps_per_episode', 3),
            verbose=True
        )
        
        self.results['rl']['model_loaded'] = self.rl_model_loaded
        print(f"✅ RL evaluation completed")
    
    def run_all_evaluations(self) -> None:
        """Esegue tutte le valutazioni configurate."""
        print(f"\n{'='*70}")
        print("STARTING COMPREHENSIVE EVALUATION")
        print(f"{'='*70}")
        
        start_time = datetime.now()
        
        # Esegui tutte le valutazioni
        self.run_baseline_evaluation()
        self.run_fixed_augmentation_evaluation()
        self.run_tta_evaluation()
        self.run_rl_evaluation()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\n🎉 All evaluations completed in {total_time:.1f} seconds")
        
        # Salva risultati
        self.save_results()
    
    def create_comparison_summary(self) -> Dict[str, Any]:
        """Crea un riassunto comparativo dei risultati."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'device': str(self.device),
            'methods_evaluated': list(self.results.keys())
        }
        
        # Estrai metriche chiave per confronto
        if 'baseline' in self.results:
            baseline_acc = self.results['baseline']['accuracy']
            summary['baseline_accuracy'] = baseline_acc
        else:
            baseline_acc = 0
        
        # Confronta accuratezza
        accuracy_comparison = {}
        improvement_comparison = {}
        
        for method, results in self.results.items():
            if method == 'baseline':
                accuracy_comparison[method] = results['accuracy']
                improvement_comparison[method] = 0.0
            elif method == 'fixed_aug':
                accuracy_comparison[method] = results['augmented_accuracy']
                improvement_comparison[method] = results['accuracy_improvement']
            elif method == 'tta':
                accuracy_comparison[method] = results['tta_accuracy']
                improvement_comparison[method] = results['accuracy_improvement']
            elif method == 'rl':
                accuracy_comparison[method] = results['final_accuracy']
                improvement_comparison[method] = results['accuracy_improvement']
        
        summary['accuracy_comparison'] = accuracy_comparison
        summary['improvement_comparison'] = improvement_comparison
        
        # Trova il metodo migliore
        if improvement_comparison:
            best_method = max(improvement_comparison.items(), key=lambda x: x[1])
            summary['best_method'] = best_method[0]
            summary['best_improvement'] = best_method[1]
        
        # Confronta tempi
        time_comparison = {}
        for method, results in self.results.items():
            if 'time_per_sample' in results:
                time_comparison[method] = results['time_per_sample'] * 1000  # in ms
        
        summary['time_comparison'] = time_comparison
        
        return summary
    
    def print_comparison_summary(self) -> None:
        """Stampa riassunto comparativo."""
        summary = self.create_comparison_summary()
        
        print(f"\n{'='*70}")
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"📊 METHODS EVALUATED: {', '.join(summary['methods_evaluated'])}")
        
        if 'accuracy_comparison' in summary:
            print(f"\n📈 ACCURACY COMPARISON:")
            for method, accuracy in summary['accuracy_comparison'].items():
                improvement = summary['improvement_comparison'].get(method, 0)
                method_name = method.upper().replace('_', ' ')
                print(f"  {method_name:15}: {accuracy:.4f} ({improvement:+.4f})")
        
        if 'best_method' in summary:
            best_method = summary['best_method'].upper().replace('_', ' ')
            print(f"\n🏆 BEST METHOD: {best_method}")
            print(f"  Improvement: {summary['best_improvement']:+.4f}")
        
        if 'time_comparison' in summary:
            print(f"\n⚡ TIME COMPARISON (ms per sample):")
            for method, time_ms in summary['time_comparison'].items():
                method_name = method.upper().replace('_', ' ')
                print(f"  {method_name:15}: {time_ms:.1f}ms")
        
        # Raccomandazioni
        print(f"\n💡 RECOMMENDATIONS:")
        
        improvements = summary.get('improvement_comparison', {})
        times = summary.get('time_comparison', {})
        
        # Analizza ogni metodo
        for method in summary['methods_evaluated']:
            if method == 'baseline':
                continue
            
            improvement = improvements.get(method, 0)
            method_name = method.upper().replace('_', ' ')
            
            if improvement > 0.01:
                recommendation = "✅ Highly Recommended: Substantial accuracy gain."
            elif improvement > 0.005:
                recommendation = "⚠️ Recommended with considerations: Small but measurable gain."
            elif improvement > 0:
                recommendation = "📊 Limited benefit: Marginal or negligible improvement."
            else:
                recommendation = "❌ Not recommended: No improvement or performance degradation."
            
            print(f"  - {method_name}: {recommendation}")

    def create_plots(self):
        """Crea i grafici di confronto."""
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print(f"{'='*70}")
        
        summary = self.create_comparison_summary()
        
        # Plot 1: Accuracy Comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = list(summary['accuracy_comparison'].keys())
        accuracies = list(summary['accuracy_comparison'].values())
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        
        ax.bar(methods, accuracies, color=colors[:len(methods)])
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Comparison Across Methods')
        ax.grid(axis='y', alpha=0.5)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'accuracy_comparison.png'))
        plt.close(fig)
        
        # Plot 2: Improvement vs Time
        fig, ax = plt.subplots(figsize=(10, 6))
        if 'time_comparison' in summary and 'improvement_comparison' in summary:
            methods_to_plot = [m for m in summary['improvement_comparison'].keys() if m != 'baseline']
            improvements = [summary['improvement_comparison'][m] for m in methods_to_plot]
            times = [summary['time_comparison'].get(m, 0) for m in methods_to_plot]
            
            ax.scatter(times, improvements, s=200, c=['green', 'purple', 'orange'][:len(methods_to_plot)], alpha=0.7, edgecolors='black')
            ax.axhline(0, color='red', linestyle='--')
            
            for i, method in enumerate(methods_to_plot):
                ax.annotate(method.upper(), (times[i], improvements[i]), xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            ax.set_xlabel('Time per Sample (ms)')
            ax.set_ylabel('Accuracy Improvement over Baseline')
            ax.set_title('Improvement vs Computational Cost')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, 'improvement_vs_time.png'))
            plt.close(fig)

        print("✅ Plots generated and saved.")

    def save_results(self) -> None:
        """Salva i risultati completi e il riassunto."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'results_{timestamp}.pkl')
        summary_file = os.path.join(self.output_dir, f'summary_{timestamp}.json')
        
        # Salva i risultati completi con pickle
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Salva il riassunto come JSON
        summary = self.create_comparison_summary()
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n💾 Evaluation results saved to {results_file}")
        print(f"📄 Summary saved to {summary_file}")