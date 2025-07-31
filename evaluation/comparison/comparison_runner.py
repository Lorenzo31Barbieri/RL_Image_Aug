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
        
        try:
            # Dataset per valutazioni che richiedono singole immagini
            self.test_dataset = get_cifar10_test_dataset(
                data_root=self.config['data_root']
            )
            
            # DataLoader per valutazione baseline
            batch_size = self.config.get('batch_size', 64)
            self.test_loader = get_cifar10_test_loader(
                data_root=self.config['data_root'],
                batch_size=batch_size
            )
            
            # Info sul dataset per il summary
            data_info = {
                'total_samples': len(self.test_dataset),
                'batch_size': batch_size,
                'num_batches': len(self.test_loader),
                'num_workers': 0,
                'pin_memory': torch.cuda.is_available(),
                'distribution': {
                    'num_classes': 10, 
                    'class_counts': {i: len(self.test_dataset)//10 for i in range(10)}
                }
            }
            
            print_data_loading_summary(self.test_loader, data_info)
            
            self._data_loaded = True
            print("✅ Data loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
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
        """Crea i grafici di confronto nel layout desiderato."""
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE PLOTS")
        print(f"{'='*70}")
        
        summary = self.create_comparison_summary()
        
        # Crea figura con layout 2x3
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison (top-left) - Tutti i 4 metodi
        ax1 = axes[0, 0]
        self._plot_accuracy_comparison_all_methods(ax1, summary)
        
        # Plot 2: Transformation Usage Frequency (top-middle)
        ax2 = axes[0, 1]
        self._plot_transformation_frequency(ax2)
        
        # Plot 3: Confidence Comparison (top-right)
        ax3 = axes[0, 2]
        self._plot_confidence_comparison(ax3, summary)
        
        # Plot 4: Classification Outcome Changes (bottom-left) - Pie Chart
        ax4 = axes[1, 0]
        self._plot_outcome_changes_pie(ax4)
        
        # Plot 5: Inference Time Comparison (bottom-middle)
        ax5 = axes[1, 1]
        self._plot_inference_time_comparison(ax5, summary)
        
        # Plot 6: Performance Summary (bottom-right)
        ax6 = axes[1, 2]
        self._plot_performance_summary(ax6, summary)
        
        plt.tight_layout()
        
        # Salva il grafico
        plot_path = os.path.join(self.plots_dir, 'comprehensive_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Comprehensive plots saved to: {plot_path}")

    def _plot_accuracy_comparison_all_methods(self, ax, summary):
        """Plot 1: Confronto accuratezza di tutti i 4 metodi."""
        methods = []
        accuracies = []
        colors = []
        
        # Ordine: Baseline, Fixed Aug, TTA, RL Agent
        method_order = ['baseline', 'fixed_aug', 'tta', 'rl']
        method_names = ['Baseline', 'Fixed Aug', 'TTA', 'RL Agent']
        method_colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, method_key in enumerate(method_order):
            if method_key in summary['accuracy_comparison']:
                methods.append(method_names[i])
                accuracies.append(summary['accuracy_comparison'][method_key])
                colors.append(method_colors[i])
        
        if methods:
            bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', alpha=0.8)
            
            # Aggiungi valori sulle barre
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
            ax.set_title('Accuracy Comparison', fontweight='bold', fontsize=12)

    def _plot_transformation_frequency(self, ax):
        """Plot 2: Frequenza d'uso delle trasformazioni dall'agente RL."""
        if 'rl' not in self.results:
            ax.text(0.5, 0.5, 'RL Agent not evaluated', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Transformation Usage Frequency', fontweight='bold', fontsize=12)
            return
        
        # Cerca action_counts nei risultati RL
        rl_results = self.results['rl']
        action_counts = rl_results.get('action_counts', {})
        
        if not action_counts:
            ax.text(0.5, 0.5, 'No action data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Transformation Usage Frequency', fontweight='bold', fontsize=12)
            return
        
        # Ordina per frequenza d'uso
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        top_n = min(15, len(sorted_actions))  # Mostra solo le top 15
        
        actions = [item[0] for item in sorted_actions[:top_n]]
        counts = [item[1] for item in sorted_actions[:top_n]]
        
        # Abbrevia nomi delle azioni se troppo lunghi
        short_actions = []
        for action in actions:
            if len(action) > 12:
                short_actions.append(action[:12] + '...')
            else:
                short_actions.append(action)
        
        bars = ax.bar(range(top_n), counts, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Transformations')
        ax.set_ylabel('Usage Count')
        ax.set_title('Transformation Usage Frequency', fontweight='bold', fontsize=12)
        ax.set_xticks(range(top_n))
        ax.set_xticklabels(short_actions, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)

    def _plot_confidence_comparison(self, ax, summary):
        """Plot 3: Confronto della confidenza media."""
        methods = []
        confidences = []
        colors = []
        
        # Estrai confidenze dai risultati
        method_order = ['baseline', 'fixed_aug', 'tta', 'rl']
        method_names = ['Baseline', 'Fixed Aug', 'TTA', 'RL Agent']
        method_colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, method_key in enumerate(method_order):
            confidence = None
            
            if method_key == 'baseline' and 'baseline' in self.results:
                confidence = self.results['baseline'].get('avg_confidence')
            elif method_key == 'fixed_aug' and 'fixed_aug' in self.results:
                confidence = self.results['fixed_aug'].get('augmented_confidence')
            elif method_key == 'tta' and 'tta' in self.results:
                confidence = self.results['tta'].get('tta_avg_confidence')
            elif method_key == 'rl' and 'rl' in self.results:
                confidence = self.results['rl'].get('final_avg_confidence')
            
            if confidence is not None:
                methods.append(method_names[i])
                confidences.append(confidence)
                colors.append(method_colors[i])
        
        if methods:
            bars = ax.bar(methods, confidences, color=colors, edgecolor='black', alpha=0.8)
            
            # Aggiungi valori sulle barre
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
            ax.set_title('Confidence Comparison', fontweight='bold', fontsize=12)

    def _plot_outcome_changes_pie(self, ax):
        """Plot 4: Cambiamenti dei risultati di classificazione (grafico a torta)."""
        # Usa i dati dell'agente RL se disponibili, altrimenti calcola dai dati disponibili
        if 'rl' in self.results:
            rl_results = self.results['rl']
            improvements = rl_results.get('improvements', 0)
            degradations = rl_results.get('degradations', 0)
            total_episodes = rl_results.get('num_episodes_evaluated', improvements + degradations)
            no_change = max(0, total_episodes - improvements - degradations)
        else:
            # Usa dati da altri metodi se disponibili
            improvements = 0
            degradations = 0
            no_change = 1000  # Valore di default
            
            for method in ['fixed_aug', 'tta']:
                if method in self.results:
                    method_results = self.results[method]
                    if 'improvements' in method_results:
                        improvements = method_results['improvements']
                        degradations = method_results.get('degradations', 0)
                        total_samples = method_results.get('total_samples', 1000)
                        no_change = total_samples - improvements - degradations
                        break
        
        if improvements + degradations + no_change == 0:
            ax.text(0.5, 0.5, 'No outcome data available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Classification Outcome Changes', fontweight='bold', fontsize=12)
            return
        
        sizes = [improvements, degradations, no_change]
        labels = ['Improvements', 'Degradations', 'No Change']
        colors = ['green', 'red', 'gray']
        
        # Filtra i segmenti con valore 0
        non_zero_sizes = []
        non_zero_labels = []
        non_zero_colors = []
        
        for size, label, color in zip(sizes, labels, colors):
            if size > 0:
                non_zero_sizes.append(size)
                non_zero_labels.append(label)
                non_zero_colors.append(color)
        
        if non_zero_sizes:
            wedges, texts, autotexts = ax.pie(non_zero_sizes, labels=non_zero_labels, 
                                            colors=non_zero_colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 10})
            ax.set_title('Classification Outcome Changes', fontweight='bold', fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No changes detected', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Classification Outcome Changes', fontweight='bold', fontsize=12)

    def _plot_inference_time_comparison(self, ax, summary):
        """Plot 5: Confronto dei tempi di inferenza."""
        methods = []
        times = []
        colors = []
        
        # Estrai tempi di inferenza
        method_order = ['baseline', 'fixed_aug', 'tta', 'rl']
        method_names = ['Baseline', 'Fixed Aug', 'TTA', 'RL Agent']
        method_colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold']
        
        for i, method_key in enumerate(method_order):
            time_per_sample = None
            
            if method_key in self.results:
                time_per_sample = self.results[method_key].get('time_per_sample')
            
            if time_per_sample is not None and time_per_sample > 0:
                methods.append(method_names[i])
                times.append(time_per_sample * 1000)  # Converti in millisecondi
                colors.append(method_colors[i])
        
        if methods:
            bars = ax.bar(methods, times, color=colors, edgecolor='black', alpha=0.8)
            
            # Aggiungi valori sulle barre
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
            ax.set_title('Inference Time Comparison', fontweight='bold', fontsize=12)

    def _plot_performance_summary(self, ax, summary):
        """Plot 6: Riassunto delle performance e raccomandazioni."""
        
        # Calcola statistiche per il riassunto
        methods_evaluated = len(summary.get('methods_evaluated', []))
        best_method = summary.get('best_method', 'None')
        best_improvement = summary.get('best_improvement', 0)
        
        # Crea il testo del riassunto
        summary_text = f"🔍 COMPREHENSIVE EVALUATION SUMMARY\n\n"
        
        summary_text += f"📊 Methods Evaluated: {methods_evaluated}\n"
        summary_text += f"🏆 Best Method: {best_method.replace('_', ' ').title()}\n"
        if best_improvement != 0:
            summary_text += f"   Improvement: {best_improvement:+.4f}\n"
        
        # Performance Overview per tutti i 4 metodi
        summary_text += f"\n📈 Performance Overview:\n"
        
        accuracy_comparison = summary.get('accuracy_comparison', {})
        improvement_comparison = summary.get('improvement_comparison', {})
        
        method_display = {
            'baseline': 'Baseline',
            'fixed_aug': 'Fixed Aug', 
            'tta': 'TTA',
            'rl': 'RL Agent'
        }
        
        for i, (method_key, display_name) in enumerate(method_display.items(), 1):
            if method_key in accuracy_comparison:
                acc = accuracy_comparison[method_key]
                imp = improvement_comparison.get(method_key, 0)
                if method_key == 'baseline':
                    summary_text += f"   {i}. {display_name}: {acc:.4f} (baseline)\n"
                else:
                    summary_text += f"   {i}. {display_name}: {acc:.4f} ({imp:+.4f})\n"
        
        # Outcome Analysis (principalmente dall'agente RL)
        summary_text += f"\n📈 Outcome Analysis:\n"
        if 'rl' in self.results:
            rl_results = self.results['rl']
            improvements = rl_results.get('improvements', 0)
            degradations = rl_results.get('degradations', 0)
            total_episodes = rl_results.get('num_episodes_evaluated', improvements + degradations)
            no_change = max(0, total_episodes - improvements - degradations)
            
            summary_text += f"   • Improvements: {improvements}\n"
            summary_text += f"   • Degradations: {degradations}\n"
            summary_text += f"   • No Change: {no_change}\n"
        else:
            summary_text += f"   • No detailed outcome data available\n"
        
        # Raccomandazione
        summary_text += f"\n✅ Recommendation:\n"
        if best_improvement > 0.015:
            recommendation = f"   {best_method.replace('_', ' ').title()}: Excellent improvement!\n   Highly recommended for production use."
        elif best_improvement > 0.005:
            recommendation = f"   {best_method.replace('_', ' ').title()}: Good improvement.\n   Consider computational cost vs benefit."
        elif best_improvement > 0:
            recommendation = f"   {best_method.replace('_', ' ').title()}: Minimal improvement.\n   Limited practical benefit."
        else:
            recommendation = f"   No significant improvements found.\n   Consider different approaches."
        
        summary_text += recommendation
        
        # Aggiungi informazioni sull'efficienza se disponibili
        time_comparison = summary.get('time_comparison', {})
        if time_comparison and best_method in time_comparison:
            best_time = time_comparison[best_method]
            baseline_time = time_comparison.get('baseline', best_time)
            if baseline_time > 0:
                slowdown = best_time / baseline_time
                summary_text += f"\n\n⚡ Efficiency Note:\n"
                summary_text += f"   Speed: {slowdown:.1f}× slower than baseline"
        
        # Mostra il testo
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", alpha=0.8))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Summary & Recommendations', fontweight='bold', fontsize=12)

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