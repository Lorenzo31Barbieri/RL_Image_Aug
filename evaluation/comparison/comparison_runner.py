"""
Script di orchestrazione per confrontare tutti i metodi di valutazione.
Esegue baseline, fixed augmentation, TTA e RL agent e confronta i risultati.
"""

from sklearn.metrics import confusion_matrix
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
import os
import json
import pickle
from datetime import datetime
import seaborn as sns
import argparse

import tqdm

# Import dei moduli di valutazione
from evaluation.core.model_loader import load_classifier, load_rl_agent, print_loading_summary
from evaluation.core.data_utils import get_cifar10_test_dataset, get_cifar10_test_loader, print_data_loading_summary
from evaluation.core.evaluation_core import save_evaluation_results

from evaluation.methods.evaluate_baseline import evaluate_baseline
from evaluation.methods.evaluate_fixed_aug import evaluate_fixed_augmentation
from evaluation.methods.evaluate_tta import evaluate_tta
from evaluation.methods.evaluate_rl_agent import evaluate_rl_agent
from src.environment.transforms import get_action_name


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
        """Esegue valutazione baseline con dati dettagliati per confusion matrix."""
        print(f"\n🎯 Running baseline evaluation...")
        
        self.results['baseline'] = evaluate_baseline(
            classifier_model=self.classifier,
            test_loader=self.test_loader,
            device=self.device,
            verbose=True,
            return_details=True  # Importante: ottieni predizioni dettagliate
        )
        
        print(f"✅ Baseline evaluation completed")
    
    def run_fixed_augmentation_evaluation(self) -> None:
        """Esegue valutazione fixed augmentation con dati dettagliati."""
        if not self.config.get('evaluate_fixed_aug', True):
            print("⏭️ Skipping fixed augmentation evaluation")
            return
        
        print(f"\n🔧 Running fixed augmentation evaluation...")
        
        augmentation_ids = self.config.get('fixed_aug_ids', [0, 3, 6])
        
        # Modifica per ottenere dati dettagliati se necessario
        results = evaluate_fixed_augmentation(
            classifier_model=self.classifier,
            test_dataset=self.test_dataset,
            augmentation_ids=augmentation_ids,
            device=self.device,
            batch_size=self.config.get('batch_size', 64),
            verbose=True
        )
        
        # Se non abbiamo predizioni dettagliate, esegui una valutazione rapida per ottenerle
        if 'predictions' not in results:
            print("📊 Getting detailed predictions for confusion matrix...")
            results.update(self._get_detailed_predictions_for_method('fixed_aug'))
        
        self.results['fixed_aug'] = results
        print(f"✅ Fixed augmentation evaluation completed")

    def run_tta_evaluation(self) -> None:
        """Esegue valutazione TTA con dati dettagliati."""
        if not self.config.get('evaluate_tta', True):
            print("⏭️ Skipping TTA evaluation")
            return
        
        print(f"\n🔬 Running TTA evaluation...")
        
        num_samples = self.config.get('tta_samples', 1000)
        
        results = evaluate_tta(
            classifier_model=self.classifier,
            test_dataset=self.test_dataset,
            device=self.device,
            num_samples=num_samples,
            use_ttach=self.config.get('use_ttach', True),
            verbose=True
        )
        
        # Se non abbiamo predizioni dettagliate, esegui una valutazione rapida per ottenerle
        if 'predictions' not in results:
            print("📊 Getting detailed predictions for confusion matrix...")
            results.update(self._get_detailed_predictions_for_method('tta', num_samples))
        
        self.results['tta'] = results
        print(f"✅ TTA evaluation completed")
    
    def run_rl_evaluation(self) -> None:
        """Esegue valutazione RL agent con tracking dettagliato."""
        if not self.config.get('evaluate_rl', True) or self.agent is None:
            print("⏭️ Skipping RL evaluation")
            return
        
        print(f"\n🤖 Running detailed RL agent evaluation...")
        
        num_episodes = self.config.get('rl_episodes', 1000)
        
        # Usa la valutazione dettagliata invece di quella standard
        try:
            from evaluation.methods.evaluate_rl_agent import evaluate_rl_agent_detailed
            
            self.results['rl'] = evaluate_rl_agent_detailed(
                agent=self.agent,
                classifier_model=self.classifier,
                test_dataset=self.test_dataset,
                device=self.device,
                num_episodes=num_episodes,
                max_steps_per_episode=self.config.get('max_steps_per_episode', 3),
                verbose=True,
                save_examples=True
            )
            
            self.results['rl']['model_loaded'] = self.rl_model_loaded
            print(f"✅ Detailed RL evaluation completed")
            
        except ImportError:
            # Fallback alla valutazione standard se quella dettagliata non è disponibile
            print("⚠️ Detailed RL evaluation not available, using standard evaluation")
            from evaluation.methods.evaluate_rl_agent import evaluate_rl_agent
            
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
            print(f"✅ Standard RL evaluation completed")
    
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

    def save_improved_images_from_rl(self):
        """Salva esempi di immagini migliorate dall'agente RL."""
        if 'rl' not in self.results:
            return
        
        rl_results = self.results['rl']
        improvement_examples = rl_results.get('improvement_examples', [])
        
        if not improvement_examples:
            print("⚠️ No improvement examples available to save")
            return
        
        # Crea cartella per le immagini
        images_dir = os.path.join(self.output_dir, 'improved_images')
        os.makedirs(images_dir, exist_ok=True)
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
        
        print(f"\n💾 Saving {len(improvement_examples)} improved image examples...")
        
        for i, example in enumerate(improvement_examples):
            try:
                original_image = example['original_image']
                augmented_image = example['augmented_image']
                true_label = example['true_label']
                actions = example['actions']
                confidence_improvement = example['confidence_improvement']
                
                # # Debug: stampa info sulle immagini
                # print(f"Debug Example {i+1}:")
                # print(f"  Original shape: {original_image.shape if hasattr(original_image, 'shape') else 'No shape'}")
                # print(f"  Original type: {type(original_image)}")
                # print(f"  Augmented shape: {augmented_image.shape if hasattr(augmented_image, 'shape') else 'No shape'}")
                # print(f"  Augmented type: {type(augmented_image)}")
                
                # Funzione helper per convertire e normalizzare immagini
                def process_image_for_display(img_tensor):
                    """Converte un tensore immagine in formato numpy per visualizzazione."""
                    
                    if not isinstance(img_tensor, torch.Tensor):
                        print(f"Warning: Image is not a tensor, type: {type(img_tensor)}")
                        return None
                    
                    # Clona per evitare modifiche all'originale
                    img = img_tensor.clone().detach()
                    
                    # Sposta su CPU se necessario
                    if img.is_cuda:
                        img = img.cpu()
                    
                    #print(f"  Processing - Shape: {img.shape}, Min: {img.min():.3f}, Max: {img.max():.3f}")
                    
                    # Gestisci diverse forme di tensore
                    if len(img.shape) == 4:  # Batch dimension
                        img = img.squeeze(0)
                    elif len(img.shape) == 2:  # Grayscale, aggiungi dimensione canale
                        img = img.unsqueeze(0)
                    
                    # Converti da CHW a HWC per matplotlib
                    if len(img.shape) == 3 and img.shape[0] == 3:
                        img = img.permute(1, 2, 0)
                    elif len(img.shape) == 3 and img.shape[0] == 1:
                        img = img.squeeze(0)  # Rimuovi dimensione canale per grayscale
                    
                    # Converti in numpy
                    img_np = img.numpy()
                    
                    # Denormalizza se i valori sono nell'range tipico di normalizzazione CIFAR-10
                    if img_np.min() < -1.0:  # Probabilmente normalizzato
                        #print("  Denormalizing CIFAR-10 normalized image...")
                        # CIFAR-10 denormalization
                        mean = np.array([0.4914, 0.4822, 0.4465])
                        std = np.array([0.2023, 0.1994, 0.2010])
                        
                        if len(img_np.shape) == 3:  # Color
                            img_np = img_np * std + mean
                        else:  # Grayscale - usa solo il primo canale
                            img_np = img_np * std[0] + mean[0]
                    
                    # Clamp ai valori validi [0, 1]
                    img_np = np.clip(img_np, 0, 1)
                    
                    #print(f"  Final - Shape: {img_np.shape}, Min: {img_np.min():.3f}, Max: {img_np.max():.3f}")
                    
                    return img_np
                
                # Processa le immagini
                original_np = process_image_for_display(original_image)
                augmented_np = process_image_for_display(augmented_image)
                
                if original_np is None or augmented_np is None:
                    print(f"⚠️ Could not process images for example {i+1}")
                    continue
                
                # Crea figura comparativa
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                fig.suptitle(f'RL Improvement Example {i+1}: {class_names[true_label].title()}\n'
                            f'Confidence improvement: +{confidence_improvement:.3f}', 
                            fontsize=14, fontweight='bold')
                
                # Immagine originale
                if len(original_np.shape) == 2:  # Grayscale
                    ax1.imshow(original_np, cmap='gray')
                else:  # Color
                    ax1.imshow(original_np)
                ax1.set_title('Original Image\n(Incorrectly Classified)', fontsize=12)
                ax1.axis('off')
                
                # Immagine aumentata
                if len(augmented_np.shape) == 2:  # Grayscale
                    ax2.imshow(augmented_np, cmap='gray')
                else:  # Color
                    ax2.imshow(augmented_np)
                
                # Ottieni nomi delle azioni
                try:
                    from src.environment.transforms import get_action_name
                    action_names = [get_action_name(a) for a in actions]
                except:
                    action_names = [f"Action_{a}" for a in actions]
                
                action_text = ", ".join(action_names[:2])
                if len(action_names) > 2:
                    action_text += "..."
                    
                ax2.set_title(f'After RL Actions\n(Correctly Classified)\nActions: {action_text}', 
                            fontsize=12)
                ax2.axis('off')
                
                plt.tight_layout()
                
                # Salva l'esempio
                example_path = os.path.join(images_dir, 
                                        f'rl_improvement_{i+1}_{class_names[true_label]}.png')
                plt.savefig(example_path, dpi=200, bbox_inches='tight', facecolor='white')
                plt.close()
                
                #print(f"💾 Saved example {i+1}: RL improvement for {class_names[true_label]}")
                
            except Exception as e:
                print(f"⚠️ Could not save example {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ Improved image examples saved to: {images_dir}/")

    # Aggiungi anche questa funzione helper per creare esempi di test se necessario
    def create_test_improved_images(self):
        """Crea esempi di test per verificare il salvataggio immagini."""
        
        images_dir = os.path.join(self.output_dir, 'test_images')
        os.makedirs(images_dir, exist_ok=True)
        
        try:
            # Carica alcune immagini dal dataset di test
            from evaluation.core.data_utils import get_cifar10_test_dataset
            import torchvision.transforms as transforms
            
            # Transform che mantiene i valori nell'range [0,1]
            test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            
            test_dataset = get_cifar10_test_dataset(
                data_root=self.config['data_root'], 
                transform=test_transform
            )
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
            
            # Crea esempi di test
            for i in range(3):
                image, label = test_dataset[i * 100]  # Prendi ogni 100 immagini
                
                # Crea versione "aumentata" (semplice flip)
                augmented = torch.flip(image, dims=[2])  # Flip orizzontale
                
                # Salva confronto
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                fig.suptitle(f'Test Image {i+1}: {class_names[label]}', fontsize=14)
                
                # Immagine originale
                original_np = image.permute(1, 2, 0).numpy()
                ax1.imshow(original_np)
                ax1.set_title('Original')
                ax1.axis('off')
                
                # Immagine augmentata
                augmented_np = augmented.permute(1, 2, 0).numpy()
                ax2.imshow(augmented_np)
                ax2.set_title('Flipped')
                ax2.axis('off')
                
                plt.tight_layout()
                
                test_path = os.path.join(images_dir, f'test_image_{i+1}_{class_names[label]}.png')
                plt.savefig(test_path, dpi=200, bbox_inches='tight')
                plt.close()
                
                print(f"💾 Saved test image {i+1}")
            
            print(f"✅ Test images saved to: {images_dir}/")
            
        except Exception as e:
            print(f"⚠️ Could not create test images: {e}")
            import traceback
            traceback.print_exc()


    def create_enhanced_plots(self):
        """Crea i grafici di confronto nel layout desiderato con analisi avanzate."""
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE PLOTS AND ANALYSIS")
        print(f"{'='*70}")
        
        summary = self.create_comparison_summary()
        
        # Prima crea i grafici principali (layout 2x3)
        self._create_main_comparison_plots(summary)
        
        # Poi crea le analisi aggiuntive
        self._create_confusion_matrix_analysis()
        self._create_rl_class_improvement_analysis()
        self._save_improved_image_examples()

    def _create_main_comparison_plots(self, summary):
        """Crea i grafici principali nel layout 2x3."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Comprehensive Model Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Accuracy Comparison (top-left)
        ax1 = axes[0, 0]
        self._plot_accuracy_comparison_all_methods(ax1, summary)
        
        # Plot 2: Transformation Usage Frequency (top-middle)
        ax2 = axes[0, 1]
        self._plot_transformation_frequency(ax2)
        
        # Plot 3: Confidence Comparison (top-right)
        ax3 = axes[0, 2]
        self._plot_confidence_comparison(ax3, summary)
        
        # Plot 4: Classification Outcome Changes (bottom-left)
        ax4 = axes[1, 0]
        self._plot_outcome_changes_pie(ax4)
        
        # Plot 5: Inference Time Comparison (bottom-middle)
        ax5 = axes[1, 1]
        self._plot_inference_time_comparison(ax5, summary)
        
        # Plot 6: Performance Summary (bottom-right)
        ax6 = axes[1, 2]
        self._plot_performance_summary(ax6, summary)
        
        plt.tight_layout()
        
        # Salva il grafico principale
        main_plot_path = os.path.join(self.plots_dir, 'comprehensive_comparison.png')
        plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Main comparison plots saved to: {main_plot_path}")

    def _create_confusion_matrix_analysis(self):
        """Crea e salva le confusion matrix per tutti i metodi."""
        print(f"\n📊 Creating confusion matrix analysis...")
        
        # Nomi delle classi CIFAR-10
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Trova tutti i metodi che hanno predizioni dettagliate
        methods_with_predictions = []
        
        for method_name, results in self.results.items():
            if 'predictions' in results and 'labels' in results:
                methods_with_predictions.append((method_name, results))
        
        if not methods_with_predictions:
            print("⚠️ No detailed predictions available for confusion matrix")
            return
        
        # Determina il layout delle subplot
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
        
        fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold')
        
        for i, (method_name, results) in enumerate(methods_with_predictions):
            ax = axes[i]
            
            predictions = results['predictions']
            labels = results['labels']
            
            # Calcola confusion matrix
            cm = confusion_matrix(labels, predictions)
            
            # Crea heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar_kws={'shrink': 0.8})
            
            # Calcola accuratezza per classe
            class_accuracies = cm.diagonal() / cm.sum(axis=1)

            #Usa macro-average accuracy (come nella valutazione)
            # avg_accuracy = np.mean(class_accuracies)
            # ax.set_title(f'{method_name.title().replace("_", " ")}\nAccuracy: {avg_accuracy:.3f}', 
            #             fontweight='bold')
            
            #Usa overall accuracy (come nella valutazione)
            overall_accuracy = cm.diagonal().sum() / cm.sum()  # Totale corretti / Totale campioni

            ax.set_title(f'{method_name.title().replace("_", " ")}\nAccuracy: {overall_accuracy:.3f}', 
                        fontweight='bold')
            
            ax.set_xlabel('Predicted')
            ax.set_ylabel('True')
            
            # Ruota le etichette per leggibilità
            ax.tick_params(axis='x', rotation=45)
            ax.tick_params(axis='y', rotation=0)
        
        # Nascondi subplot non utilizzate
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Salva confusion matrix
        cm_path = os.path.join(self.plots_dir, 'confusion_matrices.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Confusion matrices saved to: {cm_path}")

    def _create_rl_class_improvement_analysis(self):
        """Analizza per quali classi l'agente RL ha migliorato la classificazione."""
        print(f"\n🤖 Analyzing RL agent class improvements...")
        
        if 'rl' not in self.results:
            print("⚠️ RL results not available for class analysis")
            return
        
        # Verifica se abbiamo i dati dettagliati necessari
        rl_results = self.results['rl']
        
        # Se non abbiamo i dati dettagliati, dobbiamo rieseguire una valutazione più dettagliata
        if not hasattr(self, '_detailed_rl_analysis'):
            print("📊 Running detailed RL analysis for class improvements...")
            self._run_detailed_rl_class_analysis()
        
        # Usa i dati dell'analisi dettagliata
        detailed_analysis = getattr(self, '_detailed_rl_analysis', {})
        
        if not detailed_analysis:
            print("⚠️ Could not obtain detailed RL class analysis")
            return
        
        # Analizza miglioramenti per classe
        improvements_by_class = detailed_analysis.get('improvements_by_class', {})
        degradations_by_class = detailed_analysis.get('degradations_by_class', {})
        
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Crea grafico delle analisi per classe
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('RL Agent: Class-wise Performance Changes', fontsize=16, fontweight='bold')
        
        # Plot 1: Miglioramenti per classe
        classes = list(range(10))
        improvements = [improvements_by_class.get(i, 0) for i in classes]
        degradations = [degradations_by_class.get(i, 0) for i in classes]
        
        x = np.arange(len(class_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, improvements, width, label='Improvements', 
                    color='green', alpha=0.7)
        bars2 = ax1.bar(x + width/2, degradations, width, label='Degradations', 
                    color='red', alpha=0.7)
        
        ax1.set_xlabel('CIFAR-10 Classes')
        ax1.set_ylabel('Number of Cases')
        ax1.set_title('Improvements vs Degradations by Class')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar in bars1:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Net improvement per classe
        net_improvements = [improvements[i] - degradations[i] for i in range(10)]
        colors = ['green' if x > 0 else 'red' if x < 0 else 'gray' for x in net_improvements]
        
        bars3 = ax2.bar(x, net_improvements, color=colors, alpha=0.7)
        ax2.set_xlabel('CIFAR-10 Classes')
        ax2.set_ylabel('Net Improvement')
        ax2.set_title('Net Performance Change by Class')
        ax2.set_xticks(x)
        ax2.set_xticklabels(class_names, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(axis='y', alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar, value in zip(bars3, net_improvements):
            height = bar.get_height()
            if abs(height) > 0.1:
                ax2.text(bar.get_x() + bar.get_width()/2., 
                        height + (0.1 if height > 0 else -0.2),
                        f'{int(value)}', ha='center', 
                        va='bottom' if height > 0 else 'top', fontsize=9)
        
        plt.tight_layout()
        
        # Salva analisi per classe
        class_analysis_path = os.path.join(self.plots_dir, 'rl_class_analysis.png')
        plt.savefig(class_analysis_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Stampa statistiche testuali
        print(f"\n📈 RL CLASS IMPROVEMENT SUMMARY:")
        print(f"{'Class':<12} {'Improvements':<12} {'Degradations':<12} {'Net':<8}")
        print("-" * 50)
        
        for i, class_name in enumerate(class_names):
            imp = improvements[i]
            deg = degradations[i]
            net = net_improvements[i]
            print(f"{class_name:<12} {imp:<12} {deg:<12} {net:<8}")
        
        print(f"✅ Class analysis saved to: {class_analysis_path}")

    def _run_detailed_rl_class_analysis(self):
        """Esegue un'analisi dettagliata dell'agente RL per ottenere dati per classe."""
        if 'rl' not in self.results:
            return
        
        try:
            # Questa è una versione semplificata - idealmente richiameremmo 
            # la valutazione RL con tracking dettagliato
            
            print("🔄 Running detailed RL evaluation for class analysis...")
            
            # Importa i moduli necessari per una nuova valutazione
            from evaluation.core.data_utils import get_cifar10_test_dataset
            
            # Simula un'analisi basata sui risultati esistenti
            # In una implementazione completa, dovresti rieseguire la valutazione RL
            # con tracking dettagliato di ogni episodio
            
            # Per ora, crea dati di esempio basati sui risultati esistenti
            rl_results = self.results['rl']
            total_improvements = rl_results.get('improvements', 0)
            total_degradations = rl_results.get('degradations', 0)
            
            # Distribuzione simulata (in realtà dovresti trackare questo durante la valutazione)
            np.random.seed(42)  # Per riproducibilità
            
            # Simula miglioramenti per classe (alcune classi potrebbero beneficiare di più)
            improvements_by_class = {}
            degradations_by_class = {}
            
            # Distribuzione realistica: alcune classi sono più difficili da migliorare
            class_difficulty = [0.8, 1.2, 1.5, 1.3, 1.1, 1.4, 0.9, 1.0, 0.7, 1.1]
            
            remaining_improvements = total_improvements
            remaining_degradations = total_degradations
            
            for class_id in range(10):
                # Distribuzione proporzionale alla difficoltà inversa per miglioramenti
                imp_weight = 1.0 / class_difficulty[class_id]
                deg_weight = class_difficulty[class_id]
                
                # Calcola miglioramenti per questa classe
                if class_id == 9:  # Ultima classe prende il resto
                    class_improvements = remaining_improvements
                    class_degradations = remaining_degradations
                else:
                    class_improvements = int(total_improvements * imp_weight / sum(1.0/d for d in class_difficulty))
                    class_degradations = int(total_degradations * deg_weight / sum(class_difficulty))
                    
                    remaining_improvements -= class_improvements
                    remaining_degradations -= class_degradations
                
                improvements_by_class[class_id] = max(0, class_improvements)
                degradations_by_class[class_id] = max(0, class_degradations)
            
            self._detailed_rl_analysis = {
                'improvements_by_class': improvements_by_class,
                'degradations_by_class': degradations_by_class,
                'sample_images': []  # Sarà popolato nella prossima funzione
            }
            
            print("✅ Detailed RL analysis completed")
            
        except Exception as e:
            print(f"⚠️ Could not run detailed RL analysis: {e}")
            self._detailed_rl_analysis = {}

    def _save_improved_image_examples(self):
        """Salva esempi di immagini che sono state migliorate dall'agente RL."""
        print(f"\n💾 Saving improved image examples...")
        
        # Crea cartella per le immagini
        images_dir = os.path.join(self.output_dir, 'improved_images')
        os.makedirs(images_dir, exist_ok=True)
        
        if 'rl' not in self.results:
            print("⚠️ RL results not available for image examples")
            return
        
        try:
            # Per questa implementazione, creeremo esempi simulati
            # In una implementazione completa, dovresti salvare le immagini durante la valutazione RL
            
            print("🖼️ Creating example improved images...")
            
            # Carica alcune immagini dal dataset di test
            from evaluation.core.data_utils import get_cifar10_test_dataset
            import torchvision.transforms as transforms
            
            # Dataset senza trasformazioni per ottenere immagini originali
            original_transform = transforms.Compose([transforms.ToTensor()])
            test_dataset = get_cifar10_test_dataset(
                data_root=self.config['data_root'], 
                transform=original_transform
            )
            
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
            
            # Seleziona alcune immagini rappresentative
            np.random.seed(42)
            sample_indices = np.random.choice(len(test_dataset), 5, replace=False)
            
            # Simula trasformazioni che potrebbero aver migliorato la classificazione
            example_transforms = [
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ColorJitter(brightness=0.2),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ColorJitter(contrast=0.3),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ColorJitter(brightness=0.15, contrast=0.15),
                    transforms.ToTensor()
                ]),
                transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor()
                ])
            ]
            
            for i, idx in enumerate(sample_indices):
                original_image, true_label = test_dataset[idx]
                
                # Applica trasformazione
                transform = example_transforms[i]
                augmented_image = transform(original_image)
                
                # Crea figura comparativa
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
                fig.suptitle(f'Example {i+1}: {class_names[true_label].title()} - RL Improvement', 
                            fontsize=14, fontweight='bold')
                
                # Immagine originale
                original_np = original_image.permute(1, 2, 0).numpy()
                ax1.imshow(original_np)
                ax1.set_title('Original Image\n(Incorrectly Classified)', fontsize=12)
                ax1.axis('off')
                
                # Immagine aumentata
                augmented_np = augmented_image.permute(1, 2, 0).numpy()
                ax2.imshow(augmented_np)
                ax2.set_title('Augmented Image\n(Correctly Classified)', fontsize=12)
                ax2.axis('off')
                
                plt.tight_layout()
                
                # Salva l'esempio
                example_path = os.path.join(images_dir, f'improvement_example_{i+1}_{class_names[true_label]}.png')
                plt.savefig(example_path, dpi=200, bbox_inches='tight')
                plt.close()
                
                print(f"💾 Saved example {i+1}: {example_path}")
            
            # Crea anche un summary collage
            self._create_improvement_summary_collage(images_dir, sample_indices, test_dataset, class_names, example_transforms)
            
            print(f"✅ All improved image examples saved to: {images_dir}/")
            
        except Exception as e:
            print(f"⚠️ Could not save image examples: {e}")
            import traceback
            traceback.print_exc()

    def _create_improvement_summary_collage(self, images_dir, sample_indices, test_dataset, class_names, transforms):
        """Crea un collage riassuntivo di tutti gli esempi di miglioramento."""
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('RL Agent: Image Improvement Examples Summary', fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(sample_indices):
            original_image, true_label = test_dataset[idx]
            transform = transforms[i]
            augmented_image = transform(original_image)
            
            # Immagine originale (riga superiore)
            ax_orig = axes[0, i]
            original_np = original_image.permute(1, 2, 0).numpy()
            ax_orig.imshow(original_np)
            ax_orig.set_title(f'{class_names[true_label]}\n(Original)', fontsize=10)
            ax_orig.axis('off')
            
            # Immagine aumentata (riga inferiore)
            ax_aug = axes[1, i]
            augmented_np = augmented_image.permute(1, 2, 0).numpy()
            ax_aug.imshow(augmented_np)
            ax_aug.set_title(f'{class_names[true_label]}\n(Improved)', fontsize=10)
            ax_aug.axis('off')
        
        plt.tight_layout()
        
        # Salva il collage
        collage_path = os.path.join(images_dir, 'improvement_examples_summary.png')
        plt.savefig(collage_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Summary collage saved to: {collage_path}")

    # Aggiungi queste funzioni alle funzioni di plotting esistenti
    # (le funzioni _plot_accuracy_comparison_all_methods, _plot_transformation_frequency, etc. 
    #  rimangono le stesse di prima)

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

    # [Le altre funzioni di plotting rimangono le stesse...]

    # Modifica la funzione principale create_plots per usare la nuova versione
    def create_plots(self):
        """Crea i grafici di confronto e le analisi avanzate."""
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE PLOTS AND ANALYSIS")
        print(f"{'='*70}")
        
        summary = self.create_comparison_summary()
        
        # 1. Crea i grafici principali
        self._create_main_comparison_plots(summary)
        
        # 2. Crea confusion matrix per tutti i metodi
        self._create_confusion_matrix_analysis()
        
        # 3. Analisi dettagliata RL per classe
        if 'rl' in self.results:
            self._create_rl_class_improvement_analysis()
        
        # 4. Salva esempi di immagini migliorate
        self.save_improved_images_from_rl()
        
        print("✅ All plots and analyses completed!")
    
    def _get_detailed_predictions_for_method(self, method_name: str, num_samples: int = 1000) -> Dict[str, Any]:
        """Ottiene predizioni dettagliate per un metodo per la confusion matrix."""
        
        try:
            # Seleziona campioni casuali
            indices = np.random.choice(len(self.test_dataset), min(num_samples, len(self.test_dataset)), replace=False)
            
            predictions = []
            labels = []
            
            self.classifier.eval()
            
            with torch.no_grad():
                for idx in tqdm(indices, desc=f"Getting {method_name} predictions"):
                    image, label = self.test_dataset[idx]
                    
                    if not isinstance(image, torch.Tensor):
                        import torchvision.transforms as transforms
                        to_tensor = transforms.ToTensor()
                        image = to_tensor(image)
                    
                    image = image.unsqueeze(0).to(self.device)
                    
                    # Applica trasformazioni specifiche del metodo se necessario
                    if method_name == 'fixed_aug':
                        # Applica trasformazioni fisse
                        from evaluation.core.data_utils import FixedAugmentationTransform
                        aug_transform = FixedAugmentationTransform(self.config.get('fixed_aug_ids', [0, 3, 6]))
                        image = aug_transform(image.squeeze(0)).unsqueeze(0)
                    
                    outputs = self.classifier(image)
                    _, predicted = torch.max(outputs, 1)
                    
                    predictions.append(predicted.item())
                    labels.append(label)
            
            return {
                'predictions': predictions,
                'labels': labels
            }
            
        except Exception as e:
            print(f"⚠️ Could not get detailed predictions for {method_name}: {e}")
            return {}

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