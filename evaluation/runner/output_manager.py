#!/usr/bin/env python3
"""
Output Manager for Evaluation Results
=====================================

Manages all output formatting, file saving, and result presentation.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


class OutputManager:
    """Manages output generation and formatting for evaluation results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        
        # Create directories
        for directory in [self.output_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_results(self, results: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Save evaluation results in multiple formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as pickle (complete data)
        pickle_file = self.output_dir / f'results_{timestamp}.pkl'
        with open(pickle_file, 'wb') as f:
            pickle.dump({'results': results, 'config': config}, f)
        
        # Save summary as JSON
        summary = self._create_summary(results, config)
        json_file = self.output_dir / f'summary_{timestamp}.json'
        with open(json_file, 'w') as f:
            # Use a custom encoder to handle numpy types
            json.dump(summary, f, indent=4, cls=NpEncoder)
        
        # Generate text report
        self._generate_text_report(results, config, timestamp)
        
        print(f" Results saved:")
        print(f"   Complete data: {pickle_file}")
        print(f"   Summary: {json_file}")
    
    def _create_summary(self, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a JSON-serializable summary of results."""
        summary = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config': config,
                'methods_evaluated': list(results.keys())
            },
            'performance': {}
        }
        
        # Extract key metrics for each method based on comparison_runner's output
        if 'baseline' in results:
            summary['performance']['baseline'] = {
                'accuracy': results['baseline']['accuracy'],
                'avg_confidence': results['baseline'].get('avg_confidence'),
                'time_per_sample': results['baseline'].get('time_per_sample')
            }
        
        if 'fixed_aug' in results:
            # comparison_runner uses 'accuracy' for augmented, and calculates improvement later
            summary['performance']['fixed_augmentation'] = {
                'accuracy': results['fixed_aug']['accuracy'],
                'avg_confidence': results['fixed_aug']['avg_confidence'],
                'time_per_sample': results['fixed_aug'].get('time_per_sample')
            }
        
        if 'tta' in results:
            # comparison_runner uses 'accuracy' and 'tta_avg_confidence'
            summary['performance']['tta'] = {
                'accuracy': results['tta']['accuracy'],
                'avg_confidence': results['tta']['tta_avg_confidence'],
                'time_per_sample': results['tta'].get('time_per_sample'),
                'num_augmentations': results['tta'].get('num_augmentations')
            }
        
        if 'rl' in results:
            # comparison_runner uses 'accuracy' and 'final_avg_confidence'
            summary['performance']['rl_agent'] = {
                'accuracy': results['rl']['accuracy'],
                'avg_confidence': results['rl']['final_avg_confidence'],
                'avg_reward': results['rl'].get('avg_reward'),
                'time_per_sample': results['rl'].get('time_per_sample'),
                'model_loaded': results['rl'].get('model_loaded', False)
            }
        
        # Determine best method
        improvements = {}
        baseline_acc = summary['performance'].get('baseline', {}).get('accuracy', 0)
        
        for method, data in summary['performance'].items():
            if method != 'baseline':
                accuracy = data.get('accuracy', 0)
                improvement = accuracy - baseline_acc
                data['improvement'] = improvement # Add improvement to the summary for clarity
                improvements[method] = improvement
        
        if improvements:
            best_method = max(improvements, key=improvements.get)
            summary['analysis'] = {
                'best_method': best_method,
                'best_improvement': improvements[best_method],
                'all_improvements': improvements
            }
        
        return summary
    
    def _generate_text_report(self, results: Dict[str, Any], config: Dict[str, Any], timestamp: str) -> None:
        """Generate a human-readable text report."""
        report_path = self.output_dir / f'report_{timestamp}.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"Comprehensive Evaluation Report - {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            
            f.write("--- CONFIGURATION ---\n")
            f.write(json.dumps(config, indent=4, cls=NpEncoder) + "\n\n")
            
            f.write("--- RESULTS SUMMARY ---\n")
            summary = self._create_summary(results, config)
            f.write(json.dumps(summary, indent=4, cls=NpEncoder) + "\n\n")
            
            f.write("--- FORMATTED COMPARISON ---\n")
            f.write(ResultsFormatter.format_accuracy_comparison(results) + "\n\n")
            f.write(ResultsFormatter.format_timing_comparison(results) + "\n\n")
            f.write(ResultsFormatter.format_improvement_summary(results) + "\n\n")
            
            f.write("--- RECOMMENDATIONS ---\n")
            f.write(self._generate_recommendations(results) + "\n")
        
        print(f"   Text report: {report_path}")

    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate method recommendations based on results."""
        recommendations = []
        
        # Calculate improvements
        improvements = {}
        baseline_acc = results.get('baseline', {}).get('accuracy', 0)
        
        if 'fixed_aug' in results:
            improvements['Fixed Augmentation'] = results['fixed_aug']['accuracy'] - baseline_acc
        if 'tta' in results:
            improvements['TTA'] = results['tta']['accuracy'] - baseline_acc
        if 'rl' in results:
            improvements['RL Agent'] = results['rl']['accuracy'] - baseline_acc
        
        # Find best method
        if improvements:
            best_method = max(improvements, key=improvements.get)
            best_improvement = improvements[best_method]
            
            recommendations.append(f"Best performing method: {best_method}")
            recommendations.append(f"Best improvement: {best_improvement:+.4f}")
            recommendations.append("")
            
            # Method-specific recommendations
            for method, improvement in improvements.items():
                if improvement > 0.01:
                    recommendations.append(f" {method}: Highly recommended (substantial gain)")
                elif improvement > 0.005:
                    recommendations.append(f"  {method}: Consider for critical accuracy scenarios")
                elif improvement > 0:
                    recommendations.append(f" {method}: Limited practical benefit")
                else:
                    recommendations.append(f" {method}: Not recommended (no improvement)")
        else:
            recommendations.append("No improvement data available for comparison")
        
        return "\n".join(recommendations)
    
    def print_file_locations(self) -> None:
        """Print the locations of generated files."""
        print(f"\n OUTPUT LOCATIONS:")
        print(f"   Plots: {self.plots_dir}/")
        print(f"   Raw Data: {self.output_dir}/*.pkl")
    
    def clean_old_results(self, keep_latest: int = 5) -> None:
        """Clean up old result files, keeping only the most recent ones."""
        patterns = ['results_*.pkl', 'summary_*.json', 'report_*.txt']
        
        for pattern in patterns:
            files = list(self.output_dir.glob(pattern))
            if len(files) > keep_latest:
                # Sort by modification time, newest first
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # Remove older files
                for old_file in files[keep_latest:]:
                    try:
                        old_file.unlink()
                        print(f"  Cleaned up old file: {old_file.name}")
                    except Exception as e:
                        print(f"  Could not remove {old_file.name}: {e}")

class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class ResultsFormatter:
    """Utility class for formatting evaluation results for display."""
    
    @staticmethod
    def format_accuracy_comparison(results: Dict[str, Any]) -> str:
        """Format accuracy comparison as a table."""
        lines = [" ACCURACY COMPARISON:", "-" * 40]
        
        baseline_acc = results.get('baseline', {}).get('accuracy', 0)
        if baseline_acc:
            lines.append(f"Baseline:        {baseline_acc:.4f}")
        
        if 'fixed_aug' in results:
            acc = results['fixed_aug']['accuracy']
            imp = acc - baseline_acc
            lines.append(f"Fixed Aug:       {acc:.4f} ({imp:+.4f})")
        
        if 'tta' in results:
            acc = results['tta']['accuracy']
            imp = acc - baseline_acc
            lines.append(f"TTA:             {acc:.4f} ({imp:+.4f})")
        
        if 'rl' in results:
            acc = results['rl']['accuracy']
            imp = acc - baseline_acc
            loaded = "OK" if results['rl'].get('model_loaded', False) else "NO"
            lines.append(f"RL Agent:        {acc:.4f} ({imp:+.4f}) {loaded}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_timing_comparison(results: Dict[str, Any]) -> str:
        """Format timing comparison as a table."""
        lines = [" TIMING COMPARISON:", "-" * 40]
        
        baseline_time = None
        if 'baseline' in results:
            time_ms = results['baseline'].get('time_per_sample', 0) * 1000
            baseline_time = time_ms
            lines.append(f"Baseline:        {time_ms:.1f}ms")
        
        for method in ['fixed_aug', 'tta', 'rl']:
            if method in results:
                time_per_sample = results[method].get('time_per_sample', 0)
                time_ms = time_per_sample * 1000
                slowdown = f"({time_ms/baseline_time:.1f}×)" if baseline_time and baseline_time > 0 else ""
                method_name = method.replace('_', ' ').title()
                lines.append(f"{method_name:12}: {time_ms:.1f}ms {slowdown}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_improvement_summary(results: Dict[str, Any]) -> str:
        """Format improvement summary with recommendations."""
        lines = ["💡 IMPROVEMENT SUMMARY:", "-" * 40]
        
        improvements = {}
        baseline_acc = results.get('baseline', {}).get('accuracy', 0)
        
        if 'fixed_aug' in results:
            improvements['Fixed Aug'] = results['fixed_aug']['accuracy'] - baseline_acc
        if 'tta' in results:
            improvements['TTA'] = results['tta']['accuracy'] - baseline_acc
        if 'rl' in results:
            improvements['RL Agent'] = results['rl']['accuracy'] - baseline_acc
        
        if improvements:
            # Sort by improvement
            sorted_methods = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
            
            for i, (method, improvement) in enumerate(sorted_methods, 1):
                icon = "🏆" if i == 1 else "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                lines.append(f"{icon} {i}. {method}: {improvement:+.4f}")
            
            # Best method recommendation
            best_method, best_improvement = sorted_methods[0]
            if best_improvement > 0.01:
                lines.append(f"\n Recommendation: Use {best_method} (significant improvement)")
            elif best_improvement > 0.005:
                lines.append(f"\n  Recommendation: Consider {best_method} for critical scenarios")
            elif best_improvement > 0:
                lines.append(f"\n Recommendation: {best_method} shows minimal benefit")
            else:
                lines.append(f"\n Recommendation: No method shows clear improvement")
        else:
            lines.append("No improvement data available")
        
        return "\n".join(lines)