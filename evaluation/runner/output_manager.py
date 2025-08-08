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


class OutputManager:
    """Manages output generation and formatting for evaluation results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / 'plots'
        self.images_dir = self.output_dir / 'improved_images'
        self.reports_dir = self.output_dir / 'reports'
        
        # Create directories
        for directory in [self.output_dir, self.plots_dir, self.images_dir, self.reports_dir]:
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
            json.dump(summary, f, indent=4, default=str)
        
        # Generate text report
        self._generate_text_report(results, config, timestamp)
        
        print(f" Results saved:")
        print(f"   Complete data: {pickle_file}")
        print(f"   Summary: {json_file}")
        print(f"   Report: {self.reports_dir}/report_{timestamp}.txt")
    
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
        
        # Extract key metrics for each method
        if 'baseline' in results:
            summary['performance']['baseline'] = {
                'accuracy': results['baseline']['accuracy'],
                'avg_confidence': results['baseline'].get('avg_confidence'),
                'time_per_sample': results['baseline'].get('time_per_sample')
            }
        
        if 'fixed_aug' in results:
            summary['performance']['fixed_augmentation'] = {
                'accuracy': results['fixed_aug']['augmented_accuracy'],
                'improvement': results['fixed_aug']['accuracy_improvement'],
                'time_per_sample': results['fixed_aug'].get('time_per_sample')
            }
        
        if 'tta' in results:
            summary['performance']['tta'] = {
                'accuracy': results['tta']['tta_accuracy'],
                'improvement': results['tta']['accuracy_improvement'],
                'time_per_sample': results['tta'].get('time_per_sample'),
                'num_augmentations': results['tta'].get('num_augmentations')
            }
        
        if 'rl' in results:
            summary['performance']['rl_agent'] = {
                'accuracy': results['rl']['final_accuracy'],
                'improvement': results['rl']['accuracy_improvement'],
                'avg_reward': results['rl'].get('avg_reward'),
                'time_per_sample': results['rl'].get('time_per_sample'),
                'model_loaded': results['rl'].get('model_loaded', False)
            }
        
        # Determine best method
        improvements = {}
        for method, data in summary['performance'].items():
            if 'improvement' in data:
                improvements[method] = data['improvement']
        
        if improvements:
            best_method = max(improvements, key=improvements.get)
            summary['analysis'] = {
                'best_method': best_method,
                'best_improvement': improvements[best_method],
                'all_improvements': improvements
            }
        
        return summary
    
    def _generate_text_report(self, results: Dict[str, Any], config: Dict[str, Any], timestamp: str) -> None:
        """Generate a comprehensive text report."""
        report_file = self.reports_dir / f'report_{timestamp}.txt'
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            # Metadata
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {config.get('output_dir', 'Unknown')}\n")
            f.write(f"Methods Evaluated: {', '.join(results.keys())}\n\n")
            
            # Performance summary
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            
            for method_name, method_results in results.items():
                f.write(f"\n{method_name.upper().replace('_', ' ')}:\n")
                
                if method_name == 'baseline':
                    f.write(f"  Accuracy: {method_results['accuracy']:.4f}\n")
                    f.write(f"  Avg Confidence: {method_results.get('avg_confidence', 'N/A'):.4f}\n")
                
                elif method_name == 'fixed_aug':
                    f.write(f"  Accuracy: {method_results['augmented_accuracy']:.4f}\n")
                    f.write(f"  Improvement: {method_results['accuracy_improvement']:+.4f}\n")
                
                elif method_name == 'tta':
                    f.write(f"  Accuracy: {method_results['tta_accuracy']:.4f}\n")
                    f.write(f"  Improvement: {method_results['accuracy_improvement']:+.4f}\n")
                    f.write(f"  Augmentations Used: {method_results.get('num_augmentations', 'N/A')}\n")
                
                elif method_name == 'rl':
                    f.write(f"  Accuracy: {method_results['final_accuracy']:.4f}\n")
                    f.write(f"  Improvement: {method_results['accuracy_improvement']:+.4f}\n")
                    f.write(f"  Avg Reward: {method_results.get('avg_reward', 'N/A'):.3f}\n")
                    f.write(f"  Model Loaded: {'Yes' if method_results.get('model_loaded', False) else 'No (Random)'}\n")
                
                # Timing info
                time_per_sample = method_results.get('time_per_sample')
                if time_per_sample:
                    f.write(f"  Time per Sample: {time_per_sample * 1000:.1f}ms\n")
            
            # Recommendations
            f.write(f"\n\nRECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            f.write(self._generate_recommendations(results))
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Generate method recommendations based on results."""
        recommendations = []
        
        # Calculate improvements
        improvements = {}
        if 'fixed_aug' in results:
            improvements['Fixed Augmentation'] = results['fixed_aug']['accuracy_improvement']
        if 'tta' in results:
            improvements['TTA'] = results['tta']['accuracy_improvement']
        if 'rl' in results:
            improvements['RL Agent'] = results['rl']['accuracy_improvement']
        
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
        print(f"    Images: {self.images_dir}/")
        print(f"   Reports: {self.reports_dir}/")
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


class ResultsFormatter:
    """Utility class for formatting evaluation results for display."""
    
    @staticmethod
    def format_accuracy_comparison(results: Dict[str, Any]) -> str:
        """Format accuracy comparison as a table."""
        lines = [" ACCURACY COMPARISON:", "-" * 40]
        
        if 'baseline' in results:
            acc = results['baseline']['accuracy']
            lines.append(f"Baseline:        {acc:.4f}")
        
        if 'fixed_aug' in results:
            acc = results['fixed_aug']['augmented_accuracy']
            imp = results['fixed_aug']['accuracy_improvement']
            lines.append(f"Fixed Aug:       {acc:.4f} ({imp:+.4f})")
        
        if 'tta' in results:
            acc = results['tta']['tta_accuracy']
            imp = results['tta']['accuracy_improvement']
            lines.append(f"TTA:             {acc:.4f} ({imp:+.4f})")
        
        if 'rl' in results:
            acc = results['rl']['final_accuracy']
            imp = results['rl']['accuracy_improvement']
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
                time_ms = results[method].get('time_per_sample', 0) * 1000
                slowdown = f"({time_ms/baseline_time:.1f}×)" if baseline_time and baseline_time > 0 else ""
                method_name = method.replace('_', ' ').title()
                lines.append(f"{method_name:12}: {time_ms:.1f}ms {slowdown}")
        
        return "\n".join(lines)
    
    @staticmethod
    def format_improvement_summary(results: Dict[str, Any]) -> str:
        """Format improvement summary with recommendations."""
        lines = ["💡 IMPROVEMENT SUMMARY:", "-" * 40]
        
        improvements = {}
        for method in ['fixed_aug', 'tta', 'rl']:
            if method in results:
                if method == 'fixed_aug':
                    improvements['Fixed Aug'] = results[method]['accuracy_improvement']
                elif method == 'tta':
                    improvements['TTA'] = results[method]['accuracy_improvement']
                elif method == 'rl':
                    improvements['RL Agent'] = results[method]['accuracy_improvement']
        
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