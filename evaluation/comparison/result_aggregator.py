"""
Result Aggregator for evaluation comparisons.
Handles all result processing, aggregation, and summary generation.
"""

import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple


class ResultAggregator:
    """
    Handles aggregation and analysis of evaluation results.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the result aggregator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def create_comprehensive_summary(self, results: Dict[str, Any], device) -> Dict[str, Any]:
        """
        Create a comprehensive summary of all evaluation results.
        
        Args:
            results: Dictionary containing results from all evaluation methods
            device: Device used for evaluation
            
        Returns:
            Comprehensive summary dictionary
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'device': str(device),
            'methods_evaluated': list(results.keys())
        }
        
        # Extract baseline accuracy as reference
        baseline_accuracy = 0.0
        if 'baseline' in results:
            baseline_accuracy = results['baseline']['accuracy']
            summary['baseline_accuracy'] = baseline_accuracy
        
        # Compare accuracies and calculate improvements
        accuracy_comparison = {}
        improvement_comparison = {}
        confidence_comparison = {}
        
        for method, method_results in results.items():
            accuracy = method_results.get('accuracy', 0.0)
            confidence = self._extract_confidence(method, method_results)
            
            accuracy_comparison[method] = float(accuracy)
            confidence_comparison[method] = float(confidence) if confidence is not None else 0.0
            improvement_comparison[method] = float(accuracy - baseline_accuracy)
        
        summary['accuracy_comparison'] = accuracy_comparison
        summary['improvement_comparison'] = improvement_comparison
        summary['confidence_comparison'] = confidence_comparison
        
        # Find best method
        if improvement_comparison:
            best_method = max(improvement_comparison.items(), key=lambda x: x[1])
            summary['best_method'] = best_method[0]
            summary['best_improvement'] = best_method[1]
        
        # Compare timing
        time_comparison = {}
        for method, method_results in results.items():
            if 'time_per_sample' in method_results:
                time_comparison[method] = float(method_results['time_per_sample'] * 1000)  # in ms
                
        summary['time_comparison'] = time_comparison
        
        # Add method-specific analyses
        summary['method_analyses'] = self._create_method_analyses(results)
        
        return summary
    
    def _extract_confidence(self, method: str, method_results: Dict[str, Any]) -> float:
        """
        Extract confidence value for a specific method, handling different result structures.
        
        Args:
            method: Method name
            method_results: Results dictionary for the method
            
        Returns:
            Confidence value or None if not available
        """
        if method == 'baseline':
            return method_results.get('avg_confidence')
        elif method == 'fixed_aug':
            return method_results.get('avg_confidence')
        elif method == 'tta':
            return method_results.get('avg_confidence')
        elif method == 'rl':
            return method_results.get('avg_confidence')
        
        # Fallback to generic avg_confidence
        return method_results.get('avg_confidence')
    
    def _create_method_analyses(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create detailed analyses for each method.
        
        Args:
            results: Complete results dictionary
            
        Returns:
            Method-specific analyses
        """
        analyses = {}
        
        # Fixed Augmentation Analysis
        if 'fixed_aug' in results:
            fixed_results = results['fixed_aug']
            analyses['fixed_augmentation'] = {
                'transformations_used': fixed_results.get('augmentation_names', []),
                'transformation_ids': fixed_results.get('augmentation_ids', []),
                'total_samples': fixed_results.get('total_samples', 0)
            }
        
        # TTA Analysis
        if 'tta' in results:
            tta_results = results['tta']
            analyses['tta'] = {
                'num_augmentations': tta_results.get('num_augmentations', 0),
                'method_used': tta_results.get('method_used', 'unknown'),
                'samples_evaluated': tta_results.get('total_samples_evaluated', 0)
            }
        
        # RL Agent Analysis
        if 'rl' in results:
            rl_results = results['rl']
            analyses['rl_agent'] = {
                'episodes_evaluated': rl_results.get('num_episodes_evaluated', 0),
                'model_loaded': rl_results.get('model_loaded', False),
                'avg_reward': rl_results.get('avg_reward', 0.0),
                'improvements': rl_results.get('improvements', 0),
                'degradations': rl_results.get('degradations', 0),
                'net_improvement_rate': rl_results.get('net_improvement_rate', 0.0),
                'action_distribution': rl_results.get('action_counts', {}),
                'avg_sequence_length': rl_results.get('avg_sequence_length', 0.0)
            }
        
        return analyses
    
    def print_comprehensive_summary(self, summary: Dict[str, Any], results: Dict[str, Any]) -> None:
        """
        Print a comprehensive formatted summary.
        
        Args:
            summary: Summary dictionary
            results: Complete results dictionary
        """
        print(f"\n{'='*70}")
        print("COMPREHENSIVE COMPARISON SUMMARY")
        print(f"{'='*70}")
        
        print(f"METHODS EVALUATED: {', '.join(summary['methods_evaluated'])}")
        
        # Accuracy comparison table
        if 'accuracy_comparison' in summary and 'baseline' in results:
            self._print_accuracy_table(summary)
        
        # Best method
        if 'best_method' in summary and summary['best_method'] != 'baseline':
            best_method = summary['best_method'].upper().replace('_', ' ')
            print(f"\n🏆 BEST METHOD: {best_method}")
            print(f"   Improvement over baseline: {summary['best_improvement']:+.4f}")
        
        # Timing comparison
        if 'time_comparison' in summary:
            self._print_timing_table(summary)
                
        # Method-specific details
        self._print_method_specific_details(summary, results)
    
    def _print_accuracy_table(self, summary: Dict[str, Any]) -> None:
        """Print formatted accuracy comparison table."""
        baseline_acc = summary['baseline_accuracy']
        
        print(f"\nACCURACY COMPARISON:")
        print(f"{'Method':<15} {'Accuracy':<10} {'vs Baseline':<12} {'Confidence':<12}")
        print("-" * 60)
        
        for method in ['baseline', 'fixed_aug', 'tta', 'rl']:
            if method in summary['accuracy_comparison']:
                acc = summary['accuracy_comparison'][method]
                improvement = summary['improvement_comparison'][method]
                confidence = summary['confidence_comparison'].get(method, 0)
                
                method_name = method.upper().replace('_', ' ')
                improvement_str = "baseline" if method == 'baseline' else f"{improvement:+.4f}"
                
                print(f"{method_name:<15} {acc:.4f}     {improvement_str:<12} {confidence:.4f}")
    
    def _print_timing_table(self, summary: Dict[str, Any]) -> None:
        """Print formatted timing comparison table."""
        print(f"\nTIMING COMPARISON (ms per sample):")
        for method, time_ms in summary['time_comparison'].items():
            method_name = method.upper().replace('_', ' ')
            print(f"  {method_name:<15}: {time_ms:.1f}ms")
    
    def _print_method_specific_details(self, summary: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Print method-specific details."""
        analyses = summary.get('method_analyses', {})
        
        if 'fixed_augmentation' in analyses:
            analysis = analyses['fixed_augmentation']
            print(f"\n📝 FIXED AUGMENTATION DETAILS:")
            transformations = analysis.get('transformations_used', [])
            print(f"   Transformations: {', '.join(transformations[:3])}{'...' if len(transformations) > 3 else ''}")
            
        if 'tta' in analyses:
            analysis = analyses['tta']
            print(f"\n🔄 TTA DETAILS:")
            print(f"   Augmentations used: {analysis.get('num_augmentations', 'N/A')}")
            print(f"   Method: {analysis.get('method_used', 'N/A')}")
            print(f"   Samples evaluated: {analysis.get('samples_evaluated', 'N/A')}")
            
        if 'rl_agent' in analyses:
            analysis = analyses['rl_agent']
            print(f"\n🤖 RL AGENT DETAILS:")
            print(f"   Episodes evaluated: {analysis.get('episodes_evaluated', 'N/A')}")
            print(f"   Model loaded: {'✅ OK' if analysis.get('model_loaded', False) else '❌ Random'}")
            print(f"   Average reward: {analysis.get('avg_reward', 0):.3f}")
            print(f"   Improvements: {analysis.get('improvements', 0)}")
            print(f"   Degradations: {analysis.get('degradations', 0)}")
            net_rate = analysis.get('net_improvement_rate', 0)
            print(f"   Net improvement rate: {net_rate:.1%}")
    
    def calculate_statistical_significance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate statistical significance of improvements (if detailed data available).
        
        Args:
            results: Complete results dictionary
            
        Returns:
            Statistical significance analysis
        """
        significance_analysis = {}
        
        # This would require access to individual predictions to perform proper statistical tests
        # For now, return basic analysis based on available metrics
        
        baseline_accuracy = results.get('baseline', {}).get('accuracy', 0)
        
        for method, method_results in results.items():
            if method == 'baseline':
                continue
                
            method_accuracy = method_results.get('accuracy', 0)
            improvement = method_accuracy - baseline_accuracy
            
            # Simple significance assessment based on improvement magnitude
            # In a full implementation, you'd use actual statistical tests
            if improvement > 0.01:
                significance = "highly_significant"
            elif improvement > 0.005:
                significance = "significant"
            elif improvement > 0.001:
                significance = "marginally_significant"
            else:
                significance = "not_significant"
            
            significance_analysis[method] = {
                'improvement': improvement,
                'significance_level': significance,
                'baseline_accuracy': baseline_accuracy,
                'method_accuracy': method_accuracy
            }
        
        return significance_analysis
    
    def create_performance_ranking(self, results: Dict[str, Any]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Create a performance ranking of all methods.
        
        Args:
            results: Complete results dictionary
            
        Returns:
            List of tuples: (method_name, score, details)
        """
        rankings = []
        baseline_accuracy = results.get('baseline', {}).get('accuracy', 0)
        
        for method, method_results in results.items():
            accuracy = method_results.get('accuracy', 0)
            confidence = self._extract_confidence(method, method_results)
            time_per_sample = method_results.get('time_per_sample', 0)
            
            # Calculate composite score (weighted)
            accuracy_score = accuracy * 0.7  # 70% weight on accuracy
            confidence_score = (confidence or 0) * 0.2  # 20% weight on confidence
            efficiency_score = max(0, (0.01 - time_per_sample) / 0.01) * 0.1  # 10% weight on speed
            
            composite_score = accuracy_score + confidence_score + efficiency_score
            
            details = {
                'accuracy': accuracy,
                'confidence': confidence,
                'time_per_sample': time_per_sample,
                'improvement_over_baseline': accuracy - baseline_accuracy,
                'composite_score': composite_score
            }
            
            rankings.append((method, composite_score, details))
        
        # Sort by composite score (descending)
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings