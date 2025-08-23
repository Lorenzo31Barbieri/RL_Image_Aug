"""
Fixed comparison runner with adaptive RL agent support.
Handles dimension mismatches automatically.
"""

import torch
import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any

from evaluation.core.model_loader import load_classifier, load_rl_agent, print_loading_summary
from evaluation.core.data_utils import get_cifar10_test_dataset, get_cifar10_test_loader, print_data_loading_summary

from evaluation.methods.evaluate_baseline import evaluate_baseline
from evaluation.methods.evaluate_fixed_aug import evaluate_fixed_augmentation
from evaluation.methods.evaluate_tta import evaluate_tta
from evaluation.methods.evaluate_rl_agent import evaluate_rl_agent, test_agent_environment_compatibility

from .result_aggregator import ResultAggregator
from .visualization_manager import VisualizationManager


class EvaluationComparison:
    """
    Main orchestrator for running and comparing all evaluation methods.
    Now with adaptive RL agent support to handle dimension mismatches.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluation comparison system.
        
        Args:
            config: Configuration dictionary with paths and parameters
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {}
        
        # Setup output directories
        self.output_dir = config.get('output_dir', './evaluation_results')
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Initialize specialized managers
        self.result_aggregator = ResultAggregator(config)
        self.visualization_manager = VisualizationManager(self.plots_dir)
        
        print(f"Evaluation Comparison initialized with adaptive RL support")
        print(f"Output directory: {self.output_dir}")
        print(f"Device: {self.device}")
    
    def load_models(self) -> None:
        """Load all required models with dimension detection."""
        print(f"\n{'='*60}")
        print("LOADING MODELS WITH ADAPTIVE SUPPORT")
        print(f"{'='*60}")
        
        # Load classifier
        self.classifier = load_classifier(
            model_path=self.config['classifier_path'],
            device=self.device
        )
        
        # Load RL agent with automatic dimension detection
        self.agent = None
        self.rl_model_loaded = False
        
        if self.config.get('evaluate_rl', True):
            try:
                self.agent, self.rl_model_loaded = load_rl_agent(
                    model_path=self.config['rl_model_path'],
                    state_dim=self.config.get('state_dim'),  # Can be None for auto-detection
                    device=self.device
                )
                
                # Test compatibility if agent was loaded
                if self.agent and self.rl_model_loaded:
                    compatibility = test_agent_environment_compatibility(
                        self.agent, self.classifier, self.device
                    )
                    
                    if not compatibility['compatible']:
                        print(f"⚠️ Agent-environment compatibility issue detected:")
                        print(f"   Error: {compatibility['error']}")
                        print(f"   Will attempt adaptive evaluation...")
                    else:
                        print(f"✅ Agent-environment compatibility confirmed")
                
            except Exception as e:
                print(f"Could not load RL agent: {e}")
                print("RL evaluation will be skipped.")
                self.config['evaluate_rl'] = False
        
        # Print summary
        print_loading_summary(
            classifier=self.classifier,
            agent=self.agent,
            agent_loaded=self.rl_model_loaded
        )
    
    def load_data(self) -> None:
        """Load test data."""
        print(f"\n{'='*60}")
        print("LOADING DATA")
        print(f"{'='*60}")
        
        try:
            # Dataset for evaluations requiring individual images
            self.test_dataset = get_cifar10_test_dataset(
                data_root=self.config['data_root']
            )
            
            # DataLoader for baseline evaluation
            batch_size = self.config.get('batch_size', 64)
            self.test_loader = get_cifar10_test_loader(
                data_root=self.config['data_root'],
                batch_size=batch_size
            )
            
            # Dataset info for summary
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
            print("Data loaded successfully")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def run_baseline_evaluation(self) -> None:
        """Run baseline evaluation with configurable samples."""
        print(f"\nRunning baseline evaluation...")
        
        num_samples = self.config.get('baseline_samples')
        if num_samples and num_samples < len(self.test_dataset):
            print(f"Using {num_samples} samples for baseline evaluation")
        
        self.results['baseline'] = evaluate_baseline(
            classifier_model=self.classifier,
            test_loader=self.test_loader if num_samples is None else None,
            test_dataset=self.test_dataset,
            device=self.device,
            num_samples=num_samples,
            batch_size=self.config.get('batch_size', 64),
            verbose=True,
            return_details=True
        )
        
        print(f"Baseline evaluation completed")
    
    def run_fixed_augmentation_evaluation(self) -> None:
        """Run fixed augmentation evaluation with configurable samples."""
        if not self.config.get('evaluate_fixed_aug', True):
            print("Skipping fixed augmentation evaluation")
            return
        
        print(f"\nRunning fixed augmentation evaluation...")
        
        augmentation_ids = self.config.get('fixed_aug_ids', [0, 3, 6])
        num_samples = self.config.get('fixed_aug_samples')
        
        if num_samples and num_samples < len(self.test_dataset):
            print(f"Using {num_samples} samples for fixed augmentation evaluation")
        
        results = evaluate_fixed_augmentation(
            classifier_model=self.classifier,
            test_dataset=self.test_dataset,
            augmentation_ids=augmentation_ids,
            device=self.device,
            num_samples=num_samples,
            batch_size=self.config.get('batch_size', 64),
            verbose=True
        )
        
        self.results['fixed_aug'] = results
        print(f"Fixed augmentation evaluation completed")

    def run_tta_evaluation(self) -> None:
        """Run TTA evaluation with detailed data."""
        if not self.config.get('evaluate_tta', True):
            print("Skipping TTA evaluation")
            return
        
        print(f"\nRunning TTA evaluation...")
        
        num_samples = self.config.get('tta_samples', 1000)
        
        results = evaluate_tta(
            classifier_model=self.classifier,
            test_dataset=self.test_dataset,
            device=self.device,
            num_samples=num_samples,
            use_ttach=self.config.get('use_ttach', True),
            verbose=True
        )
        
        self.results['tta'] = results
        print(f"TTA evaluation completed")
    
    def run_rl_evaluation(self) -> None:
        """Run RL agent evaluation with adaptive dimension handling."""
        if not self.config.get('evaluate_rl', True) or self.agent is None:
            print("Skipping RL evaluation")
            return
        
        print(f"\nRunning adaptive RL agent evaluation...")
        
        num_episodes = self.config.get('rl_episodes', 1000)
        
        try:
            # Get agent's dimension information
            agent_state_dim = getattr(self.agent, 'state_dim', None)
            detected_state_dim = getattr(self.agent, 'detected_state_dim', None)
            detected_image_features = getattr(self.agent, 'detected_image_feature_dim', 0)
            
            print(f"Agent configuration:")
            print(f"  State dimension: {agent_state_dim}")
            print(f"  Detected state dimension: {detected_state_dim}")
            print(f"  Detected image features: {detected_image_features}")
            
            results = evaluate_rl_agent(
                agent=self.agent,
                classifier_model=self.classifier,
                test_dataset=self.test_dataset,
                device=self.device,
                num_episodes=num_episodes,
                max_steps_per_episode=self.config.get('max_steps_per_episode', 3),
                verbose=True,
                return_details=True
            )
            
            results['model_loaded'] = self.rl_model_loaded
            self.results['rl'] = results
            
            print(f"Adaptive RL evaluation completed successfully")
            print(f"State adaptation: {results.get('state_adaptation', 'Unknown')}")
            print(f"Valid episodes: {results.get('valid_episodes', 0)}/{results.get('total_episodes_attempted', 0)}")
                
        except Exception as e:
            print(f"Error in adaptive RL evaluation: {e}")
            import traceback
            traceback.print_exc()
            
            # Add error information to results
            self.results['rl'] = {
                'error': str(e),
                'model_loaded': self.rl_model_loaded,
                'evaluation_failed': True,
                'accuracy': 0.0,
                'avg_confidence': 0.0,
                'method': 'rl_agent_failed'
            }
    
    def run_all_evaluations(self) -> None:
        """Run all configured evaluations."""
        print(f"\n{'='*70}")
        print("STARTING COMPREHENSIVE EVALUATION WITH ADAPTIVE RL")
        print(f"{'='*70}")
        
        start_time = datetime.now()
        
        # Run all evaluations
        self.run_baseline_evaluation()
        self.run_fixed_augmentation_evaluation()
        self.run_tta_evaluation()
        self.run_rl_evaluation()
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        print(f"\nAll evaluations completed in {total_time:.1f} seconds")
        
        # Print summary of any issues
        if 'rl' in self.results and self.results['rl'].get('evaluation_failed'):
            print(f"\n⚠️ RL evaluation encountered issues:")
            print(f"   Error: {self.results['rl'].get('error', 'Unknown')}")
            print(f"   This may be due to model dimension mismatches")
            print(f"   Consider retraining with compatible dimensions or using a different model")
        
        # Save results
        self.save_results()
    
    def create_comparison_summary(self) -> Dict[str, Any]:
        """Create comparative summary using the result aggregator."""
        return self.result_aggregator.create_comprehensive_summary(self.results, self.device)
    
    def print_comparison_summary(self) -> None:
        """Print comparative summary using the result aggregator."""
        summary = self.create_comparison_summary()
        self.result_aggregator.print_comprehensive_summary(summary, self.results)
    
    def create_plots(self):
        """Create all visualization plots using the visualization manager."""
        print(f"\n{'='*70}")
        print("GENERATING COMPREHENSIVE PLOTS")
        print(f"{'='*70}")
        
        summary = self.create_comparison_summary()
        
        # Delegate all plotting to visualization manager
        self.visualization_manager.create_all_plots(self.results, summary)
        
        print("All plots and analyses completed!")

    def save_results(self) -> None:
        """Save complete results and summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f'results_{timestamp}.pkl')
        summary_file = os.path.join(self.output_dir, f'summary_{timestamp}.json')
        
        # Save complete results with pickle
        with open(results_file, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary as JSON
        summary = self.create_comparison_summary()
        
        # Custom JSON encoder to handle numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer, np.floating, np.bool_)):
                    return obj.item()
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4, cls=NumpyEncoder)
        
        print(f"\nEvaluation results saved to {results_file}")
        print(f"Summary saved to {summary_file}")

    def print_diagnostics(self) -> None:
        """Print diagnostic information about the evaluation."""
        print(f"\n{'='*60}")
        print("EVALUATION DIAGNOSTICS")
        print(f"{'='*60}")
        
        print(f"Configuration:")
        for key, value in self.config.items():
            if 'path' in key.lower():
                exists = "✅" if os.path.exists(str(value)) else "❌"
                print(f"  {key}: {value} {exists}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nModels:")
        print(f"  Classifier: {'✅ Loaded' if hasattr(self, 'classifier') else '❌ Not loaded'}")
        print(f"  RL Agent: {'✅ Loaded' if hasattr(self, 'agent') and self.agent else '❌ Not loaded'}")
        
        if hasattr(self, 'agent') and self.agent:
            agent_state_dim = getattr(self.agent, 'state_dim', 'Unknown')
            detected_state_dim = getattr(self.agent, 'detected_state_dim', 'Unknown')
            print(f"  Agent state dim: {agent_state_dim}")
            print(f"  Detected state dim: {detected_state_dim}")
        
        print(f"\nResults:")
        for method, result in self.results.items():
            if isinstance(result, dict):
                if 'error' in result:
                    print(f"  {method}: ❌ Failed - {result['error']}")
                elif 'accuracy' in result:
                    accuracy = result['accuracy']
                    print(f"  {method}: ✅ Accuracy {accuracy:.4f}")
                else:
                    print(f"  {method}: ⚠️ Incomplete")
            else:
                print(f"  {method}: ⚠️ Invalid format")
        
        print(f"{'='*60}")


# Import numpy at the top if not already imported
import numpy as np