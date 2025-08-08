#!/usr/bin/env python3
"""
Evaluation Orchestrator
=======================

Coordinates the execution of all evaluation methods in the correct sequence.
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional

from ..comparison import EvaluationComparison
from .config_manager import EvaluationConfig
from .requirements_checker import RequirementsChecker
from .output_manager import OutputManager


class EvaluationOrchestrator:
    """Orchestrates the complete evaluation process."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.comparison: Optional[EvaluationComparison] = None
        self.output_manager = OutputManager(config.output_dir)
        self.start_time: Optional[float] = None
        self.results: Dict[str, Any] = {}
    
    def run_complete_evaluation(self) -> Optional[Dict[str, Any]]:
        """
        Run the complete evaluation pipeline.
        
        Returns:
            Results dictionary if successful, None if failed
        """
        try:
            # 1. Validate requirements
            if not self._validate_requirements():
                return None
            
            # 2. Initialize evaluation system
            if not self._initialize_evaluation_system():
                return None
            
            # 3. Load models and data
            if not self._load_models_and_data():
                return None
            
            # 4. Run evaluations
            if not self._run_evaluations():
                return None
            
            # 5. Generate outputs
            self._generate_outputs()
            
            # 6. Print final summary
            self._print_completion_summary()
            
            return self.results
            
        except KeyboardInterrupt:
            self._handle_interruption()
            return None
        except Exception as e:
            self._handle_error(e)
            return None
    
    def _validate_requirements(self) -> bool:
        """Validate system requirements."""
        print(" COMPREHENSIVE MODEL EVALUATION")
        print("=" * 60)
        print(f" Project root: {self.config.data_root}")
        print(f" Device: {'CUDA' if self._is_cuda_available() else 'CPU'}")
        print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        checker = RequirementsChecker(self.config)
        return checker.check_all_requirements()
    
    def _initialize_evaluation_system(self) -> bool:
        """Initialize the evaluation system."""
        try:
            print("\n LOADING EVALUATION SYSTEM")
            print("-" * 40)
            
            # Print configuration
            from .config_manager import ConfigManager
            ConfigManager.print_config_summary(self.config)
            
            # Create comparison object
            self.comparison = EvaluationComparison(self.config.__dict__)
            print(" Evaluation system initialized successfully")
            return True
            
        except Exception as e:
            print(f" Failed to initialize evaluation system: {e}")
            return False
    
    def _load_models_and_data(self) -> bool:
        """Load models and data."""
        try:
            print("\n LOADING MODELS AND DATA")
            print("-" * 40)
            
            print(" Loading models...")
            self.comparison.load_models()
            
            print(" Loading data...")
            self.comparison.load_data()
            
            print(" Models and data loaded successfully")
            return True
            
        except Exception as e:
            print(f" Error loading models/data: {e}")
            return False
    
    def _run_evaluations(self) -> bool:
        """Run all evaluations."""
        try:
            self.start_time = time.time()
            
            print("\n RUNNING EVALUATIONS")
            print("-" * 40)
            print(" This may take several minutes...")
            print(f" Evaluating {self.config.tta_samples} samples for TTA")
            print(f" Evaluating {self.config.rl_episodes} episodes for RL")
            
            self.comparison.run_all_evaluations()
            self.results = self.comparison.results
            
            print(" All evaluations completed successfully")
            return True
            
        except Exception as e:
            print(f" Error during evaluation: {e}")
            return False
    
    def _generate_outputs(self) -> None:
        """Generate all outputs (plots, reports, etc.)."""
        print("\n GENERATING OUTPUTS")
        print("-" * 40)
        
        # Print results summary
        if self.results:
            self.comparison.print_comparison_summary()
        
        # Create visualizations
        if self.config.create_plots:
            print(" Creating comprehensive analysis...")
            try:
                self.comparison.create_plots()
                print(" Visualizations created successfully")
            except Exception as e:
                print(f"  Error creating plots: {e}")
        
        # Save results
        if self.config.save_results:
            try:
                self.comparison.save_results()
                print(" Results saved successfully")
            except Exception as e:
                print(f"  Error saving results: {e}")
    
    def _print_completion_summary(self) -> None:
        """Print final completion summary."""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        print("\n EVALUATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"⏱  Total time: {total_time/60:.1f} minutes")
        print(f" Results saved to: {self.config.output_dir}/")
        
        # Print key file locations
        plots_dir = f"{self.config.output_dir}/plots"
        images_dir = f"{self.config.output_dir}/improved_images"
        
        print(f" Main plots: {plots_dir}/comprehensive_comparison.png")
        print(f" Confusion matrices: {plots_dir}/confusion_matrices.png")
        print(f"  Class analysis: {plots_dir}/rl_class_analysis.png")
        print(f"  Improved images: {images_dir}/")
        print("=" * 60)
    
    def _handle_interruption(self) -> None:
        """Handle keyboard interruption."""
        print("\n\n⏹  Evaluation interrupted by user")
        print(" You can restart the evaluation anytime")
    
    def _handle_error(self, error: Exception) -> None:
        """Handle unexpected errors."""
        print(f"\n Error during evaluation: {error}")
        print(f"Error type: {type(error).__name__}")
        
        # Print helpful debugging info
        error_str = str(error).lower()
        print("\n Debugging information:")
        if "not found" in error_str:
            print(" File not found - check your model paths")
        elif "cuda" in error_str:
            print(" CUDA error - try running with CPU: export CUDA_VISIBLE_DEVICES=''")
        elif "memory" in error_str:
            print(" Memory error - try reducing batch_size in config")
        else:
            print(" General error - check the full traceback above")
        
        import traceback
        traceback.print_exc()
    
    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False


def create_and_run_evaluation(config: Optional[EvaluationConfig] = None) -> Optional[Dict[str, Any]]:
    """
    Convenience function to create and run a complete evaluation.
    
    Args:
        config: Optional configuration. If None, uses default.
    
    Returns:
        Results dictionary if successful, None if failed
    """
    if config is None:
        from .config_manager import ConfigManager
        config = ConfigManager.create_default_config()
    
    orchestrator = EvaluationOrchestrator(config)
    return orchestrator.run_complete_evaluation()