#!/usr/bin/env python3
"""
Interactive Runner for Evaluation System
========================================

Provides interactive prompts and user-friendly interface for running evaluations.
"""

import sys
from typing import Optional, Dict, Any

from .config_manager import ConfigManager, EvaluationConfig
from .evaluation_orchestrator import EvaluationOrchestrator
from .output_manager import ResultsFormatter


class InteractiveRunner:
    """Interactive interface for running evaluations."""
    
    def __init__(self):
        self.config: Optional[EvaluationConfig] = None
        self.orchestrator: Optional[EvaluationOrchestrator] = None
    
    def run_interactive_evaluation(self) -> Optional[Dict[str, Any]]:
        """Run evaluation with interactive prompts."""
        try:
            print("🚀 COMPREHENSIVE MODEL EVALUATION")
            print("=" * 60)
            
            # 1. Get configuration
            self.config = self._get_configuration()
            if not self.config:
                return None
            
            # 2. Print configuration and get confirmation
            if not self._confirm_configuration():
                return None
            
            # 3. Run evaluation
            self.orchestrator = EvaluationOrchestrator(self.config)
            results = self.orchestrator.run_complete_evaluation()
            
            # 4. Print final summary
            if results:
                self._print_final_summary(results)
            
            return results
            
        except KeyboardInterrupt:
            print("\n\n👋 Evaluation interrupted by user. Goodbye!")
            return None
        except Exception as e:
            print(f"\n\n❌ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_configuration(self) -> Optional[EvaluationConfig]:
        """Get configuration from user input."""
        print("📋 CONFIGURATION SETUP")
        print("-" * 40)
        
        # Ask for configuration type
        print("Choose configuration type:")
        print("1. Default (recommended)")
        print("2. Quick test (faster, fewer samples)")
        print("3. Custom configuration")
        
        while True:
            try:
                choice = input("\nEnter choice (1-3): ").strip()
                
                if choice == '1':
                    return ConfigManager.create_default_config()
                elif choice == '2':
                    return ConfigManager.create_quick_config()
                elif choice == '3':
                    return self._create_custom_config()
                else:
                    print("Please enter 1, 2, or 3")
                    
            except (EOFError, KeyboardInterrupt):
                print("\nOperation cancelled")
                return None
    
    def _create_custom_config(self) -> Optional[EvaluationConfig]:
        """Create custom configuration through interactive prompts."""
        print("\n🔧 CUSTOM CONFIGURATION")
        print("-" * 30)
        
        config = ConfigManager.create_default_config()
        
        try:
            # Ask for key parameters
            tta_samples = self._get_int_input(
                f"TTA samples (default {config.tta_samples}): ",
                default=config.tta_samples,
                min_val=100,
                max_val=5000
            )
            if tta_samples is not None:
                config.tta_samples = tta_samples
            
            rl_episodes = self._get_int_input(
                f"RL episodes (default {config.rl_episodes}): ",
                default=config.rl_episodes,
                min_val=100,
                max_val=10000
            )
            if rl_episodes is not None:
                config.rl_episodes = rl_episodes
            
            batch_size = self._get_int_input(
                f"Batch size (default {config.batch_size}): ",
                default=config.batch_size,
                min_val=1,
                max_val=256
            )
            if batch_size is not None:
                config.batch_size = batch_size
            
            # Ask about methods to evaluate
            print("\nMethods to evaluate:")
            config.evaluate_baseline = self._get_bool_input("Baseline? (Y/n): ", default=True)
            config.evaluate_fixed_aug = self._get_bool_input("Fixed Augmentation? (Y/n): ", default=True)
            config.evaluate_tta = self._get_bool_input("Test-Time Augmentation? (Y/n): ", default=True)
            config.evaluate_rl = self._get_bool_input("RL Agent? (Y/n): ", default=True)
            
            return config
            
        except (EOFError, KeyboardInterrupt):
            print("\nConfiguration cancelled")
            return None
    
    def _confirm_configuration(self) -> bool:
        """Show configuration and get confirmation."""
        ConfigManager.print_config_summary(self.config)
        
        # Estimate time
        total_samples = self.config.tta_samples + self.config.rl_episodes
        if total_samples > 2000:
            print(f"\n⚠️  This evaluation will process ~{total_samples:,} samples")
            print("   Estimated time: 10-30 minutes depending on your hardware")
        elif total_samples > 1000:
            print(f"\n📊 This evaluation will process ~{total_samples:,} samples")
            print("   Estimated time: 5-15 minutes")
        else:
            print(f"\n🚀 Quick evaluation: ~{total_samples:,} samples")
            print("   Estimated time: 2-5 minutes")
        
        return self._get_bool_input("\n❓ Continue with this configuration? (Y/n): ", default=True)
    
    def _print_final_summary(self, results: Dict[str, Any]) -> None:
        """Print final summary of results."""
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        # Print formatted results
        print(ResultsFormatter.format_accuracy_comparison(results))
        print()
        print(ResultsFormatter.format_timing_comparison(results))
        print()
        print(ResultsFormatter.format_improvement_summary(results))
        
        print("\n📁 RESULTS LOCATION:")
        print(f"  Main directory: {self.config.output_dir}")
        print("  📊 Plots: ./plots/comprehensive_comparison.png")
        print("  🧠 Analysis: ./plots/confusion_matrices.png")
        print("  🖼️  Examples: ./improved_images/")
        
        print("\n🎉 All done! Check the results above and in the output directory.")
    
    def _get_int_input(self, prompt: str, default: int, min_val: int = None, max_val: int = None) -> Optional[int]:
        """Get integer input with validation."""
        while True:
            try:
                response = input(prompt).strip()
                if not response:
                    return default
                
                value = int(response)
                
                if min_val is not None and value < min_val:
                    print(f"Value must be at least {min_val}")
                    continue
                
                if max_val is not None and value > max_val:
                    print(f"Value must be at most {max_val}")
                    continue
                
                return value
                
            except ValueError:
                print("Please enter a valid number")
            except (EOFError, KeyboardInterrupt):
                return None
    
    def _get_bool_input(self, prompt: str, default: bool = True) -> bool:
        """Get boolean input with Y/n prompt."""
        try:
            response = input(prompt).lower().strip()
            
            if not response:
                return default
            
            if response in ['y', 'yes', 'true', '1']:
                return True
            elif response in ['n', 'no', 'false', '0']:
                return False
            else:
                print("Please enter Y or N")
                return self._get_bool_input(prompt, default)
                
        except (EOFError, KeyboardInterrupt):
            return False


def main():
    """Main entry point for interactive evaluation."""
    runner = InteractiveRunner()
    results = runner.run_interactive_evaluation()
    
    if results:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure or cancellation


if __name__ == '__main__':
    main()