#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL EVALUATION
===========================================

Streamlined main script for comprehensive model evaluation.

Usage: python full_evaluation_refactored.py [--quick] [--config CONFIG_FILE]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import modular components
from evaluation.runner.config_manager import ConfigManager, EvaluationConfig
from evaluation.runner.evaluation_orchestrator import create_and_run_evaluation
from evaluation.runner.interactive_runner import InteractiveRunner


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Model Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python full_evaluation_refactored.py                    # Interactive mode
  python full_evaluation_refactored.py --quick           # Quick evaluation
  python full_evaluation_refactored.py --non-interactive # Use defaults
  python full_evaluation_refactored.py --tta-samples 500 # Custom TTA samples
        """
    )
    
    # Configuration options
    parser.add_argument('--quick', action='store_true',
                       help='Run quick evaluation with reduced samples')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run with default configuration without prompts')
    
    # Custom parameters
    parser.add_argument('--tta-samples', type=int, metavar='N',
                       help='Number of samples for TTA evaluation')
    parser.add_argument('--rl-episodes', type=int, metavar='N',
                       help='Number of episodes for RL evaluation')
    parser.add_argument('--batch-size', type=int, metavar='N',
                       help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, metavar='PATH',
                       help='Output directory for results')
    
    # Method toggles
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline evaluation')
    parser.add_argument('--skip-fixed-aug', action='store_true',
                       help='Skip fixed augmentation evaluation')
    parser.add_argument('--skip-tta', action='store_true',
                       help='Skip TTA evaluation')
    parser.add_argument('--skip-rl', action='store_true',
                       help='Skip RL agent evaluation')
    
    # Output options
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip plot generation')
    parser.add_argument('--no-save', action='store_true',
                       help='Skip saving results to files')
    
    return parser.parse_args()


def create_config_from_args(args) -> EvaluationConfig:
    """Create configuration from command line arguments."""
    if args.quick:
        config = ConfigManager.create_quick_config()
    else:
        config = ConfigManager.create_default_config()
    
    # Apply custom parameters
    if args.tta_samples:
        config.tta_samples = args.tta_samples
    if args.rl_episodes:
        config.rl_episodes = args.rl_episodes
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.output_dir:
        config.output_dir = args.output_dir
    
    # Apply method toggles
    if args.skip_baseline:
        config.evaluate_baseline = False
    if args.skip_fixed_aug:
        config.evaluate_fixed_aug = False
    if args.skip_tta:
        config.evaluate_tta = False
    if args.skip_rl:
        config.evaluate_rl = False
    
    # Apply output options
    if args.no_plots:
        config.create_plots = False
    if args.no_save:
        config.save_results = False
    
    return config


def validate_environment():
    """Validate that the evaluation environment is properly set up."""
    try:
        # Check if we can import the evaluation system
        from evaluation.comparison import EvaluationComparison
        return True
    except ImportError as e:
        print(f"Error: Cannot import evaluation system: {e}")
        print("Make sure you're running from the project root directory")
        print("And that all required modules are properly installed")
        return False


def main():
    """Main entry point."""
    print("COMPREHENSIVE MODEL EVALUATION SYSTEM")
    print("=" * 60)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Parse arguments
    args = parse_arguments()
    
    try:
        if args.non_interactive:
            # Non-interactive mode: use configuration from args
            config = create_config_from_args(args)
            
            print("Running in non-interactive mode")
            ConfigManager.print_config_summary(config)
            
            # Run evaluation
            results = create_and_run_evaluation(config)
            
        else:
            # Interactive mode: use interactive runner
            print("Running in interactive mode")
            print("   Use --non-interactive to skip prompts")
            print()
            
            # If specific args provided, create config and skip some prompts
            if any([args.tta_samples, args.rl_episodes, args.batch_size, args.quick]):
                config = create_config_from_args(args)
                
                # Show config and ask for confirmation
                ConfigManager.print_config_summary(config)
                
                try:
                    response = input("\nContinue with this configuration? (Y/n): ").strip().lower()
                    if response and response not in ['y', 'yes']:
                        print("Evaluation cancelled")
                        sys.exit(0)
                except (EOFError, KeyboardInterrupt):
                    print("\nEvaluation cancelled")
                    sys.exit(0)
                
                results = create_and_run_evaluation(config)
            else:
                # Full interactive mode
                runner = InteractiveRunner()
                results = runner.run_interactive_evaluation()
        
        # Handle results
        if results:
            print("\nEvaluation completed successfully!")
            sys.exit(0)
        else:
            print("\nEvaluation failed or was cancelled")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Goodbye!")
        sys.exit(0)
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Run with --help for usage information")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()