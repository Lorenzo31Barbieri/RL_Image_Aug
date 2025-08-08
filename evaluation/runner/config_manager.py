#!/usr/bin/env python3
"""
Configuration Manager for Comprehensive Evaluation
=================================================

Centralizes all configuration logic and validation for the evaluation system.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EvaluationConfig:
    """Centralized configuration for evaluation runs."""
    
    # Model paths
    classifier_path: str = './checkpoint/ckpt.pth'
    rl_model_path: str = './models/best_improved_dqn_model.pth'
    data_root: str = './data'
    
    # Evaluation parameters  
    batch_size: int = 64
    tta_samples: int = 1000
    rl_episodes: int = 1000
    max_steps_per_episode: int = 3
    state_dim: int = 15
    
    # Fixed augmentation sequence
    fixed_aug_ids: list = field(default_factory=lambda: [0, 3, 6])
    
    # Output configuration
    output_dir: str = './comprehensive_results'
    save_results: bool = True
    create_plots: bool = True
    
    # Method toggles
    evaluate_baseline: bool = True
    evaluate_fixed_aug: bool = True
    evaluate_tta: bool = True
    evaluate_rl: bool = True


class ConfigManager:
    """Manages configuration loading, validation, and display."""
    
    @staticmethod
    def create_default_config() -> EvaluationConfig:
        """Create default configuration."""
        return EvaluationConfig()
    
    @staticmethod
    def create_quick_config() -> EvaluationConfig:
        """Create configuration for quick testing."""
        config = EvaluationConfig()
        config.tta_samples = 200
        config.rl_episodes = 200
        config.output_dir = './quick_results'
        return config
    
    @staticmethod
    def validate_config(config: EvaluationConfig) -> tuple[bool, list[str]]:
        """
        Validate configuration and return (is_valid, error_messages).
        
        Returns:
            tuple: (is_valid: bool, errors: list[str])
        """
        errors = []
        
        # Check required files
        if not os.path.exists(config.classifier_path):
            errors.append(f"Classifier not found: {config.classifier_path}")
        
        # Check optional files with warnings
        warnings = []
        if not os.path.exists(config.rl_model_path):
            warnings.append(f"RL model not found: {config.rl_model_path} (will use random agent)")
        
        # Validate parameters
        if config.batch_size <= 0:
            errors.append("batch_size must be positive")
        if config.tta_samples <= 0:
            errors.append("tta_samples must be positive")
        if config.rl_episodes <= 0:
            errors.append("rl_episodes must be positive")
        
        # Create output directory
        try:
            os.makedirs(config.output_dir, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
        
        # Print warnings
        for warning in warnings:
            print(f"  {warning}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def print_config_summary(config: EvaluationConfig) -> None:
        """Print a formatted configuration summary."""
        print(" EVALUATION CONFIGURATION")
        print("-" * 40)
        
        print("  Model Paths:")
        print(f"  Classifier: {config.classifier_path}")
        print(f"  RL Model: {config.rl_model_path}")
        print(f"  Data Root: {config.data_root}")
        
        print("\n  Evaluation Parameters:")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  TTA Samples: {config.tta_samples}")
        print(f"  RL Episodes: {config.rl_episodes}")
        print(f"  Max Steps/Episode: {config.max_steps_per_episode}")
        
        print("\n🔧 Methods Enabled:")
        print(f"  Baseline: {'OK' if config.evaluate_baseline else 'NOT ENABLED'}")
        print(f"  Fixed Aug: {'OK' if config.evaluate_fixed_aug else 'NOT ENABLED'}")
        print(f"  TTA: {'OK' if config.evaluate_tta else 'NOT ENABLED'}")
        print(f"  RL Agent: {'OK' if config.evaluate_rl else 'NOT ENABLED'}")
        
        print(f"\n Output: {config.output_dir}")
        print("-" * 40)