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
    rl_model_path: str = './models/enhanced_dqn_episode_72000.pth'
    data_root: str = './data'
    
    # Evaluation parameters  
    batch_size: int = 64
    baseline_samples: int = 10000
    fixed_aug_samples: int = 10000
    tta_samples: int = 1000
    rl_episodes: int = 1000
    max_steps_per_episode: int = 3
    state_dim: int = None  # Auto-detect from model
    image_feature_dim: int = 128  # Default enhanced image features
    
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
        """Create default configuration with auto-detection."""
        config = EvaluationConfig()
        
        # Try to auto-detect state dimension from available models
        state_dim, image_feature_dim = ConfigManager._detect_model_dimensions(config.rl_model_path)
        if state_dim:
            config.state_dim = state_dim
            config.image_feature_dim = image_feature_dim
            print(f"Auto-detected state dimensions: {state_dim} (image features: {image_feature_dim})")
        else:
            # Fallback to original dimensions if no model found
            config.state_dim = 15
            config.image_feature_dim = 0
            print("Using fallback dimensions for backward compatibility")
        
        return config
    
    @staticmethod
    def create_quick_config() -> EvaluationConfig:
        """Create configuration for quick testing."""
        config = ConfigManager.create_default_config()
        config.tta_samples = 200
        config.rl_episodes = 200
        config.output_dir = './quick_results'
        return config
    
    @staticmethod
    def _detect_model_dimensions(model_path: str) -> tuple[Optional[int], int]:
        """
        Detect state dimensions from an existing model file.
        
        Returns:
            tuple: (state_dim, image_feature_dim)
        """
        if not os.path.exists(model_path):
            # Try alternative model paths
            alternative_paths = [
                './models/best_enhanced_dqn_model.pth',
                './models/final_enhanced_dqn_model.pth',
                './models/best_improved_dqn_model.pth',
                './models/final_improved_dqn_model.pth'
            ]
            
            for alt_path in alternative_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                return None, 0
        
        try:
            import torch
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Find input dimension from first layer
            for key, tensor in state_dict.items():
                if 'fc1.weight' in key:
                    state_dim = tensor.shape[1]
                    
                    # Calculate image feature dimension
                    base_dim = 15  # Original state dimension (10 logits + 5 additional)
                    if state_dim > base_dim:
                        image_feature_dim = state_dim - base_dim
                    else:
                        image_feature_dim = 0
                    
                    return state_dim, image_feature_dim
                    
        except Exception as e:
            print(f"Could not detect dimensions from {model_path}: {e}")
        
        return None, 0
    
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
        
        # Validate state dimensions
        if config.state_dim is not None and config.state_dim <= 0:
            errors.append("state_dim must be positive if specified")
        
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
        
        print("\n  State Configuration:")
        if config.state_dim:
            print(f"  State Dimension: {config.state_dim}")
            print(f"  Image Features: {config.image_feature_dim}")
            if config.image_feature_dim > 0:
                print(f"  Enhanced State: ✅ (logits: 10, additional: 5, image: {config.image_feature_dim})")
            else:
                print(f"  Original State: ⚠️ (backward compatibility mode)")
        else:
            print(f"  State Dimension: Auto-detect from model")
        
        print("\n🔧 Methods Enabled:")
        print(f"  Baseline: {'✅' if config.evaluate_baseline else '❌'}")
        print(f"  Fixed Aug: {'✅' if config.evaluate_fixed_aug else '❌'}")
        print(f"  TTA: {'✅' if config.evaluate_tta else '❌'}")
        print(f"  RL Agent: {'✅' if config.evaluate_rl else '❌'}")
        
        print(f"\n📁 Output: {config.output_dir}")
        print("-" * 40)