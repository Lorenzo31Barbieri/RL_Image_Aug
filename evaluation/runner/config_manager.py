#!/usr/bin/env python3
"""
Configuration Manager for 143D State Space
==================================================

Centralized configuration for the evaluation system.
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs with 143D state space."""
    
    # Model paths
    classifier_path: str = './checkpoint/ckpt.pth'
    rl_model_path: str = './models/enhanced_dqn_episode_72000.pth'
    data_root: str = './data'
    
    # Fixed dimensions for 143D state space
    state_dim: int = 143
    image_feature_dim: int = 128
    action_dim: int = 16
    
    # Evaluation parameters  
    batch_size: int = 64
    baseline_samples: int = 10000
    fixed_aug_samples: int = 10000
    tta_samples: int = 1000
    rl_episodes: int = 1000
    max_steps_per_episode: int = 3
    
    # Fixed augmentation sequence
    fixed_aug_ids: list = field(default_factory=lambda: [0, 3, 6])
    
    # Output configuration
    output_dir: str = './evaluation_results'
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
        config.baseline_samples = 2000
        config.fixed_aug_samples = 2000
        config.output_dir = './quick_results'
        return config
    
    @staticmethod
    def create_comprehensive_config() -> EvaluationConfig:
        """Create configuration for comprehensive evaluation."""
        config = EvaluationConfig()
        config.tta_samples = 2000
        config.rl_episodes = 2000
        config.baseline_samples = 20000
        config.fixed_aug_samples = 20000
        config.output_dir = './comprehensive_results'
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
        
        # Validate fixed dimensions
        if config.state_dim != 143:
            errors.append(f"state_dim must be 143, got {config.state_dim}")
        if config.image_feature_dim != 128:
            errors.append(f"image_feature_dim must be 128, got {config.image_feature_dim}")
        if config.action_dim != 16:
            errors.append(f"action_dim must be 16, got {config.action_dim}")
        
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
        print(f"  Baseline Samples: {config.baseline_samples}")
        print(f"  Fixed Aug Samples: {config.fixed_aug_samples}")
        print(f"  TTA Samples: {config.tta_samples}")
        print(f"  RL Episodes: {config.rl_episodes}")
        print(f"  Max Steps/Episode: {config.max_steps_per_episode}")
        
        print("\n  Fixed State Configuration:")
        print(f"  State Dimension: {config.state_dim}")
        print(f"    - Logits: 10")
        print(f"    - Additional Features: 5") 
        print(f"    - Image Features: {config.image_feature_dim}")
        print(f"  Action Dimension: {config.action_dim}")
        
        print("\n Methods Enabled:")
        print(f"  Baseline: {'' if config.evaluate_baseline else ''}")
        print(f"  Fixed Aug: {'' if config.evaluate_fixed_aug else ''}")
        print(f"  TTA: {'' if config.evaluate_tta else ''}")
        print(f"  RL Agent: {'' if config.evaluate_rl else ''}")
        
        print(f"\n Output: {config.output_dir}")
        print("-" * 40)


def create_training_config() -> Dict[str, Any]:
    """
    Create configuration dictionary for training script.
    
    Returns:
        Dict with training configuration
    """
    return {
        # Fixed dimensions
        'state_dim': 143,
        'image_feature_dim': 128,
        'logits_dim': 10,
        'additional_features_dim': 5,
        'action_dim': 16,
        
        # Training hyperparameters
        'learning_rate': 0.0003,
        'gamma': 0.95,
        'epsilon_start': 1.0,
        'epsilon_end': 0.005,
        'epsilon_decay': 0.99975,
        'buffer_size': 300000,
        'batch_size': 128,
        'target_update_freq': 1000,
        
        # Episode configuration
        'num_total_episodes': 75000,
        'max_steps_per_episode': 3,
        'images_per_cycle': 3,
        
        # Training strategy
        'warmup_episodes': 3000,
        'eval_freq': 2500,
        'eval_episodes': 200,
        'patience': 300,
        
        # Paths
        'data_root': './data',
        'classifier_path': './checkpoint/ckpt.pth',
        'models_dir': './models',
        'plots_dir': './plots'
    }


def create_evaluation_config() -> Dict[str, Any]:
    """
    Create configuration dictionary for evaluation system.
    
    Returns:
        Dict with evaluation configuration
    """
    return {
        # Model paths
        'classifier_path': './checkpoint/ckpt.pth',
        'rl_model_path': './models/enhanced_dqn_episode_72000.pth',
        'data_root': './data',
        
        # Fixed dimensions
        'state_dim': 143,
        'image_feature_dim': 128,
        'action_dim': 16,
        
        # Evaluation parameters
        'batch_size': 64,
        'baseline_samples': 10000,
        'fixed_aug_samples': 10000,
        'tta_samples': 1000,
        'rl_episodes': 1000,
        'max_steps_per_episode': 3,
        
        # Fixed augmentation sequence
        'fixed_aug_ids': [0, 3, 6],
        
        # Method toggles
        'evaluate_baseline': True,
        'evaluate_fixed_aug': True,
        'evaluate_tta': True,
        'evaluate_rl': True,
        
        # Output configuration
        'output_dir': './evaluation_results',
        'save_results': True,
        'create_plots': True,
        'use_ttach': True
    }


def print_training_config(config: Dict[str, Any]) -> None:
    """Print training configuration summary."""
    print(" TRAINING CONFIGURATION")
    print("-" * 40)
    
    print("  State Space (Fixed):")
    print(f"  Total Dimension: {config['state_dim']}")
    print(f"    - Logits: {config['logits_dim']}")
    print(f"    - Additional Features: {config['additional_features_dim']}")
    print(f"    - Image Features: {config['image_feature_dim']}")
    print(f"  Action Dimension: {config['action_dim']}")
    
    print("\n  Training Hyperparameters:")
    print(f"  Learning Rate: {config['learning_rate']}")
    print(f"  Gamma: {config['gamma']}")
    print(f"  Epsilon: {config['epsilon_start']} → {config['epsilon_end']}")
    print(f"  Buffer Size: {config['buffer_size']:,}")
    print(f"  Batch Size: {config['batch_size']}")
    
    print("\n  Episode Configuration:")
    print(f"  Total Episodes: {config['num_total_episodes']:,}")
    print(f"  Max Steps/Episode: {config['max_steps_per_episode']}")
    print(f"  Images per Cycle: {config['images_per_cycle']}")
    print(f"  Warmup Episodes: {config['warmup_episodes']:,}")
    
    print("\n  Paths:")
    print(f"  Data Root: {config['data_root']}")
    print(f"  Classifier: {config['classifier_path']}")
    print(f"  Models Dir: {config['models_dir']}")
    print(f"  Plots Dir: {config['plots_dir']}")
    
    print("-" * 40)


def validate_training_config(config: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate training configuration.
    
    Args:
        config: Training configuration dictionary
    
    Returns:
        tuple: (is_valid: bool, errors: list[str])
    """
    errors = []
    
    # Check required paths
    if not os.path.exists(config['classifier_path']):
        errors.append(f"Classifier not found: {config['classifier_path']}")
    
    # Validate dimensions
    if config['state_dim'] != 143:
        errors.append(f"state_dim must be 143, got {config['state_dim']}")
    
    if config['image_feature_dim'] != 128:
        errors.append(f"image_feature_dim must be 128, got {config['image_feature_dim']}")
    
    if config['logits_dim'] != 10:
        errors.append(f"logits_dim must be 10, got {config['logits_dim']}")
    
    if config['additional_features_dim'] != 5:
        errors.append(f"additional_features_dim must be 5, got {config['additional_features_dim']}")
    
    # Check dimension consistency
    expected_total = config['logits_dim'] + config['additional_features_dim'] + config['image_feature_dim']
    if config['state_dim'] != expected_total:
        errors.append(f"state_dim {config['state_dim']} doesn't match sum of components {expected_total}")
    
    # Validate hyperparameters
    if config['learning_rate'] <= 0:
        errors.append("learning_rate must be positive")
    
    if not (0 <= config['gamma'] <= 1):
        errors.append("gamma must be between 0 and 1")
    
    if config['batch_size'] <= 0:
        errors.append("batch_size must be positive")
    
    if config['num_total_episodes'] <= 0:
        errors.append("num_total_episodes must be positive")
    
    # Create directories
    for dir_path in [config['models_dir'], config['plots_dir']]:
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create directory {dir_path}: {e}")
    
    return len(errors) == 0, errors


def get_default_paths() -> Dict[str, str]:
    """Get default paths for the project."""
    return {
        'classifier_path': './checkpoint/ckpt.pth',
        'rl_model_path': './models/enhanced_dqn_episode_72000.pth',
        'data_root': './data',
        'models_dir': './models',
        'plots_dir': './plots',
        'evaluation_results': './evaluation_results'
    }


def check_project_structure() -> Dict[str, bool]:
    """
    Check if the project has the expected structure.
    
    Returns:
        Dict mapping paths to existence status
    """
    paths = get_default_paths()
    
    # Add directories to check
    directories_to_check = [
        './src',
        './src/models',
        './src/environment', 
        './evaluation',
        './classifier'
    ]
    
    structure_check = {}
    
    # Check files
    for name, path in paths.items():
        structure_check[f"file_{name}"] = os.path.exists(path)
    
    # Check directories
    for directory in directories_to_check:
        dir_name = directory.replace('./', '').replace('/', '_')
        structure_check[f"dir_{dir_name}"] = os.path.exists(directory)
    
    return structure_check


def print_project_structure_check() -> None:
    """Print project structure check results."""
    structure = check_project_structure()
    
    print("\n PROJECT STRUCTURE CHECK")
    print("-" * 40)
    
    # Group by type
    files = {k: v for k, v in structure.items() if k.startswith('file_')}
    directories = {k: v for k, v in structure.items() if k.startswith('dir_')}
    
    print(" Directories:")
    for name, exists in directories.items():
        clean_name = name.replace('dir_', '').replace('_', '/')
        status = "" if exists else ""
        print(f"  {status} {clean_name}")
    
    print("\n Key Files:")
    for name, exists in files.items():
        clean_name = name.replace('file_', '')
        status = "" if exists else ""
        print(f"  {status} {clean_name}")
    
    # Summary
    total_items = len(structure)
    existing_items = sum(structure.values())
    
    print(f"\n Summary: {existing_items}/{total_items} items found")
    
    if existing_items == total_items:
        print(" Project structure is complete!")
    else:
        missing = total_items - existing_items
        print(f" {missing} items missing - some functionality may be limited")
    
    print("-" * 40)