"""
Centralized Evaluation Configuration
===================================
All evaluation-related parameters and paths in one place.
"""

import os

# === MODEL PATHS ===
CLASSIFIER_PATH = './checkpoint/ckpt.pth'
RL_MODEL_PATH = './models/best_dqn_model.pth'  # Main RL model path
DATA_ROOT = './data'

# === ALTERNATIVE RL MODEL PATHS (fallbacks) ===
ALTERNATIVE_RL_PATHS = [
    './models/final_dqn_model.pth',
    './models/dqn_episode_72000.pth', 
    './models/interrupted_dqn_model.pth'
]

# === FIXED ARCHITECTURE (143D State Space) ===
STATE_DIM = 143
LOGITS_DIM = 10  # CIFAR-10 classes
ADDITIONAL_FEATURES_DIM = 5  # confidence, entropy, margin, correctness, step_ratio  
IMAGE_FEATURE_DIM = 128  # Image features from classifier
ACTION_DIM = 16

# === EVALUATION PARAMETERS ===
BATCH_SIZE = 64
BASELINE_SAMPLES = 3000
FIXED_AUG_SAMPLES = 3000
TTA_SAMPLES = 3000
RL_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 3

# === METHOD CONFIGURATION ===
EVALUATE_BASELINE = True
EVALUATE_FIXED_AUG = True
EVALUATE_TTA = True
EVALUATE_RL = True

# === FIXED AUGMENTATION SEQUENCE ===
FIXED_AUG_IDS = [0, 3, 6]  # Brightness, Contrast, Horizontal Flip

# === OUTPUT CONFIGURATION ===
OUTPUT_DIR = './evaluation_results'
SAVE_RESULTS = True
CREATE_PLOTS = True
USE_TTACH = True

# === QUICK TEST CONFIGURATION ===
QUICK_BASELINE_SAMPLES = 2000
QUICK_FIXED_AUG_SAMPLES = 2000
QUICK_TTA_SAMPLES = 200
QUICK_RL_EPISODES = 200
QUICK_OUTPUT_DIR = './quick_results'

def get_default_config():
    """Get default evaluation configuration."""
    return {
        # Model paths
        'classifier_path': CLASSIFIER_PATH,
        'rl_model_path': RL_MODEL_PATH,
        'data_root': DATA_ROOT,
        
        # Fixed dimensions
        'state_dim': STATE_DIM,
        'image_feature_dim': IMAGE_FEATURE_DIM,
        'action_dim': ACTION_DIM,
        
        # Evaluation parameters
        'batch_size': BATCH_SIZE,
        'baseline_samples': BASELINE_SAMPLES,
        'fixed_aug_samples': FIXED_AUG_SAMPLES,
        'tta_samples': TTA_SAMPLES,
        'rl_episodes': RL_EPISODES,
        'max_steps_per_episode': MAX_STEPS_PER_EPISODE,
        
        # Method toggles
        'evaluate_baseline': EVALUATE_BASELINE,
        'evaluate_fixed_aug': EVALUATE_FIXED_AUG,
        'evaluate_tta': EVALUATE_TTA,
        'evaluate_rl': EVALUATE_RL,
        
        # Fixed augmentation
        'fixed_aug_ids': FIXED_AUG_IDS,
        
        # Output options
        'output_dir': OUTPUT_DIR,
        'save_results': SAVE_RESULTS,
        'create_plots': CREATE_PLOTS,
        'use_ttach': USE_TTACH
    }

def get_quick_config():
    """Get quick test configuration with reduced samples."""
    config = get_default_config()
    config.update({
        'baseline_samples': QUICK_BASELINE_SAMPLES,
        'fixed_aug_samples': QUICK_FIXED_AUG_SAMPLES, 
        'tta_samples': QUICK_TTA_SAMPLES,
        'rl_episodes': QUICK_RL_EPISODES,
        'output_dir': QUICK_OUTPUT_DIR
    })
    return config

def print_config(config=None):
    """Print evaluation configuration summary."""
    if config is None:
        config = get_default_config()
    
    print(" EVALUATION CONFIGURATION")
    print("-" * 40)
    
    print("  Model Paths:")
    print(f"  Classifier: {config['classifier_path']}")
    print(f"  RL Model: {config['rl_model_path']}")
    print(f"  Data Root: {config['data_root']}")
    
    print("\n  State Space (Fixed 143D):")
    print(f"  Total Dimension: {config['state_dim']}")
    print(f"    - Logits: {LOGITS_DIM}")
    print(f"    - Additional Features: {ADDITIONAL_FEATURES_DIM}")
    print(f"    - Image Features: {config['image_feature_dim']}")
    print(f"  Action Dimension: {config['action_dim']}")
    
    print("\n  Evaluation Samples:")
    print(f"  Baseline: {config['baseline_samples']:,}")
    print(f"  Fixed Aug: {config['fixed_aug_samples']:,}")
    print(f"  TTA: {config['tta_samples']:,}")
    print(f"  RL Episodes: {config['rl_episodes']:,}")
    
    print("\n  Methods Enabled:")
    print(f"  Baseline: {'Yes' if config['evaluate_baseline'] else 'No'}")
    print(f"  Fixed Aug: {'Yes' if config['evaluate_fixed_aug'] else 'No'}")
    print(f"  TTA: {'Yes' if config['evaluate_tta'] else 'No'}")
    print(f"  RL Agent: {'Yes' if config['evaluate_rl'] else 'No'}")
    
    print(f"\n  Output: {config['output_dir']}")
    print("-" * 40)