"""
Centralized Training Configuration
================================
All training-related parameters and paths in one place.
"""

import os

# === MODEL PATHS ===
CLASSIFIER_PATH = './checkpoint/ckpt.pth'
DATA_ROOT = './data'
MODELS_DIR = './models'
PLOTS_DIR = './plots'

# === FIXED ARCHITECTURE (143D State Space) ===
STATE_DIM = 143
LOGITS_DIM = 10  # CIFAR-10 classes
ADDITIONAL_FEATURES_DIM = 5  # confidence, entropy, margin, correctness, step_ratio
IMAGE_FEATURE_DIM = 128  # Image features from classifier
ACTION_DIM = 16

# === TRAINING HYPERPARAMETERS ===
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.99975
BUFFER_SIZE = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQ = 2000

# === TRAINING STRATEGY ===
NUM_TOTAL_EPISODES = 50000
MAX_STEPS_PER_EPISODE = 3
IMAGES_PER_CYCLE = 4
WARMUP_EPISODES = 2000
EVAL_FREQ = 2000
EVAL_EPISODES = 200
PATIENCE = 100

# === OUTPUT FILES ===
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_dqn_model.pth')
FINAL_MODEL_PATH = os.path.join(MODELS_DIR, 'final_dqn_model.pth')
INTERRUPTED_MODEL_PATH = os.path.join(MODELS_DIR, 'interrupted_dqn_model.pth')
TRAINING_PLOTS_PATH = os.path.join(PLOTS_DIR, 'training_analysis.png')

def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

def print_config():
    """Print training configuration summary."""
    print(" TRAINING CONFIGURATION")
    print("-" * 40)
    
    print("  Model Paths:")
    print(f"  Classifier: {CLASSIFIER_PATH}")
    print(f"  Data Root: {DATA_ROOT}")
    print(f"  Models Dir: {MODELS_DIR}")
    
    print("\n  State Space (Fixed 143D):")
    print(f"  Total Dimension: {STATE_DIM}")
    print(f"    - Logits: {LOGITS_DIM}")
    print(f"    - Additional Features: {ADDITIONAL_FEATURES_DIM}")
    print(f"    - Image Features: {IMAGE_FEATURE_DIM}")
    print(f"  Action Dimension: {ACTION_DIM}")
    
    print("\n  Training Parameters:")
    print(f"  Episodes: {NUM_TOTAL_EPISODES:,}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Max Steps/Episode: {MAX_STEPS_PER_EPISODE}")
    
    print("-" * 40)