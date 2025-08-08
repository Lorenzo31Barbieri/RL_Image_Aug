# rl_image_tools/config.py
"""Configuration for RL image evaluation and saving."""

# Model paths (relative to project root)
CLASSIFIER_PATH = './checkpoint/ckpt.pth'
RL_MODEL_PATH = './models/best_improved_dqn_model.pth'
DATA_ROOT = './data'

# Evaluation parameters
NUM_IMAGES = 100
MAX_STEPS_PER_EPISODE = 3
STATE_DIM = 15

# Output directories (within rl_image_tools folder)
OUTPUT_DIR = 'output_images'
ORIGINAL_IMAGES_DIR = f'{OUTPUT_DIR}/original'
AUGMENTED_IMAGES_DIR = f'{OUTPUT_DIR}/augmented'

# Device
DEVICE = 'cuda'  # or 'cpu'