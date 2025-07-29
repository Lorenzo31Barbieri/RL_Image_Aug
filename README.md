# RL Image Augmentation Project

## Project Structure

```
project_root/
├── src/                          # Main source code
│   ├── models/                   # Model definitions
│   │   ├── vgg.py               # VGG classifier
│   │   └── agent.py             # DQN Agent
│   ├── environment/             # RL Environment
│   │   ├── environment.py       # ImageAugmentationEnv
│   │   └── transforms.py        # Image transformations
│   ├── data/                    # Data utilities
│   │   └── augmented_image_buffer.py
│   └── utils/                   # General utilities
│       └── utils.py
├── evaluation/                  # Evaluation system
│   ├── core/                   # Core evaluation utilities
│   ├── methods/                # Individual evaluation methods
│   ├── comparison/             # Comparison tools
│   └── visualization/          # Plotting tools
├── scripts/                    # Training and evaluation scripts
│   ├── training_script_improved.py
│   ├── evaluation_script.py
│   ├── fixed_augmentation_evaluation.py
│   ├── tta.py
│   └── tta_rl_agent_evaluation.py
├── data/                       # Dataset storage
├── checkpoint/                 # Model checkpoints
├── models/                     # Trained models
└── results/                    # Evaluation results
```

## Data
Link to data folder: https://drive.google.com/drive/folders/1uU31eW6cE8Kt5z5AeAnAFCGfRGmcnWID?usp=drive_link



1. **Training**: Run training script
   ```bash
   python scripts/training_script_improved.py
   ```

2. **Evaluation**: Use the modular evaluation system
   ```python
   from evaluation.comparison import EvaluationComparison
   from evaluation.core import load_classifier
   
   # Load model
   classifier = load_classifier('./checkpoint/ckpt.pth')
   
   # Run comprehensive evaluation
   comparison = EvaluationComparison(config)
   comparison.run_all_evaluations()
   ```

3. **Individual Evaluations**:
   ```python
   from evaluation.methods import evaluate_baseline, evaluate_rl_agent
   
   # Evaluate baseline
   results = evaluate_baseline(classifier, test_loader, device)
   
   # Evaluate RL agent  
   results = evaluate_rl_agent(agent, classifier, test_dataset, device)
   ```

## Import Structure

After migration, use these imports:
```python
# Models
from src.models import VGG, DQNAgent

# Environment
from src.environment import ImageAugmentationEnv, get_num_actions

# Data utilities
from src.data import AugmentedImageBuffer

# Evaluation system
from evaluation.methods import evaluate_baseline, evaluate_rl_agent
from evaluation.comparison import EvaluationComparison
```
