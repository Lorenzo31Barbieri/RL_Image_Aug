# RL-Based Image Augmentation for CIFAR-10

Deep reinforcement learning approach for adaptive image augmentation on CIFAR-10 classification tasks.

## Overview

This project implements a DQN-based RL agent that learns optimal augmentation policies for improving image classification performance. The system compares multiple augmentation approaches including baseline, fixed augmentation, test-time augmentation (TTA), and the proposed RL method.

## Project Structure

```
├── src/                           # Core implementation
├── evaluation/                    # Comprehensive evaluation system
├── classifier/                    # CIFAR-10 classifier training
├── rl_image_tools/                # Tools for saving original/augmented images
├── full_evaluation.py             # Main evaluation entry point
└── training_script_improved.py    # Script for RL agent training
```
*Check dedicated README for more details*

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rl-image-augmentation

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Classifier (Optional)

Train a VGG19 classifier on CIFAR-10:

```bash
cd classifier
python main.py
```

This creates `./checkpoint/ckpt.pth` with the trained classifier.

### 3. Train the RL Agent (Optional)

Train the DQN agent to learn optimal augmentation policies:

```bash
python training_script_improved.py
```

This creates `./models/best_improved_dqn_model.pth` with the trained agent.

### 4. Run Comprehensive Evaluation

Compare all methods (Baseline, Fixed Aug, TTA, RL Agent):

```bash
python full_evaluation.py
```

For interactive configuration:
```bash
python full_evaluation.py --interactive
```

For quick testing:
```bash
python full_evaluation.py --quick
```

## RL Environment

### State Representation (15D)
- **Logits (10D)**: Classifier output probabilities
- **Confidence Measures (3D)**: Max probability, entropy, confidence margin
- **Correctness (1D)**: Whether prediction matches true label
- **Step Info (1D)**: Current step in episode

### Action Space (16 Actions)
- **Brightness**: ±10% adjustments
- **Contrast**: ±10% adjustments  
- **Saturation**: ±20% adjustments
- **Rotation**: ±3 degree rotations
- **Spatial**: Random crop with padding, horizontal flip
- **Noise**: Mild gaussian noise
- **Color**: Color jittering
- **Sharpness**: ±30% adjustments
- **Identity**: No-op transformation

### Reward Function
- **+10**: Fix incorrect → correct prediction
- **-10**: Break correct → incorrect prediction
- **+5×conf_change**: Confidence improvement when correct
- **+2×conf_change**: Confidence improvement when incorrect
- **Penalties**: For aggressive transformations and inefficiency

## Evaluation Methods

### 1. Baseline
Standard classifier performance without augmentation.

### 2. Fixed Augmentation  
Apply a predetermined sequence of transformations.

### 3. Test-Time Augmentation (TTA)
Apply multiple transformations and average predictions.

### 4. RL Agent
Dynamic augmentation using the trained DQN agent.

## Results & Analysis

### Typical Results using a classifier with 80% accuracy
- **Baseline**: ~80% accuracy
- **Fixed Aug**: ~78% accuracy (-0.020)
- **TTA**: ~82% accuracy (+0.020)  
- **RL Agent**: ~88% accuracy (+0.080)

*Results vary based on model training and hyperparameters*

## Advanced Usage

### Custom Evaluation Configuration

```python
from evaluation.runner import ConfigManager, create_and_run_evaluation

# Create custom config
config = ConfigManager.create_default_config()
config.tta_samples = 2000
config.rl_episodes = 2000
config.output_dir = './custom_results'

# Run evaluation
results = create_and_run_evaluation(config)
```

### Individual Method Evaluation

```python
from evaluation.methods import evaluate_rl_agent
from evaluation.core import load_classifier, get_cifar10_test_dataset

# Load components
classifier = load_classifier('./checkpoint/ckpt.pth')
agent, loaded = load_rl_agent('./models/best_improved_dqn_model.pth')
test_dataset = get_cifar10_test_dataset('./data')

# Evaluate RL agent
results = evaluate_rl_agent(
    agent=agent,
    classifier_model=classifier, 
    test_dataset=test_dataset,
    device=torch.device('cuda'),
    num_episodes=1000
)
```

### Custom Transformations

Add new transformations to `src/environment/transforms.py`:

```python
# Add to _ACTIONS_MAP
16: (lambda img: your_custom_transform(img), "Custom Transform"),
```

## Configuration Options

Key parameters in evaluation config:

```python
# Model paths
classifier_path: str = './checkpoint/ckpt.pth'
rl_model_path: str = './models/best_improved_dqn_model.pth'

# Evaluation parameters  
batch_size: int = 64
tta_samples: int = 1000      # Samples for TTA evaluation
rl_episodes: int = 1000      # Episodes for RL evaluation
max_steps_per_episode: int = 3  # Max augmentation steps

# Fixed augmentation sequence
fixed_aug_ids: list = [0, 3, 6]  # Brightness, Contrast, HFlip

# Output options
output_dir: str = './comprehensive_results'
create_plots: bool = True
save_results: bool = True
```

## Output Structure

After evaluation, results are saved to:

```
./comprehensive_results/
├── plots/
│   ├── comprehensive_comparison.png    # Main comparison plots
│   ├── confusion_matrices.png          # Confusion matrices  
│   └── rl_class_analysis.png          # RL class improvements
├── results_YYYYMMDD_HHMMSS.pkl        # Complete results
└── summary_YYYYMMDD_HHMMSS.json       # Summary metrics
```

## License

This project is licensed under the MIT License

## Authors

- **Lorenzo Barbieri** - [GitHub](https://github.com/Lorenzo31Barbieri)


