# Evaluation Package

Comprehensive evaluation system for comparing different augmentation methods.

## Structure

- `core/` - Core evaluation utilities and data loading functions
- `methods/` - Individual evaluation methods for each approach
- `comparison/` - Orchestration and comparison tools
- `runner/` - Interactive and automated evaluation runners

## Evaluation Methods

### Baseline
Standard classifier performance without augmentation.

### Fixed Augmentation
Applies a predefined sequence of transformations to all images.

### Test-Time Augmentation (TTA)
Uses multiple augmented versions of each image for ensemble prediction.

### RL Agent
Applies augmentation actions selected based on DQN Q-value predictions.

## Usage

### Quick Evaluation
```bash
python -m evaluation.runner.interactive_runner
```

### Programmatic Usage
```python
from evaluation.comparison import EvaluationComparison
from evaluation.runner import ConfigManager

config = ConfigManager.create_default_config()
comparison = EvaluationComparison(config.__dict__)
comparison.load_models()
comparison.load_data()
comparison.run_all_evaluations()
```

## Output

- Comprehensive accuracy and performance comparisons
- Confusion matrices and class-wise analysis
- Visualization plots and charts
- Detailed evaluation reports
