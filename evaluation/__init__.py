# evaluation/__init__.py (root package)
"""
Modular evaluation system for RL augmentation methods.
"""

__version__ = "1.0.0"


from .core import (
    load_classifier,
    load_rl_agent, 
    get_cifar10_test_dataset,
    get_cifar10_test_loader
)

from .methods import (
    evaluate_baseline,
    evaluate_fixed_augmentation,
    evaluate_tta,
    evaluate_rl_agent
)

# Import main comparison class - this remains unchanged for backward compatibility
from .comparison import EvaluationComparison

# New imports for the refactored components
from .comparison import ResultAggregator, VisualizationManager

__all__ = [
    'load_classifier',
    'load_rl_agent',
    'get_cifar10_test_dataset', 
    'get_cifar10_test_loader',
    'evaluate_baseline',
    'evaluate_fixed_augmentation',
    'evaluate_tta', 
    'evaluate_rl_agent',
    'EvaluationComparison',
    'ResultAggregator',
    'VisualizationManager'
]