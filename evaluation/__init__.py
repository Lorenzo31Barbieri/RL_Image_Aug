# evaluation/__init__.py (root package)
"""
Modular evaluation system for RL augmentation methods.

This package provides a comprehensive, modular system for evaluating
different augmentation methods including baseline, fixed augmentation,
test-time augmentation (TTA), and reinforcement learning agents.

Main modules:
- core: Common utilities and functions
- methods: Individual evaluation methods
- comparison: Orchestration and comparison tools
- visualization: Plotting and analysis tools

Example usage:
    from evaluation.methods import evaluate_baseline, evaluate_tta
    from evaluation.core import load_classifier, get_cifar10_test_loader
    
    classifier = load_classifier('./model.pth', device)
    test_loader = get_cifar10_test_loader()
    
    baseline_results = evaluate_baseline(classifier, test_loader, device)
    tta_results = evaluate_tta(classifier, test_dataset, device)
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import principali per comodità
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

from .comparison import EvaluationComparison

__all__ = [
    'load_classifier',
    'load_rl_agent',
    'get_cifar10_test_dataset', 
    'get_cifar10_test_loader',
    'evaluate_baseline',
    'evaluate_fixed_augmentation',
    'evaluate_tta', 
    'evaluate_rl_agent',
    'EvaluationComparison'
]