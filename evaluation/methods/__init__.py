# methods/__init__.py
"""
Evaluation methods for different augmentation techniques.
"""

from .evaluate_baseline import evaluate_baseline, run_baseline_evaluation
from .evaluate_fixed_aug import evaluate_fixed_augmentation, run_fixed_augmentation_evaluation
from .evaluate_tta import evaluate_tta, run_tta_evaluation
from .evaluate_rl_agent import evaluate_rl_agent, run_rl_agent_evaluation

__all__ = [
    'evaluate_baseline',
    'run_baseline_evaluation',
    'evaluate_fixed_augmentation', 
    'run_fixed_augmentation_evaluation',
    'evaluate_tta',
    'run_tta_evaluation',
    'evaluate_rl_agent',
    'run_rl_agent_evaluation'
]
