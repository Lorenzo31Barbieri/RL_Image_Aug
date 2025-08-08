# methods/__init__.py
"""
Evaluation methods for different augmentation techniques.
After cleanup: only core evaluation functions are exposed.
"""

from .evaluate_baseline import (
    evaluate_baseline,
    evaluate_baseline_with_confidence_analysis
)

from .evaluate_fixed_aug import (
    evaluate_fixed_augmentation
)

from .evaluate_tta import (
    evaluate_tta,
    ManualTTAWrapper
)

from .evaluate_rl_agent import (
    evaluate_rl_agent
)

__all__ = [
    # Baseline evaluation
    'evaluate_baseline',
    'evaluate_baseline_with_confidence_analysis',
    
    # Fixed augmentation evaluation  
    'evaluate_fixed_augmentation',
    
    # TTA evaluation
    'evaluate_tta',
    'ManualTTAWrapper',
    
    # RL agent evaluation
    'evaluate_rl_agent'
]