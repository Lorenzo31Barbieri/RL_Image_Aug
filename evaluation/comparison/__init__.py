# comparison/__init__.py
"""
Comparison and orchestration modules.
"""

from .comparison_runner import EvaluationComparison
from .result_aggregator import ResultAggregator
from .visualization_manager import VisualizationManager

__all__ = [
    'EvaluationComparison',
    'ResultAggregator', 
    'VisualizationManager'
]