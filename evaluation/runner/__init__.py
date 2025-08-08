# evaluation/runner/__init__.py
"""
Evaluation Runner Package
=========================

Modular components for orchestrating and managing comprehensive model evaluations.
"""

from .config_manager import ConfigManager, EvaluationConfig
from .requirements_checker import RequirementsChecker
from .evaluation_orchestrator import EvaluationOrchestrator, create_and_run_evaluation
from .output_manager import OutputManager, ResultsFormatter
from .interactive_runner import InteractiveRunner

__all__ = [
    'ConfigManager',
    'EvaluationConfig', 
    'RequirementsChecker',
    'EvaluationOrchestrator',
    'create_and_run_evaluation',
    'OutputManager',
    'ResultsFormatter',
    'InteractiveRunner'
]

__version__ = "1.0.0"