# evaluation/runner/__init__.py
"""
Evaluation Runner Package
=========================

Modular components for orchestrating and managing comprehensive model evaluations.

This package provides a clean separation of concerns for the evaluation process:

- ConfigManager: Handles configuration creation and validation
- RequirementsChecker: Validates system requirements and dependencies  
- EvaluationOrchestrator: Coordinates the complete evaluation pipeline
- OutputManager: Manages result formatting, saving, and presentation
- InteractiveRunner: Provides user-friendly interactive interface

Example usage:
    from evaluation.runner import create_and_run_evaluation, ConfigManager
    
    config = ConfigManager.create_default_config()
    results = create_and_run_evaluation(config)
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