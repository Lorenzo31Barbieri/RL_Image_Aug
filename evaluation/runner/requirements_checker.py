#!/usr/bin/env python3
"""
Requirements Checker for Evaluation System
==========================================

Validates that all required files, directories, and dependencies are available.
"""

import os
import torch
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from .config_manager import EvaluationConfig


@dataclass
class RequirementResult:
    """Result of a requirement check."""
    name: str
    passed: bool
    message: str
    is_critical: bool = True


class RequirementsChecker:
    """Validates system requirements for evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.results: List[RequirementResult] = []
    
    def check_all_requirements(self) -> bool:
        """
        Run all requirement checks.
        
        Returns:
            bool: True if all critical requirements pass
        """
        self.results = []
        
        # Check files
        self._check_required_files()
        self._check_optional_files()
        
        # Check directories
        self._check_directories()
        
        # Check system
        self._check_system_requirements()
        
        # Print results
        self._print_results()
        
        # Return success status
        critical_failures = [r for r in self.results if not r.passed and r.is_critical]
        return len(critical_failures) == 0
    
    def _check_required_files(self) -> None:
        """Check for required files."""
        required_files = [
            self.config.classifier_path,
        ]
        
        for file_path in required_files:
            if os.path.exists(file_path):
                self.results.append(RequirementResult(
                    name=f"Required file: {file_path}",
                    passed=True,
                    message=f"✅ Found: {file_path}",
                    is_critical=True
                ))
            else:
                self.results.append(RequirementResult(
                    name=f"Required file: {file_path}",
                    passed=False,
                    message=f"❌ Missing: {file_path}",
                    is_critical=True
                ))
    
    def _check_optional_files(self) -> None:
        """Check for optional files."""
        optional_files = [
            (self.config.rl_model_path, "RL evaluation will use random agent"),
        ]
        
        for file_path, fallback_msg in optional_files:
            if os.path.exists(file_path):
                self.results.append(RequirementResult(
                    name=f"Optional file: {file_path}",
                    passed=True,
                    message=f"✅ Found: {file_path}",
                    is_critical=False
                ))
            else:
                self.results.append(RequirementResult(
                    name=f"Optional file: {file_path}",
                    passed=False,
                    message=f"⚠️  Missing: {file_path} ({fallback_msg})",
                    is_critical=False
                ))
    
    def _check_directories(self) -> None:
        """Check and create required directories."""
        directories = [
            self.config.data_root,
            self.config.output_dir,
        ]
        
        for dir_path in directories:
            if os.path.exists(dir_path):
                self.results.append(RequirementResult(
                    name=f"Directory: {dir_path}",
                    passed=True,
                    message=f"✅ Found directory: {dir_path}",
                    is_critical=True
                ))
            else:
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    self.results.append(RequirementResult(
                        name=f"Directory: {dir_path}",
                        passed=True,
                        message=f"📁 Created directory: {dir_path}",
                        is_critical=True
                    ))
                except Exception as e:
                    self.results.append(RequirementResult(
                        name=f"Directory: {dir_path}",
                        passed=False,
                        message=f"❌ Cannot create directory: {dir_path} ({e})",
                        is_critical=True
                    ))
    
    def _check_system_requirements(self) -> None:
        """Check system and hardware requirements."""
        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            self.results.append(RequirementResult(
                name="CUDA Support",
                passed=True,
                message=f"✅ CUDA available: {device_count} device(s), {device_name}",
                is_critical=False
            ))
        else:
            self.results.append(RequirementResult(
                name="CUDA Support",
                passed=False,
                message="⚠️  CUDA not available, using CPU (slower evaluation)",
                is_critical=False
            ))
        
        # Check Python version
        import sys
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        if sys.version_info >= (3, 8):
            self.results.append(RequirementResult(
                name="Python Version",
                passed=True,
                message=f"✅ Python {python_version} (supported)",
                is_critical=False
            ))
        else:
            self.results.append(RequirementResult(
                name="Python Version",
                passed=False,
                message=f"⚠️  Python {python_version} (recommend 3.8+)",
                is_critical=False
            ))
    
    def _print_results(self) -> None:
        """Print formatted requirement check results."""
        print("🔍 REQUIREMENTS CHECK")
        print("=" * 60)
        
        # Group results
        critical_passed = [r for r in self.results if r.passed and r.is_critical]
        critical_failed = [r for r in self.results if not r.passed and r.is_critical]
        optional_results = [r for r in self.results if not r.is_critical]
        
        # Print critical results
        if critical_passed:
            for result in critical_passed:
                print(result.message)
        
        if critical_failed:
            print("\n❌ CRITICAL ISSUES:")
            for result in critical_failed:
                print(f"  {result.message}")
        
        # Print optional results
        if optional_results:
            print("\n💡 OPTIONAL/INFO:")
            for result in optional_results:
                print(f"  {result.message}")
        
        # Summary
        if critical_failed:
            print(f"\n❌ Requirements check failed! Please fix {len(critical_failed)} critical issue(s).")
            print("💡 Required:")
            print("   - Trained VGG19 classifier in ./checkpoint/ckpt.pth")
            print("   - CIFAR-10 data will be downloaded automatically")
            print("   - RL model is optional (will use random agent if missing)")
        else:
            print(f"\n✅ All critical requirements satisfied!")
        
        print("=" * 60)
    
    def get_failed_critical_requirements(self) -> List[RequirementResult]:
        """Get list of failed critical requirements."""
        return [r for r in self.results if not r.passed and r.is_critical]