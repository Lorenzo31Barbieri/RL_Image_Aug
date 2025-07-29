#!/usr/bin/env python3
"""
Script per migrare il progetto alla nuova struttura.
Sposta i file e aggiorna gli import automaticamente.
"""

import os
import shutil
import sys
from pathlib import Path

def create_directory_structure():
    """Crea la nuova struttura di directory."""
    directories = [
        'src',
        'src/models',
        'src/environment', 
        'src/data',
        'src/utils',
        'scripts',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Crea file __init__.py
        init_file = Path(directory) / '__init__.py'
        if not init_file.exists() and directory.startswith('src'):
            init_file.touch()
    
    print("✅ Directory structure created")

def move_files():
    """Sposta i file nella nuova struttura."""
    moves = [
        # Source files to src/
        ('vgg.py', 'src/models/vgg.py'),
        ('agent.py', 'src/models/agent.py'),
        ('environment.py', 'src/environment/environment.py'),
        ('transforms.py', 'src/environment/transforms.py'),
        ('augmented_image_buffer.py', 'src/data/augmented_image_buffer.py'),
        ('utils.py', 'src/utils/utils.py'),
        
        # Scripts to scripts/
        ('training_script_improved.py', 'scripts/training_script_improved.py'),
        ('evaluation_script.py', 'scripts/evaluation_script.py'),
        ('fixed_augmentation_evaluation.py', 'scripts/fixed_augmentation_evaluation.py'),
        ('tta.py', 'scripts/tta.py'),
        ('tta_rl_agent_evaluation.py', 'scripts/tta_rl_agent_evaluation.py'),
    ]
    
    for source, destination in moves:
        if os.path.exists(source):
            try:
                shutil.move(source, destination)
                print(f"✅ Moved {source} -> {destination}")
            except Exception as e:
                print(f"❌ Error moving {source}: {e}")
        else:
            print(f"⚠️  File not found: {source}")

def update_imports_in_file(file_path, import_mappings):
    """Aggiorna gli import in un singolo file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Applica le sostituzioni degli import
        for old_import, new_import in import_mappings.items():
            content = content.replace(old_import, new_import)
        
        # Scrivi solo se ci sono stati cambiamenti
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        return False
    except Exception as e:
        print(f"❌ Error updating {file_path}: {e}")
        return False

def update_all_imports():
    """Aggiorna tutti gli import nel progetto."""
    
    # Mappings per gli import
    import_mappings = {
        # Basic imports
        'from src.models.vgg import': 'from src.models.vgg import',
        'from src.models.agent import': 'from src.models.agent import', 
        'from src.environment.environment import': 'from src.environment.environment import',
        'from src.environment.transforms import': 'from src.environment.transforms import',
        'import src.environment.transforms as transforms': 'import src.environment.transforms as transforms',
        'from src.data.augmented_image_buffer import': 'from src.data.augmented_image_buffer import',
        'from src.utils.utils import': 'from src.utils.utils import',
        
        # Specific class imports that might be used
        'from src.models.vgg import VGG': 'from src.models.vgg import VGG',
        'from src.models.agent import DQNAgent': 'from src.models.agent import DQNAgent',
        'from src.environment.environment import ImageAugmentationEnv': 'from src.environment.environment import ImageAugmentationEnv',
        
        # Transform specific imports 
        'src.environment.transforms.get_num_actions': 'src.environment.src.environment.transforms.get_num_actions',
        'src.environment.transforms.get_action_name': 'src.environment.src.environment.transforms.get_action_name',
        'src.environment.transforms.get_action_transform': 'src.environment.src.environment.transforms.get_action_transform',
        'src.environment.transforms._ACTIONS_MAP': 'src.environment.src.environment.transforms._ACTIONS_MAP',
    }
    
    # File da aggiornare
    files_to_update = []
    
    # Trova tutti i file Python
    for root, dirs, files in os.walk('.'):
        # Salta alcune directory
        if any(skip_dir in root for skip_dir in ['.git', '__pycache__', '.pytest_cache', 'venv', 'env']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                files_to_update.append(os.path.join(root, file))
    
    updated_count = 0
    for file_path in files_to_update:
        if update_imports_in_file(file_path, import_mappings):
            print(f"✅ Updated imports in {file_path}")
            updated_count += 1
    
    print(f"\n✅ Updated imports in {updated_count} files")

def create_init_files():
    """Crea i file __init__.py con i contenuti corretti."""
    
    init_files_content = {
        'src/__init__.py': '''# src/__init__.py
"""
Main source code package for the RL Image Augmentation project.
"""

__version__ = "1.0.0"

# Import commonly used classes for easier access
from .models.vgg import VGG
from .models.agent import DQNAgent
from .environment.environment import ImageAugmentationEnv
from .environment.transforms import get_num_actions, get_action_transform, get_action_name
from .data.augmented_image_buffer import AugmentedImageBuffer

__all__ = [
    'VGG',
    'DQNAgent', 
    'ImageAugmentationEnv',
    'get_num_actions',
    'get_action_transform',
    'get_action_name',
    'AugmentedImageBuffer'
]''',
        
        'src/models/__init__.py': '''# src/models/__init__.py
"""
Neural network models for the project.
"""

from .vgg import VGG
from .agent import DQNAgent, QNetwork, PrioritizedReplayBuffer

__all__ = [
    'VGG',
    'DQNAgent',
    'QNetwork', 
    'PrioritizedReplayBuffer'
]''',

        'src/environment/__init__.py': '''# src/environment/__init__.py
"""
RL Environment and transformations for image augmentation.
"""

from .environment import ImageAugmentationEnv
from .transforms import (
    get_num_actions,
    get_action_transform, 
    get_action_name,
    get_all_transforms,
    get_conservative_actions,
    get_aggressive_actions,
    ACTION_CATEGORIES
)

__all__ = [
    'ImageAugmentationEnv',
    'get_num_actions',
    'get_action_transform',
    'get_action_name', 
    'get_all_transforms',
    'get_conservative_actions',
    'get_aggressive_actions',
    'ACTION_CATEGORIES'
]''',

        'src/data/__init__.py': '''# src/data/__init__.py
"""
Data handling utilities.
"""

from .augmented_image_buffer import AugmentedImageBuffer

__all__ = [
    'AugmentedImageBuffer'
]''',

        'src/utils/__init__.py': '''# src/utils/__init__.py
"""
General utility functions.
"""

from .utils import (
    get_mean_and_std,
    init_params,
    progress_bar,
    format_time
)

__all__ = [
    'get_mean_and_std',
    'init_params', 
    'progress_bar',
    'format_time'
]'''
    }
    
    for file_path, content in init_files_content.items():
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created {file_path}")
        except Exception as e:
            print(f"❌ Error creating {file_path}: {e}")

def create_project_readme():
    """Crea un README aggiornato per la nuova struttura."""
    readme_content = '''# RL Image Augmentation Project

## Project Structure

```
project_root/
├── src/                          # Main source code
│   ├── models/                   # Model definitions
│   │   ├── vgg.py               # VGG classifier
│   │   └── agent.py             # DQN Agent
│   ├── environment/             # RL Environment
│   │   ├── environment.py       # ImageAugmentationEnv
│   │   └── transforms.py        # Image transformations
│   ├── data/                    # Data utilities
│   │   └── augmented_image_buffer.py
│   └── utils/                   # General utilities
│       └── utils.py
├── evaluation/                  # Evaluation system
│   ├── core/                   # Core evaluation utilities
│   ├── methods/                # Individual evaluation methods
│   ├── comparison/             # Comparison tools
│   └── visualization/          # Plotting tools
├── scripts/                    # Training and evaluation scripts
│   ├── training_script_improved.py
│   ├── evaluation_script.py
│   ├── fixed_augmentation_evaluation.py
│   ├── tta.py
│   └── tta_rl_agent_evaluation.py
├── data/                       # Dataset storage
├── checkpoint/                 # Model checkpoints
├── models/                     # Trained models
└── results/                    # Evaluation results
```

## Quick Start

1. **Training**: Run training script
   ```bash
   python scripts/training_script_improved.py
   ```

2. **Evaluation**: Use the modular evaluation system
   ```python
   from evaluation.comparison import EvaluationComparison
   from evaluation.core import load_classifier
   
   # Load model
   classifier = load_classifier('./checkpoint/ckpt.pth')
   
   # Run comprehensive evaluation
   comparison = EvaluationComparison(config)
   comparison.run_all_evaluations()
   ```

3. **Individual Evaluations**:
   ```python
   from evaluation.methods import evaluate_baseline, evaluate_rl_agent
   
   # Evaluate baseline
   results = evaluate_baseline(classifier, test_loader, device)
   
   # Evaluate RL agent  
   results = evaluate_rl_agent(agent, classifier, test_dataset, device)
   ```

## Import Structure

After migration, use these imports:
```python
# Models
from src.models import VGG, DQNAgent

# Environment
from src.environment import ImageAugmentationEnv, get_num_actions

# Data utilities
from src.data import AugmentedImageBuffer

# Evaluation system
from evaluation.methods import evaluate_baseline, evaluate_rl_agent
from evaluation.comparison import EvaluationComparison
```
'''
    
    try:
        with open('README_NEW.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        print("✅ Created README_NEW.md")
    except Exception as e:
        print(f"❌ Error creating README: {e}")


def main():
    """Funzione principale per la migrazione."""
    print("🚀 Starting project migration...")
    print("=" * 50)
    
    # Step 1: Crea struttura directory
    print("\n📁 Step 1: Creating directory structure...")
    create_directory_structure()
    
    # Step 2: Sposta i file
    print("\n📦 Step 2: Moving files...")
    move_files()
    
    # Step 3: Crea file __init__.py
    print("\n📝 Step 3: Creating __init__.py files...")
    create_init_files()
    
    # Step 4: Aggiorna gli import
    print("\n🔄 Step 4: Updating imports...")
    update_all_imports()
    
    # Step 5: Crea README
    print("\n📚 Step 5: Creating updated README...")
    create_project_readme()
    
    print("\n" + "=" * 50)
    print("🎉 Migration completed!")
    print("\n📋 Post-migration checklist:")
    print("1. ✅ Files moved to new structure")
    print("2. ✅ Imports updated automatically") 
    print("3. ✅ __init__.py files created")
    print("4. ✅ README updated")
    print("\n💡 Next steps:")
    print("- Test the evaluation system: python -m evaluation.usage_example")
    print("- Run training: python scripts/training_script_improved.py")
    print("- Check for any remaining import errors")
    print("- Update any custom scripts you may have")


if __name__ == '__main__':
    # Conferma prima di procedere
    print("⚠️  This will restructure your project and move files.")
    print("📋 Files to be moved:")
    print("   - vgg.py -> src/models/")
    print("   - agent.py -> src/models/")
    print("   - environment.py -> src/environment/")
    print("   - transforms.py -> src/environment/")
    print("   - augmented_image_buffer.py -> src/data/")
    print("   - utils.py -> src/utils/")
    print("   - Training/evaluation scripts -> scripts/")
    print("\n🔄 All Python files will have imports updated automatically.")
    
    response = input("\n❓ Do you want to proceed? (y/N): ").lower().strip()
    
    if response in ['y', 'yes']:
        main()
    else:
        print("❌ Migration cancelled.")
        sys.exit(0)