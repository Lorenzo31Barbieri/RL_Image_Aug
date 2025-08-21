"""
Configuration utilities for evaluation comparison.
"""

def create_default_config():
    """
    Crea una configurazione di default per EvaluationComparison.
    
    Returns:
        Dict con configurazione di default
    """
    return {
        # Percorsi dei modelli
        'classifier_path': './checkpoint/ckpt.pth',
        # 'rl_model_path': './models/best_improved_dqn_model.pth',
        'rl_model_path': './models/enhanced_dqn_episode_72000.pth',
        'data_root': './data',
        
        # Parametri di valutazione
        'batch_size': 64,
        'tta_samples': 1000,
        'rl_episodes': 1000,
        'max_steps_per_episode': 3,
        'state_dim': 15,  # NUM_CLASSES + 5
        
        # Configurazione metodi
        'evaluate_baseline': True,
        'evaluate_fixed_aug': True,
        'evaluate_tta': True,
        'evaluate_rl': True,
        
        # Parametri Fixed Augmentation
        'fixed_aug_ids': [0, 3, 6],  # Brightness, Contrast, HFlip
        
        # Parametri TTA
        'use_ttach': True,
        
        # Output
        'output_dir': './evaluation_results',
        'save_detailed_results': True,
        'create_plots': True
    }


def create_quick_test_config():
    """
    Configurazione per test rapidi con campioni ridotti.
    
    Returns:
        Dict con configurazione per test rapidi
    """
    config = create_default_config()
    config.update({
        'batch_size': 32,
        'tta_samples': 200,
        'rl_episodes': 200,
        'output_dir': './quick_test_results'
    })
    return config


def create_comprehensive_config():
    """
    Configurazione per valutazione completa e dettagliata.
    
    Returns:
        Dict con configurazione completa
    """
    config = create_default_config()
    config.update({
        'batch_size': 128,
        'tta_samples': 2000,
        'rl_episodes': 2000,
        'save_detailed_results': True,
        'create_plots': True,
        'output_dir': './comprehensive_results'
    })
    return config


def validate_config(config):
    """
    Valida una configurazione.
    
    Args:
        config: Dizionario di configurazione
    
    Raises:
        ValueError: Se la configurazione non è valida
    """
    required_keys = [
        'classifier_path', 'data_root', 'batch_size', 
        'tta_samples', 'rl_episodes', 'output_dir'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validazioni specifiche
    if config['batch_size'] <= 0:
        raise ValueError("batch_size must be positive")
    
    if config['tta_samples'] <= 0:
        raise ValueError("tta_samples must be positive")
    
    if config['rl_episodes'] <= 0:
        raise ValueError("rl_episodes must be positive")
    
    print("Configuration validated successfully")


def print_config(config):
    """
    Stampa una configurazione in modo leggibile.
    
    Args:
        config: Dizionario di configurazione
    """
    print("Evaluation Configuration:")
    print("-" * 40)
    
    print("Model Paths:")
    print(f"  Classifier: {config.get('classifier_path', 'Not set')}")
    print(f"  RL Model: {config.get('rl_model_path', 'Not set')}")
    print(f"  Data Root: {config.get('data_root', 'Not set')}")
    
    print("\nEvaluation Parameters:")
    print(f"  Batch Size: {config.get('batch_size', 'Not set')}")
    print(f"  TTA Samples: {config.get('tta_samples', 'Not set')}")
    print(f"  RL Episodes: {config.get('rl_episodes', 'Not set')}")
    print(f"  Max Steps per Episode: {config.get('max_steps_per_episode', 'Not set')}")
    
    print("\nMethods Enabled:")
    print(f"  Baseline: {'OK' if config.get('evaluate_baseline', False) else 'NOT ENABLED'}")
    print(f"  Fixed Augmentation: {'OK' if config.get('evaluate_fixed_aug', False) else 'NOT ENABLED'}")
    print(f"  TTA: {'OK' if config.get('evaluate_tta', False) else 'NOT ENABLED'}")
    print(f"  RL Agent: {'OK' if config.get('evaluate_rl', False) else 'NOT ENABLED'}")
    
    print(f"\nOutput: {config.get('output_dir', 'Not set')}")
    print("-" * 40)