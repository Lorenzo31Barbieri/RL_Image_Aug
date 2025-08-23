import torch
import os
from typing import Dict, Any, Optional, Tuple
from src.models.vgg import VGG
from src.models.agent import DQNAgent

# --- GLOBAL CONFIGURATION ---
DEFAULT_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
DEFAULT_RL_MODEL_PATH = './models/enhanced_dqn_episode_72000.pth'
DEFAULT_IMAGE_FEATURE_DIM = 128


def load_classifier(model_path: str = DEFAULT_CLASSIFIER_PATH,
                   device: torch.device = None,
                   model_architecture: str = 'VGG19') -> torch.nn.Module:
    """
    Carica il classificatore pre-trained.
    
    Args:
        model_path: Percorso del file del modello
        device: Device su cui caricare il modello
        model_architecture: Architettura del modello
    
    Returns:
        Modello caricato e configurato per valutazione
    
    Raises:
        FileNotFoundError: Se il file del modello non esiste
        ValueError: Se il checkpoint ha formato non valido
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading pre-trained {model_architecture} CIFAR10 classifier...")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    
    # Controlla se il file esiste
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model not found at {model_path}")
    
    # Inizializza il modello
    classifier_model = VGG(model_architecture).to(device)
    
    try:
        # Carica il checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Gestisce diversi formati di checkpoint
        if isinstance(checkpoint, dict):
            if 'net' in checkpoint:
                state_dict = checkpoint['net']
                accuracy_info = checkpoint.get('acc', 'Unknown')
            elif 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                accuracy_info = checkpoint.get('accuracy', 'Unknown')
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                accuracy_info = 'Unknown'
            else:
                # Assume che il checkpoint sia direttamente il state_dict
                state_dict = checkpoint
                accuracy_info = 'Unknown'
        else:
            # Assume che il checkpoint sia direttamente il state_dict
            state_dict = checkpoint
            accuracy_info = 'Unknown'
        
        # Rimuovi prefisso 'module.' se presente (per modelli DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        # Carica i pesi nel modello
        classifier_model.load_state_dict(new_state_dict, strict=True)
        
        print(f"Successfully loaded classifier from {model_path}")
        if accuracy_info != 'Unknown':
            print(f"Reported accuracy: {accuracy_info}")
        
    except Exception as e:
        print(f"Error loading classifier: {e}")
        raise ValueError(f"Failed to load classifier from {model_path}: {e}")
    
    # Configura il modello per valutazione
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    
    print("Classifier loaded and frozen for evaluation")
    return classifier_model


def detect_model_state_dimension(model_path: str) -> Optional[int]:
    """
    Detect the state dimension that a model was trained with.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        State dimension if detected, None otherwise
    """
    if not os.path.exists(model_path):
        return None
        
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        
        # Find input dimension from first layer
        for key, tensor in state_dict.items():
            if 'fc1.weight' in key:
                return tensor.shape[1]
                
    except Exception:
        pass
    
    return None


def load_rl_agent(model_path: str = DEFAULT_RL_MODEL_PATH,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: torch.device = None,
                 image_feature_dim: int = DEFAULT_IMAGE_FEATURE_DIM) -> Tuple[DQNAgent, bool]:
    """
    Load RL agent with automatic dimension detection and compatibility handling.
    
    Args:
        model_path: Path to the RL model file
        state_dim: State space dimension (auto-detected if None)
        action_dim: Action space dimension
        device: Device for computation
        image_feature_dim: Image feature dimension (ignored if model has different dims)
    
    Returns:
        Tuple (agent, model_loaded) where model_loaded indicates successful loading
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if action_dim is None:
        # Import dynamically to avoid circular dependencies
        try:
            from src.environment.transforms import get_num_actions
            action_dim = get_num_actions()
        except ImportError:
            print("Warning: Could not import get_num_actions, using default action_dim=16")
            action_dim = 16
    
    print(f"Loading RL agent...")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    
    # Try to detect state dimension from existing model
    detected_state_dim = detect_model_state_dimension(model_path)
    
    # Try alternative model paths if primary doesn't exist
    model_paths_to_try = [
        model_path,
        './models/best_enhanced_dqn_model.pth',
        './models/final_enhanced_dqn_model.pth',
        './models/enhanced_dqn_episode_72000.pth',
        './models/best_improved_dqn_model.pth',
        './models/final_improved_dqn_model.pth'
    ]
    
    actual_model_path = None
    actual_state_dim = None
    
    # Find the first existing model and detect its dimensions
    for path_to_try in model_paths_to_try:
        if os.path.exists(path_to_try):
            detected_dim = detect_model_state_dimension(path_to_try)
            if detected_dim:
                actual_model_path = path_to_try
                actual_state_dim = detected_dim
                break
    
    if actual_state_dim is None:
        # No model found, use provided dimensions or defaults
        if state_dim is None:
            print("No model found for dimension detection. Using original state dimension (15).")
            actual_state_dim = 15  # Original dimension for backward compatibility
            image_feature_dim = 0
        else:
            actual_state_dim = state_dim
            
        agent = DQNAgent(actual_state_dim, action_dim, device)
        model_loaded = False
        
        print(f"Initialized random RL agent with state_dim={actual_state_dim}, action_dim={action_dim}")
        return agent, model_loaded
    
    # Calculate image feature dimension from detected state dim
    base_dim = 15  # 10 logits + 5 additional features
    if actual_state_dim > base_dim:
        actual_image_feature_dim = actual_state_dim - base_dim
        model_type = "enhanced"
    else:
        actual_image_feature_dim = 0
        model_type = "original"
    
    print(f"Detected model: {actual_model_path}")
    print(f"Model type: {model_type}")
    print(f"State dimension: {actual_state_dim}")
    print(f"Image features: {actual_image_feature_dim}")
    
    # Initialize agent with detected dimensions
    agent = DQNAgent(actual_state_dim, action_dim, device)
    model_loaded = False
    
    try:
        # Load the model weights
        state_dict = torch.load(actual_model_path, map_location=device)
        
        # Load both Q-network and target Q-network
        agent.q_network.load_state_dict(state_dict)
        agent.target_q_network.load_state_dict(state_dict)
        
        # Configure for evaluation
        agent.q_network.eval()
        agent.target_q_network.eval()
        agent.epsilon = 0  # Disable exploration for evaluation
        
        model_loaded = True
        print(f"Successfully loaded RL agent from {actual_model_path}")
        
    except Exception as e:
        print(f"Error loading RL agent from {actual_model_path}: {e}")
        print("Using randomly initialized agent...")
    
    # Store detected dimensions in agent for compatibility
    agent.detected_state_dim = actual_state_dim
    agent.detected_image_feature_dim = actual_image_feature_dim
    
    return agent, model_loaded


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Estrae informazioni su un modello PyTorch.
    
    Args:
        model: Modello PyTorch
    
    Returns:
        Dict con informazioni sul modello
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Determina il device del modello
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = "No parameters"
    
    # Try to get input dimension if it's a DQN agent
    input_dim = "Unknown"
    if hasattr(model, 'fc1') and hasattr(model.fc1, 'weight'):
        input_dim = model.fc1.weight.shape[1]
    elif hasattr(model, 'layers') and len(model.layers) > 0:
        first_layer = model.layers[0]
        if hasattr(first_layer, 'weight'):
            input_dim = first_layer.weight.shape[1]
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_device': str(model_device),
        'is_training': model.training,
        'model_type': type(model).__name__,
        'input_dimension': input_dim
    }


def validate_model_compatibility(classifier: torch.nn.Module,
                               agent: Optional[DQNAgent] = None,
                               expected_num_classes: int = 10) -> None:
    """
    Validate model compatibility with automatic dimension detection.
    
    Args:
        classifier: Classifier model
        agent: RL agent (optional)
        expected_num_classes: Expected number of classes
    
    Raises:
        ValueError: If models are not compatible
    """
    print("Validating model compatibility...")
    
    # Validate classifier
    classifier_info = get_model_info(classifier)
    print(f"Classifier info: {classifier_info['model_type']} with {classifier_info['total_parameters']:,} parameters")
    
    # Test classifier with dummy input
    try:
        dummy_input = torch.randn(1, 3, 32, 32).to(next(classifier.parameters()).device)
        with torch.no_grad():
            output = classifier(dummy_input)
        
        if output.shape[1] != expected_num_classes:
            raise ValueError(f"Classifier output shape {output.shape[1]} doesn't match expected classes {expected_num_classes}")
        
        print(f"Classifier validation passed - output shape: {output.shape}")
        
    except Exception as e:
        raise ValueError(f"Classifier validation failed: {e}")
    
    # Validate RL agent if provided
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        actual_state_dim = getattr(agent, 'state_dim', 'Unknown')
        detected_state_dim = getattr(agent, 'detected_state_dim', 'Unknown')
        detected_image_features = getattr(agent, 'detected_image_feature_dim', 0)
        
        print(f"RL Agent info: {agent_info['model_type']} with {agent_info['total_parameters']:,} parameters")
        print(f"Agent state dimension: {actual_state_dim}")
        print(f"Detected state dimension: {detected_state_dim}")
        print(f"Detected image features: {detected_image_features}")
        
        # Test agent with dummy state
        try:
            test_state_dim = detected_state_dim if detected_state_dim != 'Unknown' else actual_state_dim
            dummy_state = torch.randn(1, test_state_dim).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(dummy_state)
            
            if q_values.shape[1] != agent.action_dim:
                raise ValueError(f"Agent output shape {q_values.shape[1]} doesn't match expected actions {agent.action_dim}")
            
            print(f"RL Agent validation passed - Q-values shape: {q_values.shape}")
            
        except Exception as e:
            raise ValueError(f"RL Agent validation failed: {e}")
    
    print("All models validated successfully!")


def print_loading_summary(classifier: torch.nn.Module,
                        agent: Optional[DQNAgent] = None,
                        agent_loaded: bool = False) -> None:
    """
    Print a summary of loaded models with dimension information.
    
    Args:
        classifier: Loaded classifier model
        agent: RL agent (optional)
        agent_loaded: Whether agent was loaded successfully
    """
    print(f"\n{'='*60}")
    print("MODEL LOADING SUMMARY")
    print(f"{'='*60}")
    
    # Classifier info
    classifier_info = get_model_info(classifier)
    print(f"🎯 CLASSIFIER ({classifier_info['model_type']}):")
    print(f"  Status: ✅ Loaded and ready")
    print(f"  Parameters: {classifier_info['total_parameters']:,}")
    print(f"  Training mode: {'ON' if classifier_info['is_training'] else 'OFF (Evaluation)'}")
    print(f"  Device: {classifier_info['model_device']}")
    
    # RL agent info
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        actual_state_dim = getattr(agent, 'state_dim', 'Unknown')
        detected_state_dim = getattr(agent, 'detected_state_dim', 'Unknown')
        detected_image_features = getattr(agent, 'detected_image_feature_dim', 0)
        
        status = "✅ Loaded from checkpoint" if agent_loaded else "⚠️ Random initialization"
        
        print(f"\n🤖 RL AGENT ({agent_info['model_type']}):")
        print(f"  Status: {status}")
        print(f"  Parameters: {agent_info['total_parameters']:,}")
        print(f"  State dim: {actual_state_dim}, Action dim: {agent.action_dim}")
        
        # Dimension breakdown
        if detected_state_dim != 'Unknown':
            if detected_image_features > 0:
                print(f"  Model Type: Enhanced (with image features)")
                print(f"  State Breakdown:")
                print(f"    - Logits: 10")
                print(f"    - Additional features: 5")
                print(f"    - Image features: {detected_image_features}")
                print(f"    - Total: {detected_state_dim}")
            else:
                print(f"  Model Type: Original (no image features)")
                print(f"  State Breakdown:")
                print(f"    - Logits: 10")
                print(f"    - Additional features: 5")
                print(f"    - Total: {detected_state_dim}")
        
        print(f"  Epsilon: {agent.epsilon} (exploration disabled)")
        print(f"  Device: {agent_info['model_device']}")
        print(f"  Network input dim: {agent_info['input_dimension']}")
    
    print(f"{'='*60}")


def detect_available_rl_models(base_dir: str = './models') -> Dict[str, Dict[str, Any]]:
    """
    Detect available RL models and their characteristics.
    
    Args:
        base_dir: Base directory to search for models
    
    Returns:
        Dict with information about found models
    """
    available_models = {}
    
    if not os.path.exists(base_dir):
        return available_models
    
    for filename in os.listdir(base_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(base_dir, filename)
            
            try:
                # Load model to inspect dimensions
                state_dict = torch.load(filepath, map_location='cpu')
                
                # Find input dimension
                input_dim = None
                for key, tensor in state_dict.items():
                    if 'fc1.weight' in key:
                        input_dim = tensor.shape[1]
                        break
                
                # Classify model type
                model_type = "unknown"
                if input_dim:
                    if input_dim == 15:  # Original state dim
                        model_type = "original"
                    elif input_dim > 15:  # Enhanced state dim
                        model_type = "enhanced"
                        
                    # Calculate image feature dimension
                    if input_dim > 15:
                        img_features = input_dim - 15  # Subtract base dimensions
                    else:
                        img_features = 0
                
                available_models[filename] = {
                    'path': filepath,
                    'type': model_type,
                    'state_dim': input_dim,
                    'image_features': img_features if input_dim else 0,
                    'file_size': os.path.getsize(filepath),
                    'enhanced': model_type == "enhanced"
                }
                
            except Exception as e:
                print(f"Could not inspect model {filename}: {e}")
    
    return available_models


def print_available_models() -> None:
    """Print available RL models."""
    models = detect_available_rl_models()
    
    if not models:
        print("No RL models found in ./models/")
        return
    
    print(f"\n📁 AVAILABLE RL MODELS:")
    print("-" * 60)
    
    enhanced_models = []
    original_models = []
    
    for filename, info in models.items():
        if info['enhanced']:
            enhanced_models.append((filename, info))
        else:
            original_models.append((filename, info))
    
    if enhanced_models:
        print("🚀 Enhanced Models (with image features):")
        for filename, info in enhanced_models:
            size_mb = info['file_size'] / (1024 * 1024)
            print(f"  {filename}")
            print(f"    State dim: {info['state_dim']}, Image features: {info['image_features']}")
            print(f"    Size: {size_mb:.1f} MB")
    
    if original_models:
        print("\n📊 Original Models (no image features):")
        for filename, info in original_models:
            size_mb = info['file_size'] / (1024 * 1024)
            print(f"  {filename}")
            print(f"    State dim: {info['state_dim']}")
            print(f"    Size: {size_mb:.1f} MB")
    
    print("-" * 60)