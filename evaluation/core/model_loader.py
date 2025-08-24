import torch
import os
from typing import Dict, Any, Tuple
from src.models.vgg import VGG
from src.models.agent import DQNAgent

# Import centralized configuration
from config.evaluation_config import *

def load_classifier(model_path: str = None,
                   device: torch.device = None,
                   model_architecture: str = 'VGG19') -> torch.nn.Module:
    """
    Load the pre-trained classifier using centralized configuration.
    """
    if model_path is None:
        model_path = CLASSIFIER_PATH
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading pre-trained {model_architecture} CIFAR10 classifier...")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model not found at {model_path}")
    
    # Initialize model
    classifier_model = VGG(model_architecture).to(device)
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats
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
                state_dict = checkpoint
                accuracy_info = 'Unknown'
        else:
            state_dict = checkpoint
            accuracy_info = 'Unknown'
        
        # Remove 'module.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        classifier_model.load_state_dict(new_state_dict, strict=True)
        
        print(f" Successfully loaded classifier from {model_path}")
        if accuracy_info != 'Unknown':
            print(f" Reported accuracy: {accuracy_info}")
        
    except Exception as e:
        print(f" Error loading classifier: {e}")
        raise ValueError(f"Failed to load classifier from {model_path}: {e}")
    
    # Configure model for evaluation
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    
    print(" Classifier loaded and frozen for evaluation")
    return classifier_model


def load_rl_agent(model_path: str = None,
                 device: torch.device = None) -> Tuple[DQNAgent, bool]:
    """
    Load RL agent using centralized configuration.
    """
    if model_path is None:
        model_path = RL_MODEL_PATH
        
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading RL agent...")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")
    print(f"State dimension: {STATE_DIM}")
    print(f"Action dimension: {ACTION_DIM}")
    
    # Initialize agent with centralized dimensions
    agent = DQNAgent(STATE_DIM, ACTION_DIM, device)
    model_loaded = False
    
    # Try to load model weights
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            
            agent.q_network.load_state_dict(state_dict)
            agent.target_q_network.load_state_dict(state_dict)
            
            agent.q_network.eval()
            agent.target_q_network.eval()
            agent.epsilon = 0  # Disable exploration for evaluation
            
            model_loaded = True
            print(f" Successfully loaded RL agent from {model_path}")
            
        except Exception as e:
            print(f" Error loading RL agent from {model_path}: {e}")
            print(" Using randomly initialized agent...")
    else:
        # Try alternative paths from configuration
        for alt_path in ALTERNATIVE_RL_PATHS:
            if os.path.exists(alt_path):
                try:
                    state_dict = torch.load(alt_path, map_location=device)
                    agent.q_network.load_state_dict(state_dict)
                    agent.target_q_network.load_state_dict(state_dict)
                    agent.q_network.eval()
                    agent.target_q_network.eval()
                    agent.epsilon = 0
                    
                    model_loaded = True
                    print(f" Successfully loaded RL agent from {alt_path}")
                    break
                    
                except Exception as e:
                    continue
        
        if not model_loaded:
            print(f" No valid RL model found. Using randomly initialized agent...")
    
    return agent, model_loaded


def get_model_info(model: torch.nn.Module) -> Dict[str, Any]:
    """Extract information about a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = "No parameters"
    
    # Try to get input dimension
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
                               agent: DQNAgent = None,
                               expected_num_classes: int = LOGITS_DIM) -> None:
    """Validate model compatibility using centralized configuration."""
    print(" Validating model compatibility...")
    
    # Validate classifier
    classifier_info = get_model_info(classifier)
    print(f" Classifier info: {classifier_info['model_type']} with {classifier_info['total_parameters']:,} parameters")
    
    # Test classifier with dummy input
    try:
        dummy_input = torch.randn(1, 3, 32, 32).to(next(classifier.parameters()).device)
        with torch.no_grad():
            output = classifier(dummy_input)
        
        if output.shape[1] != expected_num_classes:
            raise ValueError(f"Classifier output shape {output.shape[1]} doesn't match expected classes {expected_num_classes}")
        
        print(f" Classifier validation passed - output shape: {output.shape}")
        
    except Exception as e:
        raise ValueError(f" Classifier validation failed: {e}")
    
    # Validate RL agent if provided
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        
        print(f" RL Agent info: {agent_info['model_type']} with {agent_info['total_parameters']:,} parameters")
        print(f" Agent state dimension: {agent.state_dim}")
        print(f" Agent action dimension: {agent.action_dim}")
        
        # Test agent with dummy state
        try:
            dummy_state = torch.randn(1, STATE_DIM).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(dummy_state)
            
            if q_values.shape[1] != agent.action_dim:
                raise ValueError(f"Agent output shape {q_values.shape[1]} doesn't match expected actions {agent.action_dim}")
            
            print(f" RL Agent validation passed - Q-values shape: {q_values.shape}")
            
        except Exception as e:
            raise ValueError(f" RL Agent validation failed: {e}")
    
    print(" All models validated successfully!")


def print_loading_summary(classifier: torch.nn.Module,
                        agent: DQNAgent = None,
                        agent_loaded: bool = False) -> None:
    """Print a summary of loaded models using centralized configuration."""
    print(f"\n{'='*60}")
    print("MODEL LOADING SUMMARY")
    print(f"{'='*60}")
    
    # Classifier info
    classifier_info = get_model_info(classifier)
    print(f" CLASSIFIER ({classifier_info['model_type']}):")
    print(f"  Status:  Loaded and ready")
    print(f"  Parameters: {classifier_info['total_parameters']:,}")
    print(f"  Training mode: {'ON' if classifier_info['is_training'] else 'OFF (Evaluation)'}")
    print(f"  Device: {classifier_info['model_device']}")
    
    # RL agent info
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        
        status = " Loaded from checkpoint" if agent_loaded else " Random initialization"
        
        print(f"\n RL AGENT ({agent_info['model_type']}):")
        print(f"  Status: {status}")
        print(f"  Parameters: {agent_info['total_parameters']:,}")
        print(f"  State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
        print(f"  Fixed {STATE_DIM}D State Space:")
        print(f"    - Logits: {LOGITS_DIM}")
        print(f"    - Additional features: {ADDITIONAL_FEATURES_DIM}")
        print(f"    - Image features: {IMAGE_FEATURE_DIM}")
        print(f"    - Total: {STATE_DIM}")
        print(f"  Epsilon: {agent.epsilon} (exploration disabled)")
        print(f"  Device: {agent_info['model_device']}")
        print(f"  Network input dim: {agent_info['input_dimension']}")
    
    print(f"{'='*60}")


def detect_available_rl_models(base_dir: str = './models') -> Dict[str, Dict[str, Any]]:
    """Detect available RL models and their characteristics using centralized config."""
    available_models = {}
    
    if not os.path.exists(base_dir):
        return available_models
    
    for filename in os.listdir(base_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(base_dir, filename)
            
            try:
                state_dict = torch.load(filepath, map_location='cpu')
                
                # Find input dimension
                input_dim = None
                for key, tensor in state_dict.items():
                    if 'fc1.weight' in key:
                        input_dim = tensor.shape[1]
                        break
                
                # Check compatibility with centralized STATE_DIM
                compatible = input_dim == STATE_DIM if input_dim else False
                
                available_models[filename] = {
                    'path': filepath,
                    'state_dim': input_dim,
                    'compatible': compatible,
                    'file_size': os.path.getsize(filepath),
                }
                
            except Exception as e:
                print(f" Could not inspect model {filename}: {e}")
    
    return available_models


def print_available_models() -> None:
    """Print available RL models using centralized configuration."""
    models = detect_available_rl_models()
    
    if not models:
        print(" No RL models found in ./models/")
        return
    
    print(f"\n AVAILABLE RL MODELS:")
    print("-" * 60)
    
    compatible_models = []
    incompatible_models = []
    
    for filename, info in models.items():
        if info['compatible']:
            compatible_models.append((filename, info))
        else:
            incompatible_models.append((filename, info))
    
    if compatible_models:
        print(f" Compatible Models ({STATE_DIM}D state space):")
        for filename, info in compatible_models:
            size_mb = info['file_size'] / (1024 * 1024)
            print(f"   {filename}")
            print(f"    State dim: {info['state_dim']}")
            print(f"    Size: {size_mb:.1f} MB")
    
    if incompatible_models:
        print(f"\n Incompatible Models (wrong state dimension):")
        for filename, info in incompatible_models:
            size_mb = info['file_size'] / (1024 * 1024)
            print(f"   {filename}")
            print(f"    State dim: {info['state_dim']} (expected: {STATE_DIM})")
            print(f"    Size: {size_mb:.1f} MB")
    
    print("-" * 60)