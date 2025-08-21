import torch
import os
from typing import Dict, Any, Optional, Tuple
from src.models.vgg import VGG
from src.models.agent import DQNAgent

# --- GLOBAL CONFIGURATION ---
DEFAULT_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
DEFAULT_RL_MODEL_PATH = './models/best_enhanced_dqn_model.pth'  # Updated for enhanced model
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


def load_rl_agent(model_path: str = DEFAULT_RL_MODEL_PATH,
                 state_dim: int = None,
                 action_dim: int = None,
                 device: torch.device = None,
                 image_feature_dim: int = DEFAULT_IMAGE_FEATURE_DIM) -> Tuple[DQNAgent, bool]:
    """
    Carica l'agente RL pre-trained con supporto per stato migliorato.
    
    Args:
        model_path: Percorso del file del modello RL
        state_dim: Dimensione dello spazio degli stati (calcolata automaticamente se None)
        action_dim: Dimensione dello spazio delle azioni
        device: Device su cui caricare il modello
        image_feature_dim: Dimensione delle feature dell'immagine
    
    Returns:
        Tupla (agent, model_loaded) dove model_loaded indica se il modello è stato caricato con successo
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Calculate state dimension automatically if not provided
    if state_dim is None:
        logits_dim = 10  # CIFAR-10 classes
        additional_features_dim = 5  # confidence, entropy, margin, correctness, step_ratio
        state_dim = logits_dim + additional_features_dim + image_feature_dim
        print(f"Auto-calculated state dimension: {state_dim} (logits: {logits_dim}, additional: {additional_features_dim}, image_features: {image_feature_dim})")
    
    if action_dim is None:
        # Importa dinamicamente per evitare dipendenze circolari
        try:
            from src.environment.transforms import get_num_actions
            action_dim = get_num_actions()
        except ImportError:
            print("Warning: Could not import get_num_actions, using default action_dim=16")
            action_dim = 16
    
    print(f"Loading enhanced RL agent...")
    print(f"Model path: {model_path}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Image feature dim: {image_feature_dim}")
    print(f"Device: {device}")
    
    # Inizializza l'agente con dimensioni migliorate
    agent = DQNAgent(state_dim, action_dim, device)
    model_loaded = False
    
    # Try to load enhanced model first, then fallback to regular paths
    model_paths_to_try = [
        model_path,
        './models/best_enhanced_dqn_model.pth',
        './models/final_enhanced_dqn_model.pth',
        './models/best_improved_dqn_model.pth',  # Fallback to original
        './models/final_improved_dqn_model.pth'   # Fallback to original
    ]
    
    for path_to_try in model_paths_to_try:
        if os.path.exists(path_to_try):
            try:
                # Carica i pesi del modello
                state_dict = torch.load(path_to_try, map_location=device)
                
                # Verifica compatibilità dimensioni
                first_layer_key = None
                for key in state_dict.keys():
                    if 'fc1.weight' in key or '.0.weight' in key:
                        first_layer_key = key
                        break
                
                if first_layer_key:
                    loaded_input_dim = state_dict[first_layer_key].shape[1]
                    if loaded_input_dim != state_dim:
                        print(f"Warning: Model {path_to_try} expects state_dim={loaded_input_dim}, but current state_dim={state_dim}")
                        
                        # Try to adapt if the difference is only in image features
                        expected_base_dim = 10 + 5  # logits + additional
                        loaded_img_features = loaded_input_dim - expected_base_dim
                        current_img_features = state_dim - expected_base_dim
                        
                        if loaded_img_features > 0 and current_img_features > 0:
                            print(f"Model has image features dim {loaded_img_features}, current is {current_img_features}")
                            if abs(loaded_img_features - current_img_features) <= 64:  # Allow some flexibility
                                print("Attempting to adapt agent to loaded model dimensions...")
                                # Re-initialize agent with loaded dimensions
                                adapted_state_dim = loaded_input_dim
                                agent = DQNAgent(adapted_state_dim, action_dim, device)
                                print(f"Agent adapted to state_dim={adapted_state_dim}")
                        else:
                            print(f"Skipping incompatible model {path_to_try}")
                            continue
                
                # Carica sia Q-network che target Q-network
                agent.q_network.load_state_dict(state_dict)
                agent.target_q_network.load_state_dict(state_dict)
                
                # Configura per valutazione
                agent.q_network.eval()
                agent.target_q_network.eval()
                agent.epsilon = 0  # Disabilita esplorazione per valutazione
                
                model_loaded = True
                print(f"Successfully loaded enhanced RL agent from {path_to_try}")
                
                # Verify loaded model info
                actual_state_dim = getattr(agent, 'state_dim', 'Unknown')
                print(f"Loaded model state dimension: {actual_state_dim}")
                
                break
                
            except Exception as e:
                print(f"Error loading RL agent from {path_to_try}: {e}")
                continue
    
    if not model_loaded:
        print("Could not load any RL model, using randomly initialized agent for comparison...")
        print("Training an enhanced RL agent first is recommended for meaningful evaluation.")
    
    # Assicurati che epsilon sia 0 per valutazione
    agent.epsilon = 0
    
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
                               expected_num_classes: int = 10,
                               expected_state_dim: int = None,
                               image_feature_dim: int = DEFAULT_IMAGE_FEATURE_DIM) -> None:
    """
    Valida la compatibilità tra modelli e configurazione per stato migliorato.
    
    Args:
        classifier: Modello classificatore
        agent: Agente RL (opzionale)
        expected_num_classes: Numero di classi atteso
        expected_state_dim: Dimensione stato attesa (calcolata se None)
        image_feature_dim: Dimensione features immagine
    
    Raises:
        ValueError: Se i modelli non sono compatibili
    """
    print("Validating enhanced model compatibility...")
    
    # Calculate expected state dimension if not provided
    if expected_state_dim is None:
        expected_state_dim = expected_num_classes + 5 + image_feature_dim
    
    # Valida il classificatore
    classifier_info = get_model_info(classifier)
    print(f"Classifier info: {classifier_info['model_type']} with {classifier_info['total_parameters']:,} parameters")
    
    # Testa il classificatore con input dummy
    try:
        dummy_input = torch.randn(1, 3, 32, 32).to(next(classifier.parameters()).device)
        with torch.no_grad():
            output = classifier(dummy_input)
        
        if output.shape[1] != expected_num_classes:
            raise ValueError(f"Classifier output shape {output.shape[1]} doesn't match expected classes {expected_num_classes}")
        
        print(f"Classifier validation passed - output shape: {output.shape}")
        
    except Exception as e:
        raise ValueError(f"Classifier validation failed: {e}")
    
    # Valida l'agente RL se fornito
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        actual_state_dim = getattr(agent, 'state_dim', 'Unknown')
        
        print(f"RL Agent info: {agent_info['model_type']} with {agent_info['total_parameters']:,} parameters")
        print(f"Agent state dimension: {actual_state_dim}")
        print(f"Expected state dimension: {expected_state_dim}")
        print(f"Agent input dimension: {agent_info['input_dimension']}")
        
        # Check state dimension compatibility
        if actual_state_dim != 'Unknown' and actual_state_dim != expected_state_dim:
            print(f"Warning: Agent state dimension ({actual_state_dim}) doesn't match expected ({expected_state_dim})")
            print("This might indicate the agent was trained with different state configuration")
        
        # Testa l'agente con stato dummy
        try:
            dummy_state = torch.randn(1, actual_state_dim if actual_state_dim != 'Unknown' else expected_state_dim).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(dummy_state)
            
            if q_values.shape[1] != agent.action_dim:
                raise ValueError(f"Agent output shape {q_values.shape[1]} doesn't match expected actions {agent.action_dim}")
            
            print(f"RL Agent validation passed - Q-values shape: {q_values.shape}")
            
        except Exception as e:
            raise ValueError(f"RL Agent validation failed: {e}")
    
    print("All enhanced models validated successfully!")


def print_loading_summary(classifier: torch.nn.Module,
                        agent: Optional[DQNAgent] = None,
                        agent_loaded: bool = False,
                        image_feature_dim: int = DEFAULT_IMAGE_FEATURE_DIM) -> None:
    """
    Stampa un riassunto dei modelli caricati con informazioni sullo stato migliorato.
    
    Args:
        classifier: Modello classificatore caricato
        agent: Agente RL (opzionale)
        agent_loaded: Se l'agente è stato caricato con successo
        image_feature_dim: Dimensione delle feature dell'immagine
    """
    print(f"\n{'='*60}")
    print("ENHANCED MODEL LOADING SUMMARY")
    print(f"{'='*60}")
    
    # Info classificatore
    classifier_info = get_model_info(classifier)
    print(f"🎯 CLASSIFIER ({classifier_info['model_type']}):")
    print(f"  Status: Loaded and ready")
    print(f"  Parameters: {classifier_info['total_parameters']:,}")
    print(f"  Training mode: {'ON' if classifier_info['is_training'] else 'OFF (Evaluation)'}")
    print(f"  Device: {classifier_info['model_device']}")
    
    # Info agente RL
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        actual_state_dim = getattr(agent, 'state_dim', 'Unknown')
        status = "✅ Loaded from checkpoint" if agent_loaded else "⚠️ Random initialization"
        
        print(f"\n🤖 ENHANCED RL AGENT ({agent_info['model_type']}):")
        print(f"  {status}")
        print(f"  Parameters: {agent_info['total_parameters']:,}")
        print(f"  State dim: {actual_state_dim}, Action dim: {agent.action_dim}")
        
        # Enhanced state breakdown
        if actual_state_dim != 'Unknown':
            logits_dim = 10
            additional_dim = 5
            calculated_img_dim = actual_state_dim - logits_dim - additional_dim
            
            print(f"  Enhanced State Breakdown:")
            print(f"    - Logits: {logits_dim}")
            print(f"    - Additional features: {additional_dim}")
            print(f"    - Image features: {calculated_img_dim}")
            print(f"    - Expected image features: {image_feature_dim}")
            
            if calculated_img_dim == image_feature_dim:
                print(f"    ✅ State dimensions match perfectly")
            elif abs(calculated_img_dim - image_feature_dim) <= 64:
                print(f"    ⚠️ Close match (difference: {abs(calculated_img_dim - image_feature_dim)})")
            else:
                print(f"    ❌ Dimension mismatch")
        
        print(f"  Epsilon: {agent.epsilon} (exploration disabled)")
        print(f"  Device: {agent_info['model_device']}")
        print(f"  Network input dim: {agent_info['input_dimension']}")
    
    print(f"{'='*60}")


def detect_available_rl_models(base_dir: str = './models') -> Dict[str, Dict[str, Any]]:
    """
    Rileva i modelli RL disponibili e le loro caratteristiche.
    
    Args:
        base_dir: Directory base dove cercare i modelli
    
    Returns:
        Dict con informazioni sui modelli trovati
    """
    available_models = {}
    
    if not os.path.exists(base_dir):
        return available_models
    
    # Pattern di file da cercare
    model_patterns = [
        'best_enhanced_dqn_model.pth',
        'final_enhanced_dqn_model.pth',
        'enhanced_dqn_episode_*.pth',
        'best_improved_dqn_model.pth',
        'final_improved_dqn_model.pth',
        'improved_dqn_episode_*.pth'
    ]
    
    for filename in os.listdir(base_dir):
        if filename.endswith('.pth'):
            filepath = os.path.join(base_dir, filename)
            
            try:
                # Carica il modello per ispezionare le dimensioni
                state_dict = torch.load(filepath, map_location='cpu')
                
                # Trova la dimensione di input
                input_dim = None
                for key, tensor in state_dict.items():
                    if 'fc1.weight' in key:
                        input_dim = tensor.shape[1]
                        break
                
                # Classifica il tipo di modello
                model_type = "unknown"
                if input_dim:
                    if input_dim == 15:  # Original state dim
                        model_type = "original"
                    elif input_dim >= 128:  # Enhanced state dim
                        model_type = "enhanced"
                        
                    # Calculate image feature dimension
                    if input_dim > 15:
                        img_features = input_dim - 15  # Subtract logits + additional
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
    """Stampa i modelli RL disponibili."""
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