import torch
import os
from typing import Dict, Any, Optional, Tuple
from vgg import VGG
from agent import DQNAgent

# --- GLOBAL CONFIGURATION ---
DEFAULT_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
DEFAULT_RL_MODEL_PATH = './models/best_improved_dqn_model.pth'


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
        
        print(f"✅ Successfully loaded classifier from {model_path}")
        if accuracy_info != 'Unknown':
            print(f"📊 Reported accuracy: {accuracy_info}")
        
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        raise ValueError(f"Failed to load classifier from {model_path}: {e}")
    
    # Configura il modello per valutazione
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    
    print("🔒 Classifier loaded and frozen for evaluation")
    return classifier_model


def load_rl_agent(model_path: str = DEFAULT_RL_MODEL_PATH,
                 state_dim: int = 15,  # NUM_CLASSES + 5
                 action_dim: int = None,
                 device: torch.device = None) -> Tuple[DQNAgent, bool]:
    """
    Carica l'agente RL pre-trained.
    
    Args:
        model_path: Percorso del file del modello RL
        state_dim: Dimensione dello spazio degli stati
        action_dim: Dimensione dello spazio delle azioni
        device: Device su cui caricare il modello
    
    Returns:
        Tupla (agent, model_loaded) dove model_loaded indica se il modello è stato caricato con successo
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if action_dim is None:
        # Importa dinamicamente per evitare dipendenze circolari
        try:
            from transforms import get_num_actions
            action_dim = get_num_actions()
        except ImportError:
            print("⚠️ Warning: Could not import get_num_actions, using default action_dim=12")
            action_dim = 12
    
    print(f"Loading RL agent...")
    print(f"Model path: {model_path}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {device}")
    
    # Inizializza l'agente
    agent = DQNAgent(state_dim, action_dim, device)
    model_loaded = False
    
    if os.path.exists(model_path):
        try:
            # Carica i pesi del modello
            state_dict = torch.load(model_path, map_location=device)
            
            # Carica sia Q-network che target Q-network
            agent.q_network.load_state_dict(state_dict)
            agent.target_q_network.load_state_dict(state_dict)
            
            # Configura per valutazione
            agent.q_network.eval()
            agent.target_q_network.eval()
            agent.epsilon = 0  # Disabilita esplorazione per valutazione
            
            model_loaded = True
            print(f"✅ Successfully loaded RL agent from {model_path}")
            
        except Exception as e:
            print(f"❌ Error loading RL agent: {e}")
            print("📝 Using randomly initialized agent for comparison...")
    else:
        print(f"❌ RL model not found at {model_path}")
        print("📝 Using randomly initialized agent for comparison...")
    
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
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_device': str(model_device),
        'is_training': model.training,
        'model_type': type(model).__name__
    }


def validate_model_compatibility(classifier: torch.nn.Module,
                               agent: Optional[DQNAgent] = None,
                               expected_num_classes: int = 10) -> None:
    """
    Valida la compatibilità tra modelli e configurazione.
    
    Args:
        classifier: Modello classificatore
        agent: Agente RL (opzionale)
        expected_num_classes: Numero di classi atteso
    
    Raises:
        ValueError: Se i modelli non sono compatibili
    """
    print("🔍 Validating model compatibility...")
    
    # Valida il classificatore
    classifier_info = get_model_info(classifier)
    print(f"📊 Classifier info: {classifier_info['model_type']} with {classifier_info['total_parameters']:,} parameters")
    
    # Testa il classificatore con input dummy
    try:
        dummy_input = torch.randn(1, 3, 32, 32).to(next(classifier.parameters()).device)
        with torch.no_grad():
            output = classifier(dummy_input)
        
        if output.shape[1] != expected_num_classes:
            raise ValueError(f"Classifier output shape {output.shape[1]} doesn't match expected classes {expected_num_classes}")
        
        print(f"✅ Classifier validation passed - output shape: {output.shape}")
        
    except Exception as e:
        raise ValueError(f"Classifier validation failed: {e}")
    
    # Valida l'agente RL se fornito
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        print(f"🤖 RL Agent info: {agent_info['model_type']} with {agent_info['total_parameters']:,} parameters")
        
        # Testa l'agente con stato dummy
        try:
            dummy_state = torch.randn(1, agent.state_dim).to(agent.device)
            with torch.no_grad():
                q_values = agent.q_network(dummy_state)
            
            if q_values.shape[1] != agent.action_dim:
                raise ValueError(f"Agent output shape {q_values.shape[1]} doesn't match expected actions {agent.action_dim}")
            
            print(f"✅ RL Agent validation passed - Q-values shape: {q_values.shape}")
            
        except Exception as e:
            raise ValueError(f"RL Agent validation failed: {e}")
    
    print("🎉 All models validated successfully!")


def print_loading_summary(classifier: torch.nn.Module,
                        agent: Optional[DQNAgent] = None,
                        agent_loaded: bool = False) -> None:
    """
    Stampa un riassunto dei modelli caricati.
    
    Args:
        classifier: Modello classificatore caricato
        agent: Agente RL (opzionale)
        agent_loaded: Se l'agente è stato caricato con successo
    """
    print(f"\n{'='*60}")
    print("MODEL LOADING SUMMARY")
    print(f"{'='*60}")
    
    # Info classificatore
    classifier_info = get_model_info(classifier)
    print(f"🎯 CLASSIFIER ({classifier_info['model_type']}):")
    print(f"  ✅ Status: Loaded and ready")
    print(f"  📊 Parameters: {classifier_info['total_parameters']:,}")
    print(f"  🔒 Training mode: {'ON' if classifier_info['is_training'] else 'OFF (Evaluation)'}")
    print(f"  💻 Device: {classifier_info['model_device']}")
    
    # Info agente RL
    if agent is not None:
        agent_info = get_model_info(agent.q_network)
        status = "✅ Loaded from checkpoint" if agent_loaded else "⚠️ Random initialization"
        print(f"\n🤖 RL AGENT ({agent_info['model_type']}):")
        print(f"  {status}")
        print(f"  📊 Parameters: {agent_info['total_parameters']:,}")
        print(f"  🎯 State dim: {agent.state_dim}, Action dim: {agent.action_dim}")
        print(f"  🔍 Epsilon: {agent.epsilon} (exploration disabled)")
        print(f"  💻 Device: {agent_info['model_device']}")
    
    print(f"{'='*60}")