"""
Modulo per la valutazione dell'agente RL con spazio degli stati adattivo.
Valuta l'agente su episodi di augmentation dinamica con gestione automatica delle dimensioni.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
import time

from evaluation.core.evaluation_core import (
    time_evaluation_context,
    print_evaluation_summary,
    validate_evaluation_inputs
)

# Import specifici per RL (gestione import condizionali)
try:
    from src.models.agent import DQNAgent
    from src.environment.environment import ImageAugmentationEnv
    from src.environment.transforms import get_action_name, get_num_actions
    RL_MODULES_AVAILABLE = True
except ImportError:
    RL_MODULES_AVAILABLE = False
    print(" Warning: RL modules not available. Some functionality will be limited.")


def evaluate_rl_agent(agent,  # DQNAgent type hint rimosso per compatibilità
                     classifier_model: torch.nn.Module,
                     test_dataset: torch.utils.data.Dataset,
                     device: torch.device,
                     num_episodes: int = 1000,
                     max_steps_per_episode: int = 3,
                     verbose: bool = True,
                     return_details: bool = True) -> Dict[str, Any]:
    """
    Valuta l'agente RL con gestione adattiva delle dimensioni dello stato.
    
    Args:
        agent: Agente RL pre-trained (DQNAgent)
        classifier_model: Modello classificatore per environment
        test_dataset: Dataset di test
        device: Device per computazione
        num_episodes: Numero di episodi da valutare
        max_steps_per_episode: Numero massimo di passi per episodio
        verbose: Se stampare informazioni dettagliate
        return_details: Se restituire predizioni e label per confusion matrix
    
    Returns:
        Dict con risultati della valutazione RL con gestione adattiva
    """
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules not available. Cannot evaluate RL agent.")
    
    validate_evaluation_inputs(classifier_model, device=device)
    
    if num_episodes > len(test_dataset):
        num_episodes = len(test_dataset)
        if verbose:
            print(f" Adjusted num_episodes to dataset size: {num_episodes}")
    
    # Detect agent's expected state dimension
    agent_state_dim = getattr(agent, 'state_dim', None)
    detected_state_dim = getattr(agent, 'detected_state_dim', None)
    detected_image_features = getattr(agent, 'detected_image_feature_dim', 0)
    
    # Determine which state dimension to use
    if detected_state_dim is not None:
        expected_state_dim = detected_state_dim
        image_feature_dim = detected_image_features
    elif agent_state_dim is not None:
        expected_state_dim = agent_state_dim
        # Calculate image features from state dim
        if agent_state_dim > 15:
            image_feature_dim = agent_state_dim - 15
        else:
            image_feature_dim = 0
    else:
        # Fallback to original dimensions
        expected_state_dim = 15
        image_feature_dim = 0
    
    if verbose:
        print(f" Starting adaptive RL agent evaluation...")
        print(f" Dataset size: {len(test_dataset)} samples")
        print(f" Episodes: {num_episodes}")
        print(f" Max steps per episode: {max_steps_per_episode}")
        print(f" Expected state dimension: {expected_state_dim}")
        print(f" Image feature dimension: {image_feature_dim}")
        print(f" Agent state dimension: {agent_state_dim}")
        print(f" Device: {device}")
        print(f" Return details: {return_details}")
    
    # Disabilita esplorazione per valutazione
    original_epsilon = getattr(agent, 'epsilon', 0)
    agent.epsilon = 0
    
    with time_evaluation_context("ADAPTIVE RL AGENT"):
        # Seleziona episodi casuali UNA SOLA VOLTA
        indices = np.random.choice(len(test_dataset), num_episodes, replace=False)
        
        if verbose:
            print(f" Selected {len(indices)} random episodes")
        
        # Esegui valutazione RL con stato adattivo
        results = _evaluate_adaptive_rl_episodes(
            agent=agent,
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            indices=indices,
            device=device,
            max_steps_per_episode=max_steps_per_episode,
            expected_state_dim=expected_state_dim,
            image_feature_dim=image_feature_dim,
            verbose=verbose,
            return_details=return_details
        )
        
        # Aggiungi metadati specifici per RL adattivo
        results.update({
            'method': 'adaptive_rl_agent',
            'num_episodes_evaluated': num_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'expected_state_dim': expected_state_dim,
            'image_feature_dim': image_feature_dim,
            'actual_state_dim': results.get('actual_state_dim', 'Unknown'),
            'dataset_size': len(test_dataset),
            'agent_epsilon': 0,  # Sempre 0 per valutazione
            'original_epsilon': original_epsilon,
            'state_adaptation': results.get('state_adaptation', 'Unknown')
        })
        
        if verbose:
            print_evaluation_summary(results, "Adaptive RL Agent")
            print(f"State dimension: Expected {expected_state_dim}, Actual {results.get('actual_state_dim', 'Unknown')}")
            print(f"State adaptation: {results.get('state_adaptation', 'Unknown')}")
    
    # Ripristina epsilon originale
    agent.epsilon = original_epsilon
    
    return results


def _evaluate_adaptive_rl_episodes(agent,
                                  classifier_model: torch.nn.Module,
                                  test_dataset: torch.utils.data.Dataset,
                                  indices: np.ndarray,
                                  device: torch.device,
                                  max_steps_per_episode: int,
                                  expected_state_dim: int,
                                  image_feature_dim: int,
                                  verbose: bool,
                                  return_details: bool) -> Dict[str, Any]:
    """
    Esegue la valutazione RL sui episodi selezionati con gestione adattiva dello stato.
    """
    
    # Metriche finali (solo dopo RL)
    final_correct = []
    final_confidences = []
    final_predictions = []
    final_labels = []
    
    # Metriche di performance RL
    episode_rewards = []
    action_sequences = []
    action_counts = defaultdict(int)
    
    # Contatori per tracking miglioramenti
    improvements = 0
    degradations = 0
    
    # State dimension tracking
    state_dimensions = []
    state_adaptation = "Unknown"
    
    start_time = time.time()
    
    progress_desc = "Adaptive RL episodes"
    
    for idx in tqdm(indices, desc=progress_desc, disable=not verbose):
        image, true_label = test_dataset[idx]
        
        # Inizializza environment con agente per adattamento automatico
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=device,
            agent=agent,  # Pass agent for dimension detection
            image_feature_dim=image_feature_dim
        )
        
        # Reset environment con l'immagine corrente
        state = env.reset(image, true_label)
        
        # Track state dimension and adaptation type
        actual_state_dim = len(state)
        state_dimensions.append(actual_state_dim)
        
        # Determine adaptation type on first episode
        if idx == indices[0]:
            if actual_state_dim == expected_state_dim:
                state_adaptation = "Perfect match"
            elif actual_state_dim == 15 and expected_state_dim > 15:
                state_adaptation = "Downgraded to original"
            elif actual_state_dim > 15 and expected_state_dim == 15:
                state_adaptation = "Upgraded to enhanced"
            else:
                state_adaptation = f"Adapted from {expected_state_dim} to {actual_state_dim}"
            
            if verbose:
                print(f" State adaptation: {state_adaptation}")
                print(f" Environment state type: {'Enhanced' if env.use_enhanced_state else 'Original'}")
        
        # Registra stato iniziale (solo per tracking interno)
        initial_is_correct = env.initial_correct
        
        # Esegui episodio RL
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            # Verifica compatibilità dimensioni prima di ogni azione
            if len(state) != expected_state_dim:
                if verbose and idx == indices[0]:  # Print warning only once
                    print(f" Warning: State dimension mismatch! Expected {expected_state_dim}, got {len(state)}")
                    print(f" This may cause reduced performance or errors.")
            
            try:
                # Seleziona azione (senza esplorazione)
                action = agent.select_action(state, training=False)
                
                # Esegui azione nell'environment
                next_state, reward, done, info = env.step(action)
                
                # Registra azione e reward
                actions_taken.append(action)
                action_counts[get_action_name(action)] += 1
                episode_reward += reward
                
                state = next_state
                
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    if verbose:
                        print(f" Dimension error detected: {e}")
                        print(f" Skipping episode {idx} due to incompatible dimensions")
                    break
                else:
                    raise e
        
        # Ottieni metriche finali dall'environment (SOLO finali)
        try:
            metrics = env.get_improvement_metrics()
            final_is_correct = metrics['final_correct']
            final_confidence = metrics['final_confidence']
            
            # Per confusion matrix, ottieni predizione finale
            if return_details:
                with torch.no_grad():
                    final_output = classifier_model(env.augmented_image_tensor.unsqueeze(0))
                    final_prediction = torch.argmax(final_output).item()
                    final_predictions.append(final_prediction)
                    final_labels.append(true_label)
            
            # Registra SOLO risultati finali
            final_correct.append(final_is_correct)
            final_confidences.append(final_confidence)
            episode_rewards.append(episode_reward)
            action_sequences.append(actions_taken)
            
            # Conta miglioramenti/peggioramenti (per statistiche interne)
            if not initial_is_correct and final_is_correct:
                improvements += 1
            elif initial_is_correct and not final_is_correct:
                degradations += 1
                
        except Exception as e:
            if verbose:
                print(f" Error getting metrics for episode {idx}: {e}")
            # Skip this episode's metrics but continue
            continue
    
    total_time = time.time() - start_time
    num_valid_episodes = len(final_correct)
    
    if num_valid_episodes == 0:
        # No valid episodes processed
        return {
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'f1_score': 0.0,
            'total_samples': 0,
            'error': 'No valid episodes could be processed due to dimension mismatches',
            'actual_state_dim': state_dimensions[0] if state_dimensions else 'Unknown',
            'state_adaptation': state_adaptation
        }
    
    # Calcola metriche finali (solo post-RL)
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    final_accuracy = sum(final_correct) / num_valid_episodes
    avg_final_confidence = np.mean(final_confidences)
    avg_reward = np.mean(episode_rewards)
    avg_sequence_length = np.mean([len(seq) for seq in action_sequences])
    
    # Calcola F1 e confusion matrix se abbiamo i dettagli
    f1_score_val = 0.0
    conf_matrix = None
    if return_details and final_predictions:
        f1_score_val = f1_score(final_labels, final_predictions, average='weighted')
        conf_matrix = confusion_matrix(final_labels, final_predictions)
    
    # Risultati finali
    results = {
        # Metriche principali (standard per tutti i metodi)
        'accuracy': final_accuracy,
        'avg_confidence': avg_final_confidence,
        'f1_score': f1_score_val,
        'total_samples': num_valid_episodes,
        
        # Metriche specifiche RL
        'avg_reward': avg_reward,
        'reward_std': np.std(episode_rewards),
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_valid_episodes,
        'degradation_rate': degradations / num_valid_episodes,
        'net_improvement_rate': (improvements - degradations) / num_valid_episodes,
        'avg_sequence_length': avg_sequence_length,
        'action_counts': dict(action_counts),
        'episode_rewards': episode_rewards,
        'action_sequences': action_sequences,
        
        # Metriche temporali
        'inference_time': total_time,
        'time_per_sample': total_time / num_valid_episodes,
        
        # State adaptation info
        'actual_state_dim': state_dimensions[0] if state_dimensions else 'Unknown',
        'state_adaptation': state_adaptation,
        'valid_episodes': num_valid_episodes,
        'total_episodes_attempted': len(indices),
        
        # Confusion matrix data se richiesta
        'confusion_matrix': conf_matrix
    }
    
    # Aggiungi dettagli per confusion matrix se richiesti
    if return_details and final_predictions:
        results.update({
            'predictions': final_predictions,
            'labels': final_labels,
            'confidences': final_confidences
        })
    
    return results


def create_compatible_environment(agent, classifier_model, max_steps_per_episode, device):
    """
    Create an environment that's compatible with the agent's expected state dimensions.
    
    Args:
        agent: The RL agent
        classifier_model: Classifier model
        max_steps_per_episode: Maximum steps per episode
        device: Device for computation
        
    Returns:
        ImageAugmentationEnv configured for the agent
    """
    # Detect agent's expected dimensions
    agent_state_dim = getattr(agent, 'state_dim', 15)
    detected_state_dim = getattr(agent, 'detected_state_dim', agent_state_dim)
    detected_image_features = getattr(agent, 'detected_image_feature_dim', 0)
    
    # Calculate image feature dimension
    if detected_state_dim > 15:
        image_feature_dim = detected_image_features
    else:
        image_feature_dim = 0
    
    print(f"Creating compatible environment:")
    print(f"  Agent expects state_dim: {detected_state_dim}")
    print(f"  Image features: {image_feature_dim}")
    print(f"  Environment type: {'Enhanced' if image_feature_dim > 0 else 'Original'}")
    
    return ImageAugmentationEnv(
        classifier=classifier_model,
        max_steps=max_steps_per_episode,
        device=device,
        agent=agent,
        image_feature_dim=image_feature_dim
    )


def test_agent_environment_compatibility(agent, classifier_model, device):
    """
    Test compatibility between agent and environment before running full evaluation.
    
    Args:
        agent: RL agent to test
        classifier_model: Classifier model
        device: Device for computation
        
    Returns:
        Dict with compatibility test results
    """
    print("Testing agent-environment compatibility...")
    
    try:
        # Create test environment
        env = create_compatible_environment(agent, classifier_model, 3, device)
        
        # Create dummy image and label
        dummy_image = torch.randn(3, 32, 32).to(device)
        dummy_label = 0
        
        # Test environment reset
        state = env.reset(dummy_image, dummy_label)
        
        # Test agent action selection
        action = agent.select_action(state, training=False)
        
        # Test environment step
        next_state, reward, done, info = env.step(action)
        
        # Test second action if not done
        if not done:
            action2 = agent.select_action(next_state, training=False)
            final_state, reward2, done2, info2 = env.step(action2)
        
        compatibility_result = {
            'compatible': True,
            'state_dim': len(state),
            'expected_state_dim': getattr(agent, 'state_dim', 'Unknown'),
            'environment_type': 'Enhanced' if env.use_enhanced_state else 'Original',
            'test_successful': True,
            'error': None
        }
        
        print(f"✅ Compatibility test passed!")
        print(f"   State dimension: {len(state)}")
        print(f"   Environment type: {compatibility_result['environment_type']}")
        
        return compatibility_result
        
    except Exception as e:
        compatibility_result = {
            'compatible': False,
            'state_dim': 'Unknown',
            'expected_state_dim': getattr(agent, 'state_dim', 'Unknown'),
            'environment_type': 'Unknown',
            'test_successful': False,
            'error': str(e)
        }
        
        print(f"❌ Compatibility test failed: {e}")
        
        return compatibility_result