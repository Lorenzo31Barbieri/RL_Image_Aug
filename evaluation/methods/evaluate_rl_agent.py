"""
Modulo per la valutazione dell'agente RL con spazio degli stati migliorato.
Valuta l'agente su episodi di augmentation dinamica con features dell'immagine.
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

# Enhanced state configuration
DEFAULT_IMAGE_FEATURE_DIM = 128
DEFAULT_STATE_DIM = 10 + 5 + DEFAULT_IMAGE_FEATURE_DIM  # logits + additional + image_features


def evaluate_rl_agent(agent,  # DQNAgent type hint rimosso per compatibilità
                     classifier_model: torch.nn.Module,
                     test_dataset: torch.utils.data.Dataset,
                     device: torch.device,
                     num_episodes: int = 1000,
                     max_steps_per_episode: int = 3,
                     image_feature_dim: int = DEFAULT_IMAGE_FEATURE_DIM,
                     verbose: bool = True,
                     return_details: bool = True) -> Dict[str, Any]:
    """
    Valuta l'agente RL SOLO su episodi di augmentation dinamica con stato migliorato.
    
    Args:
        agent: Agente RL pre-trained (DQNAgent)
        classifier_model: Modello classificatore per environment
        test_dataset: Dataset di test
        device: Device per computazione
        num_episodes: Numero di episodi da valutare
        max_steps_per_episode: Numero massimo di passi per episodio
        image_feature_dim: Dimensione delle feature dell'immagine
        verbose: Se stampare informazioni dettagliate
        return_details: Se restituire predizioni e label per confusion matrix
    
    Returns:
        Dict con risultati della valutazione RL con stato migliorato
    """
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules not available. Cannot evaluate RL agent.")
    
    validate_evaluation_inputs(classifier_model, device=device)
    
    if num_episodes > len(test_dataset):
        num_episodes = len(test_dataset)
        if verbose:
            print(f" Adjusted num_episodes to dataset size: {num_episodes}")
    
    # Calculate expected state dimension
    expected_state_dim = 10 + 5 + image_feature_dim  # logits + additional + image_features
    
    if verbose:
        print(f" Starting enhanced RL agent evaluation...")
        print(f" Dataset size: {len(test_dataset)} samples")
        print(f" Episodes: {num_episodes}")
        print(f" Max steps per episode: {max_steps_per_episode}")
        print(f" Image feature dimension: {image_feature_dim}")
        print(f" Expected state dimension: {expected_state_dim}")
        print(f" Agent state dimension: {getattr(agent, 'state_dim', 'Unknown')}")
        print(f" Device: {device}")
        print(f" Return details: {return_details}")
    
    # Disabilita esplorazione per valutazione
    original_epsilon = getattr(agent, 'epsilon', 0)
    agent.epsilon = 0
    
    with time_evaluation_context("ENHANCED RL AGENT"):
        # Seleziona episodi casuali UNA SOLA VOLTA
        indices = np.random.choice(len(test_dataset), num_episodes, replace=False)
        
        if verbose:
            print(f" Selected {len(indices)} random episodes (seed fixed for reproducibility)")
        
        # Esegui valutazione RL con stato migliorato
        results = _evaluate_enhanced_rl_episodes(
            agent=agent,
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            indices=indices,
            device=device,
            max_steps_per_episode=max_steps_per_episode,
            image_feature_dim=image_feature_dim,
            verbose=verbose,
            return_details=return_details
        )
        
        # Aggiungi metadati specifici per RL migliorato
        results.update({
            'method': 'enhanced_rl_agent',
            'num_episodes_evaluated': num_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'image_feature_dim': image_feature_dim,
            'expected_state_dim': expected_state_dim,
            'actual_state_dim': results.get('actual_state_dim', 'Unknown'),
            'dataset_size': len(test_dataset),
            'agent_epsilon': 0,  # Sempre 0 per valutazione
            'original_epsilon': original_epsilon,
            'indices_used': indices.tolist()  # Per debug/riproducibilità
        })
        
        if verbose:
            print_evaluation_summary(results, "Enhanced RL Agent")
            print(f"State dimension verification: Expected {expected_state_dim}, Actual {results.get('actual_state_dim', 'Unknown')}")
    
    # Ripristina epsilon originale
    agent.epsilon = original_epsilon
    
    return results


def _evaluate_enhanced_rl_episodes(agent,
                                  classifier_model: torch.nn.Module,
                                  test_dataset: torch.utils.data.Dataset,
                                  indices: np.ndarray,
                                  device: torch.device,
                                  max_steps_per_episode: int,
                                  image_feature_dim: int,
                                  verbose: bool,
                                  return_details: bool) -> Dict[str, Any]:
    """
    Esegue la valutazione RL sui episodi selezionati con stato migliorato.
    Restituisce SOLO i risultati finali (dopo RL), non il confronto con baseline.
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
    
    # Contatori per tracking miglioramenti (ma non li usiamo per confronti)
    improvements = 0
    degradations = 0
    
    # State dimension tracking
    state_dimensions = []
    
    start_time = time.time()
    
    progress_desc = "Enhanced RL episodes"
    
    for idx in tqdm(indices, desc=progress_desc, disable=not verbose):
        image, true_label = test_dataset[idx]
        
        # Inizializza environment con stato migliorato
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=device,
            image_feature_dim=image_feature_dim
        )
        
        # Reset environment con l'immagine corrente
        state = env.reset(image, true_label)
        
        # Track state dimension
        state_dimensions.append(len(state))
        
        # Registra stato iniziale (solo per tracking interno)
        initial_is_correct = env.initial_correct
        
        # Esegui episodio RL
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            # Seleziona azione (senza esplorazione)
            action = agent.select_action(state, training=False)
            
            # Esegui azione nell'environment
            next_state, reward, done, info = env.step(action)
            
            # Registra azione e reward
            actions_taken.append(action)
            action_counts[get_action_name(action)] += 1
            episode_reward += reward
            
            state = next_state
        
        # Ottieni metriche finali dall'environment (SOLO finali)
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
    
    total_time = time.time() - start_time
    num_episodes = len(indices)
    
    # Calcola metriche finali (solo post-RL)
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    final_accuracy = sum(final_correct) / num_episodes
    avg_final_confidence = np.mean(final_confidences)
    avg_reward = np.mean(episode_rewards)
    avg_sequence_length = np.mean([len(seq) for seq in action_sequences])
    
    # Calcola F1 e confusion matrix se abbiamo i dettagli
    f1_score_val = 0.0
    conf_matrix = None
    if return_details and final_predictions:
        f1_score_val = f1_score(final_labels, final_predictions, average='weighted')
        conf_matrix = confusion_matrix(final_labels, final_predictions)
    
    # Risultati finali (solo post-RL)
    results = {
        # Metriche principali (standard per tutti i metodi)
        'accuracy': final_accuracy,
        'avg_confidence': avg_final_confidence,
        'f1_score': f1_score_val,
        'total_samples': num_episodes,
        
        # Metriche specifiche RL
        'avg_reward': avg_reward,
        'reward_std': np.std(episode_rewards),
        'improvements': improvements,  # Per statistiche interne
        'degradations': degradations,   # Per statistiche interne
        'improvement_rate': improvements / num_episodes,
        'degradation_rate': degradations / num_episodes,
        'net_improvement_rate': (improvements - degradations) / num_episodes,
        'avg_sequence_length': avg_sequence_length,
        'action_counts': dict(action_counts),
        'episode_rewards': episode_rewards,
        'action_sequences': action_sequences,
        
        # Metriche temporali
        'inference_time': total_time,
        'time_per_sample': total_time / num_episodes,
        
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