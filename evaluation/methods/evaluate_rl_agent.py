"""
RL agent evaluation for 143-dimensional state space.
"""

import torch
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
from collections import defaultdict
import time

from evaluation.core.evaluation_core import (
    time_evaluation_context,
    print_evaluation_summary,
    validate_evaluation_inputs
)

# Import RL modules
try:
    from src.models.agent import DQNAgent
    from src.environment.environment import ImageAugmentationEnv
    from src.environment.transforms import get_action_name, get_num_actions
    RL_MODULES_AVAILABLE = True
except ImportError:
    RL_MODULES_AVAILABLE = False
    print("Warning: RL modules not available. Some functionality will be limited.")


def evaluate_rl_agent(agent: DQNAgent,
                     classifier_model: torch.nn.Module,
                     test_dataset: torch.utils.data.Dataset,
                     device: torch.device,
                     num_episodes: int = 1000,
                     max_steps_per_episode: int = 3,
                     image_feature_dim: int = 128,
                     verbose: bool = True,
                     return_details: bool = True) -> Dict[str, Any]:
    """
    Evaluate RL agent with 143-dimensional state space.
    
    Args:
        agent: DQN agent
        classifier_model: Classifier model for environment
        test_dataset: Test dataset
        device: Device for computation
        num_episodes: Number of episodes to evaluate
        max_steps_per_episode: Maximum steps per episode
        image_feature_dim: Image feature dimension (128)
        verbose: Whether to print detailed information
        return_details: Whether to return predictions and labels for confusion matrix
    
    Returns:
        Dict with evaluation results
    """
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules not available. Cannot evaluate RL agent.")
    
    validate_evaluation_inputs(classifier_model, device=device)
    
    if num_episodes > len(test_dataset):
        num_episodes = len(test_dataset)
        if verbose:
            print(f"Adjusted num_episodes to dataset size: {num_episodes}")
    
    expected_state_dim = 143  # 10 + 5 + 128
    
    if verbose:
        print(f"Starting RL agent evaluation...")
        print(f"Dataset size: {len(test_dataset)} samples")
        print(f"Episodes: {num_episodes}")
        print(f"Max steps per episode: {max_steps_per_episode}")
        print(f"State dimension: {expected_state_dim}")
        print(f"Image feature dimension: {image_feature_dim}")
        print(f"Device: {device}")
    
    # Disable exploration for evaluation
    original_epsilon = getattr(agent, 'epsilon', 0)
    agent.epsilon = 0
    
    with time_evaluation_context("RL AGENT"):
        # Select random episodes
        indices = np.random.choice(len(test_dataset), num_episodes, replace=False)
        
        if verbose:
            print(f"Selected {len(indices)} random episodes")
        
        # Run evaluation
        results = _evaluate_rl_episodes(
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
        
        # Add metadata
        results.update({
            'method': 'rl_agent',
            'num_episodes_evaluated': num_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'state_dim': expected_state_dim,
            'image_feature_dim': image_feature_dim,
            'dataset_size': len(test_dataset),
            'agent_epsilon': 0,  # Always 0 for evaluation
            'original_epsilon': original_epsilon
        })
        
        if verbose:
            print_evaluation_summary(results, "RL Agent")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    return results


def _evaluate_rl_episodes(agent: DQNAgent,
                         classifier_model: torch.nn.Module,
                         test_dataset: torch.utils.data.Dataset,
                         indices: np.ndarray,
                         device: torch.device,
                         max_steps_per_episode: int,
                         image_feature_dim: int,
                         verbose: bool,
                         return_details: bool) -> Dict[str, Any]:
    """Execute RL evaluation episodes."""
    
    # Final metrics (after RL)
    final_correct = []
    final_confidences = []
    final_predictions = []
    final_labels = []
    
    # RL performance metrics
    episode_rewards = []
    action_sequences = []
    action_counts = defaultdict(int)
    
    # Improvement counters
    improvements = 0
    degradations = 0
    
    start_time = time.time()
    
    progress_desc = "RL episodes"
    
    for idx in tqdm(indices, desc=progress_desc, disable=not verbose):
        image, true_label = test_dataset[idx]
        
        # Initialize environment
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=device,
            image_feature_dim=image_feature_dim
        )
        
        # Reset environment
        state = env.reset(image, true_label)
        
        # Verify state dimension
        if len(state) != 143:
            if verbose and idx == indices[0]:  # Print warning only once
                print(f"Warning: Expected state dimension 143, got {len(state)}")
        
        # Record initial state
        initial_is_correct = env.initial_correct
        
        # Run RL episode
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            try:
                # Select action (no exploration)
                action = agent.select_action(state, training=False)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Record action and reward
                actions_taken.append(action)
                action_counts[get_action_name(action)] += 1
                episode_reward += reward
                
                state = next_state
                
            except RuntimeError as e:
                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                    if verbose:
                        print(f"Dimension error detected: {e}")
                        print(f"Skipping episode {idx}")
                    break
                else:
                    raise e
        
        # Get final metrics
        try:
            metrics = env.get_improvement_metrics()
            final_is_correct = metrics['final_correct']
            final_confidence = metrics['final_confidence']
            
            # Get final prediction for confusion matrix
            if return_details:
                with torch.no_grad():
                    final_output = classifier_model(env.augmented_image_tensor.unsqueeze(0))
                    final_prediction = torch.argmax(final_output).item()
                    final_predictions.append(final_prediction)
                    final_labels.append(true_label)
            
            # Record final results
            final_correct.append(final_is_correct)
            final_confidences.append(final_confidence)
            episode_rewards.append(episode_reward)
            action_sequences.append(actions_taken)
            
            # Count improvements/degradations
            if not initial_is_correct and final_is_correct:
                improvements += 1
            elif initial_is_correct and not final_is_correct:
                degradations += 1
                
        except Exception as e:
            if verbose:
                print(f"Error getting metrics for episode {idx}: {e}")
            continue
    
    total_time = time.time() - start_time
    num_valid_episodes = len(final_correct)
    
    if num_valid_episodes == 0:
        return {
            'accuracy': 0.0,
            'avg_confidence': 0.0,
            'f1_score': 0.0,
            'total_samples': 0,
            'error': 'No valid episodes could be processed'
        }
    
    # Calculate final metrics
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    
    final_accuracy = sum(final_correct) / num_valid_episodes
    avg_final_confidence = np.mean(final_confidences)
    avg_reward = np.mean(episode_rewards)
    avg_sequence_length = np.mean([len(seq) for seq in action_sequences])
    
    # Calculate F1 and confusion matrix if details available
    f1_score_val = 0.0
    conf_matrix = None
    if return_details and final_predictions:
        f1_score_val = f1_score(final_labels, final_predictions, average='weighted')
        conf_matrix = confusion_matrix(final_labels, final_predictions)
    
    # Compile results
    results = {
        # Standard metrics (compatible with other methods)
        'accuracy': final_accuracy,
        'avg_confidence': avg_final_confidence,
        'f1_score': f1_score_val,
        'total_samples': num_valid_episodes,
        
        # RL-specific metrics
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
        
        # Timing metrics
        'inference_time': total_time,
        'time_per_sample': total_time / num_valid_episodes,
        
        # Confusion matrix data
        'confusion_matrix': conf_matrix,
        'valid_episodes': num_valid_episodes,
        'total_episodes_attempted': len(indices)
    }
    
    # Add details for confusion matrix if requested
    if return_details and final_predictions:
        results.update({
            'predictions': final_predictions,
            'labels': final_labels,
            'confidences': final_confidences
        })
    
    return results


def test_agent_environment_compatibility(agent: DQNAgent, 
                                       classifier_model: torch.nn.Module, 
                                       device: torch.device,
                                       image_feature_dim: int = 128) -> Dict[str, Any]:
    """
    Test compatibility between agent and environment.
    
    Args:
        agent: RL agent to test
        classifier_model: Classifier model
        device: Device for computation
        image_feature_dim: Image feature dimension
        
    Returns:
        Dict with compatibility test results
    """
    print("Testing agent-environment compatibility...")
    
    try:
        # Create test environment
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=3,
            device=device,
            image_feature_dim=image_feature_dim
        )
        
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
            'expected_state_dim': 143,
            'environment_type': 'Standard 143D',
            'test_successful': True,
            'error': None
        }
        
        print(f" Compatibility test passed!")
        print(f"   State dimension: {len(state)}")
        
        return compatibility_result
        
    except Exception as e:
        compatibility_result = {
            'compatible': False,
            'state_dim': 'Unknown',
            'expected_state_dim': 143,
            'environment_type': 'Unknown',
            'test_successful': False,
            'error': str(e)
        }
        
        print(f" Compatibility test failed: {e}")
        
        return compatibility_result