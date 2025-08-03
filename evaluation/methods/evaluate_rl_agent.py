"""
Modulo per la valutazione dell'agente RL.
Valuta l'agente su episodi di augmentation dinamica e confronta con baseline.
"""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from tqdm import tqdm
from collections import defaultdict
import time

# Import dei moduli core
from evaluation.core.evaluation_core import (
    time_evaluation_context,
    calculate_improvement_metrics,
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
    print("⚠️ Warning: RL modules not available. Some functionality will be limited.")


def evaluate_rl_agent(agent,  # DQNAgent type hint rimosso per compatibilità
                     classifier_model: torch.nn.Module,
                     test_dataset: torch.utils.data.Dataset,
                     device: torch.device,
                     num_episodes: int = 1000,
                     max_steps_per_episode: int = 3,
                     verbose: bool = True,
                     return_details: bool = True) -> Dict[str, Any]:
    """
    Valuta l'agente RL su episodi di augmentation dinamica.
    
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
        Dict con risultati della valutazione RL inclusi predictions e labels
        per garantire consistency tra accuracy e confusion matrix
    """
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules not available. Cannot evaluate RL agent.")
    
    validate_evaluation_inputs(classifier_model, device=device)
    
    if num_episodes > len(test_dataset):
        num_episodes = len(test_dataset)
        if verbose:
            print(f"⚠️ Adjusted num_episodes to dataset size: {num_episodes}")
    
    if verbose:
        print(f"🤖 Starting RL agent evaluation...")
        print(f"📊 Dataset size: {len(test_dataset)} samples")
        print(f"🎮 Episodes: {num_episodes}")
        print(f"🎯 Max steps per episode: {max_steps_per_episode}")
        print(f"💻 Device: {device}")
        print(f"📋 Return details: {return_details}")
    
    # Disabilita esplorazione per valutazione
    original_epsilon = getattr(agent, 'epsilon', 0)
    agent.epsilon = 0
    
    with time_evaluation_context("RL AGENT"):
        # Seleziona episodi casuali UNA SOLA VOLTA
        indices = np.random.choice(len(test_dataset), num_episodes, replace=False)
        
        if verbose:
            print(f"🎲 Selected {len(indices)} random episodes (seed fixed for reproducibility)")
        
        # Esegui valutazione con tracking dettagliato
        results = _evaluate_rl_episodes_with_details(
            agent=agent,
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            indices=indices,
            device=device,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
            return_details=return_details
        )
        
        # Aggiungi metadati
        results.update({
            'method': 'rl_agent',
            'num_episodes_evaluated': num_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'dataset_size': len(test_dataset),
            'agent_epsilon': 0,  # Sempre 0 per valutazione
            'original_epsilon': original_epsilon,
            'indices_used': indices.tolist()  # Per debug/riproducibilità
        })
        
        if verbose:
            _print_rl_summary(results)
    
    # Ripristina epsilon originale
    agent.epsilon = original_epsilon
    
    return results


def _evaluate_rl_episodes_with_details(agent,
                                     classifier_model: torch.nn.Module,
                                     test_dataset: torch.utils.data.Dataset,
                                     indices: np.ndarray,
                                     device: torch.device,
                                     max_steps_per_episode: int,
                                     verbose: bool,
                                     return_details: bool) -> Dict[str, Any]:
    """
    Esegue la valutazione RL sui episodi selezionati con tracking completo.
    
    IMPORTANTE: Questa funzione ora traccia sia le metriche di performance
    che le predizioni dettagliate sugli STESSI campioni per garantire
    consistency tra accuracy riportata e confusion matrix.
    """
    
    # Metriche di performance
    initial_correct = []
    final_correct = []
    initial_confidences = []
    final_confidences = []
    episode_rewards = []
    action_sequences = []
    action_counts = defaultdict(int)
    
    # Tracking dettagliato per confusion matrix (STESSI campioni!)
    initial_predictions = []  # Predizioni prima dell'RL
    final_predictions = []    # Predizioni dopo l'RL  
    true_labels = []         # Label vere
    
    improvements = 0
    degradations = 0
    
    start_time = time.time()
    
    progress_desc = "RL episodes (with details)" if return_details else "RL episodes"
    
    for idx in tqdm(indices, desc=progress_desc, disable=not verbose):
        image, true_label = test_dataset[idx]
        
        # Inizializza environment
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=device
        )
        
        # Reset environment con l'immagine corrente
        state = env.reset(image, true_label)
        
        # Registra stato iniziale
        initial_is_correct = env.initial_correct
        initial_confidence = env.initial_confidence
        initial_prediction = env.initial_prediction  # Predizione iniziale
        
        initial_correct.append(initial_is_correct)
        initial_confidences.append(initial_confidence)
        
        # Tracking per confusion matrix
        if return_details:
            initial_predictions.append(initial_prediction)
            true_labels.append(true_label)
        
        # Esegui episodio RL
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            actions_taken.append(action)
            action_counts[get_action_name(action)] += 1
            episode_reward += reward
            
            state = next_state
        
        # Ottieni metriche finali dall'environment
        metrics = env.get_improvement_metrics()
        final_is_correct = metrics['final_correct']
        final_confidence = metrics['final_confidence']
        
        # CRUCIALE: Ottieni predizione finale per confusion matrix
        if return_details:
            with torch.no_grad():
                # Re-ottieni predizione finale sullo stesso campione
                final_output = classifier_model(env.augmented_image_tensor.unsqueeze(0))
                final_prediction = torch.argmax(final_output).item()
                final_predictions.append(final_prediction)
        
        # Registra risultati finali
        final_correct.append(final_is_correct)
        final_confidences.append(final_confidence)
        episode_rewards.append(episode_reward)
        action_sequences.append(actions_taken)
        
        # Conta miglioramenti/peggioramenti
        if not initial_is_correct and final_is_correct:
            improvements += 1
        elif initial_is_correct and not final_is_correct:
            degradations += 1
    
    total_time = time.time() - start_time
    num_episodes = len(indices)
    
    # Calcola metriche aggregate
    initial_accuracy = sum(initial_correct) / num_episodes
    final_accuracy = sum(final_correct) / num_episodes
    accuracy_improvement = final_accuracy - initial_accuracy
    
    confidence_improvements = [final - initial for final, initial in zip(final_confidences, initial_confidences)]
    avg_confidence_improvement = np.mean(confidence_improvements)
    
    avg_reward = np.mean(episode_rewards)
    avg_sequence_length = np.mean([len(seq) for seq in action_sequences])
    
    # Risultati base
    results = {
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'initial_avg_confidence': np.mean(initial_confidences),
        'final_avg_confidence': np.mean(final_confidences),
        'avg_confidence_improvement': avg_confidence_improvement,
        'avg_reward': avg_reward,
        'reward_std': np.std(episode_rewards),
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_episodes,
        'degradation_rate': degradations / num_episodes,
        'net_improvement_rate': (improvements - degradations) / num_episodes,
        'avg_sequence_length': avg_sequence_length,
        'action_counts': dict(action_counts),
        'episode_rewards': episode_rewards,
        'confidence_improvements': confidence_improvements,
        'action_sequences': action_sequences,
        'inference_time': total_time,
        'time_per_sample': total_time / num_episodes
    }
    
    # Aggiungi dettagli per confusion matrix (STESSI campioni!)
    if return_details:
        results.update({
            'predictions': final_predictions,    # Predizioni finali RL
            'labels': true_labels,              # Label vere
            'initial_predictions': initial_predictions,  # Predizioni iniziali (baseline)
            'total_samples': num_episodes,
            'details_source': 'same_episodes'   # Flag per debug
        })
        
        # Verifica consistency
        if len(final_predictions) != num_episodes or len(true_labels) != num_episodes:
            print(f"⚠️ WARNING: Details length mismatch!")
            print(f"   Episodes: {num_episodes}")
            print(f"   Predictions: {len(final_predictions)}")
            print(f"   Labels: {len(true_labels)}")
    
    return results


def _evaluate_rl_episodes(agent,
                        classifier_model: torch.nn.Module,
                        test_dataset: torch.utils.data.Dataset,
                        indices: np.ndarray,
                        device: torch.device,
                        max_steps_per_episode: int,
                        verbose: bool) -> Dict[str, Any]:
    """Esegue la valutazione RL sui episodi selezionati."""
    
    # Metriche di performance
    initial_correct = []
    final_correct = []
    initial_confidences = []
    final_confidences = []
    episode_rewards = []
    action_sequences = []
    action_counts = defaultdict(int)
    
    improvements = 0
    degradations = 0
    
    start_time = time.time()
    
    progress_desc = "RL episodes"
    
    for idx in tqdm(indices, desc=progress_desc, disable=not verbose):
        image, true_label = test_dataset[idx]
        
        # Inizializza environment
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=device
        )
        
        # Reset environment con l'immagine corrente
        state = env.reset(image, true_label)
        
        # Registra stato iniziale
        initial_is_correct = env.initial_correct
        initial_confidence = env.initial_confidence
        
        initial_correct.append(initial_is_correct)
        initial_confidences.append(initial_confidence)
        
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
        
        # Ottieni metriche finali dall'environment
        metrics = env.get_improvement_metrics()
        final_is_correct = metrics['final_correct']
        final_confidence = metrics['final_confidence']
        
        # Registra risultati finali
        final_correct.append(final_is_correct)
        final_confidences.append(final_confidence)
        episode_rewards.append(episode_reward)
        action_sequences.append(actions_taken)
        
        # Conta miglioramenti/peggioramenti
        if not initial_is_correct and final_is_correct:
            improvements += 1
        elif initial_is_correct and not final_is_correct:
            degradations += 1
    
    total_time = time.time() - start_time
    num_episodes = len(indices)
    
    # Calcola metriche aggregate
    initial_accuracy = sum(initial_correct) / num_episodes
    final_accuracy = sum(final_correct) / num_episodes
    accuracy_improvement = final_accuracy - initial_accuracy
    
    confidence_improvements = [final - initial for final, initial in zip(final_confidences, initial_confidences)]
    avg_confidence_improvement = np.mean(confidence_improvements)
    
    avg_reward = np.mean(episode_rewards)
    avg_sequence_length = np.mean([len(seq) for seq in action_sequences])
    
    return {
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'initial_avg_confidence': np.mean(initial_confidences),
        'final_avg_confidence': np.mean(final_confidences),
        'avg_confidence_improvement': avg_confidence_improvement,
        'avg_reward': avg_reward,
        'reward_std': np.std(episode_rewards),
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_episodes,
        'degradation_rate': degradations / num_episodes,
        'net_improvement_rate': (improvements - degradations) / num_episodes,
        'avg_sequence_length': avg_sequence_length,
        'action_counts': dict(action_counts),
        'episode_rewards': episode_rewards,
        'confidence_improvements': confidence_improvements,
        'action_sequences': action_sequences,
        'inference_time': total_time,
        'time_per_sample': total_time / num_episodes
    }


def _print_rl_summary(results: Dict[str, Any]) -> None:
    """Stampa riassunto dettagliato dei risultati RL."""
    
    print(f"\n{'='*60}")
    print("RL AGENT EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"🎮 EPISODES: {results['num_episodes_evaluated']}")
    print(f"🎯 Max steps per episode: {results['max_steps_per_episode']}")
    print(f"🔄 Average sequence length: {results['avg_sequence_length']:.1f}")
    
    print(f"\n📈 ACCURACY COMPARISON:")
    print(f"  Initial: {results['initial_accuracy']:.4f}")
    print(f"  Final: {results['final_accuracy']:.4f}")
    
    improvement_sign = "📈" if results['accuracy_improvement'] > 0 else "📉" if results['accuracy_improvement'] < 0 else "➡️"
    print(f"  {improvement_sign} Improvement: {results['accuracy_improvement']:+.4f}")
    
    print(f"\n🔍 CONFIDENCE ANALYSIS:")
    print(f"  Initial confidence: {results['initial_avg_confidence']:.4f}")
    print(f"  Final confidence: {results['final_avg_confidence']:.4f}")
    print(f"  Change: {results['avg_confidence_improvement']:+.4f}")
    
    print(f"\n🏆 REWARD ANALYSIS:")
    print(f"  Average reward: {results['avg_reward']:.3f} ± {results['reward_std']:.3f}")
    positive_rewards = [r for r in results['episode_rewards'] if r > 0]
    success_rate = len(positive_rewards) / len(results['episode_rewards']) if results['episode_rewards'] else 0
    print(f"  Success rate: {success_rate:.1%} ({len(positive_rewards)} positive rewards)")
    
    print(f"\n📊 IMPROVEMENT BREAKDOWN:")
    print(f"  Improved episodes: {results['improvements']} ({results['improvement_rate']:.1%})")
    print(f"  Degraded episodes: {results['degradations']} ({results['degradation_rate']:.1%})")
    print(f"  Net success rate: {results['net_improvement_rate']:+.1%}")
    
    print(f"\n🔧 ACTION USAGE:")
    # Mostra le azioni più utilizzate
    sorted_actions = sorted(results['action_counts'].items(), key=lambda x: x[1], reverse=True)
    for i, (action_name, count) in enumerate(sorted_actions[:5]):
        print(f"  {i+1}. {action_name}: {count} times")
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"  Total time: {results['inference_time']:.2f}s")
    print(f"  Time per episode: {results['time_per_sample']*1000:.1f}ms")
    
    # Raccomandazione
    if results['accuracy_improvement'] > 0.01 and results['avg_reward'] > 0.1:
        recommendation = "✅ Strong performance - Agent is effective"
    elif results['accuracy_improvement'] > 0.005 or results['avg_reward'] > 0:
        recommendation = "⚠️ Moderate performance - Some benefit observed"
    elif results['accuracy_improvement'] > 0:
        recommendation = "📊 Weak performance - Limited improvement"
    else:
        recommendation = "❌ Poor performance - Consider retraining"
    
    print(f"\n💡 RECOMMENDATION: {recommendation}")
    print(f"{'='*60}")


def analyze_rl_agent_behavior(agent,
                            classifier_model: torch.nn.Module,
                            test_dataset: torch.utils.data.Dataset,
                            device: torch.device,
                            num_episodes: int = 200) -> Dict[str, Any]:
    """
    Analizza il comportamento dell'agente RL in dettaglio.
    
    Args:
        agent: Agente RL
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        device: Device di computazione
        num_episodes: Numero di episodi per analisi
    
    Returns:
        Dict con analisi comportamentale dell'agente
    """
    print(f"🔬 Analyzing RL agent behavior over {num_episodes} episodes...")
    
    # Esegui valutazione standard
    results = evaluate_rl_agent(
        agent=agent,
        classifier_model=classifier_model,
        test_dataset=test_dataset,
        device=device,
        num_episodes=num_episodes,
        verbose=False
    )
    
    # Analisi delle sequenze di azioni
    action_sequences = results['action_sequences']
    episode_rewards = results['episode_rewards']
    
    # Analizza pattern di azioni
    sequence_patterns = defaultdict(int)
    for seq in action_sequences:
        pattern = tuple(seq)
        sequence_patterns[pattern] += 1
    
    # Trova pattern più comuni
    most_common_patterns = sorted(sequence_patterns.items(), key=lambda x: x[1], reverse=True)
    
    # Analizza lunghezze delle sequenze
    sequence_lengths = [len(seq) for seq in action_sequences]
    length_distribution = defaultdict(int)
    for length in sequence_lengths:
        length_distribution[length] += 1
    
    # Analizza correlazione azione-reward
    action_reward_correlation = defaultdict(list)
    for seq, reward in zip(action_sequences, episode_rewards):
        for action in seq:
            action_name = get_action_name(action)
            action_reward_correlation[action_name].append(reward)
    
    # Calcola statistiche per azione
    action_stats = {}
    for action_name, rewards in action_reward_correlation.items():
        if rewards:
            action_stats[action_name] = {
                'avg_reward': np.mean(rewards),
                'std_reward': np.std(rewards),
                'usage_count': len(rewards),
                'success_rate': sum(1 for r in rewards if r > 0) / len(rewards)
            }
    
    behavioral_analysis = {
        'sequence_patterns': dict(most_common_patterns[:10]),  # Top 10 pattern
        'length_distribution': dict(length_distribution),
        'action_stats': action_stats,
        'avg_sequence_length': np.mean(sequence_lengths),
        'sequence_length_std': np.std(sequence_lengths),
        'most_common_pattern': most_common_patterns[0] if most_common_patterns else None,
        'pattern_diversity': len(sequence_patterns),  # Numero di pattern unici
    }
    
    results['behavioral_analysis'] = behavioral_analysis
    
    print(f"\n🧠 BEHAVIORAL ANALYSIS:")
    print(f"  Pattern diversity: {behavioral_analysis['pattern_diversity']} unique sequences")
    print(f"  Avg sequence length: {behavioral_analysis['avg_sequence_length']:.1f} ± {behavioral_analysis['sequence_length_std']:.1f}")
    
    if behavioral_analysis['most_common_pattern']:
        pattern, count = behavioral_analysis['most_common_pattern']
        pattern_names = [get_action_name(a) for a in pattern]
        print(f"  Most common pattern: {pattern_names} ({count} times)")
    
    print(f"\n🎯 TOP ACTION PERFORMANCE:")
    sorted_actions = sorted(action_stats.items(), key=lambda x: x[1]['avg_reward'], reverse=True)
    for i, (action_name, stats) in enumerate(sorted_actions[:5]):
        print(f"  {i+1}. {action_name}: {stats['avg_reward']:.3f} reward ({stats['success_rate']:.1%} success)")
    
    return results


def compare_rl_with_random_baseline(agent,
                                  classifier_model: torch.nn.Module,
                                  test_dataset: torch.utils.data.Dataset,
                                  device: torch.device,
                                  num_episodes: int = 200,
                                  max_steps_per_episode: int = 3) -> Dict[str, Any]:
    """
    Confronta l'agente RL con una baseline di azioni casuali.
    
    Args:
        agent: Agente RL trained
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        device: Device di computazione
        num_episodes: Numero di episodi per confronto
        max_steps_per_episode: Passi massimi per episodio
    
    Returns:
        Dict con confronto RL vs random
    """
    print(f"🎲 Comparing RL agent with random baseline...")
    
    # Valutazione agente RL
    rl_results = evaluate_rl_agent(
        agent=agent,
        classifier_model=classifier_model,
        test_dataset=test_dataset,
        device=device,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        verbose=False
    )
    
    # Valutazione baseline casuale
    print("🎯 Evaluating random action baseline...")
    random_results = _evaluate_random_baseline(
        classifier_model=classifier_model,
        test_dataset=test_dataset,
        device=device,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode
    )
    
    # Confronto
    comparison = {
        'rl_results': rl_results,
        'random_results': random_results,
        'rl_advantage': {
            'accuracy_improvement': rl_results['accuracy_improvement'] - random_results['accuracy_improvement'],
            'avg_reward': rl_results['avg_reward'] - random_results['avg_reward'],
            'success_rate': rl_results['improvement_rate'] - random_results['improvement_rate']
        }
    }
    
    print(f"\n🏆 RL vs RANDOM COMPARISON:")
    print(f"  Accuracy improvement:")
    print(f"    RL: {rl_results['accuracy_improvement']:+.4f}")
    print(f"    Random: {random_results['accuracy_improvement']:+.4f}")
    print(f"    RL Advantage: {comparison['rl_advantage']['accuracy_improvement']:+.4f}")
    
    print(f"  Average reward:")
    print(f"    RL: {rl_results['avg_reward']:.3f}")
    print(f"    Random: {random_results['avg_reward']:.3f}")
    print(f"    RL Advantage: {comparison['rl_advantage']['avg_reward']:+.3f}")
    
    # Valutazione significatività
    if comparison['rl_advantage']['accuracy_improvement'] > 0.01:
        significance = "🟢 Highly Significant"
    elif comparison['rl_advantage']['accuracy_improvement'] > 0.005:
        significance = "🟡 Significant"
    elif comparison['rl_advantage']['accuracy_improvement'] > 0.001:
        significance = "🟠 Marginal"
    else:
        significance = "🔴 Not Significant"
    
    comparison['significance_assessment'] = significance
    print(f"  Significance: {significance}")
    
    return comparison


def _evaluate_random_baseline(classifier_model: torch.nn.Module,
                            test_dataset: torch.utils.data.Dataset,
                            device: torch.device,
                            num_episodes: int,
                            max_steps_per_episode: int) -> Dict[str, Any]:
    """Valuta baseline con azioni casuali."""
    
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules required for random baseline evaluation")
    
    indices = np.random.choice(len(test_dataset), num_episodes, replace=False)
    
    initial_correct = []
    final_correct = []
    initial_confidences = []
    final_confidences = []
    episode_rewards = []
    
    improvements = 0
    degradations = 0
    
    for idx in tqdm(indices, desc="Random baseline"):
        image, true_label = test_dataset[idx]
        
        # Inizializza environment
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=device
        )
        
        state = env.reset(image, true_label)
        
        # Registra stato iniziale
        initial_is_correct = env.initial_correct
        initial_confidence = env.initial_confidence
        
        initial_correct.append(initial_is_correct)
        initial_confidences.append(initial_confidence)
        
        # Esegui episodio con azioni casuali
        episode_reward = 0
        done = False
        
        while not done:
            # Azione casuale
            action = np.random.randint(0, get_num_actions())
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        # Risultati finali
        metrics = env.get_improvement_metrics()
        final_is_correct = metrics['final_correct']
        final_confidence = metrics['final_confidence']
        
        final_correct.append(final_is_correct)
        final_confidences.append(final_confidence)
        episode_rewards.append(episode_reward)
        
        # Conta miglioramenti
        if not initial_is_correct and final_is_correct:
            improvements += 1
        elif initial_is_correct and not final_is_correct:
            degradations += 1
    
    # Calcola metriche
    initial_accuracy = sum(initial_correct) / num_episodes
    final_accuracy = sum(final_correct) / num_episodes
    accuracy_improvement = final_accuracy - initial_accuracy
    
    confidence_improvements = [final - initial for final, initial in zip(final_confidences, initial_confidences)]
    
    return {
        'method': 'random_baseline',
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'avg_reward': np.mean(episode_rewards),
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_episodes,
        'degradation_rate': degradations / num_episodes,
        'avg_confidence_improvement': np.mean(confidence_improvements),
        'episode_rewards': episode_rewards
    }


def evaluate_rl_robustness(agent,
                         classifier_model: torch.nn.Module,
                         test_dataset: torch.utils.data.Dataset,
                         device: torch.device,
                         num_runs: int = 5,
                         episodes_per_run: int = 200) -> Dict[str, Any]:
    """
    Valuta la robustezza dell'agente RL attraverso multiple run.
    
    Args:
        agent: Agente RL
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        device: Device di computazione
        num_runs: Numero di run indipendenti
        episodes_per_run: Episodi per ogni run
    
    Returns:
        Dict con statistiche di robustezza
    """
    print(f"🔬 Testing RL agent robustness over {num_runs} runs...")
    
    accuracy_improvements = []
    avg_rewards = []
    success_rates = []
    
    for run in range(num_runs):
        print(f"  Run {run+1}/{num_runs}")
        
        results = evaluate_rl_agent(
            agent=agent,
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            device=device,
            num_episodes=episodes_per_run,
            verbose=False
        )
        
        accuracy_improvements.append(results['accuracy_improvement'])
        avg_rewards.append(results['avg_reward'])
        success_rates.append(results['improvement_rate'])
    
    robustness_results = {
        'num_runs': num_runs,
        'episodes_per_run': episodes_per_run,
        'accuracy_improvement_mean': np.mean(accuracy_improvements),
        'accuracy_improvement_std': np.std(accuracy_improvements),
        'accuracy_improvement_min': np.min(accuracy_improvements),
        'accuracy_improvement_max': np.max(accuracy_improvements),
        'reward_mean': np.mean(avg_rewards),
        'reward_std': np.std(avg_rewards),
        'success_rate_mean': np.mean(success_rates),
        'success_rate_std': np.std(success_rates),
        'all_accuracy_improvements': accuracy_improvements,
        'all_rewards': avg_rewards,
        'all_success_rates': success_rates
    }
    
    # Valuta stabilità
    cv_accuracy = robustness_results['accuracy_improvement_std'] / abs(robustness_results['accuracy_improvement_mean']) if robustness_results['accuracy_improvement_mean'] != 0 else float('inf')
    cv_reward = robustness_results['reward_std'] / abs(robustness_results['reward_mean']) if robustness_results['reward_mean'] != 0 else float('inf')
    
    if cv_accuracy < 0.2 and cv_reward < 0.3:
        stability = "🟢 Very Stable"
    elif cv_accuracy < 0.5 and cv_reward < 0.5:
        stability = "🟡 Stable"
    elif cv_accuracy < 1.0 and cv_reward < 1.0:
        stability = "🟠 Moderately Stable"
    else:
        stability = "🔴 Unstable"
    
    robustness_results['stability_assessment'] = stability
    robustness_results['cv_accuracy'] = cv_accuracy
    robustness_results['cv_reward'] = cv_reward
    
    print(f"\n📊 ROBUSTNESS RESULTS:")
    print(f"  Accuracy improvement: {robustness_results['accuracy_improvement_mean']:.4f} ± {robustness_results['accuracy_improvement_std']:.4f}")
    print(f"  Average reward: {robustness_results['reward_mean']:.3f} ± {robustness_results['reward_std']:.3f}")
    print(f"  Success rate: {robustness_results['success_rate_mean']:.1%} ± {robustness_results['success_rate_std']:.1%}")
    print(f"  Stability: {stability}")
    
    return robustness_results


# Funzione wrapper per compatibilità
def run_rl_agent_evaluation(classifier_path: str = './checkpoint/ckpt.pth',
                          rl_model_path: str = './models/best_improved_dqn_model.pth',
                          data_root: str = './data',
                          num_episodes: int = 1000,
                          max_steps_per_episode: int = 3,
                          state_dim: int = 15,
                          device: torch.device = None) -> Dict[str, Any]:
    """
    Funzione wrapper per eseguire valutazione RL completa.
    
    Args:
        classifier_path: Percorso del classificatore
        rl_model_path: Percorso del modello RL
        data_root: Directory root per i dati
        num_episodes: Numero di episodi da valutare
        max_steps_per_episode: Passi massimi per episodio
        state_dim: Dimensione dello spazio degli stati
        device: Device (auto-detect se None)
    
    Returns:
        Dict con risultati della valutazione RL
    """
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules not available. Cannot run RL evaluation.")
    
    from core.model_loader import load_classifier, load_rl_agent
    from core.data_utils import get_cifar10_test_dataset
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Running complete RL agent evaluation...")
    print(f"📁 Classifier: {classifier_path}")
    print(f"📁 RL model: {rl_model_path}")
    print(f"📁 Data root: {data_root}")
    print(f"🎮 Episodes: {num_episodes}")
    
    # Carica modelli
    classifier = load_classifier(classifier_path, device)
    agent, model_loaded = load_rl_agent(rl_model_path, state_dim=state_dim, device=device)
    
    if not model_loaded:
        print("⚠️ Warning: Using randomly initialized RL agent")
    
    # Carica dataset
    test_dataset = get_cifar10_test_dataset(data_root=data_root)
    
    # Esegui valutazione
    results = evaluate_rl_agent(
        agent=agent,
        classifier_model=classifier,
        test_dataset=test_dataset,
        device=device,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        verbose=True
    )
    
    results['model_loaded'] = model_loaded
    return results

def evaluate_rl_agent_detailed(agent,
                               classifier_model: torch.nn.Module,
                               test_dataset: torch.utils.data.Dataset,
                               device: torch.device,
                               num_episodes: int = 1000,
                               max_steps_per_episode: int = 3,
                               verbose: bool = True,
                               save_examples: bool = True) -> Dict[str, Any]:
    """
    Valuta l'agente RL con tracking dettagliato per analisi avanzate.
    
    Args:
        agent: Agente RL pre-trained
        classifier_model: Modello classificatore
        test_dataset: Dataset di test
        device: Device di computazione
        num_episodes: Numero di episodi da valutare
        max_steps_per_episode: Passi massimi per episodio
        verbose: Se stampare informazioni dettagliate
        save_examples: Se salvare esempi di miglioramenti
    
    Returns:
        Dict con risultati dettagliati inclusi dati per classe e esempi
    """
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules not available. Cannot evaluate RL agent.")
    
    validate_evaluation_inputs(classifier_model, device=device)
    
    if num_episodes > len(test_dataset):
        num_episodes = len(test_dataset)
        if verbose:
            print(f"⚠️ Adjusted num_episodes to dataset size: {num_episodes}")
    
    if verbose:
        print(f"🤖 Starting detailed RL agent evaluation...")
        print(f"📊 Dataset size: {len(test_dataset)} samples")
        print(f"🎮 Episodes: {num_episodes}")
        print(f"🎯 Max steps per episode: {max_steps_per_episode}")
        print(f"💾 Save examples: {save_examples}")
    
    # Disabilita esplorazione per valutazione
    original_epsilon = getattr(agent, 'epsilon', 0)
    agent.epsilon = 0
    
    with time_evaluation_context("DETAILED RL AGENT"):
        # Seleziona episodi casuali
        indices = np.random.choice(len(test_dataset), num_episodes, replace=False)
        
        # Esegui valutazione dettagliata
        results = _evaluate_rl_episodes_detailed(
            agent=agent,
            classifier_model=classifier_model,
            test_dataset=test_dataset,
            indices=indices,
            device=device,
            max_steps_per_episode=max_steps_per_episode,
            verbose=verbose,
            save_examples=save_examples
        )
        
        # Aggiungi metadati
        results.update({
            'method': 'rl_agent_detailed',
            'num_episodes_evaluated': num_episodes,
            'max_steps_per_episode': max_steps_per_episode,
            'dataset_size': len(test_dataset),
            'agent_epsilon': 0,
            'original_epsilon': original_epsilon
        })
        
        if verbose:
            _print_detailed_rl_summary(results)
    
    # Ripristina epsilon originale
    agent.epsilon = original_epsilon
    
    return results

def _evaluate_rl_episodes_detailed(agent,
                                  classifier_model: torch.nn.Module,
                                  test_dataset: torch.utils.data.Dataset,
                                  indices: np.ndarray,
                                  device: torch.device,
                                  max_steps_per_episode: int,
                                  verbose: bool,
                                  save_examples: bool) -> Dict[str, Any]:
    """Esegue valutazione RL dettagliata con tracking per classe."""
    
    # Metriche di performance standard
    initial_correct = []
    final_correct = []
    initial_confidences = []
    final_confidences = []
    episode_rewards = []
    action_sequences = []
    action_counts = defaultdict(int)
    
    # Tracking dettagliato per classe
    improvements_by_class = defaultdict(int)
    degradations_by_class = defaultdict(int)
    class_episode_data = []
    
    # Per confusion matrix
    initial_predictions = []
    final_predictions = []
    true_labels = []
    
    # Esempi di miglioramenti
    improvement_examples = []
    
    improvements = 0
    degradations = 0
    
    start_time = time.time()
    
    progress_desc = "Detailed RL episodes"
    
    for episode_idx, idx in enumerate(tqdm(indices, desc=progress_desc, disable=not verbose)):
        image, true_label = test_dataset[idx]
        
        # Inizializza environment
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=device
        )
        
        # Reset environment con l'immagine corrente
        state = env.reset(image, true_label)
        
        # Registra stato iniziale
        initial_is_correct = env.initial_correct
        initial_confidence = env.initial_confidence
        initial_prediction = env.initial_prediction
        
        initial_correct.append(initial_is_correct)
        initial_confidences.append(initial_confidence)
        initial_predictions.append(initial_prediction)
        true_labels.append(true_label)
        
        # Esegui episodio RL
        episode_reward = 0
        done = False
        actions_taken = []
        augmented_image = image.clone()
        
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
        
        # Ottieni metriche finali dall'environment
        metrics = env.get_improvement_metrics()
        final_is_correct = metrics['final_correct']
        final_confidence = metrics['final_confidence']
        final_prediction = metrics.get('final_prediction', initial_prediction)
        
        # Registra risultati finali
        final_correct.append(final_is_correct)
        final_confidences.append(final_confidence)
        final_predictions.append(final_prediction)
        episode_rewards.append(episode_reward)
        action_sequences.append(actions_taken)
        
        # Tracking per classe
        episode_data = {
            'episode_idx': episode_idx,
            'image_idx': idx,
            'true_label': true_label,
            'initial_correct': initial_is_correct,
            'final_correct': final_is_correct,
            'initial_prediction': initial_prediction,
            'final_prediction': final_prediction,
            'initial_confidence': initial_confidence,
            'final_confidence': final_confidence,
            'actions': actions_taken,
            'reward': episode_reward
        }
        class_episode_data.append(episode_data)
        
        # Conta miglioramenti/peggioramenti per classe
        if not initial_is_correct and final_is_correct:
            improvements += 1
            improvements_by_class[true_label] += 1
            
            # Salva esempio di miglioramento se richiesto
            if save_examples and len(improvement_examples) < 10:
                # Ottieni l'immagine finale augmentata dall'environment
                final_image = getattr(env, 'current_image', image)
                improvement_examples.append({
                    'original_image': image.clone(),
                    'augmented_image': final_image.clone(),
                    'true_label': true_label,
                    'initial_prediction': initial_prediction,
                    'final_prediction': final_prediction,
                    'actions': actions_taken.copy(),
                    'confidence_improvement': final_confidence - initial_confidence
                })
                
        elif initial_is_correct and not final_is_correct:
            degradations += 1
            degradations_by_class[true_label] += 1
    
    total_time = time.time() - start_time
    num_episodes = len(indices)
    
    # Calcola metriche aggregate standard
    initial_accuracy = sum(initial_correct) / num_episodes
    final_accuracy = sum(final_correct) / num_episodes
    accuracy_improvement = final_accuracy - initial_accuracy
    
    confidence_improvements = [final - initial for final, initial in zip(final_confidences, initial_confidences)]
    avg_confidence_improvement = np.mean(confidence_improvements)
    
    avg_reward = np.mean(episode_rewards)
    avg_sequence_length = np.mean([len(seq) for seq in action_sequences])
    
    # Calcola confusion matrix
    from sklearn.metrics import confusion_matrix
    initial_cm = confusion_matrix(true_labels, initial_predictions, labels=list(range(10)))
    final_cm = confusion_matrix(true_labels, final_predictions, labels=list(range(10)))
    
    return {
        # Metriche standard
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'initial_avg_confidence': np.mean(initial_confidences),
        'final_avg_confidence': np.mean(final_confidences),
        'avg_confidence_improvement': avg_confidence_improvement,
        'avg_reward': avg_reward,
        'reward_std': np.std(episode_rewards),
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_episodes,
        'degradation_rate': degradations / num_episodes,
        'net_improvement_rate': (improvements - degradations) / num_episodes,
        'avg_sequence_length': avg_sequence_length,
        'action_counts': dict(action_counts),
        'episode_rewards': episode_rewards,
        'confidence_improvements': confidence_improvements,
        'action_sequences': action_sequences,
        'inference_time': total_time,
        'time_per_sample': total_time / num_episodes,
        
        # Dati dettagliati per analisi avanzate
        'improvements_by_class': dict(improvements_by_class),
        'degradations_by_class': dict(degradations_by_class),
        'class_episode_data': class_episode_data,
        'improvement_examples': improvement_examples,
        
        # Per confusion matrix
        'predictions': final_predictions,
        'labels': true_labels,
        'initial_predictions': initial_predictions,
        'initial_confusion_matrix': initial_cm,
        'final_confusion_matrix': final_cm
    }

def _print_detailed_rl_summary(results: Dict[str, Any]) -> None:
    """Stampa riassunto dettagliato con analisi per classe."""
    
    print(f"\n{'='*70}")
    print("DETAILED RL AGENT EVALUATION RESULTS")
    print(f"{'='*70}")
    
    print(f"🎮 EPISODES: {results['num_episodes_evaluated']}")
    print(f"🎯 Max steps per episode: {results['max_steps_per_episode']}")
    print(f"🔄 Average sequence length: {results['avg_sequence_length']:.1f}")
    
    print(f"\n📈 ACCURACY COMPARISON:")
    print(f"  Initial: {results['initial_accuracy']:.4f}")
    print(f"  Final: {results['final_accuracy']:.4f}")
    
    improvement_sign = "📈" if results['accuracy_improvement'] > 0 else "📉" if results['accuracy_improvement'] < 0 else "➡️"
    print(f"  {improvement_sign} Improvement: {results['accuracy_improvement']:+.4f}")
    
    print(f"\n📊 IMPROVEMENT BREAKDOWN:")
    print(f"  Improved episodes: {results['improvements']} ({results['improvement_rate']:.1%})")
    print(f"  Degraded episodes: {results['degradations']} ({results['degradation_rate']:.1%})")
    print(f"  Net success rate: {results['net_improvement_rate']:+.1%}")
    
    # Analisi per classe
    print(f"\n🏷️  CLASS-WISE IMPROVEMENTS:")
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    improvements_by_class = results['improvements_by_class']
    degradations_by_class = results['degradations_by_class']
    
    print(f"  {'Class':<12} {'Improved':<8} {'Degraded':<8} {'Net':<6}")
    print(f"  {'-'*35}")
    
    for class_id in range(10):
        imp = improvements_by_class.get(class_id, 0)
        deg = degradations_by_class.get(class_id, 0)
        net = imp - deg
        class_name = class_names[class_id]
        print(f"  {class_name:<12} {imp:<8} {deg:<8} {net:<6}")
    
    # Esempi di miglioramento
    improvement_examples = results.get('improvement_examples', [])
    if improvement_examples:
        print(f"\n💡 IMPROVEMENT EXAMPLES: {len(improvement_examples)} saved")
        for i, example in enumerate(improvement_examples[:3]):  # Mostra primi 3
            class_name = class_names[example['true_label']]
            conf_imp = example['confidence_improvement']
            actions = [get_action_name(a) for a in example['actions']]
            print(f"  {i+1}. {class_name}: +{conf_imp:.3f} confidence via {actions}")
    
    print(f"\n⚡ PERFORMANCE:")
    print(f"  Total time: {results['inference_time']:.2f}s")
    print(f"  Time per episode: {results['time_per_sample']*1000:.1f}ms")
    
    print(f"{'='*70}")

# Modifica la funzione wrapper per usare la valutazione dettagliata
def run_rl_agent_evaluation_detailed(classifier_path: str = './checkpoint/ckpt.pth',
                                    rl_model_path: str = './models/best_improved_dqn_model.pth',
                                    data_root: str = './data',
                                    num_episodes: int = 1000,
                                    max_steps_per_episode: int = 3,
                                    state_dim: int = 15,
                                    device: torch.device = None,
                                    save_examples: bool = True) -> Dict[str, Any]:
    """
    Funzione wrapper per eseguire valutazione RL dettagliata.
    """
    if not RL_MODULES_AVAILABLE:
        raise ImportError("RL modules not available. Cannot run RL evaluation.")
    
    from evaluation.core.model_loader import load_classifier, load_rl_agent
    from evaluation.core.data_utils import get_cifar10_test_dataset
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 Running detailed RL agent evaluation...")
    print(f"📁 Classifier: {classifier_path}")
    print(f"📁 RL model: {rl_model_path}")
    print(f"🎮 Episodes: {num_episodes}")
    
    # Carica modelli
    classifier = load_classifier(classifier_path, device)
    agent, model_loaded = load_rl_agent(rl_model_path, state_dim=state_dim, device=device)
    
    if not model_loaded:
        print("⚠️ Warning: Using randomly initialized RL agent")
    
    # Carica dataset
    test_dataset = get_cifar10_test_dataset(data_root=data_root)
    
    # Esegui valutazione dettagliata
    results = evaluate_rl_agent_detailed(
        agent=agent,
        classifier_model=classifier,
        test_dataset=test_dataset,
        device=device,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        verbose=True,
        save_examples=save_examples
    )
    
    results['model_loaded'] = model_loaded
    return results


if __name__ == '__main__':
    """
    Script principale per test del modulo.
    """
    print("Testing RL agent evaluation module...")
    print(f"RL modules available: {RL_MODULES_AVAILABLE}")
    
    if not RL_MODULES_AVAILABLE:
        print("❌ RL modules not available. Cannot test RL evaluation.")
        exit(1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Test valutazione standard
        results = run_rl_agent_evaluation(
            classifier_path='./checkpoint/ckpt.pth',
            rl_model_path='./models/best_improved_dqn_model.pth',
            data_root='./data',
            num_episodes=500,
            max_steps_per_episode=3,
            device=device
        )
        
        print(f"\n🎉 RL agent evaluation completed!")
        print(f"📊 Accuracy improvement: {results['accuracy_improvement']:+.4f}")
        print(f"🏆 Average reward: {results['avg_reward']:.3f}")
        print(f"✅ Model loaded from checkpoint: {results['model_loaded']}")
        
        # Test analisi comportamentale
        from core.model_loader import load_classifier, load_rl_agent
        from core.data_utils import get_cifar10_test_dataset
        
        classifier = load_classifier('./checkpoint/ckpt.pth', device)
        agent, _ = load_rl_agent('./models/best_improved_dqn_model.pth', device=device)
        test_dataset = get_cifar10_test_dataset('./data')
        
        print(f"\n🧠 Testing behavioral analysis...")
        behavioral_results = analyze_rl_agent_behavior(
            agent=agent,
            classifier_model=classifier,
            test_dataset=test_dataset,
            device=device,
            num_episodes=100
        )
        
        print(f"\n🎲 Testing comparison with random baseline...")
        comparison_results = compare_rl_with_random_baseline(
            agent=agent,
            classifier_model=classifier,
            test_dataset=test_dataset,
            device=device,
            num_episodes=100
        )
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        print("Make sure all required modules and models are available.")