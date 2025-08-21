import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
from tqdm import tqdm
import signal
import sys


# Import improved modules
from src.models.agent import DQNAgent
from src.environment.environment import ImageAugmentationEnv
from src.environment.transforms import get_num_actions
from src.models.vgg import VGG

# --- GLOBAL CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Dataset and Path Configuration ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
IMAGE_SIZE = 32
NUM_CLASSES = 10

# Enhanced state dimension with image features
IMAGE_FEATURE_DIM = 128  # Features extracted from the classifier
LOGITS_DIM = NUM_CLASSES  # 10 for CIFAR-10
ADDITIONAL_FEATURES_DIM = 5  # confidence, entropy, margin, correctness, step_info
STATE_DIM = LOGITS_DIM + ADDITIONAL_FEATURES_DIM + IMAGE_FEATURE_DIM  # Total: 143

ACTION_DIM = get_num_actions()

print(f"Enhanced State dimension: {STATE_DIM} (logits: {LOGITS_DIM}, additional: {ADDITIONAL_FEATURES_DIM}, image features: {IMAGE_FEATURE_DIM})")
print(f"Action dimension: {ACTION_DIM}")

# Improved hyperparameters for enhanced state space
learning_rate = 0.0003  # Lower learning rate for stability
gamma = 0.95
epsilon_start = 1.0
epsilon_end = 0.005
epsilon_decay = 0.99975
buffer_size = 300000  # Larger buffer for complex state space
batch_size = 128  # Larger batch size
target_update_freq = 1000  # More frequent updates
num_total_episodes = 75000  # More episodes to handle complexity
max_steps_per_episode = 3
images_per_cycle = 3

# Training strategy parameters
warmup_episodes = 3000  # More warmup episodes
eval_freq = 2500
eval_episodes = 200
patience = 300
best_eval_reward = float('-inf')
patience_counter = 0

# Variabili globali per il signal handler
training_data = {
    'episode_rewards': [],
    'loss_history': [],
    'evaluation_history': [],
    'agent': None,
    'final_results': None
}

def load_classifier_model():
    """Load the pre-trained classifier."""
    print("Loading pre-trained VGG19 CIFAR10 classifier...")
    classifier_model = VGG('VGG19').to(DEVICE)
    
    try:
        checkpoint = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=DEVICE)
        
        new_state_dict = {}
        for k, v in checkpoint['net'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        classifier_model.load_state_dict(new_state_dict, strict=True)
        print(f"Successfully loaded classifier weights from {PRE_TRAINED_CLASSIFIER_PATH}")
        print(f"Classifier accuracy from checkpoint: {checkpoint['acc']:.2f}%")
        
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {PRE_TRAINED_CLASSIFIER_PATH}")
        exit()
    except Exception as e:
        print(f"Error loading classifier: {e}")
        exit()

    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")
    return classifier_model


def create_balanced_dataset_loader(dataset, difficulty_balance=True):
    """
    Create a data loader that balances easy and hard examples.
    """
    if not difficulty_balance:
        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    
    # You could implement difficulty-based sampling here
    # For now, just return regular loader
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


def evaluate_agent_detailed(agent, classifier_model, eval_episodes=200):
    """
    Detailed evaluation of agent performance with enhanced state space.
    """
    print(f"\nEvaluating agent for {eval_episodes} episodes...")
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    eval_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=False, transform=eval_transform)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=True, num_workers=0)
    eval_iter = iter(eval_loader)
    
    # Evaluation metrics
    total_reward = 0
    successful_episodes = 0
    improvement_episodes = 0
    confidence_improvements = 0
    correctness_fixes = 0
    
    # Detailed tracking
    initial_correct_count = 0
    final_correct_count = 0
    
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    for _ in tqdm(range(eval_episodes), desc="Evaluating"):
        try:
            image_tensor, true_label_tensor = next(eval_iter)
        except StopIteration:
            eval_iter = iter(eval_loader)
            image_tensor, true_label_tensor = next(eval_iter)
        
        image_tensor = image_tensor.squeeze(0)
        true_label = true_label_tensor.item()
        
        # Create environment with enhanced state space
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=DEVICE,
            image_feature_dim=IMAGE_FEATURE_DIM
        )
        state = env.reset(image_tensor, true_label)
        
        # Verify state dimension
        if len(state) != STATE_DIM:
            print(f"Warning: Expected state dim {STATE_DIM}, got {len(state)}")
        
        # Track initial state
        if env.initial_correct:
            initial_correct_count += 1
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        # Get final metrics
        metrics = env.get_improvement_metrics()
        
        total_reward += episode_reward
        if episode_reward > 0:
            successful_episodes += 1
        if metrics['overall_improved']:
            improvement_episodes += 1
        if metrics['confidence_improved']:
            confidence_improvements += 1
        if metrics['correctness_improved']:
            correctness_fixes += 1
        if metrics['final_correct']:
            final_correct_count += 1
    
    agent.epsilon = original_epsilon  # Restore exploration
    
    # Calculate metrics
    avg_reward = total_reward / eval_episodes
    success_rate = successful_episodes / eval_episodes
    improvement_rate = improvement_episodes / eval_episodes
    confidence_improvement_rate = confidence_improvements / eval_episodes
    correctness_fix_rate = correctness_fixes / eval_episodes
    
    initial_accuracy = initial_correct_count / eval_episodes
    final_accuracy = final_correct_count / eval_episodes
    accuracy_improvement = final_accuracy - initial_accuracy
    
    print(f"Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Overall Improvement Rate: {improvement_rate:.2%}")
    print(f"  Confidence Improvement Rate: {confidence_improvement_rate:.2%}")
    print(f"  Correctness Fix Rate: {correctness_fix_rate:.2%}")
    print(f"  Initial Accuracy: {initial_accuracy:.3f}")
    print(f"  Final Accuracy: {final_accuracy:.3f}")
    print(f"  Accuracy Improvement: {accuracy_improvement:.3f}")
    
    return {
        'avg_reward': avg_reward,
        'success_rate': success_rate,
        'improvement_rate': improvement_rate,
        'accuracy_improvement': accuracy_improvement,
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy
    }


def curriculum_learning_schedule(episode):
    """
    Implement curriculum learning by adjusting the difficulty.
    """
    if episode < 3000:
        return 'easy'  # Focus on clearly incorrect images
    elif episode < 12000:
        return 'medium'  # Mix of easy and medium difficulty
    else:
        return 'hard'  # All difficulties


def train_rl_agent():
    """
    Main training function with enhanced state space and improved strategies.
    """
    global training_data
    classifier_model = load_classifier_model()

    # Prepare dataset
    preprocess_for_rl_env = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Use training set for RL training for more variety
    rl_episode_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=True, download=True, transform=preprocess_for_rl_env)
    
    rl_episode_loader = create_balanced_dataset_loader(rl_episode_dataset)
    rl_episode_iter = iter(rl_episode_loader)

    # Initialize enhanced agent with new state dimension
    agent = DQNAgent(
        STATE_DIM, ACTION_DIM, DEVICE, gamma, learning_rate,
        epsilon_start, epsilon_end, epsilon_decay, buffer_size,
        batch_size, target_update_freq, double_dqn=True, prioritized_replay=False
    )

    training_data['agent'] = agent
    # Training metrics
    global_episode_counter = 0
    episode_rewards = []
    evaluation_history = []
    loss_history = []
    
    # Moving averages for smoother tracking
    recent_rewards = deque(maxlen=200)
    recent_losses = deque(maxlen=200)
    
    # Early stopping variables
    global best_eval_reward, patience_counter
    
    print(f"\nStarting enhanced RL Agent training...")
    print(f"Training for {num_total_episodes} episodes")
    print(f"Enhanced state dim: {STATE_DIM}, Action dim: {ACTION_DIM}")
    print(f"Image feature dimension: {IMAGE_FEATURE_DIM}")
    
    start_time = time.time()
    current_image = None
    current_label = None
    
    # Pre-fill replay buffer with random experiences
    print("Pre-filling replay buffer with enhanced states...")
    prefill_episodes = min(2000, num_total_episodes // 10)
    for _ in tqdm(range(prefill_episodes), desc="Pre-filling"):
        try:
            image_tensor, true_label_tensor = next(rl_episode_iter)
        except StopIteration:
            rl_episode_iter = iter(rl_episode_loader)
            image_tensor, true_label_tensor = next(rl_episode_iter)
        
        image_tensor = image_tensor.squeeze(0)
        true_label = true_label_tensor.item()
        
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=DEVICE,
            image_feature_dim=IMAGE_FEATURE_DIM
        )
        state = env.reset(image_tensor, true_label)
        
        # Verify state dimension during prefill
        if len(state) != STATE_DIM:
            print(f"Error: State dimension mismatch! Expected {STATE_DIM}, got {len(state)}")
            print(f"Components: logits={LOGITS_DIM}, additional={ADDITIONAL_FEATURES_DIM}, image_features={IMAGE_FEATURE_DIM}")
            exit()
        
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.store_experience(state, action, reward, next_state, done)
            state = next_state
            steps += 1
    
    print(f"Pre-filled buffer with {len(agent.replay_buffer)} experiences")
    print(f"State verification: First state shape = {len(state)}")
    
    # Main training loop
    for episode in range(num_total_episodes):
        global_episode_counter += 1
        episode_reward = 0
        
        # Curriculum learning: select appropriate difficulty
        difficulty = curriculum_learning_schedule(episode)
        
        # Get new image every 'images_per_cycle' episodes
        if episode % images_per_cycle == 0:
            try:
                image_tensor_for_episode, true_label_for_episode_tensor = next(rl_episode_iter)
            except StopIteration:
                rl_episode_iter = iter(rl_episode_loader)
                image_tensor_for_episode, true_label_for_episode_tensor = next(rl_episode_iter)
            
            current_image = image_tensor_for_episode.squeeze(0)
            current_label = true_label_for_episode_tensor.item()

        # Initialize environment with enhanced state space
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=DEVICE,
            image_feature_dim=IMAGE_FEATURE_DIM
        )
        state = env.reset(current_image, current_label)
        
        # Skip this episode if the image is already correctly classified with high confidence
        # (focus on harder examples)
        if episode > warmup_episodes and env.initial_correct and env.initial_confidence > 0.95:
            if np.random.random() < 0.7:  # Skip 70% of such easy examples
                continue
        
        # Run episode
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1

            # Learning step (multiple per episode for faster learning)
            if len(agent.replay_buffer) > batch_size and episode > warmup_episodes:
                # Learning solo ogni 2 episodi, ma più steps
                if episode % 2 == 0:
                    for _ in range(4):  # 4 steps ogni 2 episodi = stesso learning
                        loss_item = agent.learn()
                        if loss_item is not None:
                            loss_history.append(loss_item)
                            recent_losses.append(loss_item)
        
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)

        # Update global training data
        training_data['episode_rewards'] = episode_rewards
        training_data['loss_history'] = loss_history
        training_data['evaluation_history'] = evaluation_history

        # ENHANCED EPSILON DECAY - Adaptive for enhanced state space
        if episode > warmup_episodes:  # Applica decay solo dopo warmup
            if episode < 15000:  # First 20% of training (dopo warmup)
                current_epsilon_decay = 0.999925
            elif episode < 45000:  # Middle 40% of training
                current_epsilon_decay = 0.9995
            else:  # Final 40% of training
                current_epsilon_decay = 0.9990
            
            # Apply adaptive epsilon decay
            agent.epsilon = max(agent.epsilon_end, agent.epsilon * current_epsilon_decay)

        # Logging
        if global_episode_counter % 200 == 0:
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            
            elapsed_time = time.time() - start_time
            episodes_per_second = global_episode_counter / elapsed_time
            
            # Get action distribution
            action_dist = agent.get_action_distribution()
            most_used_action = np.argmax(action_dist)
            
            print(f"Episode {global_episode_counter}/{num_total_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Most Used Action: {most_used_action} | "
                  f"EPS/s: {episodes_per_second:.1f} | "
                  f"State Dim: {len(state)}")
        
        # Evaluation and early stopping
        if global_episode_counter % eval_freq == 0 and global_episode_counter > warmup_episodes:
            eval_results = evaluate_agent_detailed(agent, classifier_model, eval_episodes)
            evaluation_history.append((global_episode_counter, eval_results))
            training_data['evaluation_history'] = evaluation_history
            
            eval_reward = eval_results['avg_reward']
            
            # Early stopping check
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                patience_counter = 0
                
                # Save best model
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(agent.q_network.state_dict(), './models/best_enhanced_dqn_model.pth')
                print(f"🎉 New best model saved! Eval reward: {eval_reward:.3f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at episode {global_episode_counter}")
                    break
        
        # Periodic model saving
        if global_episode_counter % 3000 == 0:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(agent.q_network.state_dict(), 
                      f'./models/enhanced_dqn_episode_{global_episode_counter}.pth')
            print(f"Checkpoint saved at episode {global_episode_counter}")
    
    print("\nEnhanced RL Agent training finished.")
    
    # Final evaluation
    print("\n" + "="*50)
    print("FINAL EVALUATION")
    print("="*50)
    final_eval_results = evaluate_agent_detailed(agent, classifier_model, eval_episodes * 2)
    
    # Save final model
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(agent.q_network.state_dict(), './models/final_enhanced_dqn_model.pth')
    
    # Create comprehensive plots
    create_comprehensive_plots(episode_rewards, loss_history, evaluation_history, agent)
    
    return final_eval_results


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully by saving results before exit."""
    print('\n\n🛑 Training interrupted by user!')
    print('💾 Saving results and creating plots...')
    
    try:
        # Save current model
        if training_data['agent'] is not None:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(training_data['agent'].q_network.state_dict(), 
                      './models/interrupted_enhanced_dqn_model.pth')
            print('✅ Model saved as: ./models/interrupted_enhanced_dqn_model.pth')
        
        # Create plots with current data
        if len(training_data['episode_rewards']) > 0:
            create_comprehensive_plots(
                training_data['episode_rewards'],
                training_data['loss_history'], 
                training_data['evaluation_history'],
                training_data['agent']
            )
            print('✅ Plots saved to: ./plots/enhanced_comprehensive_training_analysis.png')
        
        # Print summary of what was accomplished
        if len(training_data['episode_rewards']) > 0:
            print(f'\n📊 TRAINING SUMMARY (Interrupted):')
            print(f'  Episodes completed: {len(training_data["episode_rewards"]):,}')
            print(f'  Average reward: {np.mean(training_data["episode_rewards"]):.3f}')
            print(f'  Final epsilon: {training_data["agent"].epsilon:.3f}')
            
            if training_data['evaluation_history']:
                last_eval = training_data['evaluation_history'][-1][1]
                print(f'  Last evaluation accuracy: {last_eval["final_accuracy"]:.3f}')
                print(f'  Last evaluation improvement: {last_eval["accuracy_improvement"]:.3f}')
        
        print('\n✅ All data saved successfully!')
        print('🔄 You can resume training later by loading the interrupted model.')
        
    except Exception as e:
        print(f'❌ Error during save: {e}')
    
    print('\n👋 Goodbye!')
    sys.exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def create_comprehensive_plots(episode_rewards, loss_history, evaluation_history, agent):
    """Create comprehensive training visualization plots."""
    
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Training rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
    if len(episode_rewards) > 100:
        # Moving average
        window = 100
        moving_avg = [np.mean(episode_rewards[max(0, i-window):i+1]) for i in range(len(episode_rewards))]
        ax1.plot(moving_avg, color='red', linewidth=2, label=f'Moving Average ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Enhanced Training Rewards Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    ax2 = axes[0, 1]
    if loss_history:
        ax2.plot(loss_history, alpha=0.3, color='orange', label='Raw Loss')
        if len(loss_history) > 100:
            # Smoothed loss
            window = 100
            smoothed = [np.mean(loss_history[max(0, i-window):i+1]) for i in range(len(loss_history))]
            ax2.plot(smoothed, color='red', linewidth=2, label=f'Smoothed ({window})')
        ax2.set_xlabel('Learning Step')
        ax2.set_ylabel('Loss')
        ax2.set_title('Enhanced Training Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Evaluation metrics
    ax3 = axes[0, 2]
    if evaluation_history:
        episodes, results = zip(*evaluation_history)
        rewards = [r['avg_reward'] for r in results]
        improvements = [r['accuracy_improvement'] for r in results]
        
        ax3_twin = ax3.twinx()
        line1 = ax3.plot(episodes, rewards, 'o-', color='green', label='Avg Reward')
        line2 = ax3_twin.plot(episodes, improvements, 's-', color='purple', label='Accuracy Improvement')
        
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Reward', color='green')
        ax3_twin.set_ylabel('Accuracy Improvement', color='purple')
        ax3.set_title('Enhanced Evaluation Progress')
        
        # Combine legends
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_twin.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2)
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Action distribution
    ax4 = axes[1, 0]
    action_dist = agent.get_action_distribution()
    action_names = [f"Action {i}" for i in range(len(action_dist))]
    bars = ax4.bar(action_names, action_dist, color='skyblue', edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Actions')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Action Distribution During Enhanced Training')
    ax4.tick_params(axis='x', rotation=45)
    
    # Highlight most and least used actions
    max_idx = np.argmax(action_dist)
    min_idx = np.argmin(action_dist)
    bars[max_idx].set_color('red')
    bars[min_idx].set_color('yellow')
    
    # Plot 5: Reward distribution
    ax5 = axes[1, 1]
    ax5.hist(episode_rewards, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax5.axvline(np.mean(episode_rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(episode_rewards):.2f}')
    ax5.axvline(np.median(episode_rewards), color='blue', linestyle='--', 
                label=f'Median: {np.median(episode_rewards):.2f}')
    ax5.set_xlabel('Episode Reward')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Enhanced Reward Distribution')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Training statistics
    ax6 = axes[1, 2]
    if evaluation_history:
        final_results = evaluation_history[-1][1]
        stats_text = f"""Enhanced Training Results:
        
Total Episodes: {len(episode_rewards):,}
Final Avg Reward: {final_results['avg_reward']:.3f}
Success Rate: {final_results['success_rate']:.1%}
Improvement Rate: {final_results['improvement_rate']:.1%}
Accuracy Improvement: {final_results['accuracy_improvement']:.3f}

Enhanced State Space:
State Dimension: {STATE_DIM}
- Logits: {LOGITS_DIM}
- Additional Features: {ADDITIONAL_FEATURES_DIM}
- Image Features: {IMAGE_FEATURE_DIM}

Training Stats:
Mean Reward: {np.mean(episode_rewards):.3f}
Std Reward: {np.std(episode_rewards):.3f}
Max Reward: {np.max(episode_rewards):.3f}
Min Reward: {np.min(episode_rewards):.3f}
        """
    else:
        stats_text = f"""Enhanced Training Summary:
        
Total Episodes: {len(episode_rewards):,}
State Dimension: {STATE_DIM}
Mean Reward: {np.mean(episode_rewards):.3f}
Std Reward: {np.std(episode_rewards):.3f}
Max Reward: {np.max(episode_rewards):.3f}
Min Reward: {np.min(episode_rewards):.3f}
Positive Episodes: {(np.array(episode_rewards) > 0).mean():.1%}
        """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Enhanced Training Statistics Summary')
    
    plt.tight_layout()
    plt.savefig('./plots/enhanced_comprehensive_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Enhanced training plots saved to './plots/enhanced_comprehensive_training_analysis.png'")


if __name__ == '__main__':
    final_results = train_rl_agent()
    print("\n" + "="*60)
    print("ENHANCED TRAINING COMPLETED")
    print("="*60)
    print(f"Final Average Reward: {final_results['avg_reward']:.3f}")
    print(f"Final Success Rate: {final_results['success_rate']:.1%}")
    print(f"Final Accuracy Improvement: {final_results['accuracy_improvement']:.3f}")
    print(f"Enhanced State Dimension: {STATE_DIM}")
    print("="*60)