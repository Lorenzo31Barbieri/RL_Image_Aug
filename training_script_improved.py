# training_script.py (Improved version for CIFAR10)

import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import deque
from agent import DQNAgent
from environment import ImageAugmentationEnv
from transforms import get_num_actions
from vgg import VGG

# --- GLOBAL CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Dataset and Path Configuration ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
IMAGE_SIZE = 32
NUM_CLASSES = 10

# State and action dimensions for RL Agent
STATE_DIM = NUM_CLASSES
ACTION_DIM = get_num_actions()

# Improved hyperparameters for better training stability
learning_rate = 0.001  # Slightly higher for faster initial learning
gamma = 0.95  # More conservative discount factor
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9995  # Slower decay for better exploration
buffer_size = 50000  # Larger buffer for better experience diversity
batch_size = 128  # Larger batches for more stable learning
target_update_freq = 500  # Less frequent target updates for stability
num_total_episodes = 15000  # More episodes for better convergence
max_steps_per_episode = 3  # Fewer steps for more focused learning
images_per_cycle = 5  # Use same image for multiple episodes

# Early stopping and evaluation parameters
eval_freq = 500
eval_episodes = 100
patience = 100  # Number of evaluations without improvement before stopping
best_eval_reward = float('-inf')
patience_counter = 0


def load_classifier_model():
    """
    Load the pre-trained classifier.
    """

    print("Loading pre-trained VGG19 CIFAR10 classifier...")
    classifier_model = VGG('VGG19').to(DEVICE)
    
    try:
        checkpoint = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=DEVICE)
        
        # Handle 'module.' prefix if model was saved from DataParallel
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
        print("Please ensure you have trained your CIFAR10 VGG model and saved it as ckpt.pth in the 'checkpoint' directory.")
        exit()
    except KeyError:
        print(f"Error: Invalid checkpoint format at {PRE_TRAINED_CLASSIFIER_PATH}. Expected 'net' key.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading classifier: {e}")
        exit()

    classifier_model.eval()
    # Freeze classifier weights
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")
    return classifier_model

def evaluate_agent(agent, classifier_model, eval_episodes=100):
    """
    Evaluate agent performance without exploration (epsilon = 0).
    """

    print(f"\nEvaluating agent for {eval_episodes} episodes...")
    
    # Prepare evaluation dataset
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    eval_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=False, transform=eval_transform)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=1, shuffle=True, num_workers=0)
    eval_iter = iter(eval_loader)
    
    total_reward = 0
    successful_episodes = 0
    original_epsilon = agent.epsilon
    agent.epsilon = 0  # No exploration during evaluation
    
    for _ in range(eval_episodes):
        try:
            image_tensor, true_label_tensor = next(eval_iter)
        except StopIteration:
            eval_iter = iter(eval_loader)
            image_tensor, true_label_tensor = next(eval_iter)
        
        image_tensor = image_tensor.squeeze(0)
        true_label = true_label_tensor.item()
        
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=DEVICE
        )
        state = env.reset(image_tensor, true_label)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        
        total_reward += episode_reward
        if episode_reward > 0:
            successful_episodes += 1
    
    agent.epsilon = original_epsilon  # Restore exploration
    avg_reward = total_reward / eval_episodes
    success_rate = successful_episodes / eval_episodes
    
    print(f"Evaluation results: Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.2%}")
    return avg_reward

def clear_memory_cache():
    """Clear GPU memory cache to prevent memory issues"""
    if DEVICE.type == 'mps':
        torch.mps.empty_cache()
    elif DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

def train_rl_agent():
    """
    Main training function.
    """
    
    classifier_model = load_classifier_model()

    # Prepare CIFAR10 dataset for RL episodes
    preprocess_for_rl_env = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    rl_episode_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=True, transform=preprocess_for_rl_env)
    
    rl_episode_loader = torch.utils.data.DataLoader(
        rl_episode_dataset, batch_size=1, shuffle=True, num_workers=0)
    
    rl_episode_iter = iter(rl_episode_loader)

    # Initialize agent with improved parameters
    agent = DQNAgent(STATE_DIM, ACTION_DIM, DEVICE, gamma, learning_rate,
                     epsilon_start, epsilon_end, epsilon_decay, buffer_size,
                     batch_size, target_update_freq)

    # Training metrics
    global_episode_counter = 0
    episode_rewards = []
    average_rewards_history = []
    loss_history = []
    eval_rewards_history = []
    
    # Moving averages for smoother tracking
    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    
    # Early stopping variables
    global best_eval_reward, patience_counter
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=2000, gamma=0.9)

    print("\nStarting RL Agent training...")
    print(f"Training for {num_total_episodes} episodes with {max_steps_per_episode} max steps per episode")
    print(f"Using {images_per_cycle} episodes per image")
    
    start_time = time.time()
    current_image = None
    current_label = None
    
    for episode in range(num_total_episodes):
        global_episode_counter += 1
        episode_reward = 0
        
        # Get new image every 'images_per_cycle' episodes
        if episode % images_per_cycle == 0:
            try:
                image_tensor_for_episode, true_label_for_episode_tensor = next(rl_episode_iter)
            except StopIteration:
                rl_episode_iter = iter(rl_episode_loader)
                image_tensor_for_episode, true_label_for_episode_tensor = next(rl_episode_iter)
            
            current_image = image_tensor_for_episode.squeeze(0)
            current_label = true_label_for_episode_tensor.item()

        # Initialize environment with current image
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=DEVICE
        )
        state = env.reset(current_image, current_label)
        
        # Run episode
        done = False
        steps = 0
        while not done and steps < max_steps_per_episode:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            steps += 1

            # Learning step
            if len(agent.replay_buffer) > agent.batch_size:
                loss_item = agent.learn()
                if loss_item is not None:
                    loss_history.append(loss_item)
                    recent_losses.append(loss_item)
        
        episode_rewards.append(episode_reward)
        recent_rewards.append(episode_reward)
        
        # Update learning rate
        scheduler.step()

        # Logging and evaluation
        if global_episode_counter % 100 == 0:
            avg_reward = np.mean(recent_rewards)
            average_rewards_history.append(avg_reward)
            avg_loss = np.mean(recent_losses) if recent_losses else 0
            
            elapsed_time = time.time() - start_time
            episodes_per_second = global_episode_counter / elapsed_time
            
            print(f"Episode {global_episode_counter}/{num_total_episodes} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"EPS/s: {episodes_per_second:.1f}")
        
        # Periodic evaluation and early stopping
        if global_episode_counter % eval_freq == 0:
            eval_reward = evaluate_agent(agent, classifier_model, eval_episodes)
            eval_rewards_history.append((global_episode_counter, eval_reward))
            
            # Early stopping check
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                patience_counter = 0
                
                # Save best model
                if not os.path.exists('./models'):
                    os.makedirs('./models')
                torch.save(agent.q_network.state_dict(), './models/best_dqn_model.pth')
                print(f"New best model saved! Eval reward: {eval_reward:.3f}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                
                if patience_counter >= patience:
                    print(f"Early stopping at episode {global_episode_counter}")
                    break
        
        # Periodic model saving
        if global_episode_counter % 1000 == 0:
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(agent.q_network.state_dict(), f'./models/dqn_q_network_episode_{global_episode_counter}.pth')
            print(f"Checkpoint saved at episode {global_episode_counter}")
        
        # Memory management
        if global_episode_counter % 1000 == 0:
            clear_memory_cache()

    print("\nRL Agent training finished.")
    
    # Final evaluation
    final_eval_reward = evaluate_agent(agent, classifier_model, eval_episodes * 2)
    print(f"Final evaluation reward: {final_eval_reward:.3f}")
    
    # Save final model
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(agent.q_network.state_dict(), './models/final_dqn_model.pth')
    
    # Create comprehensive plots
    create_training_plots(episode_rewards, average_rewards_history, loss_history, eval_rewards_history)

def create_training_plots(episode_rewards, average_rewards_history, loss_history, eval_rewards_history):
    """Create comprehensive training visualization plots"""
    
    # Create directory for plots
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    
    # Plot 1: Training rewards
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, alpha=0.3, color='blue', label='Episode Rewards')
    if average_rewards_history:
        plt.plot(np.arange(0, len(average_rewards_history) * 100, 100), 
                average_rewards_history, color='red', linewidth=2, label='Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss history
    plt.subplot(2, 3, 2)
    if loss_history:
        # Smooth loss for better visualization
        smoothed_loss = []
        window_size = 100
        for i in range(len(loss_history)):
            start_idx = max(0, i - window_size)
            smoothed_loss.append(np.mean(loss_history[start_idx:i+1]))
        
        plt.plot(loss_history, alpha=0.3, color='orange', label='Raw Loss')
        plt.plot(smoothed_loss, color='red', linewidth=2, label=f'Smoothed Loss ({window_size} window)')
        plt.xlabel('Learning Step')
        plt.ylabel('Q-Network Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 3: Evaluation rewards
    plt.subplot(2, 3, 3)
    if eval_rewards_history:
        episodes, eval_rewards = zip(*eval_rewards_history)
        plt.plot(episodes, eval_rewards, 'o-', color='green', linewidth=2, markersize=6)
        plt.xlabel('Episode')
        plt.ylabel('Evaluation Reward')
        plt.title('Evaluation Performance')
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Reward distribution
    plt.subplot(2, 3, 4)
    plt.hist(episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Recent performance (last 1000 episodes)
    plt.subplot(2, 3, 5)
    recent_episodes = episode_rewards[-1000:] if len(episode_rewards) > 1000 else episode_rewards
    plt.plot(recent_episodes, color='purple', alpha=0.7)
    plt.xlabel('Recent Episodes')
    plt.ylabel('Reward')
    plt.title('Recent Performance (Last 1000 Episodes)')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Training statistics summary
    plt.subplot(2, 3, 6)
    stats_text = f"""Training Summary:
Total Episodes: {len(episode_rewards)}
Mean Reward: {np.mean(episode_rewards):.3f}
Std Reward: {np.std(episode_rewards):.3f}
Max Reward: {np.max(episode_rewards):.3f}
Min Reward: {np.min(episode_rewards):.3f}
Success Rate: {(np.array(episode_rewards) > 0).mean():.2%}
"""
    plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    plt.axis('off')
    plt.title('Training Statistics')
    
    plt.tight_layout()
    plt.savefig('./plots/comprehensive_training_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training plots saved to './plots/comprehensive_training_analysis.png'")

if __name__ == '__main__':
    train_rl_agent()