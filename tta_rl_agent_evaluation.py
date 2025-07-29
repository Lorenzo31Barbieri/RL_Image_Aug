import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os
import pickle
from collections import defaultdict

# Import TTA functionality
try:
    import ttach as tta
    TTA_AVAILABLE = True
except ImportError:
    TTA_AVAILABLE = False
    print("⚠️ ttach not available, using manual TTA implementation")

# Import your improved modules
from vgg import VGG # Assicurati che VGG sia in models.py, non vgg.py
from agent import DQNAgent
from environment import ImageAugmentationEnv
from transforms import get_num_actions, get_action_name

# --- GLOBAL CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Configuration ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
RL_MODEL_PATH = './models/best_improved_dqn_model.pth' # Assicurati che questo percorso sia corretto
IMAGE_SIZE = 32
NUM_CLASSES = 10

STATE_DIM = NUM_CLASSES + 5  # Enhanced state representation ( logits + current_step, correct_before, confidence_before, correct_after, confidence_after)
ACTION_DIM = get_num_actions()
MAX_STEPS_PER_EPISODE = 3


class TTAWrapper:
    """Simple TTA wrapper for manual implementation."""
    
    def __init__(self, model):
        self.model = model
        self.transforms = self._create_transforms()
    
    def _create_transforms(self):
        """Create TTA transforms suitable for CIFAR-10."""
        transforms_list = []
        
        # Original
        transforms_list.append(lambda x: x)
        
        # Horizontal flip
        transforms_list.append(lambda x: torch.flip(x, dims=[3]))
        
        # Brightness variations
        # These operations might need to be carefully handled with normalization.
        # If input is normalized, direct multiplication might shift mean/std too much.
        # It's better to unnormalize, apply transform, then re-normalize or apply directly
        # on pixel values (0-255) before initial ToTensor().
        # For simplicity in this manual TTA, assuming operations are fine on normalized [0,1] range.
        transforms_list.append(lambda x: torch.clamp(x * 1.1, 0, 1)) 
        transforms_list.append(lambda x: torch.clamp(x * 0.9, 0, 1)) 
        
        # Contrast-like variations
        # These operate on normalized tensors. The (x-0.5)*factor + 0.5 centers and scales.
        transforms_list.append(lambda x: torch.clamp((x - 0.5) * 1.1 + 0.5, 0, 1))
        transforms_list.append(lambda x: torch.clamp((x - 0.5) * 0.9 + 0.5, 0, 1))
        
        return transforms_list
    
    def predict(self, x):
        """Make TTA prediction."""
        predictions = []
        
        with torch.no_grad():
            for transform in self.transforms:
                transformed_x = transform(x)
                pred = self.model(transformed_x)
                predictions.append(pred)
        
        # Average predictions (logits or probabilities)
        # For classification, averaging logits is generally preferred.
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction


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
        print(f"✅ Classifier loaded successfully")
        print(f"Baseline accuracy from checkpoint: {checkpoint['acc']:.2f}%") # Updated message
        
    except Exception as e:
        print(f"❌ Error loading classifier: {e}")
        exit()

    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    
    return classifier_model


def load_rl_agent():
    """Load the trained RL agent."""
    print(f"Loading RL agent from {RL_MODEL_PATH}...")
    
    agent = DQNAgent(STATE_DIM, ACTION_DIM, DEVICE)
    
    if os.path.exists(RL_MODEL_PATH):
        try:
            agent.q_network.load_state_dict(torch.load(RL_MODEL_PATH, map_location=DEVICE))
            agent.target_q_network.load_state_dict(torch.load(RL_MODEL_PATH, map_location=DEVICE))
            agent.q_network.eval()
            agent.target_q_network.eval()
            agent.epsilon = 0   # No exploration for evaluation
            print("✅ RL agent loaded successfully")
            return agent, True
        except Exception as e:
            print(f"❌ Error loading RL agent: {e}")
    else:
        print(f"❌ RL model not found at {RL_MODEL_PATH}")
    
    print("Using randomly initialized agent for comparison...")
    agent.epsilon = 0
    return agent, False


def evaluate_baseline(classifier_model, test_loader):
    """Evaluate baseline classifier performance."""
    print("\n=== BASELINE EVALUATION ===")
    
    correct = 0
    total = 0
    all_confidences = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Baseline"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = classifier_model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_confidences.extend(confidences.cpu().numpy())
    
    inference_time = time.time() - start_time
    accuracy = correct / total
    avg_confidence = np.mean(all_confidences)
    
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Total Time: {inference_time:.2f}s")
    print(f"Time per Sample: {inference_time / total * 1000:.1f}ms") # Converte in ms
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'total_time': inference_time,
        'time_per_sample': inference_time / total
    }


def evaluate_tta(classifier_model, test_dataset, num_samples=1000):
    """Evaluate TTA performance on individual images."""
    print(f"\n=== TTA EVALUATION ({num_samples} samples) ===")
    
    if TTA_AVAILABLE:
        # Using ttach library's transforms and wrapper
        tta_transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Multiply(factors=[0.9, 1.0, 1.1]), # brightness/contrast variations
        ])
        tta_model = tta.ClassificationTTAWrapper(classifier_model, tta_transforms)
        # ttach automatically adds identity transform, so len(tta_transforms.transforms) + 1 for identity
        # or, len(tta_model.aug_transformations) if you want exact count including identity
        num_augmentations = len(tta_transforms.aug_transforms) + 1 
    else:
        # Using manual TTAWrapper
        tta_model = TTAWrapper(classifier_model)
        num_augmentations = len(tta_model.transforms)
    
    # Sample random images
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    correct_baseline = 0
    correct_tta = 0
    confidence_improvements = []
    improvements = 0
    degradations = 0
    
    start_time = time.time()
    
    for idx in tqdm(indices, desc="TTA evaluation"):
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(DEVICE) # Add batch dimension
        
        with torch.no_grad():
            # Baseline prediction
            baseline_output = classifier_model(image)
            baseline_prob = torch.nn.functional.softmax(baseline_output, dim=1)
            baseline_confidence, baseline_pred = torch.max(baseline_prob, 1)
            baseline_correct = (baseline_pred.item() == label)
            
            # TTA prediction
            if TTA_AVAILABLE:
                tta_output = tta_model(image) # ttach's wrapper handles this
            else:
                tta_output = tta_model.predict(image) # Manual wrapper's predict method
            
            tta_prob = torch.nn.functional.softmax(tta_output, dim=1)
            tta_confidence, tta_pred = torch.max(tta_prob, 1)
            tta_correct = (tta_pred.item() == label)
            
            # Track metrics
            if baseline_correct:
                correct_baseline += 1
            if tta_correct:
                correct_tta += 1
            
            confidence_change = tta_confidence.item() - baseline_confidence.item()
            confidence_improvements.append(confidence_change)
            
            if not baseline_correct and tta_correct:
                improvements += 1
            elif baseline_correct and not tta_correct:
                degradations += 1
    
    total_time = time.time() - start_time
    
    baseline_accuracy = correct_baseline / num_samples
    tta_accuracy = correct_tta / num_samples
    accuracy_improvement = tta_accuracy - baseline_accuracy
    avg_confidence_improvement = np.mean(confidence_improvements)
    
    print(f"Baseline Accuracy (sampled): {baseline_accuracy:.4f}") # Adjusted label since it's on a sample
    print(f"TTA Accuracy (sampled): {tta_accuracy:.4f}")
    print(f"Accuracy Improvement: {accuracy_improvement:+.4f}")
    print(f"Confidence Change: {avg_confidence_improvement:+.4f}")
    print(f"Images Improved: {improvements} ({improvements/num_samples:.1%})")
    print(f"Images Degraded: {degradations} ({degradations/num_samples:.1%})")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Time per Sample: {total_time/num_samples*1000:.1f}ms")
    print(f"Augmentations Used (including original): {num_augmentations}")
    
    return {
        'baseline_accuracy': baseline_accuracy,
        'tta_accuracy': tta_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'avg_confidence_improvement': avg_confidence_improvement,
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_samples,
        'degradation_rate': degradations / num_samples,
        'total_time': total_time,
        'time_per_sample': total_time / num_samples,
        'num_augmentations': num_augmentations,
        'confidence_improvements': confidence_improvements
    }


def evaluate_rl_agent(agent, classifier_model, test_dataset, num_samples=1000):
    """Evaluate RL agent performance on individual images."""
    print(f"\n=== RL AGENT EVALUATION ({num_samples} samples) ===")
    
    # Sample random images
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    correct_initial = 0
    correct_final = 0
    confidence_improvements = []
    improvements = 0
    degradations = 0
    episode_rewards = []
    action_counts = defaultdict(int)
    sequence_lengths = []
    
    start_time = time.time()
    
    for idx in tqdm(indices, desc="RL evaluation"):
        image, label = test_dataset[idx]
        
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=MAX_STEPS_PER_EPISODE,
            device=DEVICE
        )
        
        state = env.reset(image, label) # Pass the original PIL image from dataset, env handles ToTensor and Normalize
        
        # Track initial state
        initial_correct = env.initial_correct
        initial_confidence = env.initial_confidence
        
        if initial_correct:
            correct_initial += 1
        
        # Run RL episode
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            actions_taken.append(action)
            action_counts[get_action_name(action)] += 1
            state = next_state
        
        # Get final metrics
        metrics = env.get_improvement_metrics()
        final_correct = metrics['final_correct']
        final_confidence = metrics['final_confidence']
        
        if final_correct:
            correct_final += 1
        
        # Track improvements
        confidence_change = final_confidence - initial_confidence
        confidence_improvements.append(confidence_change)
        episode_rewards.append(episode_reward)
        sequence_lengths.append(len(actions_taken))
        
        if not initial_correct and final_correct:
            improvements += 1
        elif initial_correct and not final_correct:
            degradations += 1
    
    total_time = time.time() - start_time
    
    initial_accuracy = correct_initial / num_samples
    final_accuracy = correct_final / num_samples
    accuracy_improvement = final_accuracy - initial_accuracy
    avg_confidence_improvement = np.mean(confidence_improvements)
    avg_reward = np.mean(episode_rewards)
    avg_sequence_length = np.mean(sequence_lengths)
    
    print(f"Initial Accuracy (sampled): {initial_accuracy:.4f}") # Adjusted label
    print(f"Final Accuracy (sampled): {final_accuracy:.4f}")
    print(f"Accuracy Improvement: {accuracy_improvement:+.4f}")
    print(f"Confidence Change: {avg_confidence_improvement:+.4f}")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Images Improved: {improvements} ({improvements/num_samples:.1%})")
    print(f"Images Degraded: {degradations} ({degradations/num_samples:.1%})")
    print(f"Average Sequence Length: {avg_sequence_length:.1f}")
    print(f"Total Time: {total_time:.2f}s")
    print(f"Time per Sample: {total_time/num_samples*1000:.1f}ms")
    
    return {
        'initial_accuracy': initial_accuracy,
        'final_accuracy': final_accuracy,
        'accuracy_improvement': accuracy_improvement,
        'avg_confidence_improvement': avg_confidence_improvement,
        'avg_reward': avg_reward,
        'improvements': improvements,
        'degradations': degradations,
        'improvement_rate': improvements / num_samples,
        'degradation_rate': degradations / num_samples,
        'total_time': total_time,
        'time_per_sample': total_time / num_samples,
        'avg_sequence_length': avg_sequence_length,
        'action_counts': dict(action_counts),
        'episode_rewards': episode_rewards,
        'confidence_improvements': confidence_improvements
    }


def create_comprehensive_comparison_plots(baseline_results, tta_results, rl_results, rl_loaded):
    """Create comprehensive comparison visualizations."""
    
    if not os.path.exists('./comparison_plots'):
        os.makedirs('./comparison_plots')
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    methods = ['Baseline', 'TTA', 'RL Initial', 'RL Final']
    accuracies = [
        baseline_results['accuracy'],
        tta_results['tta_accuracy'],
        rl_results['initial_accuracy'],
        rl_results['final_accuracy']
    ]
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
    
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison: Baseline vs TTA vs RL')
    ax1.set_ylim(0, max(accuracies) * 1.1)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Improvement Comparison
    ax2 = axes[0, 1]
    improvements = [
        tta_results['accuracy_improvement'],
        rl_results['accuracy_improvement']
    ]
    improvement_methods = ['TTA', 'RL']
    colors_imp = ['green' if x > 0 else 'red' for x in improvements]
    
    bars2 = ax2.bar(improvement_methods, improvements, color=colors_imp, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title('Accuracy Improvement Comparison')
    ax2.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    for bar, imp in zip(bars2, improvements):
        ax2.text(bar.get_x() + bar.get_width()/2, 
                 bar.get_height() + 0.002 if imp >= 0 else bar.get_height() - 0.002,
                 f'{imp:+.4f}', ha='center', 
                 va='bottom' if imp >= 0 else 'top', fontweight='bold')
    
    # Plot 3: Time Comparison
    ax3 = axes[0, 2]
    times = [
        baseline_results['time_per_sample'] * 1000,
        tta_results['time_per_sample'] * 1000,
        rl_results['time_per_sample'] * 1000
    ]
    time_methods = ['Baseline', 'TTA', 'RL']
    
    bars3 = ax3.bar(time_methods, times, color=['skyblue', 'orange', 'purple'], 
                    alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Time per Sample (ms)')
    ax3.set_title('Inference Time Comparison')
    
    for bar, time_val in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                 f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Confidence Change Distribution
    ax4 = axes[1, 0]
    ax4.hist(tta_results['confidence_improvements'], bins=30, alpha=0.5, 
             label='TTA', color='green', density=True)
    ax4.hist(rl_results['confidence_improvements'], bins=30, alpha=0.5, 
             label='RL', color='purple', density=True)
    ax4.axvline(0, color='red', linestyle='--', label='No change')
    ax4.set_xlabel('Confidence Change')
    ax4.set_ylabel('Density')
    ax4.set_title('Confidence Change Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Success Rate Comparison
    ax5 = axes[1, 1]
    success_metrics = ['Improvement\nRate', 'Degradation\nRate', 'Net Success\nRate']
    tta_values = [
        tta_results['improvement_rate'],
        tta_results['degradation_rate'],
        tta_results['improvement_rate'] - tta_results['degradation_rate']
    ]
    rl_values = [
        rl_results['improvement_rate'],
        rl_results['degradation_rate'],
        rl_results['improvement_rate'] - rl_results['degradation_rate']
    ]
    
    x = np.arange(len(success_metrics))
    width = 0.35
    
    bars5a = ax5.bar(x - width/2, tta_values, width, label='TTA', color='lightgreen', alpha=0.8)
    bars5b = ax5.bar(x + width/2, rl_values, width, label='RL', color='lightcoral', alpha=0.8)
    
    ax5.set_ylabel('Rate')
    ax5.set_title('Success Rate Comparison')
    ax5.set_xticks(x)
    ax5.set_xticklabels(success_metrics)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars5a, bars5b]:
        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.005 if height >= 0 else height - 0.005,
                     f'{height:.2f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=8)
    
    # Plot 6: Efficiency Analysis
    ax6 = axes[1, 2]
    
    # Calculate efficiency scores
    baseline_time = baseline_results['time_per_sample']
    tta_slowdown = tta_results['time_per_sample'] / baseline_time
    rl_slowdown = rl_results['time_per_sample'] / baseline_time
    
    # Efficiency score: Improvement / (Slowdown - 1)
    # Handle division by zero if slowdown is 1 (no extra time) or less (faster than baseline - unlikely but possible with floating point errors)
    tta_efficiency = tta_results['accuracy_improvement'] / (tta_slowdown - 1) if tta_slowdown > 1 else 0
    rl_efficiency = rl_results['accuracy_improvement'] / (rl_slowdown - 1) if rl_slowdown > 1 else 0
    
    efficiency_data = {
        'TTA': {
            'improvement': tta_results['accuracy_improvement'],
            'slowdown': tta_slowdown,
            'efficiency': tta_efficiency
        },
        'RL': {
            'improvement': rl_results['accuracy_improvement'],
            'slowdown': rl_slowdown,
            'efficiency': rl_efficiency
        }
    }
    
    methods_eff = list(efficiency_data.keys())
    improvements_eff = [efficiency_data[m]['improvement'] for m in methods_eff]
    slowdowns = [efficiency_data[m]['slowdown'] for m in methods_eff]
    
    scatter = ax6.scatter(slowdowns, improvements_eff, s=200, alpha=0.7, 
                          c=['green', 'purple'], edgecolors='black')
    
    for i, method in enumerate(methods_eff):
        ax6.annotate(method, (slowdowns[i], improvements_eff[i]), 
                     xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    ax6.set_xlabel('Slowdown Factor (×)')
    ax6.set_ylabel('Accuracy Improvement')
    ax6.set_title('Efficiency Plot: Improvement vs Computational Cost')
    ax6.grid(True, alpha=0.3)
    ax6.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax6.axvline(1, color='red', linestyle='--', alpha=0.5)
    
    # Plot 7: Method Characteristics
    ax7 = axes[2, 0]
    characteristics = ['Accuracy\nGain', 'Consistency', 'Speed\n(inverse)', 'Complexity\n(inverse)']
    
    # Normalize characteristics for radar chart (0-1 scale)
    # Max expected gain is 1 (0 to 1 accuracy), min -1 (1 to 0 accuracy).
    # Assuming typical improvements are small, scaling by 0.05 (5%) to make differences visible.
    # Consistency: 1 - degradation rate (higher is better)
    # Speed (inverse): 1 / slowdown (higher is better, faster)
    # Complexity (inverse): Manual assignment, RL is more complex to set up.
    tta_chars = [
        max(0, min(1, tta_results['accuracy_improvement'] / 0.05)),   # Accuracy gain
        1 - (tta_results['degradation_rate']),   # Consistency (low degradation)
        1 / tta_slowdown if tta_slowdown != 0 else 0,   # Speed (inverse of slowdown)
        0.8   # Complexity (TTA is simpler, higher score)
    ]
    
    rl_chars = [
        max(0, min(1, rl_results['accuracy_improvement'] / 0.05)),   # Accuracy gain
        1 - (rl_results['degradation_rate']),   # Consistency
        1 / rl_slowdown if rl_slowdown != 0 else 0,   # Speed
        0.3   # Complexity (RL is more complex, lower score)
    ]
    
    x_chars = np.arange(len(characteristics))
    width = 0.35
    
    bars7a = ax7.bar(x_chars - width/2, tta_chars, width, label='TTA', color='lightgreen', alpha=0.8)
    bars7b = ax7.bar(x_chars + width/2, rl_chars, width, label='RL', color='lightcoral', alpha=0.8)
    
    ax7.set_ylabel('Normalized Score (0-1)')
    ax7.set_title('Method Characteristics Comparison')
    ax7.set_xticks(x_chars)
    ax7.set_xticklabels(characteristics, fontsize=9)
    ax7.legend()
    ax7.set_ylim(0, 1)
    
    # Plot 8: RL Specific Analysis (Reward Distribution)
    ax8 = axes[2, 1]
    if rl_results['episode_rewards']: # Ensure data exists
        ax8.hist(rl_results['episode_rewards'], bins=np.arange(min(rl_results['episode_rewards']), max(rl_results['episode_rewards']) + 1) - 0.5, alpha=0.7, color='purple', edgecolor='black') # Adjusted bins for discrete rewards
        ax8.axvline(np.mean(rl_results['episode_rewards']), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(rl_results["episode_rewards"]):.2f}')
        ax8.set_xlabel('Episode Reward')
        ax8.set_ylabel('Frequency')
        ax8.set_title('RL Agent Reward Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    else:
        ax8.text(0.5, 0.5, "No RL rewards data available.", horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
        ax8.axis('off')

    # Plot 9: Summary and Recommendations
    ax9 = axes[2, 2]
    
    # Determine winner
    tta_net_improvement = tta_results['accuracy_improvement']
    rl_net_improvement = rl_results['accuracy_improvement']

    winner = "Tie"
    if tta_net_improvement > rl_net_improvement + 0.001: # Small threshold for "better"
        winner = "TTA"
    elif rl_net_improvement > tta_net_improvement + 0.001:
        winner = "RL"
    
    summary_text = f"""🏆 COMPARISON SUMMARY
📊 Performance Winner (Net Accuracy Gain): {winner}
• TTA Improvement: {tta_results['accuracy_improvement']:+.4f}
• RL Improvement: {rl_results['accuracy_improvement']:+.4f}

⚡ Speed Comparison:
• TTA: {tta_slowdown:.1f}× slower
• RL: {rl_slowdown:.1f}× slower
• Faster Method (inference time): {'TTA' if tta_slowdown < rl_slowdown else 'RL' if rl_slowdown < tta_slowdown else 'Similar'}

🎯 Use Case Recommendations:
• TTA: {'✅' if tta_results['accuracy_improvement'] > 0.0 else '⚠️'} Simpler setup, fixed aug.
• RL: {'✅' if rl_results['accuracy_improvement'] > 0.0 and rl_loaded else '⚠️'} {'Requires training, adaptive' if rl_loaded else 'Needs training'}

💡 Best Choice:
{
    'TTA - Simpler, often good baseline' if winner == "TTA"
    else 'RL - More flexible, potential for specific gains' if winner == "RL" and rl_loaded
    else 'Consider both, or neither if gain is minimal'
}

🔬 Technical Notes:
• TTA uses {tta_results['num_augmentations']} augmentations per image.
• RL uses avg {rl_results['avg_sequence_length']:.1f} steps per image.
• RL model: {'✅ Trained' if rl_loaded else '❌ Random (No real agent)'}
"""
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    ax9.set_title('Overall Comparison & Recommendations')
    
    plt.tight_layout()
    plt.savefig('./comparison_plots/tta_vs_rl_comprehensive_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive comparison plots saved to './comparison_plots/tta_vs_rl_comprehensive_comparison.png'")


def main_comparison():
    """Main comparison function."""
    print("=" * 80)
    print("COMPREHENSIVE TTA vs RL COMPARISON")
    print("=" * 80)
    
    # Load models
    classifier_model = load_classifier_model()
    agent, rl_loaded = load_rl_agent()
    
    # Prepare test dataset
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=True, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(test_dataset)} images")
    
    # Run evaluations
    baseline_results = evaluate_baseline(classifier_model, test_loader)
    tta_results = evaluate_tta(classifier_model, test_dataset, num_samples=1000)
    rl_results = evaluate_rl_agent(agent, classifier_model, test_dataset, num_samples=1000)
    
    # Comprehensive comparison
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON RESULTS")
    print("=" * 80)
    
    print(f"\n📊 ACCURACY COMPARISON:")
    print(f"  Baseline (Full Test Set): {baseline_results['accuracy']:.4f}")
    print(f"  TTA (Sampled {len(tta_results['confidence_improvements'])}): {tta_results['tta_accuracy']:.4f} (Improvement: {tta_results['accuracy_improvement']:+.4f})")
    print(f"  RL Agent (Sampled {len(rl_results['confidence_improvements'])}): {rl_results['final_accuracy']:.4f} (Improvement: {rl_results['accuracy_improvement']:+.4f})")
    
    print(f"\n🔍 CONFIDENCE ANALYSIS:")
    print(f"  TTA Average Confidence Change: {tta_results['avg_confidence_improvement']:+.4f}")
    print(f"  RL Average Confidence Change: {rl_results['avg_confidence_improvement']:+.4f}")
    
    print(f"\n⚡ PERFORMANCE (TIME) ANALYSIS:")
    baseline_time_per_sample_ms = baseline_results['time_per_sample'] * 1000
    tta_time_per_sample_ms = tta_results['time_per_sample'] * 1000
    rl_time_per_sample_ms = rl_results['time_per_sample'] * 1000

    print(f"  Baseline Time per Sample: {baseline_time_per_sample_ms:.1f}ms")
    print(f"  TTA Time per Sample: {tta_time_per_sample_ms:.1f}ms (Slowdown: {tta_time_per_sample_ms / baseline_time_per_sample_ms:.1f}x)")
    print(f"  RL Agent Time per Sample: {rl_time_per_sample_ms:.1f}ms (Slowdown: {rl_time_per_sample_ms / baseline_time_per_sample_ms:.1f}x)")

    print(f"\n📈 IMPROVEMENT/DEGRADATION RATES:")
    print(f"  TTA Improved Images: {tta_results['improvements']} ({tta_results['improvement_rate']:.1%})")
    print(f"  TTA Degraded Images: {tta_results['degradations']} ({tta_results['degradation_rate']:.1%})")
    print(f"  RL Improved Images: {rl_results['improvements']} ({rl_results['improvement_rate']:.1%})")
    print(f"  RL Degraded Images: {rl_results['degradations']} ({rl_results['degradation_rate']:.1%})")

    print(f"\n📊 RL AGENT SPECIFIC METRICS:")
    print(f"  Average Episode Reward: {rl_results['avg_reward']:.3f}")
    print(f"  Average Actions Sequence Length: {rl_results['avg_sequence_length']:.1f}")
    print(f"  Action Frequencies: {rl_results['action_counts']}")
    
    print("\n" + "=" * 80)
    print("Generating comprehensive comparison plots...")
    
    create_comprehensive_comparison_plots(baseline_results, tta_results, rl_results, rl_loaded)
    
    print("\nComparison complete. Check the 'comparison_plots' directory for visualizations.")
    print("=" * 80)


if __name__ == '__main__':
    main_comparison()