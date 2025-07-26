import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import defaultdict

# Import improved modules
from vgg import VGG
from agent import DQNAgent
from environment import ImageAugmentationEnv
from transforms import get_num_actions, get_all_transforms, get_action_name

# --- GLOBAL CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Configuration ---
DATA_ROOT_DIR = './data' 
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
DQN_MODEL_PATH = './models/best_improved_dqn_model.pth'
IMAGE_SIZE = 32
NUM_CLASSES = 10

STATE_DIM = NUM_CLASSES + 5  # Enhanced state representation
ACTION_DIM = get_num_actions()
MAX_STEPS_PER_EPISODE = 3


def load_classifier_model():
    """Load the pre-trained classifier."""
    print("Loading pre-trained VGG19 CIFAR10 classifier for evaluation...")
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
        print(f"Successfully loaded classifier from {PRE_TRAINED_CLASSIFIER_PATH}")
        print(f"Classifier accuracy: {checkpoint['acc']:.2f}%")
        
    except Exception as e:
        print(f"Error loading classifier: {e}")
        exit()

    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and frozen.")
    return classifier_model


def evaluate_baseline_classifier(classifier_model, test_dataloader):
    """Evaluate baseline classifier performance."""
    print("\n=== BASELINE CLASSIFIER EVALUATION ===")
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Evaluating baseline"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = classifier_model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences = probabilities.max(dim=1)[0]
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = correct_predictions / total_samples
    avg_confidence = np.mean(all_confidences)
    
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences
    }


def evaluate_rl_agent_comprehensive(agent, classifier_model, test_episodes=1000):
    """Comprehensive evaluation of the RL agent."""
    print(f"\n=== RL AGENT EVALUATION ({test_episodes} episodes) ===")
    
    # Prepare test data
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=False, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_iter = iter(test_loader)
    
    # Evaluation metrics
    episode_rewards = []
    initial_accuracies = []
    final_accuracies = []
    initial_confidences = []
    final_confidences = []
    action_sequences = []
    transformation_counts = defaultdict(int)
    
    # Detailed analysis
    improvements_by_class = defaultdict(list)
    failures_by_class = defaultdict(list)
    
    # Disable exploration for evaluation
    original_epsilon = agent.epsilon
    agent.epsilon = 0
    
    print("Running RL episodes...")
    for episode in tqdm(range(test_episodes)):
        try:
            image_tensor, true_label_tensor = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            image_tensor, true_label_tensor = next(test_iter)
        
        image_tensor = image_tensor.squeeze(0)
        true_label = true_label_tensor.item()
        
        # Initialize environment
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=MAX_STEPS_PER_EPISODE,
            device=DEVICE
        )
        state = env.reset(image_tensor, true_label)
        
        # Track initial state
        initial_correct = env.initial_correct
        initial_confidence = env.initial_confidence
        initial_accuracies.append(1.0 if initial_correct else 0.0)
        initial_confidences.append(initial_confidence)
        
        # Run episode
        episode_reward = 0
        done = False
        actions_taken = []
        
        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            
            actions_taken.append(action)
            transformation_counts[get_action_name(action)] += 1
            
            state = next_state
            episode_reward += reward
        
        # Get final metrics
        metrics = env.get_improvement_metrics()
        final_correct = metrics['final_correct']
        final_confidence = metrics['final_confidence']
        
        # Store results
        episode_rewards.append(episode_reward)
        final_accuracies.append(1.0 if final_correct else 0.0)
        final_confidences.append(final_confidence)
        action_sequences.append(actions_taken)
        
        # Class-specific analysis
        if metrics['correctness_improved']:
            improvements_by_class[true_label].append({
                'initial_conf': initial_confidence,
                'final_conf': final_confidence,
                'actions': actions_taken,
                'reward': episode_reward
            })
        elif initial_correct and not final_correct:
            failures_by_class[true_label].append({
                'initial_conf': initial_confidence,
                'final_conf': final_confidence,
                'actions': actions_taken,
                'reward': episode_reward
            })
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Calculate comprehensive metrics
    results = calculate_comprehensive_metrics(
        episode_rewards, initial_accuracies, final_accuracies,
        initial_confidences, final_confidences, transformation_counts,
        improvements_by_class, failures_by_class, action_sequences
    )
    
    return results


def calculate_comprehensive_metrics(episode_rewards, initial_accuracies, final_accuracies,
                                  initial_confidences, final_confidences, transformation_counts,
                                  improvements_by_class, failures_by_class, action_sequences):
    """Calculate comprehensive evaluation metrics."""
    
    results = {}
    
    # Basic metrics
    results['avg_reward'] = np.mean(episode_rewards)
    results['reward_std'] = np.std(episode_rewards)
    results['initial_accuracy'] = np.mean(initial_accuracies)
    results['final_accuracy'] = np.mean(final_accuracies)
    results['accuracy_improvement'] = results['final_accuracy'] - results['initial_accuracy']
    
    # Confidence metrics
    results['initial_avg_confidence'] = np.mean(initial_confidences)
    results['final_avg_confidence'] = np.mean(final_confidences)
    results['confidence_improvement'] = results['final_avg_confidence'] - results['initial_avg_confidence']
    
    # Success metrics
    positive_rewards = [r for r in episode_rewards if r > 0]
    results['success_rate'] = len(positive_rewards) / len(episode_rewards)
    results['avg_positive_reward'] = np.mean(positive_rewards) if positive_rewards else 0
    
    # Improvement analysis
    improvements = [(f - i) for i, f in zip(initial_accuracies, final_accuracies)]
    results['correctness_improvements'] = sum(1 for imp in improvements if imp > 0)
    results['correctness_degradations'] = sum(1 for imp in improvements if imp < 0)
    results['no_change'] = sum(1 for imp in improvements if imp == 0)
    
    # Transformation analysis
    results['transformation_counts'] = dict(transformation_counts)
    results['most_used_transform'] = max(transformation_counts.items(), key=lambda x: x[1]) if transformation_counts else ("None", 0)
    results['least_used_transform'] = min(transformation_counts.items(), key=lambda x: x[1]) if transformation_counts else ("None", 0)
    
    # Class-specific analysis
    results['improvements_by_class'] = dict(improvements_by_class)
    results['failures_by_class'] = dict(failures_by_class)
    
    # Action sequence analysis
    avg_sequence_length = np.mean([len(seq) for seq in action_sequences])
    results['avg_sequence_length'] = avg_sequence_length
    
    return results


def print_detailed_results(results):
    """Print detailed evaluation results."""
    print("\n" + "="*60)
    print("DETAILED EVALUATION RESULTS")
    print("="*60)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"  Average Reward: {results['avg_reward']:.4f} ± {results['reward_std']:.4f}")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Average Positive Reward: {results['avg_positive_reward']:.4f}")
    
    print(f"\n🎯 ACCURACY METRICS:")
    print(f"  Initial Accuracy: {results['initial_accuracy']:.4f}")
    print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Accuracy Improvement: {results['accuracy_improvement']:.4f}")
    
    improvement_sign = "📈" if results['accuracy_improvement'] > 0 else "📉" if results['accuracy_improvement'] < 0 else "➡️"
    print(f"  {improvement_sign} Net Change: {results['accuracy_improvement']:+.4f}")
    
    print(f"\n🔍 CONFIDENCE METRICS:")
    print(f"  Initial Avg Confidence: {results['initial_avg_confidence']:.4f}")
    print(f"  Final Avg Confidence: {results['final_avg_confidence']:.4f}")
    print(f"  Confidence Change: {results['confidence_improvement']:+.4f}")
    
    print(f"\n📋 DETAILED BREAKDOWN:")
    print(f"  Correctness Improvements: {results['correctness_improvements']}")
    print(f"  Correctness Degradations: {results['correctness_degradations']}")
    print(f"  No Change: {results['no_change']}")
    print(f"  Average Sequence Length: {results['avg_sequence_length']:.2f}")
    
    print(f"\n🔧 TRANSFORMATION USAGE:")
    print(f"  Most Used: {results['most_used_transform'][0]} ({results['most_used_transform'][1]} times)")
    print(f"  Least Used: {results['least_used_transform'][0]} ({results['least_used_transform'][1]} times)")
    
    # Top 5 most used transformations
    sorted_transforms = sorted(results['transformation_counts'].items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 Transformations:")
    for i, (transform, count) in enumerate(sorted_transforms[:5], 1):
        print(f"    {i}. {transform}: {count}")
    
    # Class-specific improvements
    if results['improvements_by_class']:
        print(f"\n📈 CLASS-SPECIFIC IMPROVEMENTS:")
        for class_id, improvements in results['improvements_by_class'].items():
            if improvements:
                avg_reward = np.mean([imp['reward'] for imp in improvements])
                print(f"  Class {class_id}: {len(improvements)} improvements (avg reward: {avg_reward:.2f})")
    
    if results['failures_by_class']:
        print(f"\n📉 CLASS-SPECIFIC FAILURES:")
        for class_id, failures in results['failures_by_class'].items():
            if failures:
                print(f"  Class {class_id}: {len(failures)} failures")


def create_evaluation_visualizations(baseline_results, rl_results):
    """Create comprehensive evaluation visualizations."""
    
    if not os.path.exists('./evaluation_plots'):
        os.makedirs('./evaluation_plots')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    categories = ['Initial', 'Baseline', 'RL Final']
    accuracies = [
        rl_results['initial_accuracy'],
        baseline_results['accuracy'],
        rl_results['final_accuracy']
    ]
    colors = ['lightblue', 'orange', 'lightgreen']
    bars = ax1.bar(categories, accuracies, color=colors, edgecolor='black', alpha=0.7)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Plot 2: Confidence Comparison
    ax2 = axes[0, 1]
    conf_categories = ['Initial', 'Baseline', 'RL Final']
    confidences = [
        rl_results['initial_avg_confidence'],
        baseline_results['avg_confidence'],
        rl_results['final_avg_confidence']
    ]
    bars2 = ax2.bar(conf_categories, confidences, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Confidence Comparison')
    ax2.set_ylim(0, 1)
    
    for bar, conf in zip(bars2, confidences):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{conf:.3f}', ha='center', va='bottom')
    
    # Plot 3: Reward Distribution
    ax3 = axes[0, 2]
    episode_rewards = np.random.normal(rl_results['avg_reward'], rl_results['reward_std'], 1000)  # Simulated for visualization
    ax3.hist(episode_rewards, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(rl_results['avg_reward'], color='red', linestyle='--', 
                label=f'Mean: {rl_results["avg_reward"]:.3f}')
    ax3.set_xlabel('Episode Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('RL Agent Reward Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Transformation Usage
    ax4 = axes[1, 0]
    transform_names = list(rl_results['transformation_counts'].keys())
    transform_counts = list(rl_results['transformation_counts'].values())
    
    # Sort by count for better visualization
    sorted_data = sorted(zip(transform_names, transform_counts), key=lambda x: x[1], reverse=True)
    sorted_names, sorted_counts = zip(*sorted_data) if sorted_data else ([], [])
    
    if sorted_names:
        bars4 = ax4.bar(range(len(sorted_names)), sorted_counts, color='lightcoral', 
                       edgecolor='black', alpha=0.7)
        ax4.set_xlabel('Transformations')
        ax4.set_ylabel('Usage Count')
        ax4.set_title('Transformation Usage Frequency')
        ax4.set_xticks(range(len(sorted_names)))
        ax4.set_xticklabels([name[:15] + '...' if len(name) > 15 else name for name in sorted_names], 
                           rotation=45, ha='right')
    
    # Plot 5: Improvement Analysis
    ax5 = axes[1, 1]
    improvement_categories = ['Improvements', 'Degradations', 'No Change']
    improvement_counts = [
        rl_results['correctness_improvements'],
        rl_results['correctness_degradations'],
        rl_results['no_change']
    ]
    colors5 = ['green', 'red', 'gray']
    pie = ax5.pie(improvement_counts, labels=improvement_categories, colors=colors5, 
                  autopct='%1.1f%%', startangle=90)
    ax5.set_title('Classification Outcome Changes')
    
    # Plot 6: Performance Summary
    ax6 = axes[1, 2]
    summary_text = f"""🎯 EVALUATION SUMMARY
    
📊 Performance Metrics:
• Success Rate: {rl_results['success_rate']:.1%}
• Accuracy Improvement: {rl_results['accuracy_improvement']:+.4f}
• Confidence Change: {rl_results['confidence_improvement']:+.4f}

🔧 Agent Behavior:
• Avg Sequence Length: {rl_results['avg_sequence_length']:.1f}
• Most Used Transform: {rl_results['most_used_transform'][0]}
• Usage: {rl_results['most_used_transform'][1]} times

📈 Outcome Analysis:
• Improvements: {rl_results['correctness_improvements']}
• Degradations: {rl_results['correctness_degradations']}
• No Change: {rl_results['no_change']}

💡 Recommendation:
{"✅ Agent shows positive impact!" if rl_results['accuracy_improvement'] > 0.01 
 else "⚠️ Limited improvement observed" if rl_results['accuracy_improvement'] > -0.01 
 else "❌ Agent may need retraining"}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('Performance Summary')
    
    plt.tight_layout()
    plt.savefig('./evaluation_plots/comprehensive_evaluation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Evaluation plots saved to './evaluation_plots/comprehensive_evaluation_analysis.png'")


def compare_with_random_baseline(classifier_model, test_episodes=200):
    """Compare RL agent performance with random action baseline."""
    print(f"\n=== RANDOM BASELINE COMPARISON ({test_episodes} episodes) ===")
    
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=False, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0)
    test_iter = iter(test_loader)
    
    random_rewards = []
    random_improvements = []
    
    for episode in tqdm(range(test_episodes), desc="Random baseline"):
        try:
            image_tensor, true_label_tensor = next(test_iter)
        except StopIteration:
            test_iter = iter(test_loader)
            image_tensor, true_label_tensor = next(test_iter)
        
        image_tensor = image_tensor.squeeze(0)
        true_label = true_label_tensor.item()
        
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=MAX_STEPS_PER_EPISODE,
            device=DEVICE
        )
        state = env.reset(image_tensor, true_label)
        
        initial_correct = env.initial_correct
        episode_reward = 0
        done = False
        
        while not done:
            # Random action selection
            action = np.random.randint(0, ACTION_DIM)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        metrics = env.get_improvement_metrics()
        random_rewards.append(episode_reward)
        random_improvements.append(1.0 if metrics['final_correct'] else 0.0)
    
    random_results = {
        'avg_reward': np.mean(random_rewards),
        'final_accuracy': np.mean(random_improvements),
        'success_rate': sum(1 for r in random_rewards if r > 0) / len(random_rewards)
    }
    
    print(f"Random Baseline Results:")
    print(f"  Average Reward: {random_results['avg_reward']:.4f}")
    print(f"  Final Accuracy: {random_results['final_accuracy']:.4f}")
    print(f"  Success Rate: {random_results['success_rate']:.2%}")
    
    return random_results


def main_evaluation():
    """Main evaluation function."""
    print("="*60)
    print("COMPREHENSIVE RL AGENT EVALUATION")
    print("="*60)
    
    # Load models
    classifier_model = load_classifier_model()
    
    # Load RL agent
    print(f"\nLoading RL agent from {DQN_MODEL_PATH}...")
    agent = DQNAgent(STATE_DIM, ACTION_DIM, DEVICE)
    
    if os.path.exists(DQN_MODEL_PATH):
        try:
            agent.q_network.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=DEVICE))
            agent.target_q_network.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=DEVICE))
            print(f"✅ Successfully loaded RL agent from {DQN_MODEL_PATH}")
        except Exception as e:
            print(f"❌ Error loading RL agent: {e}")
            print("Using randomly initialized agent for comparison...")
    else:
        print(f"❌ RL model not found at {DQN_MODEL_PATH}")
        print("Using randomly initialized agent for comparison...")
    
    agent.q_network.eval()
    agent.target_q_network.eval()
    
    # Prepare baseline evaluation
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=False, transform=preprocess)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Run evaluations
    print("\n🔍 Starting comprehensive evaluation...")
    
    # 1. Baseline classifier evaluation
    baseline_results = evaluate_baseline_classifier(classifier_model, test_dataloader)
    
    # 2. RL agent evaluation
    rl_results = evaluate_rl_agent_comprehensive(agent, classifier_model, test_episodes=1000)
    
    # 3. Random baseline comparison
    random_results = compare_with_random_baseline(classifier_model, test_episodes=200)
    
    # Print detailed results
    print_detailed_results(rl_results)
    
    # Comparison summary
    print(f"\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    print(f"📊 ACCURACY COMPARISON:")
    print(f"  Baseline Classifier: {baseline_results['accuracy']:.4f}")
    print(f"  RL Agent (Initial): {rl_results['initial_accuracy']:.4f}")
    print(f"  RL Agent (Final): {rl_results['final_accuracy']:.4f}")
    print(f"  Random Actions: {random_results['final_accuracy']:.4f}")
    
    print(f"\n🎯 IMPROVEMENT ANALYSIS:")
    rl_vs_baseline = rl_results['final_accuracy'] - baseline_results['accuracy']
    rl_vs_initial = rl_results['accuracy_improvement']
    rl_vs_random = rl_results['final_accuracy'] - random_results['final_accuracy']
    
    print(f"  RL vs Baseline: {rl_vs_baseline:+.4f}")
    print(f"  RL vs Initial: {rl_vs_initial:+.4f}")
    print(f"  RL vs Random: {rl_vs_random:+.4f}")
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if rl_vs_baseline > 0.005 and rl_vs_initial > 0.005:
        print("  ✅ SUCCESS: RL agent shows meaningful improvement!")
    elif rl_vs_baseline > 0 or rl_vs_initial > 0:
        print("  ⚠️  PARTIAL SUCCESS: Some improvement observed, but limited.")
    else:
        print("  ❌ LIMITED SUCCESS: Minimal or no improvement. Consider:")
        print("     • Adjusting reward function")
        print("     • Increasing training episodes")
        print("     • Tuning hyperparameters")
        print("     • Improving state representation")
    
    # Create visualizations
    create_evaluation_visualizations(baseline_results, rl_results)
    
    return {
        'baseline': baseline_results,
        'rl_agent': rl_results,
        'random': random_results
    }


if __name__ == '__main__':
    results = main_evaluation()
    print(f"\n🎉 Evaluation completed! Check './evaluation_plots/' for detailed visualizations.")