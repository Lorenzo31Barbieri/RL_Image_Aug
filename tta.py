import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from collections import defaultdict
import os

# Import ttach for TTA (install with: pip install ttach)
try:
    import ttach as tta
    TTA_AVAILABLE = True
    print("✅ ttach library found - TTA functionality enabled")
except ImportError:
    TTA_AVAILABLE = False
    print("❌ ttach library not found. Install with: pip install ttach")
    print("Falling back to manual TTA implementation...")

from vgg import VGG

# --- GLOBAL CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Configuration ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth'
IMAGE_SIZE = 32
NUM_CLASSES = 10


class TTAWrapper(nn.Module):
    """
    Wrapper for applying TTA to any PyTorch model.
    """
    def __init__(self, model, tta_transforms):
        super().__init__()
        self.model = model
        self.tta_transforms = tta_transforms
    
    def forward(self, x):
        predictions = []
        
        for transform in self.tta_transforms:
            # Apply transform
            transformed_x = transform(x)
            
            # Get prediction
            with torch.no_grad():
                pred = self.model(transformed_x)
                predictions.append(pred)
        
        # Average predictions
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction


def create_manual_tta_transforms():
    """
    Create manual TTA transforms similar to what ttach would do.
    """
    transforms_list = []
    
    # Original image
    transforms_list.append(lambda x: x)
    
    # Horizontal flip
    transforms_list.append(lambda x: torch.flip(x, dims=[3]))
    
    # Brightness adjustments
    transforms_list.append(lambda x: torch.clamp(x * 1.1, 0, 1))
    transforms_list.append(lambda x: torch.clamp(x * 0.9, 0, 1))
    
    # Contrast adjustments (approximate)
    transforms_list.append(lambda x: torch.clamp((x - 0.5) * 1.1 + 0.5, 0, 1))
    transforms_list.append(lambda x: torch.clamp((x - 0.5) * 0.9 + 0.5, 0, 1))
    
    # Small rotations (approximate using transforms)
    def rotate_90(x):
        return torch.rot90(x, k=1, dims=[2, 3])
    
    def rotate_180(x):
        return torch.rot90(x, k=2, dims=[2, 3])
    
    def rotate_270(x):
        return torch.rot90(x, k=3, dims=[2, 3])
    
    # Note: For CIFAR-10, 90/180/270 degree rotations might be too aggressive
    # but we'll include them for comprehensive TTA
    # transforms_list.extend([rotate_90, rotate_180, rotate_270])
    
    return transforms_list


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


def evaluate_baseline_classifier(model, test_loader):
    """Evaluate baseline classifier without any augmentation."""
    print("\n=== BASELINE EVALUATION (No Augmentation) ===")
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Baseline evaluation"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    baseline_time = time.time() - start_time
    accuracy = correct / total
    avg_confidence = np.mean(all_confidences)
    
    print(f"Baseline Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Inference Time: {baseline_time:.2f} seconds")
    print(f"Time per sample: {baseline_time/total*1000:.2f} ms")
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'inference_time': baseline_time,
        'time_per_sample': baseline_time/total
    }


def evaluate_tta_with_ttach(model, test_loader):
    """Evaluate using ttach library for TTA."""
    print("\n=== TTA EVALUATION (using ttach library) ===")
    
    # Create TTA transforms
    tta_transforms = tta.Compose([
        tta.HorizontalFlip(),
        tta.Multiply(factors=[0.9, 1.0, 1.1]),  # Brightness
        # tta.Rotate90(angles=[0, 90, 180, 270]),  # Too aggressive for CIFAR-10
        # tta.Scale(scales=[0.9, 1.0, 1.1]),  # Scale changes
        # tta.Add(values=[-0.1, 0, 0.1]),  # Additive changes
    ])
    
    # Wrap model with TTA
    tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="TTA evaluation (ttach)"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # TTA prediction
            outputs = tta_model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    tta_time = time.time() - start_time
    accuracy = correct / total
    avg_confidence = np.mean(all_confidences)
    
    num_transforms = len(tta_transforms.aug_transforms)
    
    print(f"TTA Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Number of augmentations: {num_transforms}")
    print(f"Inference Time: {tta_time:.2f} seconds")
    print(f"Time per sample: {tta_time/total*1000:.2f} ms")
    print(f"Slowdown factor: {tta_time/test_loader.dataset.__len__()*1000:.1f}x")
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'inference_time': tta_time,
        'time_per_sample': tta_time/total,
        'num_augmentations': num_transforms
    }


def evaluate_manual_tta(model, test_loader):
    """Evaluate using manual TTA implementation."""
    print("\n=== TTA EVALUATION (Manual Implementation) ===")
    
    tta_transforms = create_manual_tta_transforms()
    tta_model = TTAWrapper(model, tta_transforms)
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    correct = 0
    total = 0
    
    start_time = time.time()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="TTA evaluation (manual)"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # TTA prediction
            outputs = tta_model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences, predicted = torch.max(probabilities, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
    
    tta_time = time.time() - start_time
    accuracy = correct / total
    avg_confidence = np.mean(all_confidences)
    
    print(f"Manual TTA Accuracy: {accuracy:.4f}")
    print(f"Average Confidence: {avg_confidence:.4f}")
    print(f"Number of augmentations: {len(tta_transforms)}")
    print(f"Inference Time: {tta_time:.2f} seconds")
    print(f"Time per sample: {tta_time/total*1000:.2f} ms")
    
    return {
        'accuracy': accuracy,
        'avg_confidence': avg_confidence,
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'inference_time': tta_time,
        'time_per_sample': tta_time/total,
        'num_augmentations': len(tta_transforms)
    }


def evaluate_single_image_tta_analysis(model, test_dataset, num_samples=100):
    """
    Detailed analysis of TTA on individual images to understand improvement patterns.
    """
    print(f"\n=== SINGLE IMAGE TTA ANALYSIS ({num_samples} samples) ===")
    
    if TTA_AVAILABLE:
        tta_transforms = tta.Compose([
            tta.HorizontalFlip(),
            tta.Multiply(factors=[0.9, 1.0, 1.1]),
        ])
        tta_model = tta.ClassificationTTAWrapper(model, tta_transforms)
    else:
        tta_transforms = create_manual_tta_transforms()
        tta_model = TTAWrapper(model, tta_transforms)
    
    improvements = []
    degradations = []
    confidence_changes = []
    
    # Randomly sample images
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)
    
    for idx in tqdm(indices, desc="Analyzing individual images"):
        image, label = test_dataset[idx]
        image = image.unsqueeze(0).to(DEVICE)
        label_tensor = torch.tensor([label]).to(DEVICE)
        
        with torch.no_grad():
            # Baseline prediction
            baseline_output = model(image)
            baseline_prob = torch.nn.functional.softmax(baseline_output, dim=1)
            baseline_confidence, baseline_pred = torch.max(baseline_prob, 1)
            baseline_correct = (baseline_pred == label_tensor).item()
            
            # TTA prediction
            tta_output = tta_model(image)
            tta_prob = torch.nn.functional.softmax(tta_output, dim=1)
            tta_confidence, tta_pred = torch.max(tta_prob, 1)
            tta_correct = (tta_pred == label_tensor).item()
            
            # Track changes
            confidence_change = tta_confidence.item() - baseline_confidence.item()
            confidence_changes.append(confidence_change)
            
            if not baseline_correct and tta_correct:
                improvements.append({
                    'label': label,
                    'baseline_pred': baseline_pred.item(),
                    'tta_pred': tta_pred.item(),
                    'baseline_conf': baseline_confidence.item(),
                    'tta_conf': tta_confidence.item(),
                    'conf_change': confidence_change
                })
            elif baseline_correct and not tta_correct:
                degradations.append({
                    'label': label,
                    'baseline_pred': baseline_pred.item(),
                    'tta_pred': tta_pred.item(),
                    'baseline_conf': baseline_confidence.item(),
                    'tta_conf': tta_confidence.item(),
                    'conf_change': confidence_change
                })
    
    print(f"Improvements: {len(improvements)} ({len(improvements)/num_samples:.1%})")
    print(f"Degradations: {len(degradations)} ({len(degradations)/num_samples:.1%})")
    print(f"Average confidence change: {np.mean(confidence_changes):+.4f}")
    
    if improvements:
        avg_improvement_conf = np.mean([imp['conf_change'] for imp in improvements])
        print(f"Average confidence gain in improvements: {avg_improvement_conf:+.4f}")
    
    if degradations:
        avg_degradation_conf = np.mean([deg['conf_change'] for deg in degradations])
        print(f"Average confidence change in degradations: {avg_degradation_conf:+.4f}")
    
    return {
        'improvements': improvements,
        'degradations': degradations,
        'confidence_changes': confidence_changes,
        'improvement_rate': len(improvements)/num_samples,
        'degradation_rate': len(degradations)/num_samples
    }


def create_tta_comparison_plots(baseline_results, tta_results, analysis_results=None):
    """Create comprehensive comparison plots."""
    
    if not os.path.exists('./tta_plots'):
        os.makedirs('./tta_plots')
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    methods = ['Baseline', 'TTA']
    accuracies = [baseline_results['accuracy'], tta_results['accuracy']]
    colors = ['skyblue', 'lightgreen']
    
    bars = ax1.bar(methods, accuracies, color=colors, edgecolor='black', alpha=0.8)
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Accuracy Comparison: Baseline vs TTA')
    ax1.set_ylim(0, 1)
    
    # Add improvement annotation
    improvement = tta_results['accuracy'] - baseline_results['accuracy']
    ax1.annotate(f'Improvement: {improvement:+.4f}', 
                xy=(1, tta_results['accuracy']), xytext=(1, tta_results['accuracy'] + 0.05),
                ha='center', fontsize=10, 
                arrowprops=dict(arrowstyle='->', color='red'))
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Confidence Comparison
    ax2 = axes[0, 1]
    confidences = [baseline_results['avg_confidence'], tta_results['avg_confidence']]
    bars2 = ax2.bar(methods, confidences, color=colors, edgecolor='black', alpha=0.8)
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Confidence Comparison: Baseline vs TTA')
    ax2.set_ylim(0, 1)
    
    conf_improvement = tta_results['avg_confidence'] - baseline_results['avg_confidence']
    ax2.annotate(f'Change: {conf_improvement:+.4f}', 
                xy=(1, tta_results['avg_confidence']), xytext=(1, tta_results['avg_confidence'] + 0.05),
                ha='center', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    for bar, conf in zip(bars2, confidences):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{conf:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Inference Time Comparison
    ax3 = axes[0, 2]
    times = [baseline_results['time_per_sample']*1000, tta_results['time_per_sample']*1000]
    bars3 = ax3.bar(methods, times, color=['lightcoral', 'gold'], edgecolor='black', alpha=0.8)
    ax3.set_ylabel('Time per Sample (ms)')
    ax3.set_title('Inference Time Comparison')
    
    slowdown = tta_results['time_per_sample'] / baseline_results['time_per_sample']
    ax3.annotate(f'Slowdown: {slowdown:.1f}x', 
                xy=(1, times[1]), xytext=(1, times[1] + max(times)*0.1),
                ha='center', fontsize=10,
                arrowprops=dict(arrowstyle='->', color='red'))
    
    for bar, time_val in zip(bars3, times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                f'{time_val:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Per-class accuracy improvement (if we had this data)
    ax4 = axes[1, 0]
    if analysis_results:
        conf_changes = analysis_results['confidence_changes']
        ax4.hist(conf_changes, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(0, color='red', linestyle='--', label='No change')
        ax4.axvline(np.mean(conf_changes), color='green', linestyle='-', 
                   label=f'Mean: {np.mean(conf_changes):+.3f}')
        ax4.set_xlabel('Confidence Change')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Confidence Changes')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Individual image\nanalysis not available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Confidence Change Distribution')
    
    # Plot 5: Efficiency Analysis
    ax5 = axes[1, 1]
    
    # Efficiency metrics
    accuracy_gain = tta_results['accuracy'] - baseline_results['accuracy']
    time_cost = slowdown - 1  # Additional time cost as ratio
    
    # Efficiency score: accuracy gain per unit time cost
    if time_cost > 0:
        efficiency = accuracy_gain / time_cost
    else:
        efficiency = float('inf') if accuracy_gain > 0 else 0
    
    metrics = ['Accuracy\nGain', 'Time Cost\n(ratio)', 'Efficiency\nScore']
    values = [accuracy_gain, time_cost, efficiency]
    colors_eff = ['green' if v > 0 else 'red' for v in values]
    
    bars5 = ax5.bar(metrics, values, color=colors_eff, alpha=0.7, edgecolor='black')
    ax5.set_ylabel('Value')
    ax5.set_title('TTA Efficiency Analysis')
    ax5.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    for bar, val in zip(bars5, values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if val >= 0 else val - 0.01,
                f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
    
    # Plot 6: Summary Statistics
    ax6 = axes[1, 2]
    
    summary_text = f"""🎯 TTA EVALUATION SUMMARY
    
📊 Performance Metrics:
• Accuracy Improvement: {improvement:+.4f}
• Confidence Change: {conf_improvement:+.4f}
• Success: {'✅' if improvement > 0 else '❌'}

⚡ Efficiency Metrics:
• Time Slowdown: {slowdown:.1f}x
• Augmentations Used: {tta_results.get('num_augmentations', 'N/A')}
• Time per Sample: {tta_results['time_per_sample']*1000:.1f}ms

🔍 Analysis:
• Method: {'ttach library' if TTA_AVAILABLE else 'Manual TTA'}
• Cost-Effectiveness: {'High' if efficiency > 0.001 else 'Medium' if efficiency > 0 else 'Low'}
• Recommendation: {'Use TTA' if improvement > 0.01 else 'Limited benefit'}

💡 TTA vs RL:
TTA is {'faster' if slowdown < 10 else 'slower'} but {'less flexible' if improvement < 0.02 else 'competitive'}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    ax6.set_xlim(0, 1)
    ax6.set_ylim(0, 1)
    ax6.axis('off')
    ax6.set_title('TTA Summary & RL Comparison')
    
    plt.tight_layout()
    plt.savefig('./tta_plots/tta_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("TTA analysis plots saved to './tta_plots/tta_comprehensive_analysis.png'")


def main_tta_evaluation():
    """Main TTA evaluation function."""
    print("="*60)
    print("TEST TIME AUGMENTATION (TTA) COMPREHENSIVE EVALUATION")
    print("="*60)
    
    # Load classifier
    classifier_model = load_classifier_model()
    
    # Prepare test dataset
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=True, transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Test dataset size: {len(test_dataset)} images")
    print(f"Batch size: {test_loader.batch_size}")
    
    # 1. Baseline evaluation
    baseline_results = evaluate_baseline_classifier(classifier_model, test_loader)
    
    # 2. TTA evaluation
    if TTA_AVAILABLE:
        tta_results = evaluate_tta_with_ttach(classifier_model, test_loader)
    else:
        tta_results = evaluate_manual_tta(classifier_model, test_loader)
    
    # 3. Individual image analysis
    analysis_results = evaluate_single_image_tta_analysis(classifier_model, test_dataset, num_samples=200)
    
    # Print comprehensive comparison
    print("\n" + "="*60)
    print("COMPREHENSIVE TTA ANALYSIS")
    print("="*60)
    
    improvement = tta_results['accuracy'] - baseline_results['accuracy']
    conf_improvement = tta_results['avg_confidence'] - baseline_results['avg_confidence']
    slowdown = tta_results['time_per_sample'] / baseline_results['time_per_sample']
    
    print(f"\n📊 ACCURACY ANALYSIS:")
    print(f"  Baseline Accuracy: {baseline_results['accuracy']:.4f}")
    print(f"  TTA Accuracy: {tta_results['accuracy']:.4f}")
    print(f"  Improvement: {improvement:+.4f} ({improvement/baseline_results['accuracy']*100:+.2f}%)")
    
    print(f"\n🔍 CONFIDENCE ANALYSIS:")
    print(f"  Baseline Confidence: {baseline_results['avg_confidence']:.4f}")
    print(f"  TTA Confidence: {tta_results['avg_confidence']:.4f}")
    print(f"  Change: {conf_improvement:+.4f}")
    
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    print(f"  Baseline Time/Sample: {baseline_results['time_per_sample']*1000:.2f} ms")
    print(f"  TTA Time/Sample: {tta_results['time_per_sample']*1000:.2f} ms")
    print(f"  Slowdown Factor: {slowdown:.1f}x")
    print(f"  Augmentations Used: {tta_results.get('num_augmentations', 'N/A')}")
    
    print(f"\n🎯 INDIVIDUAL IMAGE ANALYSIS:")
    print(f"  Images Improved: {len(analysis_results['improvements'])} ({analysis_results['improvement_rate']:.1%})")
    print(f"  Images Degraded: {len(analysis_results['degradations'])} ({analysis_results['degradation_rate']:.1%})")
    print(f"  Net Improvement Rate: {analysis_results['improvement_rate'] - analysis_results['degradation_rate']:.1%}")
    
    # Efficiency calculation
    if slowdown > 1:
        efficiency = improvement / (slowdown - 1)
        print(f"\n💡 EFFICIENCY SCORE: {efficiency:.6f} (accuracy gain per unit time cost)")
    
    print(f"\n🏆 OVERALL ASSESSMENT:")
    if improvement > 0.01:
        print("  ✅ SUCCESS: TTA provides meaningful accuracy improvement!")
        recommendation = "Recommended for production use"
    elif improvement > 0.005:
        print("  ⚠️  MODERATE SUCCESS: TTA provides some improvement")
        recommendation = "Consider for scenarios where accuracy is critical"
    elif improvement > 0:
        print("  📊 MINIMAL SUCCESS: Small improvement observed")
        recommendation = "Limited practical benefit given computational cost"
    else:
        print("  ❌ NO IMPROVEMENT: TTA does not help for this model/dataset")
        recommendation = "Not recommended - may indicate overfitting or inappropriate augmentations"
    
    print(f"  Recommendation: {recommendation}")
    
    # Create visualizations
    create_tta_comparison_plots(baseline_results, tta_results, analysis_results)
    
    # Save results for comparison with RL
    results_summary = {
        'baseline': baseline_results,
        'tta': tta_results,
        'analysis': analysis_results,
        'improvement': improvement,
        'slowdown': slowdown,
        'efficiency': efficiency if 'efficiency' in locals() else 0,
        'recommendation': recommendation
    }
    
    # Save to file for later comparison
    import pickle
    with open('./tta_results.pkl', 'wb') as f:
        pickle.dump(results_summary, f)
    print(f"\n💾 Results saved to './tta_results.pkl' for comparison with RL model")
    
    return results_summary


if __name__ == '__main__':
    if not TTA_AVAILABLE:
        print("\n" + "="*60)
        print("INSTALLING TTACH LIBRARY")
        print("="*60)
        print("To get the full TTA functionality, install ttach:")
        print("pip install ttach")
        print("Continuing with manual TTA implementation...")
        print("="*60 + "\n")
    
    results = main_tta_evaluation()
    
    print(f"\n🎉 TTA evaluation completed!")
    print(f"Check './tta_plots/' for detailed visualizations.")
    print(f"Use the saved results in './tta_results.pkl' to compare with your RL model.")