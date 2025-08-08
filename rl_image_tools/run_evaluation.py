#!/usr/bin/env python3
"""
RL Image Evaluation Script
Runs RL agent on images and saves original + augmented versions.

Usage: python -m rl_image_tools.run_evaluation
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import configurations
from rl_image_tools.config import *

# Import evaluation modules
from evaluation.core.model_loader import load_classifier, load_rl_agent
from evaluation.core.data_utils import get_cifar10_test_dataset
from src.environment.environment import ImageAugmentationEnv
from src.environment.transforms import get_action_name


def tensor_to_pil(tensor):
    """Convert normalized tensor to PIL Image."""
    tensor = tensor.detach().cpu()
    
    # Denormalize CIFAR-10
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    tensor = (tensor * 255).byte()
    from torchvision import transforms
    image = transforms.ToPILImage()(tensor)
    return image


def run_rl_evaluation():
    """Main evaluation function using evaluation modules."""
    device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get the current script directory (rl_image_tools)
    script_dir = Path(__file__).parent
    
    # Create output directories within rl_image_tools
    output_dir = script_dir / OUTPUT_DIR
    original_dir = script_dir / ORIGINAL_IMAGES_DIR
    augmented_dir = script_dir / AUGMENTED_IMAGES_DIR
    
    original_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output will be saved in: {output_dir}")
    
    # Change to project root for model/data loading
    original_cwd = os.getcwd()
    os.chdir(project_root)
    
    try:
        # Load models using evaluation modules
        classifier = load_classifier(CLASSIFIER_PATH, device)
        agent, model_loaded = load_rl_agent(RL_MODEL_PATH, state_dim=STATE_DIM, device=device)
        print(f"RL model loaded: {'OK' if model_loaded else 'ERROR (using random)'}")
        
        # Load dataset using evaluation modules
        test_dataset = get_cifar10_test_dataset(data_root=DATA_ROOT)
        
        # CIFAR-10 class names
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Select random images
        indices = np.random.choice(len(test_dataset), NUM_IMAGES, replace=False)
        
        print(f"Processing {NUM_IMAGES} images...")
        
        improvements = 0
        total_reward = 0
        
        for i, idx in enumerate(tqdm(indices, desc="Processing images")):
            image, true_label = test_dataset[idx]
            class_name = class_names[true_label]
            
            # Save original image
            original_pil = tensor_to_pil(image.clone())
            original_filename = f"{i+1:03d}_{class_name}_{idx}.png"
            original_pil.save(original_dir / original_filename)
            
            # Initialize RL environment
            env = ImageAugmentationEnv(
                classifier=classifier,
                max_steps=MAX_STEPS_PER_EPISODE,
                device=device
            )
            
            # Run RL episode
            state = env.reset(image, true_label)
            initial_correct = env.initial_correct
            episode_reward = 0
            actions_taken = []
            
            done = False
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                actions_taken.append(get_action_name(action))
                episode_reward += reward
                state = next_state
            
            # Get final results
            metrics = env.get_improvement_metrics()
            final_correct = metrics['final_correct']
            
            # Save augmented image
            augmented_pil = tensor_to_pil(env.augmented_image_tensor.clone())
            
            # Create descriptive filename
            status = "improved" if not initial_correct and final_correct else "degraded" if initial_correct and not final_correct else "nochange"
            actions_str = "_".join(actions_taken[:2])  # First 2 actions
            if len(actions_taken) > 2:
                actions_str += "_etc"
            
            augmented_filename = f"{i+1:03d}_{class_name}_{status}_{actions_str}_{idx}.png"
            augmented_pil.save(augmented_dir / augmented_filename)
            
            # Track improvements
            if not initial_correct and final_correct:
                improvements += 1
            
            total_reward += episode_reward
        
        # Print summary
        print(f"\n EVALUATION SUMMARY:")
        print(f"  Images processed: {NUM_IMAGES}")
        print(f"  Improvements: {improvements} ({improvements/NUM_IMAGES:.1%})")
        print(f"  Original images saved to: {original_dir}")
        print(f"  Augmented images saved to: {augmented_dir}")
        
        return {
            'images_processed': NUM_IMAGES,
            'improvements': improvements,
            'avg_reward': total_reward/NUM_IMAGES
        }
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == '__main__':
    try:
        results = run_rl_evaluation()
        print("Evaluation completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()