import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from .transforms import get_action_transform


class ImageAugmentationEnv:
    """
    Improved environment with better reward function and state representation.
    """

    def __init__(self, classifier, max_steps, device):
        self.classifier = classifier
        self.max_steps = max_steps
        self.device = device
        self.current_step = 0
        self.original_image_tensor = None
        self.augmented_image_tensor = None
        self.true_label = None
        
        # Track initial state for comparison
        self.initial_prediction = None
        self.initial_confidence = None
        self.initial_correct = False

        # Standard preprocessing for CIFAR10
        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.feature_extractor = self.classifier

    def reset(self, image_tensor, true_label):
        """
        Initialize a new RL episode with enhanced state representation.
        """
        self.original_image_tensor = image_tensor.to(self.device)
        self.augmented_image_tensor = self.original_image_tensor.clone()
        self.true_label = true_label
        self.current_step = 0

        # Get initial prediction and confidence
        with torch.no_grad():
            output = self.classifier(self.augmented_image_tensor.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            
            self.initial_prediction = torch.argmax(output).item()
            self.initial_confidence = probabilities.max().item()
            self.initial_correct = (self.initial_prediction == self.true_label)
            
            # Enhanced state representation
            state = self._create_enhanced_state(output, probabilities)

        return state

    def _create_enhanced_state(self, logits, probabilities):
        """
        Create enhanced state representation combining multiple information sources.
        """
        # Basic logits (10 dimensions)
        logits_norm = F.normalize(logits.squeeze(0), dim=0)
        
        # Confidence measures (3 dimensions)
        max_prob = probabilities.max().item()
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()
        margin = torch.topk(probabilities, 2)[0]
        confidence_margin = (margin[0, 0] - margin[0, 1]).item()
        
        # Prediction correctness indicator (1 dimension)
        is_correct = float(torch.argmax(logits).item() == self.true_label)
        
        # Step information (1 dimension)
        step_ratio = self.current_step / self.max_steps
        
        # Combine all features
        enhanced_state = torch.cat([
            logits_norm.cpu(),
            torch.tensor([max_prob, entropy, confidence_margin, is_correct, step_ratio])
        ])
        
        return enhanced_state.numpy()

    def step(self, action):
        """
        Execute action with improved reward function.
        """
        self.current_step += 1

        # Store previous state for comparison
        with torch.no_grad():
            prev_output = self.classifier(self.augmented_image_tensor.unsqueeze(0))
            prev_probabilities = F.softmax(prev_output, dim=1)
            prev_prediction = torch.argmax(prev_output).item()
            prev_confidence = prev_probabilities.max().item()
            prev_correct = (prev_prediction == self.true_label)

        # Apply transformation
        transform_func = get_action_transform(action)
        self.augmented_image_tensor = transform_func(self.augmented_image_tensor)

        # Get new state
        with torch.no_grad():
            output = self.classifier(self.augmented_image_tensor.unsqueeze(0))
            probabilities = F.softmax(output, dim=1)
            prediction = torch.argmax(output).item()
            confidence = probabilities.max().item()
            is_correct = (prediction == self.true_label)

        # Improved reward calculation
        reward = self._calculate_improved_reward(
            prev_correct, is_correct, prev_confidence, confidence, action
        )

        done = self.current_step >= self.max_steps
        next_state = self._create_enhanced_state(output, probabilities)

        info = {
            'prediction': prediction,
            'confidence': confidence,
            'true_label': self.true_label,
            'is_correct': is_correct,
            'confidence_change': confidence - prev_confidence,
            'action_taken': action
        }

        return next_state, reward, done, info

    def _calculate_improved_reward(self, prev_correct, curr_correct, prev_conf, curr_conf, action):
        """
        Calculate reward based on multiple factors:
        1. Correctness improvement
        2. Confidence changes
        3. Action type penalties
        """
        reward = 0.0
        
        # Primary reward: correctness improvement
        if not prev_correct and curr_correct:
            reward += 10.0  # Major positive reward for fixing incorrect prediction
        elif prev_correct and not curr_correct:
            reward -= 10.0  # Major penalty for breaking correct prediction
        elif prev_correct and curr_correct:
            # Both correct: reward confidence improvement
            conf_improvement = curr_conf - prev_conf
            reward += conf_improvement * 5.0
        else:
            # Both incorrect: small reward for confidence improvement toward correct class
            conf_improvement = curr_conf - prev_conf
            reward += conf_improvement * 2.0
        
        # Secondary reward: general confidence improvement (when correct)
        if curr_correct:
            conf_bonus = max(0, curr_conf - 0.9) * 2.0  # Bonus for high confidence
            reward += conf_bonus
        
        # Action-specific penalties to discourage overuse of aggressive transforms
        action_penalties = {
            4: -0.1,  # Rotation penalties
            5: -0.1,
            6: -0.2,  # Horizontal flip penalty (can be harmful for CIFAR-10)
        }
        reward += action_penalties.get(action, 0)
        
        # Small penalty for taking steps (encourage efficiency)
        reward -= 0.1
        
        return reward

    def get_improvement_metrics(self):
        """
        Get metrics comparing initial vs final state.
        """
        with torch.no_grad():
            final_output = self.classifier(self.augmented_image_tensor.unsqueeze(0))
            final_probabilities = F.softmax(final_output, dim=1)
            final_prediction = torch.argmax(final_output).item()
            final_confidence = final_probabilities.max().item()
            final_correct = (final_prediction == self.true_label)

        return {
            'initial_correct': self.initial_correct,
            'final_correct': final_correct,
            'initial_confidence': self.initial_confidence,
            'final_confidence': final_confidence,
            'correctness_improved': (not self.initial_correct) and final_correct,
            'confidence_improved': final_confidence > self.initial_confidence,
            'overall_improved': final_correct and (final_confidence > self.initial_confidence or not self.initial_correct)
        }