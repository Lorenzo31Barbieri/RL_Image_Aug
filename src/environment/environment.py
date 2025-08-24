import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from .transforms import get_action_transform


class ImageAugmentationEnv:
    """
    RL environment with 143-dimensional state representation.
    State components:
    - Logits (10D): Classifier output probabilities
    - Additional features (5D): confidence, entropy, margin, correctness, step_ratio  
    - Image features (128D): Features extracted from classifier's intermediate layers
    """

    def __init__(self, classifier, max_steps, device, image_feature_dim=128):
        self.classifier = classifier
        self.max_steps = max_steps
        self.device = device
        self.current_step = 0
        
        # Fixed state dimensions
        self.logits_dim = 10  # CIFAR-10 classes
        self.additional_features_dim = 5  # confidence, entropy, margin, correctness, step_ratio
        self.image_feature_dim = image_feature_dim
        self.state_dim = self.logits_dim + self.additional_features_dim + self.image_feature_dim  # 143
        
        # Image tensors
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

        # Setup feature extraction
        self.feature_extractor = self.classifier
        self._setup_feature_extraction()

    def _setup_feature_extraction(self):
        """Setup feature extraction from classifier's intermediate layers."""
        self.image_features = None
        
        def feature_hook(module, input, output):
            # Extract features from the layer before the final classifier
            if hasattr(output, 'shape') and len(output.shape) >= 2:
                # Global average pooling if spatial dimensions exist
                if len(output.shape) == 4:  # [batch, channels, height, width]
                    pooled = F.adaptive_avg_pool2d(output, (1, 1))
                    self.image_features = pooled.view(output.size(0), -1)
                else:  # Already flattened
                    self.image_features = output
        
        # Register hook on the appropriate layer
        hook_registered = False
        
        # For VGG, try to hook before the final linear layer
        if hasattr(self.classifier, 'classifier'):
            # Check if classifier is Sequential or single layer
            if hasattr(self.classifier.classifier, '__len__'):
                # Sequential module
                if len(self.classifier.classifier) > 0:
                    self.classifier.classifier[0].register_forward_hook(feature_hook)
                    hook_registered = True
            elif hasattr(self.classifier.classifier, 'register_forward_hook'):
                # Single layer
                self.classifier.classifier.register_forward_hook(feature_hook)
                hook_registered = True
        
        # If no hook registered, try alternative approaches
        if not hook_registered:
            if hasattr(self.classifier, 'fc'):
                # ResNet-style architectures
                self.classifier.fc.register_forward_hook(feature_hook)
                hook_registered = True
        
        # If still no hook, use fallback method
        if not hook_registered:
            self._use_penultimate_layer_features = True
            print("Warning: Using fallback feature extraction method")

    def _extract_image_features(self, image_tensor):
        """
        Extract 128-dimensional image features using the classifier's intermediate representations.
        
        Args:
            image_tensor: Input image tensor [1, 3, 32, 32]
            
        Returns:
            Feature vector of shape [128]
        """
        with torch.no_grad():
            if hasattr(self, '_use_penultimate_layer_features'):
                # Alternative approach: modify forward pass to get intermediate features
                features = self._get_penultimate_features(image_tensor)
            else:
                # Use the hook-based approach
                self.image_features = None  # Reset before forward pass
                _ = self.classifier(image_tensor)  # Forward pass triggers the hook
                features = self.image_features
                
            if features is None:
                # Final fallback: use output logits as features
                print("Warning: Using logits as image features (fallback)")
                logits = self.classifier(image_tensor)
                features = logits
            
            # Ensure features have the right shape and are detached
            if isinstance(features, torch.Tensor):
                features = features.detach()
            else:
                features = torch.tensor(features, device=image_tensor.device)
            
            # Handle different tensor shapes
            if features.dim() > 2:
                features = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
            elif features.dim() == 1:
                features = features.unsqueeze(0)
                
            # Reduce to target dimension if necessary
            if features.size(1) > self.image_feature_dim:
                # Use simple truncation (could also use PCA in the future)
                features = features[:, :self.image_feature_dim]
            elif features.size(1) < self.image_feature_dim:
                # Pad with zeros if features are smaller than target
                padding = torch.zeros(features.size(0), 
                                    self.image_feature_dim - features.size(1), 
                                    device=features.device)
                features = torch.cat([features, padding], dim=1)
            
            # Normalize features for stability
            features = F.normalize(features, p=2, dim=1)
            
            return features.squeeze(0)  # Remove batch dimension

    def _get_penultimate_features(self, image_tensor):
        """Alternative method to extract features from penultimate layer."""
        if hasattr(self.classifier, 'features') and hasattr(self.classifier, 'classifier'):
            # VGG-style architecture
            with torch.no_grad():
                x = self.classifier.features(image_tensor)
                x = x.view(x.size(0), -1)  # Flatten
                
                # Check if classifier is a Sequential module or single Linear layer
                if hasattr(self.classifier.classifier, '__len__'):
                    # It's a Sequential module with multiple layers
                    if len(self.classifier.classifier) > 1:
                        # Pass through all but the last layer
                        for i, layer in enumerate(self.classifier.classifier[:-1]):
                            x = layer(x)
                elif hasattr(self.classifier.classifier, 'weight'):
                    # It's a single Linear layer, x is already the input features we want
                    pass
                else:
                    # Try to iterate through the classifier
                    try:
                        layers = list(self.classifier.classifier.children())
                        if len(layers) > 1:
                            for layer in layers[:-1]:
                                x = layer(x)
                    except:
                        # If all fails, return x as is (features after CNN)
                        pass
                
                return x
        else:
            # For other architectures, return None to use logits
            return None

    def reset(self, image_tensor, true_label):
        """Initialize a new RL episode."""
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
            
            # Create 143-dimensional state
            state = self._create_state(output, probabilities)

        return state

    def _create_state(self, logits, probabilities):
        """
        Create 143-dimensional state representation.
        
        Returns:
            numpy array of shape [143] = [10 + 5 + 128]
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
        
        # Extract image features (128 dimensions)
        image_features = self._extract_image_features(self.augmented_image_tensor.unsqueeze(0))
        
        # Combine all features
        state = torch.cat([
            logits_norm.cpu(),
            torch.tensor([max_prob, entropy, confidence_margin, is_correct, step_ratio]),
            image_features.cpu()
        ])
        
        return state.numpy()

    def step(self, action):
        """Execute action and return next state."""
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

        # Calculate reward
        reward = self._calculate_reward(
            prev_correct, is_correct, prev_confidence, confidence, action
        )

        done = self.current_step >= self.max_steps
        
        # Create next state
        next_state = self._create_state(output, probabilities)

        info = {
            'prediction': prediction,
            'confidence': confidence,
            'true_label': self.true_label,
            'is_correct': is_correct,
            'confidence_change': confidence - prev_confidence,
            'action_taken': action,
            'state_dim': len(next_state)
        }

        return next_state, reward, done, info

    def _calculate_reward(self, prev_correct, curr_correct, prev_conf, curr_conf, action):
        """Calculate reward based on correctness and confidence improvements."""
        reward = 0.0
        
        # Reward più bilanciato
        if not prev_correct and curr_correct:
            reward += 5.0
        elif prev_correct and not curr_correct:
            reward -= 8.0
        elif prev_correct and curr_correct:
            # Bonus per confidence improvement
            conf_improvement = curr_conf - prev_conf
            reward += conf_improvement * 3.0
        else:
            # Reward anche quando sbagliato ma migliora confidence
            conf_improvement = curr_conf - prev_conf
            reward += conf_improvement * 1.5
        
        reward -= 0.05
        
        return reward

    def get_improvement_metrics(self):
        """Get metrics comparing initial vs final state."""
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
            'overall_improved': final_correct and (final_confidence > self.initial_confidence or not self.initial_correct),
            'state_dim': self.state_dim
        }