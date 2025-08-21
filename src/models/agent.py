import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    """
    Enhanced QNetwork to handle larger state space with image features
    """

    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super(QNetwork, self).__init__()
        
        # Larger hidden dimensions to handle the increased state space
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc5 = nn.Linear(hidden_dim // 4, action_dim)
        
        self.dropout = nn.Dropout(0.2)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim // 2)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        x = F.relu(self.layer_norm1(self.fc1(state)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.layer_norm3(self.fc3(x)))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class DQNAgent:
    """
    Enhanced DQN Agent with support for larger state spaces
    """
    
    def __init__(self, state_dim, action_dim, device, gamma=0.95, lr=0.0001,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995,
                 buffer_size=100000, batch_size=128, target_update_freq=1000,
                 double_dqn=True, prioritized_replay=False):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_counter = 0
        self.double_dqn = double_dqn
        self.prioritized_replay = prioritized_replay

        # Enhanced networks for larger state space
        hidden_dim = max(512, state_dim * 2)  # Scale hidden dimension with state size
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_network = QNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        # Optimizer with lower learning rate for stability with larger networks
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=1e-5)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.9)
        
        # Experience replay with larger buffer for complex state space
        buffer_size = max(buffer_size, 100000)  # Ensure minimum buffer size
        if prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = deque(maxlen=buffer_size)
        
        # Action frequency tracking for exploration bonus
        self.action_counts = np.zeros(action_dim)
        self.total_actions = 0

    def select_action(self, state, training=True):
        """
        Enhanced action selection with exploration bonuses.
        """
        if training and random.random() < self.epsilon:
            # Epsilon-greedy with action frequency consideration
            if self.total_actions > 1000:  # After some experience
                # Favor less-used actions for better exploration
                action_probs = 1.0 / (self.action_counts + 1)
                action_probs = action_probs / action_probs.sum()
                return np.random.choice(self.action_dim, p=action_probs)
            else:
                return random.randrange(self.action_dim)
        else:
            # Greedy action selection
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            
            action = torch.argmax(q_values).item()
            
            # Track action frequency
            if training:
                self.action_counts[action] += 1
                self.total_actions += 1
            
            return action

    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        if self.prioritized_replay:
            # Calculate TD error for prioritization
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                next_state_tensor = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                
                current_q = self.q_network(state_tensor)[0, action]
                if done:
                    target_q = reward
                else:
                    next_q = self.target_q_network(next_state_tensor).max(1)[0]
                    target_q = reward + self.gamma * next_q
                
                td_error = abs(current_q - target_q).item()
            
            self.replay_buffer.add(state, action, reward, next_state, done, td_error)
        else:
            self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self):
        """
        Enhanced learning with Double DQN and improved loss calculation.
        """
        if (self.prioritized_replay and len(self.replay_buffer) < self.batch_size) or \
           (not self.prioritized_replay and len(self.replay_buffer) < self.batch_size):
            return None

        # Sample experiences
        if self.prioritized_replay:
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)
            weights = torch.from_numpy(weights).float().to(self.device)
        else:
            experiences = random.sample(self.replay_buffer, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*experiences)
            weights = torch.ones(self.batch_size).to(self.device)

        # Convert to tensors - handle variable state dimensions
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions)

        # Calculate target Q values
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use main network to select actions, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
                next_q_values = self.target_q_network(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute weighted loss
        td_errors = current_q_values - target_q_values
        loss = (weights * td_errors.pow(2)).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability with larger networks
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        
        self.optimizer.step()
        self.scheduler.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Update prioritized replay buffer
        if self.prioritized_replay:
            td_errors_np = td_errors.detach().cpu().numpy()
            for i, td_error in enumerate(td_errors_np):
                self.replay_buffer.update_priority(indices[i], abs(td_error[0]))

        # Decay epsilon
        # NOTE: Epsilon decay is now handled in the training script for enhanced control
        # self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def get_action_distribution(self):
        """Get the distribution of actions taken during training."""
        if self.total_actions == 0:
            return np.ones(self.action_dim) / self.action_dim
        return self.action_counts / self.total_actions


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    """
    
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, state, action, reward, next_state, done, td_error):
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        self.frame += 1
        
        return experiences, indices, weights
    
    def update_priority(self, idx, td_error):
        self.priorities[idx] = td_error + 1e-6
    
    def __len__(self):
        return len(self.buffer)