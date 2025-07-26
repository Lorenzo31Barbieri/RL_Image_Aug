import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque


class QNetwork(nn.Module):
    """
    Attributes:
        state_dim (int): number of classifier logits, set to 10.
        action_dim (int): number of actions.
    """

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        return self.fc2(x)

class DQNAgent:
    """
        The RL agent.

        Attributes:
            state_dim (int): number of classifier logits, set to 10.
            action_dim (int): number of actions.
            device (string): gpu device (cuda or mps).
            gamma (float): discount factor.
            lr (float): learning rate.
            epsilon_start(float): hyperparameter.
            epsilon_end (float): hyperparameter.
            epsilon_decay (float): hyperparameter.
            buffer_size (int): buffer dimension.
            batch_size(int): batch dimension.
            target_update_freq (int): how often to update the target network.
    """
    
    def __init__(self, state_dim, action_dim, device, gamma=0.99, lr=0.0005,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64, target_update_freq=100):
        
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

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.target_q_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state):
        """
        Given a state, returns the action with the highest Q-value (epsilon-greedy policy).
        """

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self):
        """
        The heart of the DQN algorithm.
        1. Extract a batch of experiences from the replay buffer.
        2. Calcutate Q-values from the Q network.
        3. Calculate Q-values from the Target network using Bellman equation.
        4. Calculate MSE between the two sets of Q-values.
        5. Backpropagation to update Q-network weights.
        6. Periodically, update the Target network with Q-network weights.
        """
        
        if len(self.replay_buffer) < self.batch_size:
            return None # Not enough experiences to learn

        experiences = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(self.device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(self.device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(self.device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(self.device)

        # Get Q values from current Q-network
        q_values = self.q_network(states).gather(1, actions)

        # Get max Q values from target Q-network for next states
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states).max(1)[0].unsqueeze(1)
            
        # Compute target Q values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = F.mse_loss(q_values, target_q_values)

        # Optimize the Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()