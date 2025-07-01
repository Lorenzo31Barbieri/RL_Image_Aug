import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque # Per il replay buffer

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim) # Output: Q-value per ogni azione

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, lr=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.01, epsilon_decay=0.995, replay_buffer_size=10000,
                 batch_size=64, target_update_freq=100):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma # Fattore di sconto per le ricompense future
        self.epsilon = epsilon_start # Probabilità di esplorazione
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device

        # Inizializza Q-Network e Target Q-Network
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict()) # Copia i pesi
        self.target_q_network.eval() # La target network è in modalità valutazione

        # Ottimizzatore
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss() # Funzione di perdita (Huber loss)

        # Replay Buffer
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.t_step = 0 # Contatore per gli aggiornamenti della target network

    def select_action(self, state):
        """
        Sceglie un'azione usando la strategia epsilon-greedy.
        State: Un tensore (state_dim,)
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim) # Esplorazione
        else:
            with torch.no_grad():
                # Assumiamo che 'state' arrivi già come torch.Tensor (state_dim,)
                state_tensor = state.unsqueeze(0).to(self.device) # Aggiungi dimensione batch
                q_values = self.q_network(state_tensor)
                return torch.argmax(q_values).item() # Sfruttamento

    def store_experience(self, state, action, reward, next_state, done):
        """Memorizza un'esperienza nel replay buffer."""
        self.replay_buffer.append((state.cpu(), action, reward, next_state.cpu(), done))

    def learn(self):
        """
        Apprende dall'esperienza campionando dal replay buffer.
        """
        if len(self.replay_buffer) < self.batch_size:
            return # Non abbastanza esperienze per imparare

        # Campiona un batch di esperienze
        experiences = random.sample(self.replay_buffer, self.batch_size)

        # Separa le esperienze in tensori PyTorch e spostale sul device
        states = torch.stack([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).unsqueeze(-1).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).unsqueeze(-1).to(self.device)
        next_states = torch.stack([e[3] for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in experiences]).unsqueeze(-1).to(self.device)

        # Calcola i Q-values dalla rete principale
        q_values_expected = self.q_network(states).gather(1, actions)

        # Calcola i Q-values target usando la target network
        target_q_values_next = self.target_q_network(next_states).max(1)[0].unsqueeze(-1).detach()

        # Calcola i Q-values target effettivi per l'aggiornamento
        q_values_target = rewards + self.gamma * target_q_values_next * (1 - dones)
        
        # Calcola la perdita e aggiorna la rete principale
        loss = self.criterion(q_values_expected, q_values_target)
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip dei gradienti (per stabilità)
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0) 
        
        self.optimizer.step()

        # Aggiorna epsilon (decadimento)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # Aggiorna la target network periodicamente
        self.t_step = (self.t_step + 1) % self.target_update_freq
        if self.t_step == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())