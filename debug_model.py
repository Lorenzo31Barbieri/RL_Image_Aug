# Salva questo come debug_model.py
import torch
import sys
sys.path.append('.')

from src.models.agent import DQNAgent
from evaluation.core.model_loader import detect_model_state_dimension

model_path = './models/best_improved_dqn_model.pth'  # Cambia questo
state_dim = detect_model_state_dimension(model_path)
print(f"Modello: {model_path}")
print(f"Dimensioni stato: {state_dim}")

if state_dim:
    agent = DQNAgent(state_dim, 16, torch.device('cpu'))
    try:
        state_dict = torch.load(model_path, map_location='cpu')
        agent.q_network.load_state_dict(state_dict)
        print("✅ Modello caricato correttamente")
        
        # Test basic functionality
        dummy_state = torch.randn(state_dim)
        with torch.no_grad():
            q_vals = agent.q_network(dummy_state.unsqueeze(0))
        print(f"✅ Forward pass OK, output shape: {q_vals.shape}")
        
    except Exception as e:
        print(f"❌ Errore: {e}")