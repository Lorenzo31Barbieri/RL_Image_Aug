from PIL import Image
import torch
import random

from agent import DQNAgent
from classifier import ImageClassifier
from environment import ImageAugmentationEnv
from transforms import get_num_actions

test_image_paths = [
    './data/Cat/1.jpg',
    './data/Cat/2.jpg',
    './data/Dog/1.jpg'
]

# 1. Inizializzazione Classificatore e Encoder
print("\n1. Inizializzazione Classificatore e Encoder...")
classifier_instance = ImageClassifier(model_name='resnet18', pretrained=True)

# L'encoder è la parte del modello del classificatore prima del layer finale (fc)
# Rimuoviamo l'ultimo layer per ottenere le feature (embeddings)
encoder_model = torch.nn.Sequential(*(list(classifier_instance.model.children())[:-1]))
encoder_model.eval() # Imposta l'encoder in modalità valutazione
encoder_model.to(classifier_instance.device) # Sposta l'encoder sul device corretto

# 1. Definire Hyperparameters
num_episodes = 1000 # O più, a seconda della complessità
max_steps_per_episode = 5 # Limite di passi per episodio
state_dim = 512 # La dimensione delle feature dell'encoder
action_dim = get_num_actions() # Ottieni il numero di azioni dal tuo transforms.py

# 2. Inizializzare Agente e Ambiente
agent = DQNAgent(state_dim, action_dim)

# Per un training reale, dovresti iterare sul tuo dataset di immagini
# Qui useremo un'immagine dummy per mostrare la struttura
# (Nel tuo codice reale, avrai un DataLoader che fornisce (image_tensor, true_label))
dummy_image_tensor = torch.randn(3, 224, 224) # Immagine dummy pre-processata
dummy_true_label = 0 # Etichetta dummy (es. 'gatto')

print("Inizio training dell'agente...")
for episode in range(num_episodes):
    # Esempio per caricare un'immagine casuale dal tuo test_image_paths:
    try:
        # Scegli casualmente un'immagine dal tuo set di test
        current_image_path = random.choice(test_image_paths)
        raw_image_pil = Image.open(current_image_path).convert('RGB')

        # Devi definire la true_label in base al nome del file o alla struttura della cartella
        # Assumiamo che la cartella 'Cat' significhi true_label = 0 e 'Dog' significhi true_label = 1
        if 'Cat' in current_image_path:
            true_label_for_episode = 0
        elif 'Dog' in current_image_path:
            true_label_for_episode = 1
        else:
            true_label_for_episode = 0 # Fallback

        preprocessed_image_tensor = classifier_instance.preprocess_image(raw_image_pil).squeeze(0)

    except Exception as e:
        print(f"Errore nel caricamento dell'immagine per l'episodio {episode+1}: {e}. Usando immagine dummy.")
        preprocessed_image_tensor = torch.randn(3, 224, 224)
        true_label_for_episode = 0 # Fallback

    
    env = ImageAugmentationEnv(
        original_image=preprocessed_image_tensor,
        true_label=true_label_for_episode,
        classifier=classifier_instance,
        encoder=encoder_model,
        max_steps=max_steps_per_episode
    )
    
    state = env.reset() # Ottieni lo stato iniziale (embedding dell'immagine originale)
    
    episode_reward = 0
    done = False
    
    while not done:
        action = agent.select_action(state) # L'agente sceglie un'azione
        next_state, reward, done, info = env.step(action) # Esegue l'azione nell'ambiente
        
        agent.store_experience(state, action, reward, next_state, done) # Memorizza l'esperienza
        agent.learn() # L'agente impara dal replay buffer
        
        state = next_state # Aggiorna lo stato
        episode_reward += reward
        
    print(f"Episodio {episode+1}/{num_episodes}, Ricompensa: {episode_reward:.2f}, Epsilon: {agent.epsilon:.3f}")

print("Training completato!")