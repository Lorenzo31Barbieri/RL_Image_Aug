import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Importa i tuoi moduli personalizzati
from vgg_classifier import VGGClassifierWrapper
from agent import DQNAgent
from environment import ImageAugmentationEnv
from transforms import get_num_actions, get_all_transforms
from dataset import PetImagesDataset # Assicurati che PetImagesDataset funzioni per test set
from torch.utils.data import DataLoader
from torchvision import transforms

# --- CONFIGURAZIONE GLOBALE ---
# Imposta il dispositivo una sola volta qui!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Se hai una GPU NVIDIA e vuoi usare CUDA, decommenta la riga sotto e commenta quella sopra:
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Configurazione del Dataset e Percorsi ---
DATA_ROOT_DIR = './data' # Assicurati che questo sia il percorso della tua cartella 'data'
PRE_TRAINED_CLASSIFIER_PATH = './output/VGG16_trained.pth' # Percorso del classificatore pre-addestrato
DQN_MODEL_PATH = './models/dqn_q_network_episode_20000.pth' # **CAMBIA QUESTO**: Percorso del checkpoint dell'agente DQN da valutare
IMAGE_SIZE = 224
NUM_CLASSES = 2
MAX_STEPS_PER_EPISODE = 5 # Numero massimo di trasformazioni per episodio RL

# Dimensioni dello stato e delle azioni per l'Agente RL
# Il state_dim (features estratte dal VGG) dovrebbe essere 25088 per VGG16
STATE_DIM = 25088 
ACTION_DIM = get_num_actions()

# --- Funzioni di Valutazione ---

def evaluate_classifier(classifier_model, test_dataloader, device):
    """
    Valuta le performance del classificatore su un dataset di test.
    """
    classifier_model.eval() # Metti il modello in modalità valutazione
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    print("Evaluating Classifier on test set...")
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc="Classifying test images"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = classifier_model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(test_dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"\n--- Classificatore Metrics ---")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print("-" * 30)
    
    return accuracy, f1, avg_loss, conf_matrix

def evaluate_agent():
    """
    Funzione principale per valutare l'agente RL e le correlazioni.
    """
    device = DEVICE # Usa la variabile globale DEVICE

    # 1. Caricamento e configurazione del classificatore pre-addestrato
    print("\n1. Loading pre-trained VGG16 classifier...")
    classifier_model = VGGClassifierWrapper(num_classes=NUM_CLASSES).to(device)
    try:
        state_dict = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=device)
        # Ajusta le chiavi se necessario (per la compatibilità con il tuo VGGClassifierWrapper)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.') or k.startswith('classifier.'):
                new_state_dict['vgg16.' + k] = v
            else:
                new_state_dict[k] = v
        classifier_model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded classifier weights from {PRE_TRAINED_CLASSIFIER_PATH}")
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {PRE_TRAINED_CLASSIFIER_PATH}")
        print("Please ensure you have trained and saved your VGG16 classifier.")
        return
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")

    # 2. Inizializzazione e Caricamento dell'Agente RL
    print("\n2. Initializing and Loading RL Agent...")
    agent = DQNAgent(STATE_DIM, ACTION_DIM, device) # Passa il DEVICE all'agente
    
    if os.path.exists(DQN_MODEL_PATH):
        try:
            agent.q_network.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=device))
            agent.target_q_network.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=device))
            print(f"Successfully loaded DQN agent from {DQN_MODEL_PATH}")
            agent.q_network.eval() # Metti la rete in modalità valutazione
            agent.target_q_network.eval()
            agent.epsilon = 0.0 # Disabilita l'esplorazione per la valutazione (greedy policy)
        except Exception as e:
            print(f"Error loading DQN model for evaluation: {e}")
            print("Starting evaluation with a randomly initialized agent.")
            agent.q_network.eval()
            agent.target_q_network.eval()
            agent.epsilon = 0.0
    else:
        print(f"Error: DQN model not found at {DQN_MODEL_PATH}. Starting evaluation with a randomly initialized agent.")
        agent.q_network.eval()
        agent.target_q_network.eval()
        agent.epsilon = 0.0

    # 3. Preparazione del Dataset per la Valutazione
    # Pre-elaborazione per il classificatore (stessa usata per il training del classificatore e dell'agente)
    preprocess_for_classifier = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset completo (sarà diviso per test set del classificatore e per episodi RL)
    dataset = PetImagesDataset(root_dir=DATA_ROOT_DIR, transform=preprocess_for_classifier)
    
    # DataLoader per la valutazione del classificatore (usiamo lo stesso dataset, ma potresti averne uno dedicato)
    test_classifier_dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # DataLoader per gli episodi RL (batch_size=1 per singola immagine per episodio)
    rl_episode_dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    rl_episode_iter = iter(rl_episode_dataloader)

    all_transforms_list = get_all_transforms() # Ottieni la lista di (funzione, nome_stringa)

    # --- INIZIO VALUTAZIONE ---

    # Fase 1: Valutazione del Classificatore Base
    print("\n--- Phase 1: Evaluating the base Classifier ---")
    base_accuracy, base_f1, base_loss, base_conf_matrix = evaluate_classifier(classifier_model, test_classifier_dataloader, device)

    # Fase 2: Valutazione dell'Agente RL
    print("\n--- Phase 2: Evaluating the RL Agent Performance ---")
    
    num_evaluation_episodes = 500 # Numero di episodi per testare l'agente
    total_rewards_rl = []
    transform_counts = {tf_name: 0 for tf_func, tf_name in all_transforms_list} # Inizializza contatore trasformazioni

    # Dati per la correlazione
    episode_final_rewards = []
    episode_final_accuracies = []
    episode_accuracy_improvements = []
    
    print(f"Running {num_evaluation_episodes} evaluation episodes for the RL Agent...")
    for eval_episode in tqdm(range(num_evaluation_episodes), desc="Evaluating RL Agent"):
        # Recupera una nuova immagine per l'episodio di valutazione
        try:
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)
        except StopIteration:
            rl_episode_iter = iter(rl_episode_dataloader)
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)

        image_tensor = image_tensor.squeeze(0).to(device) # Assicurati che l'immagine sia sul device
        true_label_for_episode = true_label_for_episode_tensor.item()

        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=MAX_STEPS_PER_EPISODE,
            device=device # Passa il DEVICE all'ambiente
        )
        
        state = env.reset(image_tensor, true_label_for_episode)
        
        episode_reward = 0
        done = False
        
        initial_pred_info = env.initial_prediction_info
        initial_accuracy = 1.0 if initial_pred_info['pred'] == true_label_for_episode else 0.0

        while not done:
            action = agent.select_action(state)
            action_name = all_transforms_list[action][1] # Ottieni il nome dalla tupla
            transform_counts[action_name] += 1

            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            
        total_rewards_rl.append(episode_reward)

        # Raccogli dati per la correlazione
        final_prediction_info = info 
        final_accuracy = 1.0 if final_prediction_info['prediction'] == true_label_for_episode else 0.0
        
        episode_final_rewards.append(episode_reward)
        episode_final_accuracies.append(final_accuracy)
        episode_accuracy_improvements.append(final_accuracy - initial_accuracy)

    avg_rl_reward = np.mean(total_rewards_rl)
    
    print(f"\n--- Agente RL Metrics ({num_evaluation_episodes} episodes) ---")
    print(f"Average Episode Reward: {avg_rl_reward:.2f}")
    print(f"Transformation Frequencies:")
    for name, count in sorted(transform_counts.items()):
        print(f"  - {name}: {count}")
    print("-" * 30)

    # Plot delle frequenze delle trasformazioni
    transform_names_for_plot = list(transform_counts.keys())
    counts_for_plot = list(transform_counts.values())

    plt.figure(figsize=(12, 7))
    plt.bar(transform_names_for_plot, counts_for_plot, color='skyblue')
    plt.xlabel('Transformation', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Frequency of Applied Transformations by RL Agent', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('transformation_frequencies.png')
    plt.show()

    # Fase 3: Correlazione Reward vs Accuracy
    print("\n--- Phase 3: Correlating RL Metrics ---")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(episode_final_rewards, episode_final_accuracies, alpha=0.6, edgecolors='w', s=50)
    plt.xlabel('Final Episode Reward', fontsize=12)
    plt.ylabel('Final Image Classification Accuracy (0 or 1)', fontsize=12)
    plt.title('RL Agent: Reward vs. Final Accuracy (Each Point is an Episode)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('reward_vs_accuracy.png')
    plt.show()

    # Distribuzione dell'Incremento di Accuracy
    plt.figure(figsize=(10, 6))
    plt.hist(episode_accuracy_improvements, bins=np.linspace(-1, 1, 21), edgecolor='black', alpha=0.7)
    plt.xlabel('Accuracy Improvement (Final - Initial)', fontsize=12)
    plt.ylabel('Number of Episodes', fontsize=12)
    plt.title('Distribution of Accuracy Improvement by RL Agent', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('accuracy_improvement_distribution.png')
    plt.show()
    
    print(f"Average Accuracy Improvement across episodes: {np.mean(episode_accuracy_improvements):.4f}")
    print("-" * 30)

# Esegui la valutazione
if __name__ == '__main__':
    evaluate_agent()