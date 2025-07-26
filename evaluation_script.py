# evaluation_script.py (Updated for CIFAR10)

import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vgg import VGG
from agent import DQNAgent 
from environment import ImageAugmentationEnv
from transforms import get_num_actions, get_all_transforms

# --- CONFIGURAZIONE GLOBALE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Configurazione del Dataset e Percorsi ---
DATA_ROOT_DIR = './data' 
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth' # Percorso del classificatore CIFAR10
# **CAMBIA QUESTO**: Percorso del checkpoint dell'agente DQN da valutare
DQN_MODEL_PATH = './models/best_dqn_model.pth' 
IMAGE_SIZE = 32 # CIFAR10 images are 32x32
NUM_CLASSES = 10 # CIFAR10 has 10 classes

# Dimensioni dello stato e delle azioni per l'Agente RL
STATE_DIM = NUM_CLASSES # Lo stato è l'output (logits) del VGG
ACTION_DIM = get_num_actions()
MAX_STEPS_PER_EPISODE = 5 # Numero massimo di trasformazioni per episodio RL

# --- Funzioni di Valutazione ---

def load_classifier_model_for_eval():
    print("Loading pre-trained VGG19 CIFAR10 classifier for evaluation...")
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
        print(f"Successfully loaded classifier weights from {PRE_TRAINED_CLASSIFIER_PATH}")
        print(f"Classifier accuracy from checkpoint: {checkpoint['acc']:.2f}%")
        
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {PRE_TRAINED_CLASSIFIER_PATH}")
        print("Please ensure you have trained your CIFAR10 VGG model and saved it as ckpt.pth in the 'checkpoint' directory.")
        exit()
    except KeyError:
        print(f"Error: Invalid checkpoint format at {PRE_TRAINED_CLASSIFIER_PATH}. Expected 'net' key.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading classifier: {e}")
        exit()

    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")
    return classifier_model


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
    device = DEVICE # Usa la variabile globale DEVICE

    # 1. Caricamento e configurazione del classificatore pre-addestrato
    classifier_model = load_classifier_model_for_eval()

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
    # Pre-elaborazione per il classificatore (stessa usata per il training del classificatore)
    preprocess_for_classifier = transforms.Compose([
        transforms.ToTensor(), # Immagini CIFAR10 sono già 32x32
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Usiamo il testset CIFAR10 per la valutazione del classificatore e per gli episodi RL
    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=True, transform=preprocess_for_classifier)
    
    test_classifier_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0) # num_workers=0
    
    # DataLoader per gli episodi RL (batch_size=1 per singola immagine per episodio)
    rl_episode_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0) # num_workers=0
    rl_episode_iter = iter(rl_episode_loader)

    all_transforms_list = get_all_transforms() 

    # --- INIZIO VALUTAZIONE ---

    # Fase 1: Valutazione del Classificatore Puro
    print("\n--- Phase 1: Evaluating the base Classifier ---")
    base_accuracy, base_f1, base_loss, base_conf_matrix = evaluate_classifier(classifier_model, test_classifier_dataloader, device)

    # Fase 2: Valutazione dell'Agente RL (con l'interazione nell'ambiente)
    print("\n--- Phase 2: Evaluating the RL Agent Performance ---")
    
    num_evaluation_episodes = 500 
    total_rewards_rl = []
    transform_counts = {tf_name: 0 for tf_func, tf_name in all_transforms_list} 
    
    episode_final_rewards = []
    episode_final_accuracies = [] 
    episode_accuracy_improvements = [] 

    print(f"Running {num_evaluation_episodes} evaluation episodes for the RL Agent...")
    for eval_episode in tqdm(range(num_evaluation_episodes), desc="Evaluating RL Agent"):
        try:
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)
        except StopIteration:
            rl_episode_iter = iter(rl_episode_loader)
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)

        image_tensor = image_tensor.squeeze(0).to(device)
        true_label_for_episode = true_label_for_episode_tensor.item()

        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=MAX_STEPS_PER_EPISODE,
            device=device 
        )
        
        state = env.reset(image_tensor, true_label_for_episode)
        
        episode_reward = 0
        done = False
        
        initial_pred_info = env.initial_prediction_info
        initial_accuracy = 1.0 if initial_pred_info['pred'] == true_label_for_episode else 0.0

        while not done:
            action = agent.select_action(state)
            action_name = all_transforms_list[action][1] 
            transform_counts[action_name] += 1

            next_state, reward, done, info = env.step(action)
            
            state = next_state
            episode_reward += reward
            
        total_rewards_rl.append(episode_reward)

        final_prediction_info = info 
        final_accuracy = 1.0 if final_prediction_info['prediction'] == true_label_for_episode else 0.0
        
        episode_final_rewards.append(episode_reward)
        episode_final_accuracies.append(final_accuracy)
        episode_accuracy_improvements.append(final_accuracy - initial_accuracy)

    avg_rl_reward = np.mean(total_rewards_rl)
    avg_final_accuracy_with_augmentation = np.mean(episode_final_accuracies) # Nuova metrica
    
    print(f"\n--- Agente RL Metrics ({num_evaluation_episodes} episodes) ---")
    print(f"Average Episode Reward: {avg_rl_reward:.2f}")
    print(f"Average Final Accuracy (with augmentation): {avg_final_accuracy_with_augmentation:.4f}")
    print(f"Transformation Frequencies:")
    for name, count in sorted(transform_counts.items()):
        print(f"  - {name}: {count}")
    print("-" * 30)

    import matplotlib.pyplot as plt
    
    transform_names = list(transform_counts.keys())
    counts = list(transform_counts.values())

    plt.figure(figsize=(12, 7))
    plt.bar(transform_names, counts, color='skyblue')
    plt.xlabel('Transformation', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Frequency of Applied Transformations by RL Agent', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig('transformation_frequencies.png')
    plt.show()

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

if __name__ == '__main__':
    evaluate_agent()