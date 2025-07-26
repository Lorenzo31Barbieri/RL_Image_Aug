# training_script.py (Updated for CIFAR10)

import torch
import torchvision
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import numpy as np

# Importa i tuoi moduli personalizzati
from agent import DQNAgent # Assicurati che DQNAgent sia aggiornato con la nuova QNetwork
from environment import ImageAugmentationEnv # Assicurati che ImageAugmentationEnv sia aggiornato
from transforms import get_num_actions

# Importa l'architettura VGG dal tuo nuovo file models/vgg.py
from vgg import VGG 

# --- CONFIGURAZIONE GLOBALE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Configurazione del Dataset e Percorsi ---
DATA_ROOT_DIR = './data' # Percorso per i dati CIFAR10
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth' # Percorso del classificatore CIFAR10
IMAGE_SIZE = 32 # CIFAR10 images are 32x32
NUM_CLASSES = 10 # CIFAR10 has 10 classes

# Dimensioni dello stato e delle azioni per l'Agente RL
# Lo stato sarà l'output (logits) del VGG, quindi 10 classi
STATE_DIM = NUM_CLASSES 
ACTION_DIM = get_num_actions()

# Hyperparameters for training
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.9995 # Decadimento dell'esplorazione (epsilon)
buffer_size = 10000
batch_size = 64
target_update_freq = 100
num_total_episodes = 10000 # Numero di episodi di training per l'agente RL
max_steps_per_episode = 5 # Numero massimo di trasformazioni per episodio RL

# --- Caricamento del Classificatore Pre-addestrato ---
def load_classifier_model():
    print("Loading pre-trained VGG19 CIFAR10 classifier...")
    # Assicurati che 'VGG19' corrisponda al nome del tuo modello in models/vgg.py
    classifier_model = VGG('VGG19').to(DEVICE) 
    
    try:
        checkpoint = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=DEVICE)
        
        # Gestione del prefisso 'module.' se il modello è stato salvato da DataParallel
        new_state_dict = {}
        for k, v in checkpoint['net'].items(): # Il tuo checkpoint ha la chiave 'net'
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Rimuovi 'module.'
            else:
                new_state_dict[k] = v
        
        classifier_model.load_state_dict(new_state_dict, strict=True) # strict=True per controllo completo
        print(f"Successfully loaded classifier weights from {PRE_TRAINED_CLASSIFIER_PATH}")
        print(f"Classifier accuracy from checkpoint: {checkpoint['acc']:.2f}%")
        
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {PRE_TRAINED_CLASSIFIER_PATH}")
        print("Please ensure you have trained your CIFAR10 VGG model and saved it as ckpt.pth in the 'checkpoint' directory.")
        exit() # Esci se il modello non è trovato
    except KeyError:
        print(f"Error: Invalid checkpoint format at {PRE_TRAINED_CLASSIFIER_PATH}. Expected 'net' key.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading classifier: {e}")
        exit()

    classifier_model.eval() # Metti il modello in modalità valutazione
    # Freeza i pesi del classificatore, non vogliamo che l'agente li modifichi
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")
    return classifier_model

# --- Main Training Function ---
def train_rl_agent():
    classifier_model = load_classifier_model()

    # Prepara il dataset CIFAR10 per gli episodi RL
    # Le immagini devono essere trasformate come il testset del classificatore.
    preprocess_for_rl_env = transforms.Compose([
        transforms.ToTensor(), # Converti PIL Image in Tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalizzazione CIFAR10
    ])

    # Usiamo il testset CIFAR10 per gli episodi RL per coerenza
    # (così l'agente impara a migliorare le immagini che il classificatore vede già nel test)
    # Potresti usare il trainset se vuoi.
    rl_episode_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=True, transform=preprocess_for_rl_env)
    
    # Batch size di 1 per gli episodi RL (un'immagine per episodio)
    rl_episode_loader = torch.utils.data.DataLoader(
        rl_episode_dataset, batch_size=1, shuffle=True, num_workers=0) # num_workers=0 per Windows/lambda issue
    
    # Iteratore per prelevare immagini per ogni episodio
    rl_episode_iter = iter(rl_episode_loader)

    agent = DQNAgent(STATE_DIM, ACTION_DIM, DEVICE, gamma, learning_rate,
                     epsilon_start, epsilon_end, epsilon_decay, buffer_size,
                     batch_size, target_update_freq)

    global_episode_counter = 0
    episode_rewards = []
    average_rewards_history = []
    loss_history = []

    print("\nStarting RL Agent training...")
    for episode in range(num_total_episodes):
        global_episode_counter += 1
        episode_reward = 0
        
        # Preleva una nuova immagine per l'episodio
        try:
            image_tensor_for_episode, true_label_for_episode_tensor = next(rl_episode_iter)
        except StopIteration:
            # Se l'iteratore è finito, resetta per ricominciare dal dataset
            rl_episode_iter = iter(rl_episode_loader)
            image_tensor_for_episode, true_label_for_episode_tensor = next(rl_episode_iter)

        # Rimuovi la dimensione del batch (era 1) e sposta sul device
        image_tensor_for_episode = image_tensor_for_episode.squeeze(0) 
        true_label_for_episode = true_label_for_episode_tensor.item()

        # Inizializza l'ambiente con la nuova immagine
        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode,
            device=DEVICE
        )
        state = env.reset(image_tensor_for_episode, true_label_for_episode)
        
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward

            if len(agent.replay_buffer) > agent.batch_size:
                loss_item = agent.learn()
                if loss_item is not None:
                    loss_history.append(loss_item)
        
        episode_rewards.append(episode_reward)

        if global_episode_counter % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            average_rewards_history.append(avg_reward)
            avg_loss = np.mean(loss_history[-100:]) if loss_history else 0 # Media della loss
            print(f"Episodio RL {global_episode_counter}/{num_total_episodes}, Ricompensa Media (ultimi 100): {avg_reward:.2f}, Loss Media: {avg_loss:.4f}, Epsilon: {agent.epsilon:.3f}")
            
            # Salvataggio del modello DQN
            # Crea la directory 'models' se non esiste
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(agent.q_network.state_dict(), f'./models/dqn_q_network_episode_{global_episode_counter}.pth')
            print(f"Agent model saved at episode {global_episode_counter}")

    print("\nRL Agent training finished.")
    
    # Plot della convergenza delle ricompense
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(0, len(average_rewards_history) * 100, 100), average_rewards_history)
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (last 100 episodes)')
    plt.title('RL Agent Training Convergence')
    plt.grid(True)
    plt.savefig('rl_training_convergence.png')
    plt.show()

    # Plot della convergenza della loss (se ci sono dati)
    if loss_history:
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(loss_history)), loss_history)
        plt.xlabel('Learning Step')
        plt.ylabel('Q-Network Loss')
        plt.title('Q-Network Loss during Training')
        plt.grid(True)
        plt.savefig('q_network_loss.png')
        plt.show()


if __name__ == '__main__':
    train_rl_agent()