# training_script.py
import torch
import torch.nn as nn
from vgg_classifier import VGGClassifierWrapper
import os
from PIL import Image
from agent import DQNAgent 
from environment import ImageAugmentationEnv
from transforms import get_num_actions, get_all_transforms
from dataset import PetImagesDataset
from torch.utils.data import DataLoader

# --- Configurazione del Dataset ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = './output/VGG16_trained.pth'
IMAGE_SIZE = 224
NUM_CLASSES = 2

# Percorso per caricare/salvare i checkpoint dell'agente DQN
DQN_CHECKPOINT_DIR = './models' # La tua cartella 'models'
# Specifica il percorso completo del checkpoint da caricare, se presente.
# Lascia None se vuoi iniziare un nuovo training da zero.
# Esempio: LOAD_DQN_CHECKPOINT = os.path.join(DQN_CHECKPOINT_DIR, 'dqn_q_network_episode_10000.pth')
LOAD_DQN_CHECKPOINT = './models/dqn_q_network_episode_13600.pth' # Imposta a None per iniziare da zero

# Assicurati che la cartella 'models' esista per salvare i modelli dell'agente
if not os.path.exists(DQN_CHECKPOINT_DIR):
    os.makedirs(DQN_CHECKPOINT_DIR)

if __name__ == '__main__':
    num_total_episodes = 20000
    max_steps_per_episode = 5
    state_dim = 25088
    action_dim = get_num_actions()
    
    # Impostazione del Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Caricamento e configurazione del classificatore pre-addestrato (UGUALE A PRIMA)
    print("Loading pre-trained VGG16 classifier...")
    classifier_model = VGGClassifierWrapper(num_classes=NUM_CLASSES).to(device)
    try:
        state_dict = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=device)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.') or k.startswith('classifier.'):
                new_state_dict['vgg16.' + k] = v
            else:
                new_state_dict[k] = v
        classifier_model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded classifier weights from {PRE_TRAINED_CLASSIFIER_PATH} (prefix adjusted, strict=False)")
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {PRE_TRAINED_CLASSIFIER_PATH}")
        print("Please ensure you have run the VGG16 training script from the GitHub repo")
        print("and that the path 'PRE_TRAINED_CLASSIFIER_PATH' is correct.")
        exit()
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")

    # Inizializza la lista di tutte le trasformazioni
    all_transforms_list = get_all_transforms(IMAGE_SIZE) 

    # Prepara il Dataset per gli EPISODI RL
    from torchvision import transforms
    preprocess_for_classifier = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = PetImagesDataset(root_dir=DATA_ROOT_DIR, transform=preprocess_for_classifier)
    rl_episode_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    rl_episode_iter = iter(rl_episode_loader) 

    # 2. Inizializzare/Caricare Agente RL
    print("\n2. Inizializzazione/Caricamento Agente RL...")
    agent = DQNAgent(state_dim, action_dim) # Inizializza l'agente
    
    global_episode_counter = 0 # Inizializza il contatore degli episodi
    
    if LOAD_DQN_CHECKPOINT:
        try:
            agent.q_network.load_state_dict(torch.load(LOAD_DQN_CHECKPOINT, map_location=device))
            agent.target_q_network.load_state_dict(torch.load(LOAD_DQN_CHECKPOINT, map_location=device))
            
            # Estrai il numero dell'episodio dal nome del file per ripartire
            # Questo è un approccio semplice, potresti voler salvare il counter esplicitamente
            try:
                # Assuming format 'dqn_q_network_episode_XXXXX.pth'
                episode_num_str = LOAD_DQN_CHECKPOINT.split('_')[-1].split('.')[0]
                global_episode_counter = int(episode_num_str)
                print(f"Successfully loaded DQN agent from {LOAD_DQN_CHECKPOINT}. Resuming from episode {global_episode_counter + 1}.")
            except ValueError:
                print(f"Warning: Could not parse episode number from {LOAD_DQN_CHECKPOINT}. Starting episode counter from 0.")
                global_episode_counter = 0

            # Puoi anche voler caricare lo stato dell'ottimizzatore, se lo salvi
            # if os.path.exists(OPTIMIZER_CHECKPOINT_PATH):
            #     agent.optimizer.load_state_dict(torch.load(OPTIMIZER_CHECKPOINT_PATH, map_location=device))

            # Potresti anche voler resettare l'epsilon per riprendere l'esplorazione gradualmente
            # Per esempio, re-inizializzare epsilon decay se il training continua
            # agent.epsilon = agent.epsilon_start # O un valore intermedio calcolato
            
        except FileNotFoundError:
            print(f"Error: DQN checkpoint file not found at {LOAD_DQN_CHECKPOINT}. Starting new training.")
            global_episode_counter = 0 # Inizia da zero se il checkpoint non viene trovato
        except Exception as e:
            print(f"Error loading DQN checkpoint: {e}. Starting new training.")
            global_episode_counter = 0 # Inizia da zero se c'è un errore di caricamento
    else:
        print("Starting new DQN agent training from scratch.")

    print(f"\nInizio addestramento Agente RL per {num_total_episodes} episodi...")
    print(f"Ogni episodio RL userà una nuova immagine dal dataset.")

    episode_rewards = [] # Per tracciare le ricompense medie

    # Loop principale di addestramento dell'Agente RL
    while global_episode_counter < num_total_episodes: # Continua dal contatore caricato
        # Recupera una nuova immagine per l'episodio
        try:
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)
        except StopIteration:
            rl_episode_iter = iter(rl_episode_loader)
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)

        image_tensor = image_tensor.squeeze(0) 
        true_label_for_episode = true_label_for_episode_tensor.item()

        env = ImageAugmentationEnv(
            classifier=classifier_model,
            max_steps=max_steps_per_episode
        )
        
        state = env.reset(image_tensor, true_label_for_episode)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            if len(agent.replay_buffer) > agent.batch_size:
                agent.learn()
            
            state = next_state
            episode_reward += reward
            
        global_episode_counter += 1 # Incrementa dopo ogni episodio completato
        episode_rewards.append(episode_reward)

        # Stampa lo stato di avanzamento e salva i modelli periodicamente
        if global_episode_counter % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100 if len(episode_rewards) >= 100 else sum(episode_rewards) / len(episode_rewards)
            print(f"Episodio RL {global_episode_counter}/{num_total_episodes}, Ricompensa Media (ultimi 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # Salva il modello dell'agente periodicamente
            torch.save(agent.q_network.state_dict(), os.path.join(DQN_CHECKPOINT_DIR, f'dqn_q_network_episode_{global_episode_counter}.pth'))
            print(f"Modello DQN salvato a {os.path.join(DQN_CHECKPOINT_DIR, f'dqn_q_network_episode_{global_episode_counter}.pth')}")

    print("\nAddestramento Agente RL completato!")