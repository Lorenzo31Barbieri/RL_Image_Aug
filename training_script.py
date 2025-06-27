# training_script.py
import torch
import random
import os
from PIL import Image
from agent import DQNAgent 
from environment import ImageAugmentationEnv
from transforms import get_num_actions, get_all_transforms
from dataset import PetImagesDataset
from torch.utils.data import DataLoader
from my_cnn_model import MyCNN

# --- Configurazione del Dataset ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = 'pre_trained_classifier.pth'
IMAGE_SIZE = 224

if __name__ == '__main__':
    num_total_episodes = 20000
    max_steps_per_episode = 5
    state_dim = 512
    action_dim = get_num_actions()
    
    # Impostazione del Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Caricamento e Congelamento del Classificatore CNN Pre-addestrato
    print("\n1. Caricamento e congelamento del Classificatore (MyCNN)...")
    classifier_model = MyCNN(num_classes=2).to(device)
    
    # Carica i pesi pre-addestrati.
    if os.path.exists(PRE_TRAINED_CLASSIFIER_PATH):
        classifier_model.load_state_dict(torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=device))
        print(f"Classificatore caricato da: {PRE_TRAINED_CLASSIFIER_PATH}")
    else:
        print(f"ERRORE: File '{PRE_TRAINED_CLASSIFIER_PATH}' non trovato. Assicurati di aver pre-addestrato e salvato la tua CNN.")
        print("Uscita. Si prega di pre-addestrare il classificatore prima di eseguire questo script.")
        exit() # Termina lo script se il modello non è trovato

    # Congela i pesi del classificatore
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Pesi del classificatore congelati. Sarà usato solo per inferenza.")

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

    # DataLoader per gli EPISODI RL
    rl_episode_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    rl_episode_iter = iter(rl_episode_loader) 

    # 2. Inizializzare Agente RL
    print("\n2. Inizializzazione Agente RL...")
    agent = DQNAgent(state_dim, action_dim) # Iperparametri per DQNAgent sono definiti al suo interno
    print("Agente RL inizializzato.")

    print(f"\nInizio addestramento Agente RL per {num_total_episodes} episodi...")
    print(f"Ogni episodio RL userà una nuova immagine dal dataset.")

    global_episode_counter = 0
    episode_rewards = [] # Per tracciare le ricompense medie

    # Loop principale di addestramento dell'Agente RL
    while global_episode_counter < num_total_episodes:
        # Recupera una nuova immagine per l'episodio
        try:
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)
        except StopIteration:
            # Se l'iteratore finisce, resettalo per ricominciare dal dataset
            rl_episode_iter = iter(rl_episode_loader)
            image_tensor, true_label_for_episode_tensor = next(rl_episode_iter)

        image_tensor = image_tensor.squeeze(0) 
        true_label_for_episode = true_label_for_episode_tensor.item()

        # Inizializza l'ambiente per questo episodio
        env = ImageAugmentationEnv(
            classifier=classifier_model, # Passiamo il MyCNN congelato
            all_transforms_list=all_transforms_list, # Lista delle funzioni di trasformazione
            max_steps=max_steps_per_episode
        )
        
        # Resetta l'ambiente con l'immagine e l'etichetta dell'episodio
        state = env.reset(image_tensor, true_label_for_episode)
        
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_experience(state, action, reward, next_state, done)
            
            # Addestra l'agente solo se il replay buffer ha abbastanza esperienze
            if len(agent.replay_buffer) > agent.batch_size:
                agent.learn()
            
            state = next_state
            episode_reward += reward
            
        global_episode_counter += 1
        episode_rewards.append(episode_reward)

        # Stampa lo stato di avanzamento e salva i modelli periodicamente
        if global_episode_counter % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100 if len(episode_rewards) >= 100 else sum(episode_rewards) / len(episode_rewards)
            print(f"Episodio RL {global_episode_counter}/{num_total_episodes}, Ricompensa Media (ultimi 100): {avg_reward:.2f}, Epsilon: {agent.epsilon:.3f}")
            
            # Salva il modello dell'agente periodicamente
            torch.save(agent.q_network.state_dict(), f'./models/dqn_q_network_episode_{global_episode_counter}.pth')
            print(f"Modello DQN salvato a ./models/dqn_q_network_episode_{global_episode_counter}.pth")

    print("\nAddestramento Agente RL completato!")