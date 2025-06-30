# training_script.py
import torch
import torch.nn as nn # Necessario per nn.Module, utile per il debug
from vgg_classifier import VGGClassifierWrapper
import os
from PIL import Image
from agent import DQNAgent 
from environment import ImageAugmentationEnv
from transforms import get_num_actions, get_all_transforms # get_all_transforms è ancora usato per creare la lista, ma non passata all'env
from dataset import PetImagesDataset
from torch.utils.data import DataLoader
# from my_cnn_model import MyCNN # Rimuovi, non più usata come classificatore principale

# --- Configurazione del Dataset ---
# Assicurati che DATA_ROOT_DIR punti alla tua cartella 'data_classifier'
# se l'hai creata con lo script prepare_classifier_data.py.
# Altrimenti, punta alla cartella 'data' originale, ma la sua struttura
# non è quella attesa dal classificatore VGG16 per il suo addestramento.
# Per l'agente RL, PetImagesDataset legge da 'data/cat' e 'data/dog', quindi
# se il tuo dataset originale è così, va bene.
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = './output/VGG16_trained.pth' # Percorso del tuo .pth dal training del VGG16
IMAGE_SIZE = 224
NUM_CLASSES = 2

# Assicurati che la cartella 'models' esista per salvare i modelli dell'agente
if not os.path.exists('./models'):
    os.makedirs('./models')

if __name__ == '__main__':
    num_total_episodes = 20000
    max_steps_per_episode = 5
    state_dim = 25088 # Conferma che questo sia l'output size dell'encoder VGG16
    action_dim = get_num_actions()
    
    # Impostazione del Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Caricamento e configurazione del classificatore pre-addestrato
    print("Loading pre-trained VGG16 classifier...")
    
    # Crea un'istanza del tuo wrapper VGG16 (che caricherà VGG16 pre-addestrato su ImageNet)
    classifier_model = VGGClassifierWrapper(num_classes=NUM_CLASSES).to(device)

    # Carica i pesi del tuo VGG16 addestrato sui gatti/cani dal tuo repo.
    # Il percorso deve puntare al .pth che hai generato dal file del repo
    try:
        # Carica il state_dict originale
        state_dict = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=device)
        
        # Crea un nuovo state_dict con i prefissi corretti
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('features.') or k.startswith('classifier.'):
                new_state_dict['vgg16.' + k] = v
            else:
                new_state_dict[k] = v
        
        # Carica il state_dict modificato. Usiamo strict=False.
        classifier_model.load_state_dict(new_state_dict, strict=False)
        print(f"Successfully loaded classifier weights from {PRE_TRAINED_CLASSIFIER_PATH} (prefix adjusted, strict=False)")
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {PRE_TRAINED_CLASSIFIER_PATH}")
        print("Please ensure you have run the VGG16 training script from the GitHub repo")
        print("and that the path 'PRE_TRAINED_CLASSIFIER_PATH' is correct.")
        exit() # Esci dal programma se il file non è trovato

    # Il congelamento dei pesi "features" è già gestito nella VGGClassifierWrapper
    # e verrà ri-applicato qui per sicurezza all'intero modello:
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    
    print("Classifier loaded and weights frozen.")

    # Inizializza la lista di tutte le trasformazioni (questa è necessaria per l'agente e get_action_transform)
    # ma non viene passata all'ambiente direttamente.
    all_transforms_list = get_all_transforms(IMAGE_SIZE) 

    # Prepara il Dataset per gli EPISODI RL
    from torchvision import transforms # Già importato
    preprocess_for_classifier = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 224
        transforms.ToTensor(),
        # Questa normalizzazione è CRUCIALE e deve corrispondere a quella usata per addestrare VGG16
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
            classifier=classifier_model, # Passiamo il classificatore VGG16 congelato
            # RIMOSSO: all_transforms_list=all_transforms_list, # Questo parametro non esiste in ImageAugmentationEnv.__init__
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