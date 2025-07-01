# evaluation_script.py (un nuovo file, o una sezione nel tuo training_script.py)
import torch
import os
from agent import DQNAgent
from environment import ImageAugmentationEnv
from vgg_classifier import VGGClassifierWrapper
from transforms import get_num_actions, get_all_transforms
from torchvision import transforms
from PIL import Image

# ... (stesse configurazioni di training_script.py) ...
PRE_TRAINED_CLASSIFIER_PATH = './output/VGG16_trained.pth'
DQN_MODEL_PATH = './models/dqn_q_network_episode_20000.pth' # IL CHECKPOINT DA VALUTARE
IMAGE_SIZE = 224
NUM_CLASSES = 2
state_dim = 25088
action_dim = get_num_actions()
max_steps_per_episode = 5 # Puoi anche non applicare un max_steps qui, se l'obiettivo è solo testare

def evaluate_agent():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Carica il classificatore VGG16 (come prima)
    print("Loading pre-trained VGG16 classifier...")
    classifier_model = VGGClassifierWrapper(num_classes=NUM_CLASSES).to(device)
    # ... (logica di caricamento del classificatore identica a training_script.py) ...
    try:
        state_dict = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=device)
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
        return
    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")

    # Inizializza l'agente DQN
    agent = DQNAgent(state_dim, action_dim)
    
    # *** Carica i pesi dell'agente DQN ***
    if os.path.exists(DQN_MODEL_PATH):
        try:
            agent.q_network.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=device))
            # Per la valutazione, non serve caricare la target network, ma è buona pratica
            agent.target_q_network.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=device))
            print(f"Successfully loaded DQN agent from {DQN_MODEL_PATH}")
            agent.q_network.eval() # Metti la rete in modalità valutazione
            agent.target_q_network.eval() # E anche la target
            agent.epsilon = 0.0 # Disabilita l'esplorazione per la valutazione (greedy policy)
        except Exception as e:
            print(f"Error loading DQN model for evaluation: {e}")
            return
    else:
        print(f"Error: DQN model not found at {DQN_MODEL_PATH}.")
        return

    # Prepara le trasformazioni per le immagini di test
    preprocess_for_classifier = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    all_transforms_list = get_all_transforms(IMAGE_SIZE)

    # Esempio: Carica un'immagine specifica per testare
    # Puoi adattare questo per iterare su un dataset di test
    test_image_path = './data/Cat/1.jpg' # CAMBIA QUESTO PER UNA TUA IMMAGINE DI TEST
    true_label = 0 # 0 per cat, 1 per dog, a seconda di come hai mappato
    
    if not os.path.exists(test_image_path):
        print(f"Test image not found at {test_image_path}. Please update the path.")
        return

    original_image = Image.open(test_image_path).convert('RGB')
    image_tensor = preprocess_for_classifier(original_image).to(device).unsqueeze(0).squeeze(0) 

    env = ImageAugmentationEnv(
        classifier=classifier_model,
        max_steps=max_steps_per_episode
    )
    
    print(f"\nEvaluating agent on image: {test_image_path} (True Label: {true_label})")
    state = env.reset(image_tensor, true_label)
    
    done = False
    total_reward = 0
    applied_transforms = []

    initial_pred, initial_conf = classifier_model.classify(image_tensor.unsqueeze(0))
    print(f"Initial Classification: Predicted {initial_pred} (Conf: {initial_conf:.2f})")

    while not done:
        action = agent.select_action(state) # L'agente è in eval mode, quindi sarà greedy
        
        transform_object = all_transforms_list[action]
        
        # Prova a ottenere il nome in modo robusto
        action_name = "Unknown Transform" # Valore di default
        
        if hasattr(transform_object, '__name__'):
            # Questo funziona per funzioni Python definite con def
            action_name = transform_object.__name__
        elif hasattr(transform_object, '__class__') and hasattr(transform_object.__class__, '__name__'):
            # Questo funziona per istanze di classi (come quelle di torchvision.transforms)
            action_name = transform_object.__class__.__name__
        
        applied_transforms.append(action_name)

        next_state, reward, done, info = env.step(action)
        state = next_state
        total_reward += reward
        
        print(f"Step {info['steps_taken']}: Applied {action_name}. Reward: {reward:.2f}. "
              f"Current Pred: {info['prediction']} (Conf: {info['confidence']:.2f}). Status: {info['status']}")

    print(f"\nEvaluation Complete for {test_image_path}:")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Applied Transforms: {applied_transforms}")
    print(f"Final Classification: Predicted {info['prediction']} (Conf: {info['confidence']:.2f})")
    print(f"Correctly Classified: {info['prediction'] == true_label}")


evaluate_agent()
