import torch
import torchvision.transforms as transforms
from transforms import get_action_transform


class ImageAugmentationEnv:
    """
    Attributes:
        classifier (VGG): the classifier istance.
        max_steps (int): maximum number of transormations per image.
        device (string): gpu device (cuda or mps).
        preprocess (Compose): standard transformations to be applied before classification.
        feature_extractor (VGG): used to extract features that reprents the state.
    """

    def __init__(self, classifier, max_steps, device):
        self.classifier = classifier
        self.max_steps = max_steps
        self.device = device
        self.current_step = 0
        self.original_image_tensor = None
        self.augmented_image_tensor = None
        self.true_label = None

        # Trasformazioni standard per il classificatore (CIFAR10-specific)
        self.preprocess = transforms.Compose([
            transforms.ToTensor(), # Immagini CIFAR10 sono già 32x32, ToTensor() prima di normalize
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.feature_extractor = self.classifier


    def reset(self, image_tensor, true_label):
        """
        Initialize a new RL episode.
        1. Get an image and its label.
        2. Reset parameters
        3. Execute a pre-evaluation of the image, calculating the prediction without augs and the confidence.
        4. Define the state.
        """

        self.original_image_tensor = image_tensor.to(self.device) # Ora è già un tensore normalizzato
        self.augmented_image_tensor = self.original_image_tensor.clone() # Clona per le modifiche
        self.true_label = true_label
        self.current_step = 0

        # Calcola le feature iniziali e la predizione
        with torch.no_grad():
            output = self.classifier(self.augmented_image_tensor.unsqueeze(0))
            initial_prediction = torch.argmax(output).item()
            initial_confidence = torch.nn.functional.softmax(output, dim=1).max().item()
            
            state = output.squeeze(0).cpu().numpy()

        self.initial_prediction_info = {
            'pred': initial_prediction,
            'conf': initial_confidence,
        }

        return state

    def step(self, action):
        """
        Execute a single step in the environment, given an action.
        1. Search the transoformation associated with the action.
        2. Give the image to the classifier in order to get new prediction/confidence.
        3. Calculate reward.
        4. Return new state, reward, and other infos. 
        """

        self.current_step += 1

        # Applica la trasformazione scelta
        transform_func = get_action_transform(action)
        # Assicurati che transform_func possa essere applicata al tensore.
        # Le trasformazioni da TF (torchvision.transforms.functional) operano su tensori.
        self.augmented_image_tensor = transform_func(self.augmented_image_tensor)

        # Calcola la nuova predizione e ricompensa
        with torch.no_grad():
            output = self.classifier(self.augmented_image_tensor.unsqueeze(0)) # Aggiungi batch dim
            prediction = torch.argmax(output).item()
            confidence = torch.nn.functional.softmax(output, dim=1).max().item()

        # Definizione della funzione di ricompensa (stessa logica di prima)
        if prediction == self.true_label:
            reward = 1.0 # Immagine ora classificata correttamente
        else:
            reward = -1.0 # Immagine non classificata correttamente

        done = self.current_step >= self.max_steps

        # Il next_state sarà l'output (logits) del VGG per l'immagine modificata
        next_state = output.squeeze(0).cpu().numpy() # Remove batch dim, move to CPU, convert to numpy

        info = {
            'prediction': prediction,
            'confidence': confidence,
            'true_label': self.true_label
        }

        return next_state, reward, done, info