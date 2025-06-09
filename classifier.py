# classifier.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

class ImageClassifier:
    def __init__(self, model_name='resnet18', pretrained=True, num_classes_binary=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            # Non modificare il layer fc qui se vuoi usare le predizioni ImageNet originali
            # per la mappatura.
            # Se addestrassi il classificatore su un dataset binario, qui metteresti:
            # self.model.fc = nn.Linear(num_ftrs, num_classes_binary)
        else:
            raise ValueError("Modello non supportato.")

        self.model = self.model.to(self.device)
        self.model.eval() # Imposta il modello in modalità valutazione

        # Definizione delle trasformazioni per il pre-processing dell'immagine
        # Queste dovrebbero essere standard per ResNet18 addestrato su ImageNet
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # --- NUOVA LOGICA DI MAPPATURA ---
        # Definisci i range delle classi ImageNet per gatti e cani
        self.imagenet_cat_indices = list(range(281, 286)) # Indici 281-285 per i gatti
        self.imagenet_dog_indices = list(range(151, 269)) # Indici 151-268 per i cani

    def preprocess_image(self, image):
        """
        Pre-processa una PIL Image o un tensore per l'input del modello.
        """
        if isinstance(image, Image.Image):
            return self.preprocess(image)
        elif isinstance(image, torch.Tensor):
            # Se è già un tensore, assumiamo che sia già pre-processato
            # ma assicuriamoci che abbia la forma corretta (C, H, W)
            if image.ndim == 4:
                image = image.squeeze(0)
            # Potresti voler applicare la normalizzazione se non è già stata fatta
            # O semplicemente restituire l'immagine così com'è se l'ambiente lo gestisce
            return image
        else:
            raise TypeError("Input must be a PIL Image or a torch.Tensor")

    def classify_image(self, input_tensor):
        """
        Classifica un tensore immagine (C, H, W o 1, C, H, W).
        Restituisce la label binaria predetta (0 per gatto, 1 per cane) e la confidenza.
        """
        # Assicurati che l'input abbia la dimensione del batch
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0) # Aggiunge dimensione batch (1, C, H, W)

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor) # Output: (1, 1000)
            probabilities = torch.softmax(logits, dim=1)

            # Ottieni la predizione ImageNet con la massima probabilità
            max_prob, predicted_imagenet_idx = torch.max(probabilities, 1)
            predicted_imagenet_idx = predicted_imagenet_idx.item() # Estrai il valore scalare

            # --- NUOVA LOGICA DI MAPPATURA ---
            binary_label = -1 # Valore di default per indicare "non cane/gatto specifico"
            conf = 0.0 # Confidenza relativa alla classe mappata

            if predicted_imagenet_idx in self.imagenet_cat_indices:
                binary_label = 0 # Mappa a 'gatto'
                # Per la confidenza, puoi usare la probabilità della classe specifica del gatto
                # o la somma delle probabilità delle classi di gatto se vuoi una confidenza aggregata.
                # Per semplicità, usiamo la confidenza della classe singola più probabile.
                conf = max_prob.item()
            elif predicted_imagenet_idx in self.imagenet_dog_indices:
                binary_label = 1 # Mappa a 'cane'
                conf = max_prob.item()
            else:
                # Se il modello predice qualcos'altro (es. "tazza", "bicicletta"),
                # consideralo come "non una delle nostre classi target" per la ricompensa.
                # Puoi decidere come gestire questa confidenza:
                # Potresti volerla impostare a 0 o usare la confidenza della classe non-cat/dog
                # per penalizzare ulteriormente l'agente.
                binary_label = -1 # O qualsiasi valore indichi "non classificato come cat/dog"
                conf = 0.0 # Bassa confidenza se non è una delle classi che ci interessano.
                # Puoi anche decidere di ritornare la max_prob, ma la tua logica di ricompensa
                # dovrebbe penalizzare questa predizione se true_label è 0 o 1.

        return binary_label, conf