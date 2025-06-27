# classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.amp import autocast, GradScaler
from PIL import Image

# Importa la tua classe CNN personalizzata
from my_cnn_model import MyCNN 

class ImageClassifier:
    def __init__(self, num_classes=2, lr=1e-3, model_path=None):
        """
        Inizializza il classificatore.
        Args:
            num_classes (int): Numero di classi per il classificatore (2 per cani/gatti).
            lr (float): Learning rate per l'ottimizzatore del classificatore.
            model_path (str, optional): Percorso da cui caricare i pesi pre-esistenti del modello.
                                        Se None, il modello viene inizializzato casualmente.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = MyCNN(num_classes=num_classes).to(self.device)
        
        # Carica i pesi se è stato fornito un percorso
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Modello classificatore caricato da: {model_path}")
            except FileNotFoundError:
                print(f"Attenzione: file modello non trovato a {model_path}. Inizializzazione casuale.")
            except Exception as e:
                print(f"Errore nel caricamento del modello da {model_path}: {e}. Inizializzazione casuale.")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss() # Usiamo CrossEntropyLoss per classificazione multliclasse (qui binaria)

        # Inizializza lo scaler per l'AMP
        self.scaler = GradScaler()

        # Trasformazioni standard per il pre-processing dell'immagine
        # Queste dovrebbero corrispondere a quelle usate per l'addestramento iniziale della CNN
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def preprocess_image(self, image):
        """
        Pre-processa una PIL Image o un tensore per l'input del modello.
        Converte in tensore PyTorch (C, H, W) e normalizza.
        """
        if isinstance(image, Image.Image):
            return self.preprocess(image)
        elif isinstance(image, torch.Tensor):
            # Se è già un tensore, assumiamo che sia già pre-processato e normalizzato
            # Assicuriamoci che abbia la forma (C, H, W) se non è già così
            if image.ndim == 4: # Se ha una dimensione batch
                image = image.squeeze(0)
            return image
        else:
            raise TypeError("Input must be a PIL Image or a torch.Tensor")

    def classify_image(self, input_tensor):
        """
        Classifica un tensore immagine.
        Args:
            input_tensor (torch.Tensor): Tensore immagine (C, H, W) o (1, C, H, W).
        Returns:
            tuple: (predicted_label, confidence)
                predicted_label (int): La label binaria predetta (0 per gatto, 1 per cane).
                confidence (float): La confidenza della predizione (probabilità).
        """
        # Aggiungi una dimensione batch se non presente
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0) # Forma diventa (1, C, H, W)

        input_tensor = input_tensor.to(self.device)

        self.model.eval() # Imposta il modello in modalità valutazione per la classificazione
        with torch.no_grad():
            # Il modello restituisce i logits
            logits = self.model(input_tensor) # Output: (1, num_classes)
            
            # Applica softmax per ottenere le probabilità
            probabilities = torch.softmax(logits, dim=1)
            
            # Trova la classe con la massima probabilità
            max_prob, predicted_label = torch.max(probabilities, 1)
            
            # Estrai i valori scalari
            predicted_label = predicted_label.item()
            confidence = max_prob.item()

        return predicted_label, confidence

    def train_step(self, images, labels):
        """
        Esegue un singolo passo di training per il classificatore.
        Args:
            images (torch.Tensor): Tensore batch di immagini (N, C, H, W).
            labels (torch.Tensor): Tensore batch di etichette (N).
        Returns:
            float: La perdita di training per il batch.
        """
        self.model.train() # Imposta il modello in modalità training
        
        images = images.to(self.device)
        labels = labels.to(self.device)

        # Zero i gradienti dell'ottimizzatore
        self.optimizer.zero_grad()
        
        # Abilita autocast per le operazioni in mixed precision
        with autocast(device_type=self.device.type):
            # Forward pass
            outputs = self.model(images)
        
            # Calcola la perdita
            loss = self.criterion(outputs, labels)

        # Usa lo scaler per la backward pass e l'ottimizzazione
        self.scaler.scale(loss).backward() 
        self.scaler.step(self.optimizer)  
        self.scaler.update()
        
        return loss.item()

    def get_encoder(self):
        """
        Restituisce la parte del modello che funge da encoder per l'estrazione delle feature.
        """
        # Per la tua MyCNN, l'encoder sarà la parte 'features' seguita dal 'avgpool'.
        # L'obiettivo è ottenere un output 1D (batch_size, state_dim)
        encoder = nn.Sequential(self.model.features, self.model.avgpool)
        encoder.eval() # L'encoder deve rimanere in modalità valutazione durante l'interazione RL
        encoder.to(self.device)
        return encoder

    def save_model(self, path):
        """Salva i pesi del modello classificatore."""
        torch.save(self.model.state_dict(), path)
        print(f"Modello classificatore salvato a: {path}")