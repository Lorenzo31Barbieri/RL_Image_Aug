# my_cnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F # Importa F per le funzioni di attivazione e softmax

class MyCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(MyCNN, self).__init__()
        # Definire l'architettura della tua CNN
        # Esempio semplificato:
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch_size, 32, 112, 112)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch_size, 64, 56, 56)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch_size, 128, 28, 28)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # Output: (batch_size, 256, 14, 14)

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)   # Output: (batch_size, 512, 7, 7)
        )

        # Calcola la dimensione dell'input per il layer lineare finale
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Per avere un output (batch_size, 512, 1, 1)
        self.classifier_head = nn.Linear(512, num_classes) # num_classes dovrebbe essere 2 (gatto/cane)

    def forward(self, x):
        # Il percorso completo attraverso la rete per la classificazione
        # Questo è quello che userai per l'addestramento supervisionato della CNN
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # Appiattisce da (batch_size, 512, 1, 1) a (batch_size, 512)
        x = self.classifier_head(x)
        return x

    def encoder(self, x):
        """
        Estrae l'embedding dell'immagine da usare come stato per l'agente RL.
        Questo metodo non include il classificatore finale.
        """
        # Assicurati che l'input x sia un tensore PyTorch e abbia la dimensione del batch
        # Se ricevi un'immagine singola senza dimensione batch, aggiungila:
        # if x.dim() == 3:
        #     x = x.unsqueeze(0) # Aggiunge una dimensione per il batch

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # L'embedding sarà un vettore di dimensione 512
        return x # Questo è il tensore [batch_size, 512]

    def classify(self, image_tensor):
        """
        Esegue la classificazione su un tensore immagine e restituisce la classe predetta
        e la confidenza per il primo elemento del batch.
        Assume image_tensor è già pre-processato e su device corretto.
        """
        # Assicurati che il modello sia in modalità valutazione per l'inferenza
        # (disabilita dropout, batchnorm, ecc.)
        self.eval() 
        with torch.no_grad(): # Disabilita il calcolo del gradiente per l'inferenza
            logits = self.forward(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, dim=1)
            # Restituisce per il primo (e presumibilmente unico) elemento del batch
            return predicted_class.item(), confidence.item()