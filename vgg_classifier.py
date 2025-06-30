import torch
import torch.nn as nn
from torchvision import models

class VGGClassifierWrapper(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        # Carica il VGG16 pre-addestrato su ImageNet
        # Usiamo VGG16_BN_Weights.DEFAULT per ottenere i pesi pre-addestrati più recenti
        self.vgg16 = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)

        # Congela i layer convoluzionali (features)
        for param in self.vgg16.features.parameters():
            param.requires_grad = False

        # Modifica il classificatore finale per il tuo numero di classi
        # Il classificatore di VGG16 è un nn.Sequential di 6 strati. L'ultimo è nn.Linear.
        num_ftrs = self.vgg16.classifier[-1].in_features

        # Rimuovi l'ultimo strato e aggiungine uno nuovo
        features_list = list(self.vgg16.classifier.children())[:-1] # Prende tutti i layer tranne l'ultimo
        features_list.extend([nn.Linear(num_ftrs, num_classes)]) # Aggiunge il nuovo layer
        self.vgg16.classifier = nn.Sequential(*features_list) # Assegna il nuovo Sequential

        # Assicurati che l'ultimo strato (quello che abbiamo appena aggiunto) sia addestrabile
        # Di default, nn.Linear ha requires_grad=True, ma è bene verificare se in futuro si volesse fine-tunare
        for param in self.vgg16.classifier.parameters():
            param.requires_grad = True
            
        # Aggiungi un Softmax per ottenere le probabilità nel metodo classify
        self.softmax = nn.Softmax(dim=1) # Dimensione 1 per le classi

    def forward(self, x):
        # Il forward pass del wrapper è semplicemente il forward del VGG16
        # Questo restituisce le logits (output prima del softmax)
        return self.vgg16(x)

    def encoder(self, x):
        # Questo è il metodo per estrarre le feature (lo stato RL)
        # VGG16.features è l'encoder convoluzionale.
        # Dopo le features, c'è un AdaptiveAvgPool2d (self.vgg16.avgpool) e poi un Flatten implicito nel classificatore.
        
        x = self.vgg16.features(x)
        x = self.vgg16.avgpool(x) # Applica l'AdaptiveAvgPool2d che è già parte del VGG16
        x = torch.flatten(x, 1) # Flatten l'output in un vettore 1D

        # Questo x è il vettore di feature che usiamo come stato RL.
        # L'output_size di avgpool è (7,7) per VGG, quindi 512 * 7 * 7 = 25088 feature.
        # Devi ASSICURARTI che state_dim nel training_script.py sia 25088.
        return x

    def classify(self, x):
        # Questo è il metodo per ottenere la label predetta e la confidenza
        # x è l'immagine (o un batch di immagini) già pre-processata
        
        # Ottieni le logits usando il forward del VGG16
        logits = self.forward(x) # Chiamiamo il forward del wrapper, che a sua volta chiama self.vgg16(x)

        # Applica Softmax per ottenere le probabilità
        probabilities = self.softmax(logits)
        
        # La label predetta è l'indice della classe con la probabilità più alta
        # Se il batch size è 1 (come nel tuo caso per l'ambiente), torch.argmax(..., dim=1)[0].item() è corretto.
        predicted_label = torch.argmax(probabilities, dim=1).item() 

        # La confidenza è la probabilità della label predetta
        confidence = probabilities[0, predicted_label].item() 

        # RESTITUISCE DUE VALORI: predicted_label e confidence
        return predicted_label, confidence