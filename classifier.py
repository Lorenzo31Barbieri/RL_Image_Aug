import torch
import torchvision.models as models
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image # Per la gestione delle immagini PIL

class ImageClassifier:
    def __init__(self, model_name='resnet18', num_classes=1000, pretrained=True):
        """
        Inizializza il classificatore di immagini.

        Args:
            model_name (str): Nome del modello pre-addestrato da caricare (es. 'resnet18', 'resnet50').
            num_classes (int): Numero di classi del dataset su cui è stato addestrato il classificatore (default 1000 per ImageNet).
            pretrained (bool): Se caricare pesi pre-addestrati.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_name, pretrained, num_classes)
        self.model.to(self.device)
        self.model.eval() # Imposta il modello in modalità valutazione
        self.preprocess = self._define_preprocess_transforms()

    def _load_model(self, model_name, pretrained, num_classes):
        """Carica il modello pre-addestrato."""
        if model_name == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = models.resnet50(pretrained=pretrained)
        # Aggiungi altri modelli se necessario
        else:
            raise ValueError(f"Modello '{model_name}' non supportato.")

        # Se stai usando un modello pre-addestrato su ImageNet (1000 classi)
        # e vuoi riutilizzarlo per un problema a 2 classi, dovrai modificare l'ultimo layer.
        # Per ora, manteniamo l'originale se l'obiettivo è solo testare la classificazione
        # e poi mappare le label. Tuttavia, per un progetto reale, ti suggerisco
        # di fare il fine-tuning del classificatore sul tuo dataset.
        if num_classes != 1000 and pretrained:
            # Esempio per ResNet: sostituisci l'ultimo layer completamente connesso
            num_ftrs = model.fc.in_features
            model.fc = torch.nn.Linear(num_ftrs, num_classes)
            print(f"Sostituito il layer finale di {model_name} per {num_classes} classi.")

        return model

    def _define_preprocess_transforms(self):
        """Definisce le trasformazioni di preprocessing standard."""
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def preprocess_image(self, image):
        """
        Applica le trasformazioni di preprocessing a un'immagine.

        Args:
            image (PIL.Image o torch.Tensor): L'immagine da pre-processare.

        Returns:
            torch.Tensor: L'immagine pre-processata e pronta per il modello.
        """
        if isinstance(image, torch.Tensor) and image.ndim == 4:
            # Se è già un batch (B, C, H, W) e normalizzato, non fare nulla
            # Altrimenti, se è (C, H, W) o (H, W, C), le trasformazioni lo gestiranno
            return image
        elif isinstance(image, torch.Tensor) and image.ndim == 3:
            # Se è (C, H, W), applica le trasformazioni (ToTensor lo gestisce)
            return self.preprocess(T.ToPILImage()(image)) # Converti in PIL per le trasformazioni
        elif isinstance(image, Image.Image):
            return self.preprocess(image).unsqueeze(0) # Aggiungi la dimensione del batch
        else:
            raise TypeError("L'input 'image' deve essere una PIL Image o un torch.Tensor.")

    def classify_image(self, image):
        """
        Classifica una singola immagine.

        Args:
            image (PIL.Image o torch.Tensor): L'immagine da classificare.

        Returns:
            tuple: (pred_label, confidence), dove pred_label è 0 o 1, e confidence è la probabilità.
        """
        # Applica il preprocessing se l'immagine non è già un tensore pronto per il modello
        if isinstance(image, Image.Image) or (isinstance(image, torch.Tensor) and image.ndim != 4):
            input_tensor = self.preprocess_image(image)
        else:
            input_tensor = image.to(self.device) # Assume che sia già preprocessata e in formato batch

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            original_pred_label_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, original_pred_label_idx].item()

            # Mappatura delle etichette (questo è specifico del tuo problema)
            # Hai detto che vuoi classificare in 0 o 1.
            # Se le classi ImageNet da 151 a 268 corrispondono alla classe "1"
            # e le altre alla classe "0".
            # if 151 <= original_pred_label_idx <= 268:
            #     final_pred_label = 1
            # else:
            #     final_pred_label = 0

            return original_pred_label_idx, confidence