import torch
from transforms import get_action_transform, get_num_actions


class ImageAugmentationEnv:
    def __init__(self, original_image, true_label, classifier, encoder, max_steps=5):
        """
        Inizializza l'ambiente di Image Augmentation.

        Args:
            original_image (torch.Tensor): L'immagine originale da aumentare.
                                           Si assume sia già pre-processata (C, H, W) e normalizzata.
            true_label (int): L'etichetta vera (0 o 1) dell'original_image.
            classifier: L'istanza del tuo classificatore (es. ImageClassifier).
            encoder: Il modello PyTorch usato come feature extractor (encoder),
                     che prende un tensore (1, C, H, W) e restituisce un embedding.
            max_steps (int): Numero massimo di azioni che l'agente può eseguire in un episodio.
        """
        self.original_image = original_image
        self.true_label = true_label
        self.classifier = classifier
        self.encoder = encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inizializza lo stato interno dell'ambiente
        self.current_image = None
        self.steps_taken = 0
        self.max_steps = max_steps

        # Assicurati che encoder e classifier siano sul device corretto
        self.encoder.to(self.device).eval() # L'encoder deve essere in modalità valutazione
        if hasattr(self.classifier, 'model'): # Se usi la classe ImageClassifier
            self.classifier.model.to(self.device).eval()
        else: # Se passi solo un modello grezzo
            self.classifier.to(self.device).eval()


    def reset(self):
        """
        Reimposta l'ambiente per un nuovo episodio.
        Restituisce lo stato iniziale (features dell'immagine originale).
        """
        self.current_image = self.original_image.clone().to(self.device) # Metti l'immagine sul device
        self.steps_taken = 0
        initial_state_features = self.get_features(self.current_image)
        return initial_state_features

    def step(self, action_id):
        """
        Esegue un'azione nell'ambiente.

        Args:
            action_id (int): L'ID dell'azione da eseguire.

        Returns:
            tuple: (next_state, reward, done, info)
                next_state (torch.Tensor): Features dell'immagine dopo l'azione.
                reward (float): Ricompensa ottenuta.
                done (bool): True se l'episodio è terminato.
                info (dict): Informazioni aggiuntive (opzionale).
        """
        self.steps_taken += 1

        # Ottieni la funzione di trasformazione per l'azione scelta
        transform_func = get_action_transform(action_id)

        # Applica la trasformazione.
        # È importante che la trasformazione gestisca il tipo di input di current_image
        # (Tensor o PIL Image). Assumiamo che get_action_transform ritorni una funzione
        # che opera su un torch.Tensor (C, H, W). Se opera su PIL, dovrai convertire.
        self.current_image = transform_func(self.current_image)
        
        # Assicurati che l'immagine sia sul device corretto prima di classificarla
        image_for_classification = self.current_image.unsqueeze(0) # Aggiungi dimensione batch

        # Classifica l'immagine aumentata
        # Se usi la classe ImageClassifier, il suo metodo classify_image è già robusto
        # e si aspetta una PIL Image o un Tensor (C,H,W) o (B,C,H,W) già pre-processato.
        # Qui, passiamo un tensore che dovrebbe essere già normalizzato.
        # Assumiamo che classifier.classify_image gestisca il tensore normalizzato
        # e lo metta sul device corretto internamente.
        predicted_label, confidence = self.classifier.classify_image(image_for_classification)


        # Calcola la ricompensa
        reward = 0.0
        done = False
        info = {'steps_taken': self.steps_taken}

        if predicted_label == self.true_label:
            # Ricompensa positiva basata sulla confidenza per la classificazione corretta
            reward = 1.0 + confidence # Incentiva alta confidenza
            done = True # L'episodio termina con successo
            info['status'] = 'Correctly classified'
        else:
            # Penalità fissa per classificazione errata
            reward = -1.0
            info['status'] = 'Incorrectly classified'

        # Penalità per ogni azione per incentivare la brevità della sequenza
        reward -= 0.01

        # Criterio di terminazione: numero massimo di passi raggiunto
        if self.steps_taken >= self.max_steps and not done:
            done = True
            reward -= 0.5 # Penalità aggiuntiva per non aver raggiunto l'obiettivo in tempo
            info['status'] = 'Max steps reached without correct classification'

        # Ottieni il nuovo stato (features dell'immagine aumentata)
        next_state = self.get_features(self.current_image)

        return next_state, reward, done, info

    def get_features(self, image_tensor):
        """
        Estrae le feature dall'immagine utilizzando l'encoder.
        image_tensor: un tensore (C, H, W) già pre-processato e normalizzato.
        """
        image_batch = image_tensor.unsqueeze(0).to(self.device) # (1, C, H, W)
        with torch.no_grad():
            features = self.encoder(image_batch) # Output tipico di ResNet: (1, 512, 1, 1)

            # Appiattisce il tensore per rimuovere tutte le dimensioni di 1 e renderlo 1D
            # Ad esempio, da (1, 512, 1, 1) a (512,) dopo unsqueeze(0)
            return features.squeeze(0).flatten() # <--- USA .flatten() QUI
            # O equivalentemente: return features.view(features.size(0), -1).squeeze(0)
            # O anche più semplicemente se sai che è (1, N, 1, 1) -> return features.view(-1)

    def get_state_shape(self):
        dummy_image = self.original_image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.encoder(dummy_image)
            # Deve restituire la forma del tensore 1D
        return features.squeeze(0).flatten().shape # <--- Assicurati che anche qui sia 1D

    def get_num_actions(self):
        return get_num_actions()