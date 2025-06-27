import torch
from my_cnn_model import MyCNN
from transforms import get_action_transform, get_num_actions

class ImageAugmentationEnv:
    def __init__(self, classifier, max_steps=5): 
        """
        Inizializza l'ambiente di Image Augmentation.

        Args:
            classifier (MyCNN): L'istanza del classificatore, già pre-addestrato.
            max_steps (int): Numero massimo di azioni che l'agente può eseguire in un episodio.
        """
        self.classifier = classifier
        self.classifier.eval() 
        for param in self.classifier.parameters():
            param.requires_grad = False 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Inizializza lo stato interno dell'ambiente
        self.original_image = None # Sarà impostato in reset()
        self.true_label = None     # Sarà impostato in reset()
        self.current_image = None
        self.steps_taken = 0
        self.max_steps = max_steps
        
        # Per la ricompensa: memorizza la predizione iniziale dell'immagine non aumentata
        self.initial_prediction_info = {'pred': -1, 'conf': 0.0} # Verrà aggiornato in reset()


    def reset(self, original_image_tensor, true_label): 
        """
        Reimposta l'ambiente per un nuovo episodio con una nuova immagine.
        Restituisce lo stato iniziale (features dell'immagine originale).
        
        Args:
            original_image_tensor (torch.Tensor): L'immagine originale per il nuovo episodio (C, H, W).
            true_label (int): L'etichetta vera dell'immagine.
        """
        self.original_image = original_image_tensor.to(self.device) # Sposta l'originale sul device
        self.true_label = true_label
        self.current_image = self.original_image.clone() # Clona per non modificare l'originale
        self.steps_taken = 0
        
        # Calcola la predizione e confidenza iniziale per l'immagine originale
        initial_pred, initial_conf = self.classifier.classify(self.original_image.unsqueeze(0))
        self.initial_prediction_info = {'pred': initial_pred, 'conf': initial_conf}

        # Ottieni lo stato iniziale (features dall'encoder del classificatore)
        initial_state_features = self.classifier.encoder(self.current_image.unsqueeze(0)).squeeze(0)
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

        # Applica la trasformazione all'immagine corrente
        self.current_image = transform_func(self.current_image)
        
        image_for_classification = self.current_image.unsqueeze(0) 

        # Classifica l'immagine aumentata
        predicted_label, confidence = self.classifier.classify(image_for_classification)

        # Calcola la ricompensa
        reward = 0.0
        done = False
        info = {'steps_taken': self.steps_taken, 'prediction': predicted_label, 'confidence': confidence}

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
            # Penalità aggiuntiva per non aver raggiunto l'obiettivo in tempo
            reward -= 0.5 
            info['status'] = 'Max steps reached without correct classification'

        # Ottieni il nuovo stato (features dell'immagine aumentata)
        next_state = self.classifier.encoder(self.current_image.unsqueeze(0)).squeeze(0)

        return next_state, reward, done, info
    

    def get_state_shape(self):
        """
        Restituisce la forma dell'embedding di stato.
        """
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device) # Esempio: 224x224
        with torch.no_grad():
            features = self.classifier.encoder(dummy_input)
        return features.squeeze(0).shape # Dovrebbe essere torch.Size([512])

    def get_num_actions(self):
        """Restituisce il numero totale di azioni disponibili."""
        return get_num_actions()