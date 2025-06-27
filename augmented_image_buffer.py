# augmented_image_buffer.py
import random
import torch

class AugmentedImageBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = [] # Lista di tuple: (image_tensor, true_label)
        self.ptr = 0 # Puntatore per sovrascrivere gli elementi più vecchi

    def add(self, image_tensor, true_label):
        """Aggiunge un'immagine aumentata e la sua etichetta originale al buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # Aggiungi placeholder se non al massimo della capacità
        
        # Salviamo una copia per evitare problemi di riferimento o modifiche in-place
        # Assicurati che image_tensor sia (C, H, W) e non (1, C, H, W)
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.squeeze(0)
            
        self.buffer[self.ptr] = (image_tensor.clone().detach(), true_label)
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size):
        """Campiona un batch di immagini aumentate dal buffer."""
        if len(self.buffer) < batch_size:
            return None, None # Non ci sono abbastanza immagini nel buffer
        
        batch = random.sample(self.buffer, batch_size)
        images, labels = zip(*batch) # Separa immagini e label
        return torch.stack(images), torch.tensor(labels) # Restituisce come tensori
    
    def __len__(self):
        return len(self.buffer)

    def reset(self):
        self.buffer = []
        self.ptr = 0