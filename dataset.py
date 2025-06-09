# dataset.py
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class PetImagesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory con tutte le sottocartelle delle immagini (es. 'data/PetImages').
            transform (callable, optional): Trasformazioni da applicare all'immagine.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = [] # 0 per Cat, 1 per Dog

        # Assumiamo una struttura root_dir/Cat/ e root_dir/Dog/
        cat_dir = os.path.join(root_dir, 'Cat')
        dog_dir = os.path.join(root_dir, 'Dog')

        # Carica per i gatti
        for img_name in os.listdir(cat_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.image_paths.append(os.path.join(cat_dir, img_name))
                self.labels.append(0) # Etichetta 0 per i gatti

        # Carica per i cani
        for img_name in os.listdir(dog_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.image_paths.append(os.path.join(dog_dir, img_name))
                self.labels.append(1) # Etichetta 1 per i cani
        
        print(f"Dataset caricato: {len(self.image_paths)} immagini totali.")
        print(f"  Gatti: {self.labels.count(0)} | Cani: {self.labels.count(1)}")


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        
        # Restituisce l'immagine pre-processata (tensor C, H, W) e la sua etichetta
        return image, label