import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import matplotlib.pyplot as plt
from tqdm import tqdm # Per una progress bar

# Assicurati che questi import siano corretti in base alla tua struttura di file
from my_cnn_model import MyCNN
from dataset import PetImagesDataset # Assumi che PetImagesDataset esista e sia corretto

# --- Configurazione ---
DATA_ROOT_DIR = './data' # Cartella radice del tuo dataset (es. data/train, data/test)
PRE_TRAINED_CLASSIFIER_PATH = 'pre_trained_classifier.pth' # Nome del file dove salvare il modello
NUM_CLASSES = 2 # Gatti e Cani
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10 # Puoi aggiustare questo. 10-20 di solito è un buon inizio per un modello da zero.
IMAGE_SIZE = 224 # Deve corrispondere a quello che la tua MyCNN si aspetta

def pretrain_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Definizione delle Trasformazioni per il Dataset
    # Queste trasformazioni sono per l'addestramento del classificatore.
    # Includono data augmentation standard per il training.
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Caricamento del Dataset
    print("Loading datasets...")
    # Assumiamo che il tuo PetImagesDataset possa gestire una sottocartella 'train' e 'test' o 'val'
    # Se il tuo dataset ha una struttura diversa (es. tutte le immagini in una cartella e un CSV per split),
    # dovrai adattare la classe PetImagesDataset o il modo in cui carichi i dati.
    train_dataset = PetImagesDataset(root_dir=os.path.join(DATA_ROOT_DIR, 'train'), transform=train_transforms)
    val_dataset = PetImagesDataset(root_dir=os.path.join(DATA_ROOT_DIR, 'test'), transform=val_transforms) # O 'val'

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) # num_workers=0 per Windows
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # 3. Inizializzazione del Modello, Funzione di Perdita e Ottimizzatore
    model = MyCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")

    # Lists per salvare i dati di training per il plot
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 4. Loop di Addestramento
    for epoch in range(NUM_EPOCHS):
        model.train() # Imposta il modello in modalità training
        running_loss = 0.0
        # tqdm aggiunge una progress bar al loop
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        # 5. Valutazione sul Validation Set
        model.eval() # Imposta il modello in modalità valutazione
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad(): # Disabilita il calcolo dei gradienti per la valutazione
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_dataset)
        epoch_val_accuracy = correct_predictions / total_predictions
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

    print("\nTraining complete.")

    # 6. Salvataggio del Modello Addestrato
    # Crea la cartella 'models' se non esiste
    os.makedirs(os.path.dirname(PRE_TRAINED_CLASSIFIER_PATH), exist_ok=True)
    torch.save(model.state_dict(), PRE_TRAINED_CLASSIFIER_PATH)
    print(f"Pre-trained classifier saved to {PRE_TRAINED_CLASSIFIER_PATH}")

    # 7. Plot delle Curve di Addestramento
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
    plt.title('Validation Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    pretrain_classifier()