import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # AGGIUNTO random_split
from torchvision import transforms, datasets # AGGIUNTO datasets (ImageFolder)
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from my_cnn_model import MyCNN
# from dataset import PetImagesDataset # NON USEREMO PIÙ LA TUA CLASSE SPECIFICA SE USI ImageFolder

# --- Configurazione ---
DATA_ROOT_DIR = './data'
PRE_TRAINED_CLASSIFIER_PATH = 'pre_trained_classifier.pth'
NUM_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 10
IMAGE_SIZE = 224

# Percentuale di dati da usare per il training
TRAIN_SPLIT_RATIO = 0.8 # 80% per training, 20% per validazione

def pretrain_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Definizione delle Trasformazioni per il Dataset
    # Trasformazioni per tutto il dataset (prima dello split)
    full_dataset_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Qui potresti voler applicare RandomHorizontalFlip, RandomRotation SOLO al set di training
        # Ma ImageFolder lo carica una volta e poi lo splitta.
        # È meglio applicare queste augmentation SOLO nel DataLoader del training, come fatto prima.
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Caricamento del Dataset e Split
    print("Loading full dataset and splitting into train/validation...")
    
    # Usa ImageFolder che gestisce la struttura data/cat, data/dog
    # Questa classe è parte di torchvision.datasets e funziona esattamente con la tua struttura.
    full_dataset = datasets.ImageFolder(root=DATA_ROOT_DIR, transform=full_dataset_transforms)

    # Calcola le dimensioni dei subset
    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Dividi il dataset in modo casuale
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Se vuoi applicare le *diverse* trasformazioni (con augmentation) solo al training set
    # e solo quelle di base (senza augmentation) al validation set, devi fare un piccolo workaround:
    # 1. Carica il dataset completo con solo le trasformazioni di base (resize, toTensor, normalize).
    # 2. Esegui lo split.
    # 3. Assegna le trasformazioni specifiche (con augmentation) al train_dataset.

    # Rivediamo la sezione trasformazioni per gestire l'augmentation solo sul training set dopo lo split.
    # Versione 2: gestione delle trasformazioni dopo lo split
    
    # 1. Trasformazioni di base (solo resize, toTensor, normalize) per tutti i dati
    base_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset_for_split = datasets.ImageFolder(root=DATA_ROOT_DIR, transform=base_transforms)
    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset_for_split))
    val_size = len(full_dataset_for_split) - train_size
    train_subset, val_subset = random_split(full_dataset_for_split, [train_size, val_size])

    # Applica le trasformazioni di augmentation solo al training subset
    # Per fare ciò, dobbiamo creare un nuovo Dataset wrapper o modificare direttamente la transform del subset.
    # Il modo più pulito è creare un wrapper o modificare la classe PetImagesDataset se la usi.
    # Ma se stai usando ImageFolder e random_split, è più semplice applicare le augmentation nel __getitem__
    # O, più semplicemente, usare le augmentation solo nel DataLoader del training come segue:

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        # Qui potresti voler applicare altre trasformazioni on-the-fly, ma ImageFolder + random_split
        # rende un po' più complessa l'applicazione separata di train_transforms vs val_transforms.
        # Il modo più semplice è che le trasformazioni siano già parte del dataset all'inizio,
        # e accettare che le immagini di validazione vengano ridimensionate/normalizzate una volta,
        # e quelle di training vengano ulteriormente aumentate.
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    print(f"Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}")

    # Assicurati di definire `train_transforms` e `val_transforms` come funzioni che prendono un tensore
    # se la tua MyCNN si aspetta un tensore dopo il caricamento iniziale.
    # Dato che `full_dataset_transforms` include `ToTensor()` e `Normalize()`,
    # le immagini restituite da `train_subset` e `val_subset` saranno già tensori normalizzati.
    # Quindi, le augmentation come RandomHorizontalFlip e RandomRotation, che normalmente
    # lavorano su PIL Image, dovrebbero essere applicate *prima* di ToTensor,
    # oppure usare `torchvision.transforms.v2` che supporta trasformazioni su tensori.

    # DATASET: Come ho fatto nel primo script di pretraining (quello che ti ho dato per primo)
    # la soluzione migliore è avere la struttura /data/train/cat,dog e /data/test/cat,dog
    # Se hai solo /data/cat,dog e vuoi splittare, allora la tua PetImagesDataset dovrebbe
    # gestire lo split interno o usare ImageFolder con un approccio di subsetting.

    # Riconsideriamo l'approccio più robusto per lo split e le trasformazioni:
    # Il problema di `random_split` su `ImageFolder` con trasformazioni *già applicate*
    # è che non puoi avere trasformazioni diverse per train e val.
    # La soluzione migliore se vuoi augmentation solo sul train set è questa:

    # 1. Definisci le trasformazioni per il training (con augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Definisci le trasformazioni per la validazione (senza augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. Carica l'intero dataset con una trasformazione neutra (o minima)
    # È meglio che la tua classe PetImagesDataset faccia lo split.
    # Se hai solo data/cat, dog, e NON data/train/cat,dog, data/test/cat,dog,
    # allora la maniera più semplice è usare ImageFolder ma fare lo split dei *percorsi*
    # e poi creare due dataset separati, oppure modificare PetImagesDataset per gestire questo.

    # Data la tua affermazione "la mia cartella delle immagini contiene data -> cat- dog",
    # il modo più pulito è usare `ImageFolder` e poi creare due istanze di `Subset`
    # con le trasformazioni specifiche.

    # Ecco la revisione più pulita per la tua struttura `data/cat, dog`:

    # Inizializza l'intero dataset con trasformazioni di base (solo resize, ToTensor, Normalize)
    # Questo è lo stesso per entrambi i set (train e val) per garantire consistenza nelle dimensioni.
    # L'augmentation verrà gestita separatamente.
    full_dataset = datasets.ImageFolder(
        root=DATA_ROOT_DIR,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )

    # Calcola le dimensioni dei subset
    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Crea gli indici per lo split
    generator = torch.Generator().manual_seed(42) # Per riproducibilità
    train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)

    # Crea due subset usando gli indici
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)

    # Applica le trasformazioni specifiche ai subset.
    # Questo è il modo corretto per applicare trasformazioni diverse a train/val dopo random_split.
    # Nota: torchvision.datasets.ImageFolder.transform è applicata a PIL.Image
    # Quindi le tue lambda in `transforms.py` funzionano per i tensori, ma qui stiamo
    # lavorando con PIL.Image. Se la tua MyCNN aspetta tensori, le transform devono includere ToTensor().
    # Quello che stai usando è `torchvision.transforms.functional` che opera su tensori.
    # Ma il `transform` di `ImageFolder` opera su `PIL.Image`.

    # Per il training set, aggiungi le augmentation
    # Se PetImagesDataset restituisce già tensori, allora le transform devono operare su tensori.
    # Se ImageFolder restituisce PIL.Image, allora le transform devono operare su PIL.Image.
    # Assumiamo che `ImageFolder` restituisca PIL.Image, quindi le transform devono essere adatte.
    # La tua `MyCNN` aspetta tensori.

    # Rivediamo le trasformazioni per il pretrain_classifier.py
    # Train Transform:
    train_transform_for_classifier = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(), # Importante: PIL Image -> Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Validation Transform:
    val_transform_for_classifier = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(), # Importante: PIL Image -> Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Ora, invece di creare due Dataset separati, Pytorch offre un modo più pulito:
    # Crea l'ImageFolder due volte, con diverse trasformazioni, e poi usa gli indici.

    # Primo dataset: per il training
    train_full_dataset = datasets.ImageFolder(
        root=DATA_ROOT_DIR,
        transform=train_transform_for_classifier
    )
    
    # Secondo dataset: per la validazione
    val_full_dataset = datasets.ImageFolder(
        root=DATA_ROOT_DIR,
        transform=val_transform_for_classifier
    )

    # Usa gli indici generati prima per creare i loader
    train_loader = DataLoader(
        torch.utils.data.Subset(train_full_dataset, train_indices), # Applica train_transform_for_classifier
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4 # 0 for Windows if issues
    )

    val_loader = DataLoader(
        torch.utils.data.Subset(val_full_dataset, val_indices), # Applica val_transform_for_classifier
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4 # 0 for Windows if issues
    )

    print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")


    # Resto del codice rimane invariato (sezione 3, 4, 5, 6, 7)
    # ... (Il resto del tuo script `pretrain_classifier.py` da qui in poi) ...
    # 3. Inizializzazione del Modello, Funzione di Perdita e Ottimizzatore
    model = MyCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
        
        epoch_train_loss = running_loss / len(train_loader.dataset) # Use len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval() 
        val_running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset) # Use len(val_loader.dataset)
        epoch_val_accuracy = correct_predictions / total_predictions
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} -> Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_accuracy:.4f}")

    print("\nTraining complete.")

    os.makedirs(os.path.dirname(PRE_TRAINED_CLASSIFIER_PATH) or '.', exist_ok=True) # Assicurati che la cartella 'models' esista se il percorso è 'models/...'
    torch.save(model.state_dict(), PRE_TRAINED_CLASSIFIER_PATH)
    print(f"Pre-trained classifier saved to {PRE_TRAINED_CLASSIFIER_PATH}")

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