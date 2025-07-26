import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# Importa l'architettura del tuo modello (assicurati che sia la stessa usata in main.py)
from models import VGG # <-- Assicurati che sia il modello CORRETTO (es. VGG19)

# --- Funzione per mostrare le immagini ---
def imshow(img):
    img = img / 2 + 0.5     # de-normalizza
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# --- Configurazione e Caricamento Modello ---
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Carica le stesse trasformazioni del testset usate durante l'allenamento
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Prepara il testset (verrà scaricato se non esiste)
    print('==> Preparing data for prediction..')
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=True, num_workers=0) # num_workers=0 per semplicità

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    # Inizializza il modello con la stessa architettura usata in main.py
    print('==> Building model for prediction..')
    net = VGG('VGG19') # <--- ASSICURATI CHE QUESTO CORRISPONDA AL MODELLO ALLENATO!

    net = net.to(device)

    # Carica lo stato del modello (i pesi)
    checkpoint_path = './checkpoint/ckpt.pth'
    if os.path.isdir('checkpoint') and os.path.exists(checkpoint_path):
        print(f'==> Loading checkpoint from {checkpoint_path}..')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # --- INIZIO MODIFICA CRUCIALE ---
        # Crea un nuovo state_dict senza il prefisso 'module.'
        # Questo è necessario se il modello è stato salvato da torch.nn.DataParallel
        new_state_dict = {}
        for k, v in checkpoint['net'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # Rimuovi 'module.' dal prefisso
            else:
                new_state_dict[k] = v
        net.load_state_dict(new_state_dict) # Carica il nuovo state_dict
        # --- FINE MODIFICA CRUCIALE ---

        print(f"Model loaded from epoch {checkpoint['epoch']} with accuracy {checkpoint['acc']:.2f}%")
    else:
        print("Error: No checkpoint found! Please run main.py to train and save a model first.")
        exit()

    # Metti il modello in modalità valutazione (freeza i pesi)
    net.eval()
    print('Model set to evaluation mode (weights frozen).')

    # --- Esegui Predizioni ---

    # Prendi un batch casuale di immagini dal testloader
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Mostra le immagini
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))
    imshow(torchvision.utils.make_grid(images))

    # Sposta le immagini sul dispositivo corretto
    images = images.to(device)

    # Fai la predizione (senza calcolo del gradiente)
    with torch.no_grad():
        outputs = net(images)

    # Ottieni le classi predette
    _, predicted = torch.max(outputs, 1)

    print('Predicted:   ', ' '.join(f'{classes[predicted[j]]:5s}' for j in range(len(predicted))))

    # --- Puoi anche calcolare l'accuratezza sull'intero testset ---
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'\nAccuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

    print("\nPrediction script finished.")