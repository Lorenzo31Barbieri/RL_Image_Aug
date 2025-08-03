# fixed_augmentation_evaluation.py (Updated for CIFAR10)

import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision # Per CIFAR10 dataset
from src.models.vgg import VGG
from src.environment.transforms import _ACTIONS_MAP # Importa direttamente _ACTIONS_MAP

# --- CONFIGURAZIONE GLOBALE ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- Configurazione del Dataset e Percorsi ---
DATA_ROOT_DIR = './data' 
PRE_TRAINED_CLASSIFIER_PATH = './checkpoint/ckpt.pth' # Percorso del classificatore CIFAR10
IMAGE_SIZE = 32 # CIFAR10 images are 32x32
NUM_CLASSES = 10 # CIFAR10 has 10 classes


class FixedAugmentationTransform:
    """
    Apply a fixed sequence of transformation defined in _ACTIONS_MAP.
    """
    def __init__(self, action_ids_to_apply):
        self.transforms_to_apply = []
        for action_id in action_ids_to_apply:
            if action_id in _ACTIONS_MAP:
                self.transforms_to_apply.append(_ACTIONS_MAP[action_id][0])
            else:
                raise ValueError(f"Action ID {action_id} not found in _ACTIONS_MAP.")

    def __call__(self, img_tensor):
        for transform_func in self.transforms_to_apply:
            img_tensor = transform_func(img_tensor)
        return img_tensor

# Transformations used
FIXED_AUGMENTATION_ACTION_IDS = [0, 3, 6] # Esempio: Brightness +20%, Contrast -20%, Horizontal Flip


# --- Pipeline di pre-elaborazione per il classificatore ---
# Queste trasformazioni vengono sempre applicate per prepare l'immagine al modello.
# Notare che ora non c'è Resize, e i valori di Normalizzazione sono quelli di CIFAR10.
PRE_PROCESSING_STEPS = transforms.Compose([
    transforms.ToTensor(),                       # Converti in tensore
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalizzazione CIFAR10
])

# Crea la trasformazione completa per la valutazione con fixed augmentation
FIXED_AUGMENTED_TRANSFORM = transforms.Compose([
    transforms.ToTensor(), # Converti PIL Image in Tensor
    FixedAugmentationTransform(FIXED_AUGMENTATION_ACTION_IDS), # Applica le tue trasformazioni qui!
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), # Normalizzazione CIFAR10
])


# --- Funzioni di Valutazione ---

def load_classifier_model_for_eval_fixed_aug():
    print("Loading pre-trained VGG19 CIFAR10 classifier...")
    classifier_model = VGG('VGG19').to(DEVICE) 
    
    try:
        checkpoint = torch.load(PRE_TRAINED_CLASSIFIER_PATH, map_location=DEVICE)
        
        new_state_dict = {}
        for k, v in checkpoint['net'].items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        classifier_model.load_state_dict(new_state_dict, strict=True)
        print(f"Successfully loaded classifier weights from {PRE_TRAINED_CLASSIFIER_PATH}")
        print(f"Classifier accuracy from checkpoint: {checkpoint['acc']:.2f}%")
        
    except FileNotFoundError:
        print(f"Error: Classifier .pth file not found at {PRE_TRAINED_CLASSIFIER_PATH}")
        print("Please ensure you have trained your CIFAR10 VGG model and saved it as ckpt.pth in the 'checkpoint' directory.")
        exit()
    except KeyError:
        print(f"Error: Invalid checkpoint format at {PRE_TRAINED_CLASSIFIER_PATH}. Expected 'net' key.")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading classifier: {e}")
        exit()

    classifier_model.eval()
    for param in classifier_model.parameters():
        param.requires_grad = False
    print("Classifier loaded and weights frozen.")
    return classifier_model


def evaluate_model_with_transforms(model, dataloader, device, title="Evaluation"):
    """
    Valuta le performance di un modello di classificazione su un dataloader dato.
    """
    model.eval() # Metti il modello in modalità valutazione
    all_labels = []
    all_predictions = []
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\n--- {title} ---")
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc=title):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (weighted): {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print("-" * 30)
    
    return accuracy, f1, avg_loss, conf_matrix

def main():
    """
    Funzione principale per eseguire la valutazione con fixed augmentation.
    """
    classifier_model = load_classifier_model_for_eval_fixed_aug()

    # 2. Preparazione dei Dataset e DataLoader
    # Dataset per la valutazione standard (baseline)
    standard_test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=True, transform=PRE_PROCESSING_STEPS)
    standard_test_dataloader = DataLoader(standard_test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Dataset per la valutazione con fixed augmentation
    augmented_test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT_DIR, train=False, download=True, transform=FIXED_AUGMENTED_TRANSFORM)
    augmented_test_dataloader = DataLoader(augmented_test_dataset, batch_size=64, shuffle=False, num_workers=0)

    # --- INIZIO VALUTAZIONE ---

    # Fase 1: Valutazione del Classificatore Senza Augmentation Fissa
    base_accuracy, _, _, _ = evaluate_model_with_transforms(
        classifier_model, standard_test_dataloader, DEVICE, 
        title="Classificatore Baseline (Pre-processing Standard)"
    )

    # Fase 2: Valutazione del Classificatore Con Fixed Augmentation
    fixed_aug_accuracy, _, _, _ = evaluate_model_with_transforms(
        classifier_model, augmented_test_dataloader, DEVICE,
        title=f"Classificatore con Fixed Augmentation ({[ _ACTIONS_MAP[i][1] for i in FIXED_AUGMENTATION_ACTION_IDS ]})"
    )

    # Confronto finale
    print("\n--- Confronto delle Accuratezze ---")
    print(f"Accuratezza Classificatore Baseline: {base_accuracy:.4f}")
    print(f"Accuratezza Classificatore con Fixed Augmentation: {fixed_aug_accuracy:.4f}")

    if fixed_aug_accuracy > base_accuracy:
        print(f"L'augmentation fissa ha migliorato l'accuratezza di: {(fixed_aug_accuracy - base_accuracy):.4f}")
    elif fixed_aug_accuracy < base_accuracy:
        print(f"L'augmentation fissa ha peggiorato l'accuratezza di: {(base_accuracy - fixed_aug_accuracy):.4f}")
    else:
        print("L'augmentation fissa non ha avuto un impatto significativo sull'accuratezza.")

if __name__ == '__main__':
    main()