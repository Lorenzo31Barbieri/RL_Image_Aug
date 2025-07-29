import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import Optional, Tuple, List

# --- GLOBAL CONFIGURATION ---
DEFAULT_DATA_ROOT = './data'
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
DEFAULT_IMAGE_SIZE = 32
DEFAULT_NUM_CLASSES = 10


def create_standard_preprocessing(normalize: bool = True) -> transforms.Compose:
    """
    Crea la pipeline di preprocessing standard per CIFAR-10.
    
    Args:
        normalize: Se applicare normalizzazione
    
    Returns:
        Composizione di trasformazioni
    """
    transform_list = [transforms.ToTensor()]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        )
    
    return transforms.Compose(transform_list)


def get_cifar10_test_dataset(data_root: str = DEFAULT_DATA_ROOT,
                           transform: Optional[transforms.Compose] = None,
                           download: bool = True) -> torchvision.datasets.CIFAR10:
    """
    Carica il dataset CIFAR-10 di test.
    
    Args:
        data_root: Directory root per i dati
        transform: Trasformazioni da applicare (None per default)
        download: Se scaricare il dataset se non presente
    
    Returns:
        Dataset CIFAR-10 di test
    """
    if transform is None:
        transform = create_standard_preprocessing()
    
    print(f"Loading CIFAR-10 test dataset from {data_root}")
    print(f"Transform pipeline: {transform}")
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=False,
        download=download,
        transform=transform
    )
    
    print(f"✅ Test dataset loaded: {len(test_dataset)} samples")
    return test_dataset


def get_cifar10_test_loader(data_root: str = DEFAULT_DATA_ROOT,
                          batch_size: int = 64,
                          shuffle: bool = False,
                          num_workers: int = 0,
                          transform: Optional[transforms.Compose] = None,
                          download: bool = True) -> DataLoader:
    """
    Crea un DataLoader per il dataset CIFAR-10 di test.
    
    Args:
        data_root: Directory root per i dati
        batch_size: Dimensione del batch
        shuffle: Se mescolare i dati
        num_workers: Numero di worker per il caricamento
        transform: Trasformazioni da applicare
        download: Se scaricare il dataset se non presente
    
    Returns:
        DataLoader configurato
    """
    test_dataset = get_cifar10_test_dataset(
        data_root=data_root,
        transform=transform,
        download=download
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"✅ DataLoader created: batch_size={batch_size}, shuffle={shuffle}")
    print(f"📊 Total batches: {len(test_loader)}")
    
    return test_loader


def create_sample_subset(dataset: torchvision.datasets.CIFAR10,
                        num_samples: int,
                        random_seed: Optional[int] = None) -> Subset:
    """
    Crea un subset casuale del dataset.
    
    Args:
        dataset: Dataset originale
        num_samples: Numero di campioni da selezionare
        random_seed: Seed per riproducibilità
    
    Returns:
        Subset del dataset
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    total_samples = len(dataset)
    if num_samples > total_samples:
        print(f"⚠️ Warning: Requested {num_samples} samples but dataset has only {total_samples}")
        num_samples = total_samples
    
    indices = np.random.choice(total_samples, num_samples, replace=False)
    subset = Subset(dataset, indices)
    
    print(f"📊 Created subset with {len(subset)} samples from {total_samples}")
    return subset


def get_cifar10_class_names() -> List[str]:
    """
    Restituisce i nomi delle classi CIFAR-10.
    
    Returns:
        Lista dei nomi delle classi
    """
    return [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]


def analyze_dataset_distribution(dataset: torchvision.datasets.CIFAR10) -> dict:
    """
    Analizza la distribuzione delle classi nel dataset.
    
    Args:
        dataset: Dataset da analizzare
    
    Returns:
        Dict con statistiche della distribuzione
    """
    print("🔍 Analyzing dataset distribution...")
    
    # Conta le classi
    class_counts = {}
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total_samples = len(dataset)
    class_names = get_cifar10_class_names()
    
    print(f"📊 Dataset Distribution:")
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_samples) * 100
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        print(f"  {class_name:12}: {count:5d} samples ({percentage:5.1f}%)")
    
    return {
        'class_counts': class_counts,
        'total_samples': total_samples,
        'num_classes': len(class_counts),
        'class_names': class_names
    }


def validate_dataset_compatibility(dataset: torchvision.datasets.CIFAR10) -> None:
    """
    Valida che il dataset sia compatibile con le aspettative.
    
    Args:
        dataset: Dataset da validare
    
    Raises:
        ValueError: Se il dataset non è compatibile
    """
    print("🔍 Validating dataset compatibility...")
    
    if len(dataset) == 0:
        raise ValueError("Dataset is empty")
    
    # Controlla il primo campione
    try:
        sample_image, sample_label = dataset[0]
        
        # Controlla le dimensioni dell'immagine
        if isinstance(sample_image, torch.Tensor):
            if sample_image.shape != (3, 32, 32):
                print(f"⚠️ Warning: Unexpected image shape {sample_image.shape}, expected (3, 32, 32)")
        
        # Controlla il range delle label
        if not (0 <= sample_label < DEFAULT_NUM_CLASSES):
            raise ValueError(f"Label {sample_label} out of expected range [0, {DEFAULT_NUM_CLASSES-1}]")
        
        print(f"✅ Dataset validation passed")
        print(f"📊 Sample image shape: {sample_image.shape if isinstance(sample_image, torch.Tensor) else 'PIL Image'}")
        print(f"🏷️  Sample label: {sample_label}")
        
    except Exception as e:
        raise ValueError(f"Dataset validation failed: {e}")


def create_evaluation_dataloader(data_root: str = DEFAULT_DATA_ROOT,
                               batch_size: int = 64,
                               num_samples: Optional[int] = None,
                               random_seed: Optional[int] = 42,
                               device_optimized: bool = True) -> Tuple[DataLoader, dict]:
    """
    Crea un DataLoader ottimizzato per valutazione.
    
    Args:
        data_root: Directory root per i dati
        batch_size: Dimensione del batch
        num_samples: Numero di campioni (None per tutti)
        random_seed: Seed per riproducibilità
        device_optimized: Se ottimizzare per GPU
    
    Returns:
        Tupla (DataLoader, info_dict)
    """
    # Carica dataset completo
    dataset = get_cifar10_test_dataset(data_root=data_root)
    
    # Crea subset se richiesto
    if num_samples is not None:
        dataset = create_sample_subset(dataset, num_samples, random_seed)
    
    # Analizza distribuzione
    distribution_info = analyze_dataset_distribution(dataset)
    
    # Valida compatibilità
    validate_dataset_compatibility(dataset)
    
    # Configura DataLoader
    num_workers = 4 if device_optimized and torch.cuda.is_available() else 0
    pin_memory = device_optimized and torch.cuda.is_available()
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Per valutazione, tipicamente non si mescola
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False  # Mantieni tutti i campioni
    )
    
    info = {
        'total_samples': len(dataset),
        'batch_size': batch_size,
        'num_batches': len(dataloader),
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'distribution': distribution_info
    }
    
    print(f"🚀 Evaluation DataLoader ready:")
    print(f"  📦 Batches: {info['num_batches']} × {batch_size}")
    print(f"  👥 Workers: {num_workers}")
    print(f"  📌 Pin memory: {pin_memory}")
    
    return dataloader, info


def print_data_loading_summary(dataloader: DataLoader, info: dict) -> None:
    """
    Stampa un riassunto del caricamento dati.
    
    Args:
        dataloader: DataLoader configurato
        info: Informazioni sul dataset
    """
    print(f"\n{'='*60}")
    print("DATA LOADING SUMMARY")
    print(f"{'='*60}")
    
    print(f"📊 DATASET INFO:")
    print(f"  Total samples: {info['total_samples']:,}")
    print(f"  Classes: {info['distribution']['num_classes']}")
    print(f"  Batch size: {info['batch_size']}")
    print(f"  Total batches: {info['num_batches']}")
    
    print(f"\n⚡ PERFORMANCE CONFIG:")
    print(f"  Workers: {info['num_workers']}")
    print(f"  Pin memory: {'✅' if info['pin_memory'] else '❌'}")
    print(f"  CUDA available: {'✅' if torch.cuda.is_available() else '❌'}")
    
    print(f"\n🏷️  CLASS DISTRIBUTION:")
    class_names = get_cifar10_class_names()
    for class_id, count in info['distribution']['class_counts'].items():
        percentage = (count / info['total_samples']) * 100
        class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
        print(f"  {class_name:12}: {count:5d} ({percentage:4.1f}%)")
    
    print(f"{'='*60}")


# Utility per trasformazioni specifiche per fixed augmentation
class FixedAugmentationTransform:
    """
    Applica una sequenza fissa di trasformazioni definite in _ACTIONS_MAP.
    """
    def __init__(self, action_ids_to_apply: List[int]):
        """
        Args:
            action_ids_to_apply: Lista di ID delle azioni da applicare
        """
        try:
            from transforms import _ACTIONS_MAP
            self.transforms_to_apply = []
            for action_id in action_ids_to_apply:
                if action_id in _ACTIONS_MAP:
                    self.transforms_to_apply.append(_ACTIONS_MAP[action_id][0])
                else:
                    raise ValueError(f"Action ID {action_id} not found in _ACTIONS_MAP.")
            print(f"✅ Fixed augmentation configured with {len(self.transforms_to_apply)} transforms")
        except ImportError:
            raise ImportError("Cannot import _ACTIONS_MAP from transforms module")

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Applica le trasformazioni al tensore immagine.
        
        Args:
            img_tensor: Tensore immagine input
        
        Returns:
            Tensore immagine trasformato
        """
        for transform_func in self.transforms_to_apply:
            img_tensor = transform_func(img_tensor)
        return img_tensor


def create_fixed_augmentation_transform(action_ids: List[int],
                                      normalize: bool = True) -> transforms.Compose:
    """
    Crea una pipeline di trasformazioni con augmentation fissa.
    
    Args:
        action_ids: Lista di ID delle azioni da applicare
        normalize: Se applicare normalizzazione
    
    Returns:
        Composizione di trasformazioni
    """
    transform_list = [transforms.ToTensor()]
    
    # Aggiungi augmentation fissa
    if action_ids:
        transform_list.append(FixedAugmentationTransform(action_ids))
    
    # Aggiungi normalizzazione
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
        )
    
    return transforms.Compose(transform_list)