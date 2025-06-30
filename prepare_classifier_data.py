import os
import shutil
import random

from tqdm import tqdm

# --- Configurazione ---
SOURCE_DATA_DIR = './data'  # La tua cartella attuale: data/cat, data/dog
TARGET_DATA_DIR = './data_classifier' # La nuova cartella per il classificatore

TRAIN_RATIO = 0.70  # 70% per training
VAL_RATIO = 0.15    # 15% per validation
TEST_RATIO = 0.15   # 15% per test (70 + 15 + 15 = 100)

RANDOM_SEED = 42 # Per rendere lo split riproducibile

def prepare_data_for_classifier():
    print(f"Preparing data from '{SOURCE_DATA_DIR}' to '{TARGET_DATA_DIR}'...")

    # Assicurati che la cartella target non esista già per evitare sovrascritture accidentali
    if os.path.exists(TARGET_DATA_DIR):
        print(f"Warning: Target directory '{TARGET_DATA_DIR}' already exists. "
              "Please delete it manually or choose a different name.")
        # Se vuoi che lo script cancelli automaticamente la cartella esistente, decommenta la riga sotto:
        # shutil.rmtree(TARGET_DATA_DIR)
        # print(f"Existing directory '{TARGET_DATA_DIR}' deleted.")
        return

    # Crea le cartelle di destinazione
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'train', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'train', 'dog'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'val', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'val', 'dog'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'test', 'cat'), exist_ok=True)
    os.makedirs(os.path.join(TARGET_DATA_DIR, 'test', 'dog'), exist_ok=True)
    print("Target directories created.")

    random.seed(RANDOM_SEED) # Imposta il seed per la riproducibilità

    # Processa ogni classe (cat, dog)
    for class_name in ['cat', 'dog']:
        source_class_dir = os.path.join(SOURCE_DATA_DIR, class_name)
        if not os.path.exists(source_class_dir):
            print(f"Error: Source class directory '{source_class_dir}' not found. Skipping {class_name}.")
            continue

        images = [f for f in os.listdir(source_class_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images) # Mescola le immagini per uno split casuale

        num_images = len(images)
        num_train = int(num_images * TRAIN_RATIO)
        num_val = int(num_images * VAL_RATIO)
        # Il resto va al test per gestire eventuali arrotondamenti
        num_test = num_images - num_train - num_val

        train_files = images[:num_train]
        val_files = images[num_train : num_train + num_val]
        test_files = images[num_train + num_val : ]

        print(f"\nProcessing class: '{class_name}' (Total: {num_images} images)")
        print(f"  Train: {len(train_files)} ({TRAIN_RATIO*100}%)")
        print(f"  Validation: {len(val_files)} ({VAL_RATIO*100}%)")
        print(f"  Test: {len(test_files)} ({TEST_RATIO*100}%)")

        # Copia i file nelle nuove directory
        print(f"Copying files for '{class_name}'...")
        for img_file in tqdm(train_files, desc=f"Copying {class_name} train"):
            shutil.copy(os.path.join(source_class_dir, img_file),
                        os.path.join(TARGET_DATA_DIR, 'train', class_name, img_file))

        for img_file in tqdm(val_files, desc=f"Copying {class_name} val"):
            shutil.copy(os.path.join(source_class_dir, img_file),
                        os.path.join(TARGET_DATA_DIR, 'val', class_name, img_file))

        for img_file in tqdm(test_files, desc=f"Copying {class_name} test"):
            shutil.copy(os.path.join(source_class_dir, img_file),
                        os.path.join(TARGET_DATA_DIR, 'test', class_name, img_file))
        print(f"Finished copying files for '{class_name}'.")

    print("\nData preparation complete! New directory structure is ready at "
          f"'{TARGET_DATA_DIR}' for the VGG16 classifier.")

if __name__ == '__main__':
    prepare_data_for_classifier()