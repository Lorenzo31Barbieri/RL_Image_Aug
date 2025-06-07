import os
import shutil


def create_subsets():
    N = 2000

    # Cartelle sorgente
    src_root = "images/archive/PetImages"
    dst_root = "dataset_subset"

    os.makedirs(dst_root, exist_ok=True)

    for cls in ['Cat', 'Dog']:
        src_dir = os.path.join(src_root, cls)
        dst_dir = os.path.join(dst_root, cls)
        os.makedirs(dst_dir, exist_ok=True)

        count = 0
        for i in range(N*2):  # consideriamo più del necessario (perché alcune sono corrotte)
            file = os.path.join(src_dir, f"{i}.jpg")
            if os.path.exists(file):
                try:
                    with open(file, 'rb') as f:
                        f.read()  # test veloce per evitare immagini corrotte
                    shutil.copy(file, dst_dir)
                    count += 1
                    if count >= N:
                        break
                except:
                    continue

    print("Subset pronto in:", dst_root)

create_subsets()