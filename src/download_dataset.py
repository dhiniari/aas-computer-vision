"""
Step 1 - Download dataset "Indonesian License Plate Recognition" via kagglehub.

Dataset di-cache otomatis oleh kagglehub (biasanya di ~/.cache/kagglehub/...),
jadi tidak perlu disimpan manual / di-commit ke repo.

Jalankan:
    python src/download_dataset.py
"""

import os
import kagglehub

DATASET_SLUG = "juanthomaswijaya/indonesian-license-plate-dataset"


def main():
    dataset_path = kagglehub.dataset_download(DATASET_SLUG)
    print("Dataset berhasil di-download / sudah ada di cache.")
    print("Lokasi:", dataset_path)
    print("Isi folder dataset:", os.listdir(dataset_path))

    test_images = os.path.join(dataset_path, "Indonesian License Plate Dataset", "images", "test")
    test_labels = os.path.join(dataset_path, "Indonesian License Plate Dataset", "labelswithLP", "test")

    if os.path.isdir(test_images):
        print(f"\nJumlah gambar di folder test: {len(os.listdir(test_images))}")
    if os.path.isdir(test_labels):
        print(f"Jumlah label (plat nomor) di folder test: {len(os.listdir(test_labels))}")

    print("\nCatatan: path dataset di atas dipakai otomatis oleh src/infer.py")
    print("(tidak perlu disalin manual ke folder data/).")


if __name__ == "__main__":
    main()
