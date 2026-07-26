"""
Step 2 - Inferensi OCR plat nomor kendaraan menggunakan Visual Language Model (VLM)
yang dijalankan via LM Studio (SDK resmi `lmstudio-python`), sesuai referensi:
https://lmstudio.ai/docs/python/llm-prediction/image-input

Prasyarat sebelum menjalankan:
1. Buka aplikasi LM Studio (harus running, bukan cuma ter-install).
2. Load model vision, mis. `qwen2-vl-2b-instruct` (atau llava / bakllava / moondream2).
3. Nyalakan "Server" di tab Developer LM Studio (status harus "Running").

Jalankan:
    python src/infer.py

Output:
    results/raw_predictions.csv  (kolom: image, ground_truth, prediction)
"""

import os
import re
import csv

import lmstudio as lms
import kagglehub
from PIL import Image

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
DATASET_SLUG = "juanthomaswijaya/indonesian-license-plate-dataset"
SERVER_API_HOST = "127.0.0.1:1234"
MODEL_NAME = "qwen2-vl-2b-instruct"
PROMPT = "What is the license plate number shown in this image? Respond only with the plate number."
TEMP_IMAGE_PATH = "temp_resized.jpg"

# ---------------------------------------------------------------------------
# Setup dataset path & koneksi LM Studio
# ---------------------------------------------------------------------------
dataset_path = kagglehub.dataset_download(DATASET_SLUG)

TEST_IMAGE_FOLDER = os.path.join(dataset_path, "Indonesian License Plate Dataset", "images", "test")
TEST_LABEL_FOLDER = os.path.join(dataset_path, "Indonesian License Plate Dataset", "labelswithLP", "test")
YOLO_LABEL_FOLDER = os.path.join(dataset_path, "Indonesian License Plate Dataset", "labels", "test")

lms.configure_default_client(SERVER_API_HOST)
model = lms.llm(MODEL_NAME)


# ---------------------------------------------------------------------------
# Fungsi bantu
# ---------------------------------------------------------------------------
def clean_text(text):
    """Buang semua karakter non-alfanumerik, ubah ke huruf besar."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def read_ground_truth(fname):
    """Ambil teks plat nomor asli dari file label (folder labelswithLP)."""
    label_path = os.path.join(TEST_LABEL_FOLDER, os.path.splitext(fname)[0] + ".txt")
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if not lines:
        return ""
    return lines[0].split()[-1]


def crop_to_plate(image_path, fname):
    """Crop gambar ke area plat nomor memakai bounding box YOLO (kalau ada),
    supaya VLM fokus ke plat, bukan seluruh badan kendaraan."""
    yolo_path = os.path.join(YOLO_LABEL_FOLDER, os.path.splitext(fname)[0] + ".txt")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    try:
        with open(yolo_path) as f:
            line = f.readline().strip()
        _, xc, yc, bw, bh = map(float, line.split()[:5])
        xc, yc, bw, bh = xc * w, yc * h, bw * w, bh * h
        pad = 0.15
        x1 = max(0, xc - bw / 2 - bw * pad)
        y1 = max(0, yc - bh / 2 - bh * pad)
        x2 = min(w, xc + bw / 2 + bw * pad)
        y2 = min(h, yc + bh / 2 + bh * pad)
        return img.crop((x1, y1, x2, y2))
    except Exception:
        return img


def extract_plate(text):
    """Ekstrak substring yang paling mirip pola plat nomor dari jawaban mentah VLM."""
    text = text.upper()
    matches = re.findall(r"[A-Z]{1,2}\d{1,4}[A-Z]{0,3}", text)
    if matches:
        return max(matches, key=len)
    return "UNKNOWN"


def predict_plate(image_path, fname):
    """Kirim gambar (yang sudah di-crop & di-resize) ke VLM via LM Studio,
    lalu kembalikan jawaban mentah model."""
    cropped = crop_to_plate(image_path, fname)
    cropped.thumbnail((480, 480))
    cropped.save(TEMP_IMAGE_PATH, quality=90)

    image_handle = lms.prepare_image(TEMP_IMAGE_PATH)
    chat = lms.Chat()
    chat.add_user_message(PROMPT, images=[image_handle])
    prediction = model.respond(chat, config={"maxTokens": 25, "temperature": 0.1})
    return str(prediction).strip()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    image_files = sorted(
        f for f in os.listdir(TEST_IMAGE_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    total = len(image_files)
    print(f"Ditemukan {total} gambar di folder test.")

    rows = []
    errors = []

    for idx, fname in enumerate(image_files, start=1):
        img_path = os.path.join(TEST_IMAGE_FOLDER, fname)

        try:
            gt = clean_text(read_ground_truth(fname))
        except Exception as e:
            gt = ""
            errors.append(f"Ground truth gagal - {fname}: {e}")

        try:
            raw_pred = predict_plate(img_path, fname)
            pred = clean_text(extract_plate(raw_pred))
        except Exception as e:
            pred = ""
            errors.append(f"Inferensi gagal - {fname}: {e}")

        rows.append([fname, gt, pred])
        print(f"[{idx}/{total}] {fname} -> GT: {gt} | Prediksi: {pred}")

    os.makedirs("results", exist_ok=True)
    with open("results/raw_predictions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ground_truth", "prediction"])
        writer.writerows(rows)

    print(f"\nSelesai. {len(rows)} gambar diproses, disimpan di results/raw_predictions.csv")
    if errors:
        print(f"{len(errors)} error ditemukan:")
        for e in errors[:10]:
            print(" -", e)


if __name__ == "__main__":
    main()
