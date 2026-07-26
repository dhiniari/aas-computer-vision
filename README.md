# OCR Plat Nomor Kendaraan Menggunakan Visual Language Model (LM Studio + Python)

## Deskripsi

Project ini merupakan implementasi **Optical Character Recognition (OCR)** pada plat nomor kendaraan Indonesia menggunakan **Visual Language Model (VLM)** yang dijalankan melalui **LM Studio** dan diintegrasikan dengan **Python**.

Program membaca gambar dari dataset **Indonesian License Plate Recognition**, mengirimkan gambar ke model multimodal pada LM Studio, menghasilkan prediksi nomor plat kendaraan, kemudian mengevaluasi hasilnya menggunakan **Character Error Rate (CER)**.

---

## Fitur

- OCR plat nomor menggunakan Visual Language Model (VLM)
- Integrasi LM Studio dengan Python
- Mendukung model:
  - Qwen2-VL-2B-Instruct
  - LLaVA
  - BakLLaVA
- Menghasilkan file prediksi dalam format CSV
- Evaluasi menggunakan Character Error Rate (CER)

---

## Dataset

Dataset yang digunakan:

**Indonesian License Plate Recognition Dataset**

https://www.kaggle.com/datasets/juanthomaswijaya/indonesian-license-plate-dataset

Dataset yang digunakan pada project ini adalah folder **test**.

---

# Struktur Project

```text
aas-computer-vision/
│
├── OCR_Plat_Nomor_VLM.ipynb
├── README.md
├── requirements.txt
├── temp_resized.jpg
│
├── src/
│   ├── download_dataset.py
│   ├── infer.py
│   └── evaluate.py
│
└── results/
    ├── raw_predictions.csv
    └── prediction_results.csv
```

---

# Instalasi

## 1. Clone Repository

```bash
git clone https://github.com/dhiniari/aas-computer-vision.git

cd aas-computer-vision
```

---

## 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Install LM Studio

Download LM Studio dari:

https://lmstudio.ai

---

## 4. Download Model VLM

Contoh model yang dapat digunakan:

- Qwen2-VL-2B-Instruct
- LLaVA
- BakLLaVA

Load model tersebut pada LM Studio.

---

## 5. Jalankan Local Server

Buka:

Developer → Local Server

Pastikan status menjadi:

```
Status : Running
```

Endpoint server:

```
http://127.0.0.1:1234
```

API Model Identifier:

```
qwen2-vl-2b-instruct
```

---

# Download Dataset

Jalankan:

```bash
python src/download_dataset.py
```

Script akan mengunduh dataset secara otomatis menggunakan **kagglehub**.

---

# Menjalankan OCR

Jalankan:

```bash
python src/infer.py
```

Program akan melakukan proses berikut:

1. Membaca seluruh gambar pada folder **test**.
2. Melakukan crop area plat nomor.
3. Mengirim gambar ke LM Studio.
4. Memberikan prompt:

> What is the license plate number shown in this image? Respond only with the plate number.

5. Menerima hasil prediksi dari model.
6. Menyimpan hasil ke:

```
results/raw_predictions.csv
```

---

# Evaluasi Character Error Rate (CER)

Jalankan:

```bash
python src/evaluate.py
```

Output:

```
results/prediction_results.csv
```

Kolom yang dihasilkan:

- image
- ground_truth
- prediction
- CER_score

---

# Rumus Character Error Rate

\[
CER = \frac{S + D + I}{N}
\]

Keterangan:

- **S** = Substitution
- **D** = Deletion
- **I** = Insertion
- **N** = Jumlah karakter ground truth

---

# Contoh Output

```
test001.jpg

Ground Truth : B1234ABC

Prediction   : B1234ABC

CER Score    : 0.000
```

---

# Hasil

Program menghasilkan:

- OCR nomor plat kendaraan
- File CSV hasil prediksi
- Nilai Character Error Rate (CER)
- Ringkasan evaluasi performa model

---

# Teknologi yang Digunakan

- Python
- LM Studio
- Visual Language Model (Qwen2-VL)
- OpenCV
- Pandas
- Jiwer
- KaggleHub
- Jupyter Notebook

---

# Author

**Dhini Ari Minarti**
**4222311022**

Program Studi Teknik Robotika

Politeknik Negeri Batam

Tahun Akademik 2025/2026
