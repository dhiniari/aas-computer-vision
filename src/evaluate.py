"""
Step 3 - Evaluasi hasil prediksi OCR memakai metrik Character Error Rate (CER).

    CER = (S + D + I) / N

    S = jumlah karakter substitusi salah
    D = jumlah karakter terhapus (deletion)
    I = jumlah karakter tersisip (insertion)
    N = jumlah karakter pada ground truth

Dihitung dengan `jiwer.cer`, yang berbasis Levenshtein distance sehingga
sudah sesuai dengan formula CER = (S + D + I) / N.

Jalankan:
    python src/evaluate.py

Input : results/raw_predictions.csv   (image, ground_truth, prediction)
Output: results/prediction_results.csv (image, ground_truth, prediction, CER_score)
"""

import csv
import pandas as pd
from jiwer import cer

RAW_INPUT = "results/raw_predictions.csv"
FINAL_OUTPUT = "results/prediction_results.csv"


def calc_cer(ground_truth: str, prediction: str) -> float:
    """CER = (S + D + I) / N. Kasus khusus: ground truth kosong."""
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
    return cer(ground_truth, prediction)


def main():
    df_raw = pd.read_csv(RAW_INPUT, dtype=str, keep_default_na=False)

    final_rows = []
    for _, row in df_raw.iterrows():
        image, gt, pred = row["image"], row["ground_truth"], row["prediction"]
        score = round(calc_cer(gt, pred), 4)
        final_rows.append([image, gt, pred, score])

    with open(FINAL_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "ground_truth", "prediction", "CER_score"])
        writer.writerows(final_rows)

    df = pd.DataFrame(final_rows, columns=["image", "ground_truth", "prediction", "CER_score"])
    avg_cer = df["CER_score"].mean()

    print(f"Hasil evaluasi disimpan di {FINAL_OUTPUT}")
    print(f"Rata-rata CER seluruh dataset ({len(df)} gambar): {avg_cer:.4f}\n")

    df_sorted = df.sort_values("CER_score")

    print("=== 3 Contoh SUKSES (CER terendah) ===")
    print(df_sorted.head(3).to_string(index=False))

    print("\n=== 3 Contoh GAGAL (CER tertinggi) ===")
    print(df_sorted.tail(3).to_string(index=False))


if __name__ == "__main__":
    main()
