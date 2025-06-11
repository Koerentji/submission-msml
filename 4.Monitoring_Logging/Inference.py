import requests
import json
import time
import random
import pandas as pd

try:
    # Menggunakan path relatif dari lokasi skrip dijalankan
    sample_df = pd.read_csv('../2.Membangun_model/data_preprocessing/preprocessed_test.csv')
    feature_columns = sample_df.drop('satisfaction', axis=1).columns.tolist()
except FileNotFoundError:
    print("Error: File 'preprocessed_test.csv' tidak ditemukan. Pastikan path relatif sudah benar.")
    print("Lokasi yang dicari: '../2.Membangun_model/data_preprocessing/preprocessed_test.csv'")
    exit()

# Definisikan URL endpoint model dan exporter
MODEL_API_URL = "http://127.0.0.1:5002/invocations"
EXPORTER_URL_PREDICTION = "http://127.0.0.1:8000/track_prediction"
EXPORTER_URL_ERROR = "http://127.0.0.1:8000/track_error"

def generate_sample_data(columns):
    """Membuat satu baris data random sebagai contoh."""
    data = {}
    for col in columns:
        # Menghasilkan nilai random antara -2 dan 2 (karena data sudah di-scale)
        data[col] = random.uniform(-2, 2)
    return data

print("Memulai script inferensi untuk menghasilkan traffic...")
while True:
    try:
        start_time = time.time()
        
        # 1. Buat data inferensi
        sample_data = generate_sample_data(feature_columns)
        inference_request = {
            "dataframe_split": {
                "columns": feature_columns,
                "data": [list(sample_data.values())]
            }
        }

        # 2. Kirim request ke model API
        response = requests.post(MODEL_API_URL, json=inference_request, timeout=5)
        response.raise_for_status() # Akan error jika status code bukan 200-299

        # 3. Hitung latensi
        end_time = time.time()
        latency = end_time - start_time
        
        # 4. Ekstrak hasil prediksi
        prediction_result = response.json()['predictions'][0]

        # 5. Kirim data metrik ke exporter
        requests.post(EXPORTER_URL_PREDICTION, json={'latency': latency, 'prediction': prediction_result})

        print(f"Prediksi sukses: {prediction_result}, Latensi: {latency:.4f}s")
    
    except requests.exceptions.RequestException as e:
        # Jika model API error, catat sebagai error prediksi
        requests.post(EXPORTER_URL_ERROR)
        print(f"Error saat menghubungi model API: {e}")

    # Jeda 2 detik sebelum mengirim request berikutnya
    time.sleep(2)