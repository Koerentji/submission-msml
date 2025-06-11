from flask import Flask, Response, request
from prometheus_client import Counter, Histogram, generate_latest
import time

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Definisikan 5 metrik untuk memenuhi kriteria "Skilled"
# 1. Counter untuk total jumlah prediksi
PREDICTION_COUNT = Counter(
    'prediction_total', 
    'Total number of predictions made by the model'
)
# 2. Counter untuk total prediksi "satisfied"
SATISFIED_COUNT = Counter(
    'prediction_satisfied_total', 
    'Total number of satisfied predictions'
)
# 3. Counter untuk total prediksi "dissatisfied"
DISSATISFIED_COUNT = Counter(
    'prediction_dissatisfied_total', 
    'Total number of neutral or dissatisfied predictions'
)
# 4. Histogram untuk latensi prediksi (waktu yang dibutuhkan untuk prediksi)
PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds', 
    'Prediction latency in seconds'
)
# 5. Counter untuk jumlah error saat prediksi
PREDICTION_ERRORS = Counter(
    'prediction_errors_total', 
    'Total number of errors encountered during prediction'
)

# Endpoint utama di '/metrics' yang akan di-scrape oleh Prometheus
@app.route('/metrics')
def metrics():
    # Mengembalikan metrik dalam format teks yang standar
    return Response(generate_latest(), mimetype='text/plain')

# Endpoint ini akan dipanggil oleh inference.py setelah prediksi berhasil
@app.route('/track_prediction', methods=['POST'])
def track_prediction():
    data = request.json
    latency = data.get('latency')
    prediction = data.get('prediction')

    # Tingkatkan counter dan observasi histogram
    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(latency)
    
    # Asumsi 'satisfied' di-encode menjadi 1, dan 'dissatisfied' menjadi 0
    if prediction == 1:
        SATISFIED_COUNT.inc()
    else:
        DISSATISFIED_COUNT.inc()

    return "OK"

# Endpoint ini akan dipanggil oleh inference.py jika terjadi error
@app.route('/track_error', methods=['POST'])
def track_error():
    PREDICTION_ERRORS.inc()
    return "OK"

if __name__ == '__main__':
    # Jalankan server Flask di port 8000, bisa diakses dari mana saja
    app.run(host='0.0.0.0', port=8000)