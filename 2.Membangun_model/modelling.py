import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow

# Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Airline Satisfaction Baseline")

# Muat data yang sudah diproses
df = pd.read_csv('data_preprocessing/preprocessed_train.csv')

# --- OPTIMASI: Ambil sampel kecil (misal: 5000 baris) ---
df_sample = df.sample(n=5000, random_state=42)

# Pisahkan fitur dan target dari sampel
X = df_sample.drop('satisfaction', axis=1)
y = df_sample['satisfaction']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Aktifkan autologging
mlflow.autolog(log_model_signatures=True, log_input_examples=True)

with mlflow.start_run(run_name="Baseline_RandomForest_Fast"):
    print("Memulai training model baseline dengan data sampel...")

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    print("Model baseline selesai dilatih.")