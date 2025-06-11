import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow

# MLflow autolog akan menangkap semua parameter dan metrik
mlflow.autolog()

with mlflow.start_run():
    # Muat data dari path relatif di dalam proyek
    df = pd.read_csv('../data_preprocessing/preprocessed_train.csv')

    # Pisahkan fitur dan target
    X = df.drop('satisfaction', axis=1)
    y = df['satisfaction']

    # Inisialisasi dan latih model
    # Parameter ini bisa kita override dari MLproject file jika perlu
    rf = RandomForestClassifier(n_estimators=50, max_depth=20, min_samples_leaf=2, random_state=42)
    rf.fit(X, y)

    print("Model training complete.")