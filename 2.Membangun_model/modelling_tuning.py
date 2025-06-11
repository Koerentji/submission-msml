import pandas as pd
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set MLflow tracking URI dan nama eksperimen
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Airline Satisfaction Tuning")

# Muat data train dan test yang sudah diproses
try:
    train_df = pd.read_csv('data_preprocessing/preprocessed_train.csv')
    test_df = pd.read_csv('data_preprocessing/preprocessed_test.csv')
except FileNotFoundError:
    print("Error: Pastikan file preprocessed ada di folder 'data_preprocessing/'")
    exit()

# Pisahkan fitur dan target
X_train = train_df.drop('satisfaction', axis=1)
y_train = train_df['satisfaction']
X_test = test_df.drop('satisfaction', axis=1)
y_test = test_df['satisfaction']

# Definisikan hyperparameter grid (dibuat lebih kecil agar cepat)
param_grid = {
    'n_estimators': [50],
    'max_depth': [20],
    'min_samples_leaf': [2]
}

# Inisialisasi model dan GridSearchCV
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Latih model
print("Memulai Hyperparameter Tuning...")
grid_search.fit(X_train, y_train)

# Dapatkan model dan parameter terbaik
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Parameter terbaik: {best_params}")

# Mulai MLflow Run untuk mencatat semua hasil secara manual
with mlflow.start_run(run_name="Tuned_RandomForest_Final"):
    print("Mencatat hasil terbaik ke MLflow...")

    # Evaluasi model pada data test
    y_pred = best_model.predict(X_test)

    # Hitung metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # 1. Log Parameter
    mlflow.log_params(best_params)

    # 2. Log Metrik
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # 3. Log Model
    signature = mlflow.models.infer_signature(X_train, best_model.predict(X_train))
    input_example = X_train.head(5)
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        signature=signature,
        input_example=input_example
    )

    # 4. Log Artefak Tambahan (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, "plots")
    plt.close()

    print("Proses logging manual selesai.")