# Import library yang dibutuhkan
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import argparse

def preprocess_data(train_input_path, test_input_path, output_dir):
    """
    Fungsi untuk memuat data train dan test, melakukan preprocessing 
    (fit di train, transform di keduanya), dan menyimpan hasilnya.
    
    Args:
        train_input_path (str): Path ke file train.csv mentah.
        test_input_path (str): Path ke file test.csv mentah.
        output_dir (str): Folder untuk menyimpan hasil preprocessing.
    """
    print("Memulai proses preprocessing...")
    
    # Baca data train dan test
    try:
        df_train = pd.read_csv(train_input_path)
        df_test = pd.read_csv(test_input_path)
    except FileNotFoundError as e:
        print(f"Error: {e}. Pastikan path file input sudah benar.")
        return

    # --- 1. Hapus kolom tidak penting ---
    df_train = df_train.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
    df_test = df_test.drop(columns=['Unnamed: 0', 'id'], errors='ignore')
    print("Kolom 'Unnamed: 0' dan 'id' telah dihapus.")

    # --- 2. Tangani nilai hilang ---
    # Hitung median HANYA dari data train
    median_arrival_delay = df_train['Arrival Delay in Minutes'].median()
    # Isi nilai yang hilang di kedua dataset
    df_train['Arrival Delay in Minutes'].fillna(median_arrival_delay, inplace=True)
    df_test['Arrival Delay in Minutes'].fillna(median_arrival_delay, inplace=True)
    print("Nilai hilang ('Arrival Delay in Minutes') telah ditangani.")

    # --- 3. Encoding data kategorikal ---
    # Identifikasi kolom kategorikal (selain target)
    categorical_cols = df_train.select_dtypes(include=['object']).columns.tolist()
    # Pastikan 'satisfaction' ada di list jika belum
    if 'satisfaction' not in categorical_cols:
        categorical_cols.append('satisfaction')

    for col in categorical_cols:
        le = LabelEncoder()
        # Fit encoder HANYA pada data train
        df_train[col] = le.fit_transform(df_train[col])
        # Transform data test menggunakan encoder yang sudah di-fit
        # Gunakan try-except untuk menangani nilai baru di test set jika ada
        try:
            df_test[col] = le.transform(df_test[col])
        except ValueError as e:
            print(f"Warning: Nilai baru di kolom '{col}' pada data test tidak bisa di-transform. Error: {e}")
            # Opsi: beri nilai default seperti -1 atau modus
            # Buat mapping yang aman untuk nilai yang tidak dikenal
            def safe_transform(value):
                if value in le.classes_:
                    return le.transform([value])[0]
                else:
                    return -1  # Nilai default untuk kategori baru
            df_test[col] = df_test[col].map(safe_transform)

    print("Encoding kolom kategorikal selesai.")

    # --- 4. Standarisasi fitur numerik ---
    numerical_cols = df_train.select_dtypes(include=np.number).columns.tolist()
    
    # Hapus kolom target dari daftar kolom numerik yang akan di-scale
    if 'satisfaction' in numerical_cols:
        numerical_cols.remove('satisfaction')
            
    scaler = StandardScaler()
    # Fit scaler HANYA pada data train
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    # Transform data test menggunakan scaler yang sama
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])
    print("Standarisasi fitur numerik selesai.")

    # --- 5. Simpan hasil ---
    # Buat direktori output jika belum ada
    os.makedirs(output_dir, exist_ok=True)
    
    # Tentukan path file output
    train_output_path = os.path.join(output_dir, 'preprocessed_train.csv')
    test_output_path = os.path.join(output_dir, 'preprocessed_test.csv')
    
    # Simpan data yang sudah diproses
    df_train.to_csv(train_output_path, index=False)
    df_test.to_csv(test_output_path, index=False)
    
    print(f"\nPreprocessing selesai. Data bersih disimpan di:")
    print(f"- {train_output_path}")
    print(f"- {test_output_path}")

# Blok ini memungkinkan skrip untuk dijalankan dari command line
if __name__ == '__main__':
    # Setup argumen parser untuk menerima path dari command line
    parser = argparse.ArgumentParser(description="Data Preprocessing Script")
    parser.add_argument("--train_input", type=str, default="../data-raw/train.csv", help="Path ke file train.csv mentah")
    parser.add_argument("--test_input", type=str, default="../data-raw/test.csv", help="Path ke file test.csv mentah")
    parser.add_argument("--output_dir", type=str, default="data_preprocessing", help="Folder untuk menyimpan hasil")
    
    args = parser.parse_args()
    
    # Jalankan fungsi utama
    preprocess_data(args.train_input, args.test_input, args.output_dir)