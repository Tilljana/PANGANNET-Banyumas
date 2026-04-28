from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd
import os
import traceback

app = Flask(__name__)
# Sederhanakan CORS agar tidak bentrok dengan sistem Vercel
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURE_NAMES = [
    'Akses Jangkauan', 'Jumlah Keluarga Miskin', 'Rasio Penduduk Miskin Desil 1', 
    'Rumah tangga tanpa akses listrik', 'Produksi pangan', 'Luas lahan', 
    'Rasio Sarana Pangan', 'Persentase balita stunting', 'Proporsi Penduduk Lanjut Usia', 
    'Rasio Rumah Tangga Tanpa Air Bersih', 'Rasio Tenaga Kesehatan', 
    'Total Keluarga Beresiko Stunting dan Keluarga rentan'
]

FEATURE_MAPPING = {
    'X1': 'Akses Jangkauan', 'X2': 'Jumlah Keluarga Miskin', 'X3': 'Rasio Penduduk Miskin Desil 1',
    'X4': 'Rumah tangga tanpa akses listrik', 'X5': 'Produksi pangan', 'X6': 'Luas lahan',
    'X7': 'Rasio Sarana Pangan', 'X8': 'Persentase balita stunting', 'X9': 'Proporsi Penduduk Lanjut Usia',
    'X10': 'Rasio Rumah Tangga Tanpa Air Bersih', 'X11': 'Rasio Tenaga Kesehatan',
    'X12': 'Total Keluarga Beresiko Stunting dan Keluarga rentan'
}

# Inisialisasi model dan scaler sebagai None
model = None
scaler = None

def lazy_load():
    """Fungsi untuk memuat model hanya saat dibutuhkan (lebih stabil di Vercel)"""
    global model, scaler
    if model is None:
        try:
            model_path = os.path.join(BASE_DIR, 'best_model_XGB.pkl')
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("✓ Model loaded")
        except Exception as e:
            print(f"✗ Model Load Error: {e}")
            
    if scaler is None:
        try:
            scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✓ Scaler loaded")
        except Exception as e:
            print(f"✗ Scaler Load Error: {e}")

@app.route('/')
def home():
    # Pastikan file index.html ada di root folder kamu
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    lazy_load() # Pastikan model dimuat sebelum prediksi
    
    if not model or not scaler:
        return jsonify({'success': False, 'error': 'Server gagal memuat model ML.'}), 500
    
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'success': False, 'error': 'No data received'}), 400
        
        # Ekstraksi fitur
        features_dict = {}
        for feature_key, feature_name in FEATURE_MAPPING.items():
            val = data.get(feature_key)
            if val is None:
                return jsonify({'success': False, 'error': f'Missing {feature_key}'}), 400
            features_dict[feature_name] = float(val)
        
        # DataFrame & Prediction
        features_df = pd.DataFrame([features_dict], columns=FEATURE_NAMES)
        features_scaled = scaler.transform(features_df)
        prediction = model.predict(features_scaled)
        
        return jsonify({
            'success': True,
            'score': round(float(prediction[0]), 3),
            'confidence': 96.8, # Default value sesuai kode awalmu
            'message': 'Prediksi berhasil'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Untuk local development
if __name__ == '__main__':
    app.run(debug=True)
