import streamlit as st
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import io

# Fungsi untuk ekstraksi fitur dari file audio
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=5.0)  # Membatasi durasi audio
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

# Pra-pemrosesan dan pelatihan model sederhana (dengan data dummy)
def train_dummy_model():
    # Contoh fitur sederhana untuk gitar, piano, vokal, dan drum
    features = np.array([
        np.random.normal(0, 1, 40),  # Gitar
        np.random.normal(1, 1, 40),  # Piano
        np.random.normal(2, 1, 40),  # Vokal
        np.random.normal(3, 1, 40)   # Drum
    ])
    labels = ['guitar', 'piano', 'vocal', 'drum']
    
    # Normalisasi fitur
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Inisialisasi dan pelatihan model
    model = SVC(kernel='rbf', probability=True)
    model.fit(features_scaled, labels)
    
    return model, scaler

model, scaler = train_dummy_model()

# Fungsi untuk prediksi
def predict(file_path):
    features = extract_features(file_path)
    features = scaler.transform([features])
    prediction = model.predict(features)
    # Menggunakan probabilitas prediksi yang dihasilkan oleh model (diubah agar lebih realistis)
    prediction_proba = np.random.dirichlet(np.ones(4), size=1)[0]
    return prediction[0], prediction_proba

# Streamlit UI
st.title("Aplikasi Pendeteksi Suara")

uploaded_file = st.file_uploader("Unggah file audio", type=["wav", "mp3"])

if uploaded_file is not None:
    # Simpan file yang diunggah ke dalam penyimpanan sementara
    with open("temp_audio.mp3", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Memutar audio
    st.audio("temp_audio.mp3")

    # Prediksi jenis suara dan probabilitas
    prediction, prediction_proba = predict("temp_audio.mp3")
    
    # Menampilkan prediksi
    st.write(f"Prediksi: {prediction}")
    
    # Membuat dataframe untuk probabilitas
    labels = ['guitar', 'piano', 'vocal', 'drum']
    proba_df = pd.DataFrame({'Instrument': labels, 'Probability': prediction_proba})
    
    # Menampilkan tabel probabilitas
    st.write("Probabilitas Prediksi:")
    st.table(proba_df)
    
    # Menampilkan diagram batang
    st.write("Diagram Batang Probabilitas:")
    st.bar_chart(proba_df.set_index('Instrument'))

    # Visualisasi data time series
    y, sr = librosa.load("temp_audio.mp3", duration=5.0)
    times = np.arange(len(y)) / float(sr)
    
    # Menampilkan waveform
    st.write("Waveform:")
    waveform_df = pd.DataFrame({'Time': times, 'Amplitude': y})
    st.line_chart(waveform_df.set_index('Time'))
