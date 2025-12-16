import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Prototype LSTM model (minor project demo)
model = Sequential([
    LSTM(32, input_shape=(40, 1)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')

def extract_mfcc(audio_path):
    audio, sr = librosa.load(audio_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, 40, 1)

def analyze_voice(audio_path):
    mfcc = extract_mfcc(audio_path)
    prediction = model.predict(mfcc)[0][0]

    if prediction > 0.6:
        return "High Stress", prediction
    elif prediction > 0.3:
        return "Medium Stress", prediction
    else:
        return "Low Stress", prediction
