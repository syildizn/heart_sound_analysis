import librosa
import numpy as np

def extract_mfcc(y, sr=2000, n_mfcc=13):
    """
    Bir ses dosyasından MFCC özelliklerini çıkarır.
    
    Args:
    - y: Ses dosyasının yüklenmiş verisi.
    - sr: Örnekleme oranı (default: 2000 Hz).
    - n_mfcc: Çıkarılacak MFCC bileşeni sayısı (default: 13).
    
    Returns:
    - mfccs_scaled: Ortalama MFCC değerleri.
    """
    # MFCC çıkarımı
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # Ortalama MFCC değerlerini hesapla
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    
    return mfccs_scaled
