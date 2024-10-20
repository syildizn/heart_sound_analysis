import librosa
import pandas as pd
import os
from feature_extraction import extract_mfcc
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(data_dir, reference_file):
    """
    Ses dosyalarını yükler ve etiketlerle birleştirir.
    
    Args:
    - data_dir: Ses dosyalarının bulunduğu dizin.
    - reference_file: Etiketlerin bulunduğu CSV dosyası.
    
    Returns:
    - data: Ses dosyaları (y) listesi.
    - labels: Etiketler listesi (1: abnormal, -1: normal).
    """
    # Etiket dosyasını yükle
    reference = pd.read_csv(reference_file, header=None)

    data = []
    labels = []

    # Her bir ses dosyasını yükleyip etiketlerle eşleştir
    for index, row in reference.iterrows():
        file_name = row[0]  # Dosya adı (örn: a0001)
        label = row[1]  # Etiket (örn: 1 veya -1)

        # Etiketlerin geçerli olup olmadığını kontrol et (sadece -1 veya 1 olmalı)
        if label not in [-1, 1]:
            print(f"Geçersiz etiket tespit edildi: {file_name} - {label}")
            continue
        
        file_path = os.path.join(data_dir, f"{file_name}.wav")
        
        try:
            # Ses dosyasını yükle (örnekleme oranı 2000 Hz)
            y, sr = librosa.load(file_path, sr=2000)
            data.append(y)
            labels.append(label)
        except Exception as e:
            print(f"Hata oluştu: {file_name} dosyası yüklenemedi. Hata: {str(e)}")
    
    return data, labels


if __name__ == "__main__":
    # Training ses dosyalarının olduğu klasör
    data_dir = "data/training"
    reference_file = "/Users/syildizn/Documents/heart_sound_analysis/data/training/newCsv.csv"
    
    # Ses dosyalarını ve etiketleri yükle
    data, labels = load_data(data_dir, reference_file)
    
    print(f"Toplam yüklü ses dosyası sayısı: {len(data)}")
    
    # MFCC özelliklerini çıkarma
    mfcc_features = [extract_mfcc(y) for y in data]
    
    # Özellikleri ve etiketleri numpy dizilerine çevir
    X = np.array(mfcc_features)
    y = np.array(labels)
    
    # sklearn'den normalizasyon için StandardScaler'ı içe aktar
    scaler = StandardScaler()
    
    # MFCC özelliklerini çıkardıktan sonra veriyi normalleştir
    X = scaler.fit_transform(X)

    # Sonuçları kontrol et
    print(f"İlk ses dosyasının MFCC özellikleri: {mfcc_features[0]}")
    print(f"İlk ses dosyasının etiketi: {labels[0]}")
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
