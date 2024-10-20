from data_preprocessing import load_data
from feature_extraction import extract_mfcc
from model import create_model  # CNN Modeli
from model_lstm import create_lstm_model  # LSTM Modeli
import numpy as np
from sklearn.model_selection import train_test_split

# Veri Yükleme
data_dir = "data/training"
reference_file = "data/training/REFERENCE.csv"
data, labels = load_data(data_dir, reference_file)

# MFCC Özellik Çıkarma
mfcc_features = [extract_mfcc(y) for y in data]

# Verileri NumPy dizilerine çevirme
X = np.array(mfcc_features)
y = np.array(labels)

# Veriyi 3 boyutlu hale getirme (CNN ve LSTM modeli için gerekli)
X = np.expand_dims(X, axis=2)

# Eğitim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# CNN Modelini Oluşturma ve Eğitme
input_shape = X_train.shape[1]
cnn_model = create_model(input_shape)
cnn_history = cnn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# CNN Modelini Test Etme
cnn_test_loss, cnn_test_acc = cnn_model.evaluate(X_test, y_test)
print(f"CNN Model Test Doğruluğu: {cnn_test_acc}")

# LSTM Modelini Oluşturma ve Eğitme
lstm_model = create_lstm_model(input_shape)
lstm_history = lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# LSTM Modelini Test Etme
lstm_test_loss, lstm_test_acc = lstm_model.evaluate(X_test, y_test)
print(f"LSTM Model Test Doğruluğu: {lstm_test_acc}")

# Sonuçları Karşılaştırma
print(f"CNN Modeli Test Doğruluğu: {cnn_test_acc}")
print(f"LSTM Modeli Test Doğruluğu: {lstm_test_acc}")
