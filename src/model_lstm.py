import tensorflow as tf

def create_lstm_model(input_shape):
    """
    Ses sınıflandırma modeli (LSTM tabanlı).
    
    Args:
    - input_shape: Giriş verisinin boyutu (MFCC özellik sayısı).
    
    Returns:
    - model: Keras modeli
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(128, input_shape=(input_shape, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary sınıflandırma
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
