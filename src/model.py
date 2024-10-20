import tensorflow as tf

def create_model(input_shape):
    """
    Ses sınıflandırma modeli (CNN tabanlı).
    
    Args:
    - input_shape: Giriş verisinin boyutu (MFCC özellik sayısı).
    
    Returns:
    - model: Keras modeli
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary sınıflandırma
    ])
    
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model
