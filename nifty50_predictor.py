import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
import tensorflow as tf
import datetime
import pickle

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.d_model = d_model

        pos = np.arange(sequence_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        pos_encoding = np.zeros((sequence_len, d_model))
        pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
        pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_len": self.sequence_len,
            "d_model": self.d_model,
        })
        return config

def create_sequences(data, seq_len):
    x, y = [], []
    for i in range(len(data) - seq_len):
        x.append(data[i:i+seq_len])
        y.append(data[i+seq_len][[0, 3]])  # Open and Close
    return np.array(x), np.array(y)

def transformer_model(seq_len, n_features, d_model=64, num_heads=4, ff_dim=128):
    inputs = layers.Input(shape=(seq_len, n_features))
    
    x = layers.Dense(d_model)(inputs)
    x = PositionalEncoding(seq_len, d_model)(x)
    
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    
    ffn_output = layers.Dense(ff_dim, activation="relu")(x)
    ffn_output = layers.Dense(d_model)(ffn_output)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(2)(x)
    
    return models.Model(inputs=inputs, outputs=outputs)

def train_nifty_model():
    print("Training NIFTY 50 index model...")
    
    # Download NIFTY 50 data
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    df = yf.download("^NSEI", start="2015-01-01", end=today)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    SEQ_LEN = 30
    x_all, y_all = create_sequences(scaled_data, SEQ_LEN)
    
    # Train-test split
    split = int(len(x_all) * 0.8)
    x_train, y_train = x_all[:split], y_all[:split]
    
    # Create and train model
    model = transformer_model(seq_len=SEQ_LEN, n_features=5)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    epochs = int(os.environ.get("TF_EPOCHS", 10))
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=1)
    # ============================
    #  TEST SET PREPARATION
    # ============================
    x_test, y_test = x_all[split:], y_all[split:]
    
    # Predict on test set
    y_pred = model.predict(x_test, verbose=0)

    # ============================
    #  INVERSE TRANSFORM
    # ============================
    n_test = y_test.shape[0]

    # prepare full arrays because scaler expects 5 features
    y_test_full = np.zeros((n_test, 5))
    y_pred_full = np.zeros((n_test, 5))

    # Fill only Open and Close (feature index 0 and 3)
    y_test_full[:, 0] = y_test[:, 0]
    y_test_full[:, 3] = y_test[:, 1]

    y_pred_full[:, 0] = y_pred[:, 0]
    y_pred_full[:, 3] = y_pred[:, 1]

    # Inverse scaling
    y_test_inv = scaler.inverse_transform(y_test_full)
    y_pred_inv = scaler.inverse_transform(y_pred_full)

    actual_open = y_test_inv[:, 0]
    actual_close = y_test_inv[:, 3]
    pred_open = y_pred_inv[:, 0]
    pred_close = y_pred_inv[:, 3]

    # ============================
    #  METRICS CALCULATION
    # ============================
    # Errors for Close price (main indicator)
    abs_error = np.abs(actual_close - pred_close)
    mse = np.mean((actual_close - pred_close) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_error)
    mape = np.mean(abs_error / actual_close) * 100
    avg_percent_error = np.mean((abs_error / actual_close) * 100)

    # Direction accuracy
    actual_dir = np.sign(np.diff(actual_close))
    pred_dir = np.sign(np.diff(pred_close))
    direction_accuracy = np.mean(actual_dir == pred_dir) * 100

    # ============================
    #  PRINT FINAL REPORT
    # ============================
    print("\n========== MODEL PERFORMANCE REPORT ==========")
    print(f"Test Samples: {n_test}")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Average % Error: {avg_percent_error:.2f}%")
    print(f"Direction Accuracy: {direction_accuracy:.2f}%")
    print("===============================================\n")

    # Save model and scaler
    model.save("nifty_index_model.keras")
    with open("nifty_index_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print("✅ NIFTY 50 model saved successfully")

if __name__ == "__main__":
    import os
    
    # GPU setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Training on GPU: {gpus[0].name}")
        except RuntimeError as e:
            print(e)
    else:
        print("Training on CPU")
    
    train_nifty_model()
