import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
import tensorflow as tf
import datetime
import os
import pickle

# NIFTY 50 stock symbols
NIFTY50_STOCKS = {
    "RELIANCE": "RELIANCE.NS", "TCS": "TCS.NS", "HDFCBANK": "HDFCBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS", "ICICIBANK": "ICICIBANK.NS", "INFOSYS": "INFOSYS.NS",
    "SBIN": "SBIN.NS", "LICI": "LICI.NS", "ITC": "ITC.NS", "HINDUNILVR": "HINDUNILVR.NS",
    "LT": "LT.NS", "HCLTECH": "HCLTECH.NS", "MARUTI": "MARUTI.NS", "SUNPHARMA": "SUNPHARMA.NS",
    "TITAN": "TITAN.NS", "ONGC": "ONGC.NS", "TATAMOTORS": "TATAMOTORS.NS", "AXISBANK": "AXISBANK.NS",
    "NESTLEIND": "NESTLEIND.NS", "KOTAKBANK": "KOTAKBANK.NS", "NTPC": "NTPC.NS",
    "ASIANPAINT": "ASIANPAINT.NS", "ULTRACEMCO": "ULTRACEMCO.NS", "DMART": "DMART.NS",
    "BAJFINANCE": "BAJFINANCE.NS", "M&M": "M&M.NS", "WIPRO": "WIPRO.NS", "JSWSTEEL": "JSWSTEEL.NS",
    "POWERGRID": "POWERGRID.NS", "LTIM": "LTIM.NS", "TECHM": "TECHM.NS", "HINDALCO": "HINDALCO.NS",
    "COALINDIA": "COALINDIA.NS", "TATASTEEL": "TATASTEEL.NS", "CIPLA": "CIPLA.NS",
    "GRASIM": "GRASIM.NS", "HDFCLIFE": "HDFCLIFE.NS", "BPCL": "BPCL.NS", "EICHERMOT": "EICHERMOT.NS",
    "SBILIFE": "SBILIFE.NS", "ADANIENT": "ADANIENT.NS", "TRENT": "TRENT.NS",
    "INDUSINDBK": "INDUSINDBK.NS", "APOLLOHOSP": "APOLLOHOSP.NS", "BAJAJFINSV": "BAJAJFINSV.NS",
    "DRREDDY": "DRREDDY.NS", "BRITANNIA": "BRITANNIA.NS", "HEROMOTOCO": "HEROMOTOCO.NS",
    "ADANIPORTS": "ADANIPORTS.NS", "DIVISLAB": "DIVISLAB.NS"
}

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

def train_stock_model(symbol, symbol_ns, epochs=10):
    print(f"Training model for {symbol}...")
    
    # Download data
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    df = yf.download(symbol_ns, start="2015-01-01", end=today)[["Open", "High", "Low", "Close", "Volume"]].dropna()
    
    if len(df) < 100:
        print(f"Insufficient data for {symbol}")
        return False
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Create sequences
    SEQ_LEN = 30
    x_all, y_all = create_sequences(scaled_data, SEQ_LEN)
    
    if len(x_all) < 50:
        print(f"Insufficient sequences for {symbol}")
        return False
    
    # Train-test split
    split = int(len(x_all) * 0.8)
    x_train, y_train = x_all[:split], y_all[:split]
    
    # Create and train model
    model = transformer_model(seq_len=SEQ_LEN, n_features=5)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.1, verbose=0)
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    os.makedirs("scalers", exist_ok=True)
    
    model.save(f"models/{symbol}_model.keras")
    with open(f"scalers/{symbol}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Model saved for {symbol}")
    return True

def train_all_stocks():
    epochs = int(os.environ.get("TF_EPOCHS", 5))
    print(f"Training {len(NIFTY50_STOCKS)} stock models with {epochs} epochs each...")
    
    success_count = 0
    for symbol, symbol_ns in NIFTY50_STOCKS.items():
        try:
            if train_stock_model(symbol, symbol_ns, epochs):
                success_count += 1
        except Exception as e:
            print(f"❌ Failed to train {symbol}: {e}")
    
    print(f"\n🎯 Training complete: {success_count}/{len(NIFTY50_STOCKS)} models trained successfully")

if __name__ == "__main__":
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
    
    train_all_stocks()
