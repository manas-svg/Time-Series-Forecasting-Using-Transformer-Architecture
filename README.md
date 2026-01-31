📈 NIFTY 50 Index Prediction using Transformer Neural Network

This project implements a Transformer-based deep learning model to predict the next-day Open and Close prices of the NIFTY 50 index using historical OHLCV market data.

By leveraging self-attention, positional encoding, and time-series sequence modeling, the model learns complex market patterns and temporal dependencies that traditional models often miss.

🚀 Key Features

📊 Next-day price prediction

Predicts Open and Close prices for the following trading day

🧠 Transformer Architecture

Multi-Head Self-Attention

Positional Encoding

Feed-Forward Neural Network

🔄 Time-Series Modeling

Sliding window sequence generation (30-day lookback)

📉 Comprehensive Evaluation Metrics

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Mean Absolute Error (MAE)

Mean Absolute Percentage Error (MAPE)

Directional Accuracy

📈 Visual Analysis

Actual vs Predicted price plots using Matplotlib

💾 Model Persistence

Saves trained model and feature scaler for reuse

🛡️ Robust Data Validation

Handles empty, missing, or failed Yahoo Finance downloads gracefully

🏗️ Model Architecture
Input: OHLCV sequence (30 days × 5 features)
        ↓
Dense Projection (64 units)
        ↓
Positional Encoding
        ↓
Multi-Head Self Attention (4 heads)
        ↓
Add & Layer Normalization
        ↓
Feed Forward Network (128 → 64)
        ↓
Add & Layer Normalization
        ↓
Global Average Pooling
        ↓
Dense Output Layer (Open, Close)

📊 Dataset Details

Source: Yahoo Finance (yfinance)

Index: NIFTY 50

Symbol: ^NSEI

Date Range: 2015 – Present

Features Used

Open

High

Low

Close

Volume

🔄 Data Processing Pipeline

Download historical OHLCV data using yfinance

Validate dataset integrity (empty or failed downloads handled)

Normalize features using a scaler

Create sliding window sequences (30-day lookback)

Split data into training and testing sets

📉 Evaluation Strategy

Model performance is assessed using both error-based and trend-based metrics:

Regression Accuracy

MSE, RMSE, MAE, MAPE

Market Direction Accuracy

Measures correctness of predicted price movement direction

📈 Visualization

Actual vs Predicted Open prices

Actual vs Predicted Close prices

Clear trend comparison using Matplotlib plots

💾 Model Saving & Reusability

Trained Transformer model is saved for inference

Feature scaler is stored to ensure consistent preprocessing during deployment

⚠️ Important Notes

This project is intended for educational and research purposes only

Stock market predictions involve risk and should not be used for financial advice

🧠 Technologies Used

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib

yfinance

Scikit-learn

📌 Future Improvements

Add multi-step forecasting

Incorporate technical indicators

Experiment with encoder stacking

Deploy model via a web dashboard or API
