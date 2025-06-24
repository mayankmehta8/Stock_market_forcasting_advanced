import streamlit as st
import pandas as pd
import yfinance as yf
import ta
import torch
import joblib
from model import xLSTMModel
import matplotlib.pyplot as plt

# Constants
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'stoch_k', 'macd', 'macd_signal',
            'sma_20', 'ema_20', 'bb_upper', 'bb_lower',
            'atr', 'adx']
LOOKBACK = 30

# Load model + scaler
@st.cache_resource
def load_model_and_scaler():
    model = xLSTMModel(input_size=len(FEATURES))
    model.load_state_dict(torch.load("xlstm_technical_model.pt", map_location=torch.device('cpu')))
    model.eval()
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_model_and_scaler()

# Title
st.title("📈 Stock Movement Predictor (xLSTM)")
ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")

if st.button("Predict"):
    with st.spinner("🔍 Fetching and analyzing data..."):
        df = yf.download(ticker, period="6mo")
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.dropna(inplace=True)

        # Technical indicators
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
        macd = ta.trend.MACD(df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['sma_20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
        df['ema_20'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
        df.dropna(inplace=True)

        try:
            X = df[FEATURES].values
            X_scaled = scaler.transform(X)
            X_last_30 = X_scaled[-LOOKBACK:].reshape(1, LOOKBACK, -1)
            X_tensor = torch.tensor(X_last_30, dtype=torch.float32)

            with torch.no_grad():
                pred = model(X_tensor).item()
                direction = "📈 Up" if pred > 0.5 else "📉 Down"
                confidence = round(pred * 100, 2)

            st.success(f"**Prediction:** {direction} with **{confidence}%** confidence")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        st.subheader("📊 Close Price (Last 90 days)")
        st.line_chart(df["Close"].tail(90))
