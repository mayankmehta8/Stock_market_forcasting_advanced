from fastapi import FastAPI
from pydantic import BaseModel
import torch, pandas as pd, joblib, yfinance as yf, ta
from model import xLSTMModel

app = FastAPI()

# Load model and scaler once
model = xLSTMModel(input_size=15)
model.load_state_dict(torch.load("xlstm_technical_model.pt"))
model.eval()
scaler = joblib.load("scaler.pkl")

FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'stoch_k', 'macd', 'macd_signal',
            'sma_20', 'ema_20', 'bb_upper', 'bb_lower',
            'atr', 'adx']

@app.get("/")
def root():
    return {"message": "xLSTM stock forecaster API is live!"}

@app.get("/predict/{ticker}")
def predict(ticker: str):
    df = yf.download(ticker, period="6mo")
    df = df.dropna()
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['stoch_k'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close']).stoch()
    macd = ta.trend.MACD(df['Close'])
    df['macd'], df['macd_signal'] = macd.macd(), macd.macd_signal()
    df['sma_20'] = ta.trend.SMAIndicator(df['Close'], 20).sma_indicator()
    df['ema_20'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_upper'], df['bb_lower'] = bb.bollinger_hband(), bb.bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['adx'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
    df.dropna(inplace=True)

    X = df[FEATURES].values
    X_scaled = scaler.transform(X)
    X_last_30 = X_scaled[-30:].reshape(1, 30, -1)
    X_tensor = torch.tensor(X_last_30, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_tensor).item()
    direction = "Up" if pred > 0.5 else "Down"
    return {"ticker": ticker, "prediction": direction, "confidence": round(pred, 4)}
