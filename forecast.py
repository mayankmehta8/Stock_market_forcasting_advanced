import torch
import pandas as pd
import numpy as np
from model import xLSTMModel  # or TransformerModel, LSTMModel
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy
import yfinance as yf
import pandas as pd
import ta

ticker = "AAPL"
start_date = "2020-01-01"
end_date = "2025-06-23"
output_file = f"Latest_{ticker}_ta_dataset.csv"

df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)
df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

df['rsi'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
df['stoch_k'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()
macd = ta.trend.MACD(close=df['Close'])
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

df['sma_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
df['ema_20'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
bb = ta.volatility.BollingerBands(close=df['Close'])
df['bb_upper'] = bb.bollinger_hband()
df['bb_lower'] = bb.bollinger_lband()

df['atr'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close']).average_true_range()
df['adx'] = ta.trend.ADXIndicator(high=df['High'], low=df['Low'], close=df['Close']).adx()
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df.dropna(inplace=True)
df.reset_index(inplace=True)
df.to_csv(output_file, index=False)
print(f"✅ Saved to {output_file}")



# Load model + scaler
model = xLSTMModel(input_size=15)
model.load_state_dict(torch.load('xlstm_technical_model.pt'))
model.eval()


scaler = joblib.load('scaler.pkl')

# Load new data (last 30 days)
df = pd.read_csv('./Latest_AAPL_ta_dataset.csv')  # must contain last 30 days
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'rsi', 'stoch_k', 'macd', 'macd_signal',
            'sma_20', 'ema_20', 'bb_upper', 'bb_lower',
            'atr', 'adx']

# Preprocess
X = df[features].values
X_scaled = scaler.transform(X)
X_last_30 = X_scaled[-30:]
X_seq = torch.tensor(X_last_30.reshape(1, 30, -1), dtype=torch.float32)

# Predict
with torch.no_grad():
    pred = model(X_seq).item()
    direction = 1 if pred > 0.5 else 0
    print(f"Prediction: {'Up' if direction else 'Down'}, Confidence: {pred:.2f}")
