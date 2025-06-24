
# 📈 Stock Market Forecasting (Advanced)

![Model](https://img.shields.io/badge/model-xLSTM-blue)
![API](https://img.shields.io/badge/deployment-Streamlit-green)
![Data](https://img.shields.io/badge/data-Technical%20Indicators%20%2B%20Yahoo%20Finance-orange)
![License](https://img.shields.io/github/license/mayankmehta8/Stock_market_forcasting_advanced)

A hybrid AI-based forecasting project using technical indicators and deep learning to predict short-term (1-day) stock price direction.

> 🚀 Developed by [Mayank Mehta](https://github.com/mayankmehta8)

---

## 🔍 Project Overview

This project aims to forecast stock price movement (up/down) using:

- ✅ Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- ✅ Deep learning models like **xLSTM**, **LSTM**, and **Transformer**
- ✅ Optional FinBERT-based sentiment scoring (for future fusion)
- ✅ Deployment via **Streamlit** and **Cron automation**

---

## 🌐 Live Demo

Try the deployed model here:

👉 **[Launch Streamlit App 🚀](https://stockmarketforcastingadvanced-1234abcd9876.streamlit.app/)**

---

## 🧠 Model Architecture

Currently supported:
- `xLSTM`: Extended LSTM model trained on 30-day historical windows
- `Transformer` and `LSTM`: baseline comparisons
- Model output: binary classification (1 = price will go up, 0 = down)

---

## 📦 Features

| Feature                  | Status | Description                                              |
|--------------------------|--------|----------------------------------------------------------|
| 📉 Technical Indicators  | ✅     | Using `ta` library for 14+ indicators                    |
| 🤖 Deep Learning Model   | ✅     | Custom PyTorch models (xLSTM, Transformer)               |
| 📊 Streamlit App         | ✅     | Web interface for selecting ticker and predicting        |
| 🔄 Cron Automation       | ✅     | Automates daily forecasts on macOS                       |
| 💬 FinBERT Sentiment     | 🔜     | NLP-based news sentiment scoring (optional phase 2)      |

---

## 🧪 Demo

### ▶️ Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/mayankmehta8/Stock_market_forcasting_advanced
   cd Stock_market_forcasting_advanced
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

4. OR run local forecast:
   ```bash
   python forecast.py
   ```

---

## 📁 Project Structure

```
📦 Stock_market_forcasting_advanced/
├── model.py                  # xLSTM class
├── forecast.py               # CLI inference script
├── streamlit_app.py          # Web UI
├── scaler.pkl                # Fitted scaler
├── xlstm_technical_model.pt  # Trained PyTorch model
├── run_forecast.sh           # Cron-compatible script
└── requirements.txt
```

---

## 🛠 Cron Job (Mac Automation)

Use this to run prediction every weekday at 4 PM:

```bash
0 16 * * 1-5 /Users/yourname/Projects/Stock_market_forcasting_advanced/run_forecast.sh >> ~/forecast_log.txt 2>&1
```

---

## 📈 Indicators Used

- RSI (Relative Strength Index)
- MACD & MACD Signal
- Bollinger Bands (Upper/Lower)
- SMA/EMA (20-day)
- ATR (Average True Range)
- ADX (Directional Index)
- Stochastic Oscillator

---

## 🧠 Model Input Format

The model expects a sequence of the past **30 days** of 15 technical features as input and predicts a binary outcome (1-day ahead direction).

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## 🔗 Links

- 🔬 Author: [Mayank Mehta](https://www.linkedin.com/in/mayank-mehta-123734229/)
- 💻 GitHub: [github.com/mayankmehta8](https://github.com/mayankmehta8)
- 📊 Project: [Stock_market_forcasting_advanced](https://github.com/mayankmehta8/Stock_market_forcasting_advanced)
- 🌐 Live App: [stockmarketforcastingadvanced.streamlit.app](https://stockmarketforcastingadvanced-1234abcd9876.streamlit.app/)
