from RNN_train import train_lstm
import yfinance as yf

data = yf.download("AAPL", period="5d", interval="1m")
model, scaler = train_lstm(data)