import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf

def get_market_data(ticker, period="1y", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval)
    # Basic cleaning
    data = data[['Close']].copy()
    return data

# 1. RSI Strategy (Functional Style)
def get_rsi_signals(df, period=14, overbought=70, oversold=30):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Signal: 1 (Buy) if RSI < 30, -1 (Sell) if RSI > 70
    df['Signal'] = np.where(df['RSI'] < oversold, 1, np.where(df['RSI'] > overbought, -1, 0))
    return df

# 2. Simple RNN Agent (PyTorch)
class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]) # Take the last time step
    

def run_backtest(df, initial_capital=10000):
    capital = initial_capital
    shares = 0
    equity_curve = []

    for i in range(len(df)):
        price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]

        # Simplified Logic: 1 = Buy All, -1 = Sell All
        if signal == 1 and capital > price:
            shares = capital // price
            capital -= shares * price
        elif signal == -1 and shares > 0:
            capital += shares * price
            shares = 0
        
        equity_curve.append(capital + (shares * price))
    
    return np.array(equity_curve)
