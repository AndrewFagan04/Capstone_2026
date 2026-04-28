import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf

def get_market_data(ticker, period="1y", interval="1d"):
    # 1. Download data
    data = yf.download(ticker, period=period, interval=interval)

    # 2. FLATTEN MULTIINDEX (The Missing Step)
    # This removes the 'Ticker' level from the columns if it exists
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # 3. Now 'Close' is a standard Series
    data = data[['Close']].copy()

    # Optional: Ensure data is clean (no strings/objects)
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')

    return data


# 1. Moving Averages
def get_movingaverage_signals(df):
    df['5D_MA'] = df['Close'].rolling(window=5).mean()
    df['20D_MA'] = df['Close'].rolling(window=20).mean()
    df['50D_MA'] = df['Close'].rolling(window=50).mean()
    df['200D_MA'] = df['Close'].rolling(window=200).mean()

#
    # Create Signal column
    df['Signal'] = 0

    for i in range(1, len(df)):
        # BUY signal (20 crosses above 50)
        if df['20D_MA'].iloc[i] > df['50D_MA'].iloc[i] and \
           df['20D_MA'].iloc[i-1] <= df['50D_MA'].iloc[i-1]:
            df.at[df.index[i], 'Signal'] = 1

        # SELL signal (20 crosses below 50)
        elif df['20D_MA'].iloc[i] < df['50D_MA'].iloc[i] and \
             df['20D_MA'].iloc[i-1] >= df['50D_MA'].iloc[i-1]:
            df.at[df.index[i], 'Signal'] = -1
#
    return df

# 2. RSI Strategy (Functional Style)
def get_rsi_signals(df, period=14, overbought=70, oversold=30):
    delta = df['Close'].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Signal: 1 (Buy) if RSI < 30, -1 (Sell) if RSI > 70
    df['Signal'] = np.where(df['RSI'] < oversold, 1, np.where(df['RSI'] > overbought, -1, 0))
    return df

# 3. Simple RNN Agent (PyTorch) 
class SimpleRNN(nn.Module): 
    def __init__(self, input_size=1, hidden_size=32, output_size=1): 
        super(SimpleRNN, self).__init__() 
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True) 
        self.fc = nn.Linear(hidden_size, output_size) 
        
    def forward(self, x): 
        out, _ = self.rnn(x) 
        return self.fc(out[:, -1, :]) # Take the last time step
    
def train_rnn(model, prices, lookback=29, epochs=10, lr=0.001):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()

    # 🔍 Clean data (important for stock data)
    prices = np.array(prices)
    prices = prices[~np.isnan(prices)]

    # 🚨 Guard clause
    if len(prices) <= lookback:
        raise ValueError(
            f"Not enough data for RNN. Got {len(prices)} points, need > {lookback}"
        )

    X_data = []
    y_data = []

    # build sequences
    for i in range(lookback, len(prices)):
        X_data.append(prices[i - lookback:i])
        y_data.append(prices[i])

    # convert to tensors
    X_data = torch.tensor(X_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)

    # reshape for RNN
    X_data = X_data.unsqueeze(-1)  # (batch, seq, 1)
    y_data = y_data.unsqueeze(-1)

    # 🔍 Debug shapes
    print("X shape:", X_data.shape)
    print("y shape:", y_data.shape)

    for epoch in range(epochs):
        optimizer.zero_grad()

        preds = model(X_data)
        loss = loss_fn(preds, y_data)

        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    return model

def get_rnn_signals(df, model, lookback=50):
    model.eval()  # important for inference

    prices = df['Close'].values
    signals = np.zeros(len(df))

    # --- create rolling sequences like RSI rolling window ---
    for i in range(lookback, len(df)):
        window = prices[i - lookback:i]

        # normalize locally (simple, RSI-like behavior)
        window = (window - window.mean()) / (window.std() + 1e-8)

        # convert to tensor: (batch=1, seq_len, features=1)
        x = torch.tensor(window, dtype=torch.float32).view(1, lookback, 1)

        # --- RNN prediction ---
        with torch.no_grad():
            pred = model(x).item()

        # --- signal logic (RSI-style thresholding) ---
        if pred > 0.5:
            signals[i] = 1
        elif pred < -0.5:
            signals[i] = -1
        else:
            signals[i] = 0

    df['Signal'] = signals
    return df
    

def run_backtest(df, agent, initial_capital=250.00):
    capital = initial_capital
    shares = 0
    equity_curve = []
    position_size = 0.25 # use 25% capital

    if df.empty:
        return {
            "equity_curve": [],
            "return": 0,
            "sharpe": 0,
            "drawdown": 0
        }

    for i in range(len(df)):
        price = float(df['Close'].iloc[i])
        signal = float(df['Signal'].iloc[i])

        # ensure scalar
        if not isinstance(signal, (int, float)):
            signal = signal.item()

        if signal == 1 and capital > price:
            shares = int((capital * position_size) // price)
            capital -= shares * price
        elif signal == -1 and shares > 0:
            capital += shares * price
            shares = 0

        equity_curve.append(capital + (shares * price))
        
    # metrics
    total_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    if len(equity_curve) < 2:
        sharpe = 0
    else:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252 * 26)

    peak = np.maximum.accumulate(equity_curve)
    drawdown = np.min((equity_curve - peak) / peak) * 100

    return {
        "equity_curve": equity_curve,
        "return": total_return,
        "sharpe": sharpe,
        "drawdown": drawdown
    }


def all_agents(df, model):
    MA_signals = get_movingaverage_signals(df)
    RSI_signals = get_rsi_signals(df)
    RNN_signals = get_rnn_signals(df, model)

    MA_results = run_backtest(df, MA_signals)
    RSI_results = run_backtest(df, RSI_signals)
    RNN_results = run_backtest(df, RNN_signals)

    return MA_results, RSI_results, RNN_results
