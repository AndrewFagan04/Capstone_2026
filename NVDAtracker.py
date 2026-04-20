import yfinance as yf
import pandas as pd



# Fetch IT data
ticker = "AAPL" 
df = yf.download(ticker, start="2024-01-01", end="2026-04-20")

# 1. Calculate Indicators
df['SMA_Fast'] = df['Close'].rolling(window=20).mean() # 20-day
df['SMA_Slow'] = df['Close'].rolling(window=50).mean() # 50-day

# 2. Generate Signals (The "Agent's" Brain)
# Signal is 1 when Fast > Slow (Uptrend), else 0
df['Signal'] = 0.0
df.loc[df['SMA_Fast'] > df['SMA_Slow'], 'Signal'] = 1.0

# 3. Calculate Daily Returns
df['Market_Return'] = df['Close'].pct_change()
df['Strategy_Return'] = df['Market_Return'] * df['Signal'].shift(1)

# 4. Cumulative Performance
df['Portfolio_Value'] = (1 + df['Strategy_Return']).cumprod()