import yfinance as yf
import time
import pandas as pd
from datetime import datetime

def get_prediction(ticker_symbol):
    # Fetch 5 days of 15-minute data to ensure we have enough for a 50-period SMA
    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period="5d", interval="15m")
    
    if df.empty or len(df) < 50:
        return "WAITING (Insufficient Data)", 0

    # 1. Calculate Technical Indicators (SMAs)
    # Simple Moving Average: smoothed average of the last N closing prices
    df['SMA_15'] = df['Close'].rolling(window=15).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # 2. Extract the latest values
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    current_price = latest['Close']
    sma15 = latest['SMA_15']
    sma50 = latest['SMA_50']
    
    # 3. Prediction Logic: "Golden Cross" and "Death Cross"
    # Buy if short-term trend moves above long-term trend
    if sma15 > sma50 and prev['SMA_15'] <= prev['SMA_50']:
        signal = "BUY (Golden Cross)"
    # Sell if short-term trend moves below long-term trend
    elif sma15 < sma50 and prev['SMA_15'] >= prev['SMA_50']:
        signal = "SELL (Death Cross)"
    # Maintain status based on current position
    elif sma15 > sma50:
        signal = "HOLD (Bullish Trend)"
    else:
        signal = "HOLD (Bearish Trend)"
        
    return signal, current_price

def monitor_with_predictions(ticker_symbol):
    print(f"--- Monitoring {ticker_symbol} with Trend Model ---")
    try:
        while True:
            signal, price = get_prediction(ticker_symbol)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"[{timestamp}] Price: ${price:.2f} | Signal: {signal}")
            
            # Wait for 15 minutes
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    monitor_with_predictions("AAPL")