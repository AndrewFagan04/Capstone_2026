import yfinance as yf
import time
from datetime import datetime

def monitor_market(ticker_symbol):
    print(f"Monitoring {ticker_symbol} every 15 minutes. Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # 1. Fetch the ticker object
            ticker = yf.Ticker(ticker_symbol)
            
            # 2. Get the latest 1-minute data for today
            # 'period="1d"' gets the current day's data
            # 'interval="1m"' provides the most recent minute-by-minute resolution
            data = ticker.history(period="1d", interval="1m")

            if not data.empty:
                # Get the most recent closing price from the data frame
                latest_price = data['Close'].iloc[-1]
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] {ticker_symbol} Price: ${latest_price:.2f}")
            else:
                # Useful for when the market is closed or on weekends
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No data available. Market might be closed.")

            # 3. Wait for 15 minutes (900 seconds)
            time.sleep(900)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

# Usage: Run the monitor for Apple (AAPL)
if __name__ == "__main__":
    monitor_market("AAPL")