import yfinance as yf

ticker = yf.Ticker("AAPL")
status = ticker.info.get('marketState')

print(f"Current Market State: {status}")

# Common states: 'REGULAR', 'CLOSED', 'PRE', 'POST', 'PREPRE'
if status == 'REGULAR':
    print("The market is currently in regular trading hours.")