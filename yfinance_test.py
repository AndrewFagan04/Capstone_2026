# documentation for yfinance: https://pypi.org/project/yfinance/
# geeksforgeeks for yfinance: https://www.geeksforgeeks.org/python/how-to-use-yfinance-api-with-python/

import yfinance as yf

# Get ticker data
msft = yf.Ticker("MSFT")

# Get stock info (price, sector, etc.)
print(msft.info)
print()

# Get historical market data (e.g., last 1 month)
hist = msft.history(period="1mo")
print(hist)
print()

# Get financial statements
print(msft.financials)
print()
print(msft.balance_sheet)
print()

