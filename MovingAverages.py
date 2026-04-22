
import yfinance as yf
import pandas as pd

def calculate_moving_averages(ticker, start_date, end_date):
    """
    Calculates 5-day, 20-day, 50-day, and 200-day moving averages for a given stock.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").
        start_date (str): The start date for the data (e.g., "2023-01-01").
        end_date (str): The end date for the data (e.g., "2024-01-01").

    Returns:
        pandas.DataFrame: A DataFrame containing the Adjusted Close price and the calculated moving averages.
                          Returns None if there's an error fetching data.
    """

    try:
        # Download stock data from Yahoo Finance
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"No data found for {ticker} between {start_date} and {end_date}.")
            return None

        # Calculate Moving Averages
        data['5D_MA'] = data['Close'].rolling(window=5).mean()
        data['20D_MA'] = data['Close'].rolling(window=20).mean()
        data['50D_MA'] = data['Close'].rolling(window=50).mean()
        data['200D_MA'] = data['Close'].rolling(window=200).mean()

        return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == '__main__':
    # Example usage:
    ticker_symbol = "AAPL"  # Example: Apple Inc.
    start = "2023-01-01"
    end = "2024-01-01"

    ma_data = calculate_moving_averages(ticker_symbol, start, end)

    if ma_data is not None:
        print(ma_data.tail())  # Display the last few rows with the moving averages
        # Optionally, save the data to a CSV file:
        ma_data.to_csv(f"{ticker_symbol}_moving_averages.csv")
