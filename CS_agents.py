import yfinance as yf
import streamlit as st
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from RNN_train import load_lstm

# FETCHING THE DATA
@st.cache_data(ttl=60)
def fetch_data(symbol):
    df = yf.download(symbol, period="1d", interval="1m")
    return df

@st.cache_resource
def get_model():
    return load_lstm()

# Mapping Yahoo Exchange codes to mcal names
EXCHANGE_MAP = {
    "NMS": "NASDAQ", "NGM": "NASDAQ", "NCM": "NASDAQ",  # NASDAQ levels
    "NYQ": "NYSE", "NYS": "NYSE", "ASE": "NYSE", "PCX": "NYSE", # NYSE/Amex/Arca
    "LSE": "LSE",    # London
    "TOR": "TSX", "VAN": "TSX", # Toronto / Venture
    "TYO": "JPX",    # Tokyo
    "HKG": "HKEX",   # Hong Kong
    "ASX": "ASX",    # Australia
    "FRA": "XETR",   # Frankfurt/Xetra
    "MIL": "BorsaItaliana",
}

@st.cache_data(ttl=60)
def get_market_calendar(symbol):
    t = yf.Ticker(symbol)
    exchange = t.fast_info.get("exchange")

    if exchange == "CCC" or "USD" in symbol:
        # It's Crypto, so it's always open
        return True 

    try:
        # Look up the mcal name
        mcal_name = EXCHANGE_MAP.get(exchange, "NYSE")
        calender = mcal.get_calendar(mcal_name)
        return calender.is_open_now()
    except Exception:
        return False


# GETTING SIGNALS FROM EACH AGENT
def get_signals(df, model, scaler):
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.loc[:, ~df.columns.duplicated()]

    # SMA
    sma_fast = df['Close'].rolling(window=5).mean()
    sma_sig = 1 if float(df['Close'].iloc[-1]) > float(sma_fast.iloc[-1]) else -1

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    rs = gain / loss.replace(0, float('nan'))
    rsi_val = float((100 - (100 / (1 + rs))).iloc[-1])

    rsi_sig = 1 if rsi_val < 40 else (-1 if rsi_val > 60 else 0)

    # RNN Logic
    lookback = 60

    if len(df) >= lookback and model is not None and scaler is not None:
        # 1. Get last 60 closes
        recent_data = df[['Close']].tail(lookback)

        # 2. Use TRAINED scaler
        scaled_data = scaler.transform(recent_data)

        # 3. Shape for LSTM
        input_data = scaled_data.reshape(1, lookback, 1)

        # 4. Predict (scaled output)
        prediction_scaled = model.predict(input_data, verbose=0)

        # 5. Convert back to real price
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]

        current_price = float(df['Close'].iloc[-1])

        # 6. Signal
        rnn_sig = 1 if prediction > current_price else -1

    else:
        rnn_sig = 0

    return {'SMA': sma_sig, 'RSI': rsi_sig, 'RNN': rnn_sig}, rsi_val

# PORTFOLIO TRACKING
def update_portfolios(current_price, signals, timestamp, pos_size_pct):
    new_entry = {'Timestamp': timestamp}
    
    for agent in ['SMA', 'RSI', 'RNN']:
        asset = st.session_state.agent_assets[agent]
        sig = signals[agent]
        
        # BUY LOGIC: Only spend a percentage of current cash
        if sig == 1 and asset['cash'] > 10:  # Check if we have at least $10
            cash_to_spend = asset['cash'] * pos_size_pct
            shares_to_buy = cash_to_spend / current_price
            asset['shares'] += shares_to_buy
            asset['cash'] -= cash_to_spend
            
            # Log the trade
            st.session_state.trade_log = pd.concat([st.session_state.trade_log, pd.DataFrame([{
                'Timestamp': timestamp, 'Agent': agent, 'Action': 'BUY', 
                'Price': current_price, 'Shares': shares_to_buy, 'Total Value': asset['cash'] + (asset['shares'] * current_price)
            }])], ignore_index=True)

        # SELL LOGIC: Sell all shares (or modify to sell a percentage)
        elif sig == -1 and asset['shares'] > 0:
            cash_received = asset['shares'] * current_price
            asset['cash'] += cash_received
            
            # Log the trade
            st.session_state.trade_log = pd.concat([st.session_state.trade_log, pd.DataFrame([{
                'Timestamp': timestamp, 'Agent': agent, 'Action': 'SELL', 
                'Price': current_price, 'Shares': asset['shares'], 'Total Value': asset['cash']
            }])], ignore_index=True)
            
            asset['shares'] = 0

        # Calculate Current Value for history
        new_entry[agent] = asset['cash'] + (asset['shares'] * current_price)


    # Update Baseline (Buy and Hold)
    if st.session_state.agent_assets['Baseline']['shares'] == (10000/150): # First run init
         st.session_state.agent_assets['Baseline']['shares'] = 10000 / current_price
    
    new_entry['Baseline'] = st.session_state.agent_assets['Baseline']['shares'] * current_price
    
    # Append to History
    st.session_state.portfolio_history = pd.concat([
        st.session_state.portfolio_history, 
        pd.DataFrame([new_entry])
    ], ignore_index=True)
