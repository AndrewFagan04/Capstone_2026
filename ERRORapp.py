import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN

# --- CONFIGURATION & UI SETUP ---
st.set_page_config(page_title="AI Trading Agent Lab", layout="wide")
st.title("AI Stock Market Agent Simulator")

# Sidebar Controls
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
timeframe = st.sidebar.selectbox("History for Backtesting", ["1d","1mo", "6mo", "1y", "2y"])
interval = "1m"
initial_capital = st.sidebar.number_input("Starting Capital ($)", value=10000)

# DATA ENGINE
@st.cache_data(ttl=60) # Cache for 1 minutes
def load_data(symbol, period):
    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    return data

data = load_data(ticker, timeframe)

# STRATEGY DEFINITIONS

def apply_sma_strategy(df, window=20):
    # Simple Moving Average
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['Signal'] = np.nan
    df.iloc[window:, df.columns.get_loc('Signal')] = np.where(
        df['Close'].iloc[window:] > df['SMA'].iloc[window:], 1, -1
    )
    
    return df

def apply_rsi_strategy(df, periods=14):
    # Relative Strength Index
    df = df.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Signal'] = 0
    # Buy if RSI < 30, Sell if RSI > 70
    df.loc[df['RSI'] < 30, 'Signal'] = 1
    df.loc[df['RSI'] > 70, 'Signal'] = -1
    return df

def train_rnn_model(df):
    # Simple RNN Strategy
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1,1))
    
    X, y = [], []
    lookback = 10
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Build Model
    model = Sequential([
        SimpleRNN(units=50, activation='relu', input_shape=(X.shape[1], 1)),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    
    # Predict
    predictions = model.predict(X)
    
    # Generate Signals: Buy if prediction > current price
    signals = np.zeros(len(df))
    # Align signals with original dataframe indices
    pred_signals = np.where(predictions[1:] > predictions[:-1], 1, -1).flatten()
    signals[lookback+1:] = pred_signals
    
    df['Signal'] = signals
    return df

# BACKTESTING ENGINE
def backtest_strategy(df, capital):
    position = 0 # 0: Cash, 1: Long, -1: Short
    cash = capital
    shares = 0
    portfolio_value = []

    for i in range(len(df)):
        current_price = df['Close'].iloc[i].item()
        signal = df['Signal'].iloc[i]

        # Execute trades
        if signal == 1 and position <= 0: # Buy
            shares = cash / current_price
            cash = 0
            position = 1
        elif signal == -1 and position >= 0: # Sell
            cash = shares * current_price
            shares = 0
            position = -1
        
        # Track Value
        current_val = cash + (shares * current_price)
        portfolio_value.append(current_val)
    
    df['Portfolio_Value'] = portfolio_value
    return df

# MAIN DASHBOARD
tab1, tab2, tab3 = st.tabs(["Market View", "Agent Backtesting", "Documentation"])

with tab1:
    st.subheader(f"Real-time Market Data: {ticker}")
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                                 low=data['Low'], close=data['Close'], name="Market"))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Strategy Performance Evaluation")
    agent_type = st.selectbox("Select AI Agent", ["SMA (Moving Average)", "RSI (Oscillator)", "RNN (Neural Network)"])
    
    if st.button("Run Simulation"):
        with st.spinner("Agent is analyzing patterns..."):
            if agent_type == "SMA (Moving Average)":
                processed_df = apply_sma_strategy(data)
            elif agent_type == "RSI (Oscillator)":
                processed_df = apply_rsi_strategy(data)
            else:
                processed_df = train_rnn_model(data)
            
            results = backtest_strategy(processed_df, initial_capital)
            
            # Metrics
            final_val = results['Portfolio_Value'].iloc[-1]
            roi = ((final_val - initial_capital) / initial_capital) * 100
            
            col1, col2 = st.columns(2)
            col1.metric("Final Portfolio Value", f"${final_val:,.2f}")
            col2.metric("Return on Investment (ROI)", f"{roi:.2f}%")

            # Chart Performance
            perf_fig = go.Figure()
            perf_fig.add_trace(go.Scatter(x=results.index, y=results['Portfolio_Value'], name="Portfolio Value", line=dict(color='gold')))
            st.plotly_chart(perf_fig, use_container_width=True)
            
            st.dataframe(results[['Close', 'Signal', 'Portfolio_Value']].tail(20))

with tab3:
    st.header("Project Documentation")
    st.markdown("""
    # System Design
    1. Data Ingestion: Uses `yfinance` to pull 15-minute interval data.
    2. Agent Logic:
        - SMA Agent: Trend-following logic.
        - RSI Agent: Mean-reversion logic.
        - RNN Agent: Deep learning time-series prediction using Keras.
    3. Backtesting: A simulated brokerage loop that tracks `Cash` vs `Shares` to calculate real-world ROI.
    
    # Findings Summary
    - SMA works best in trending markets but fails in sideways markets.
    - RSI excels in volatile, range-bound environments.
    - RNN requires more data but can pick up on non-linear patterns the others miss.
    """)