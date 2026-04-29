import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

import sys
import os

# Adds the current directory to the path so it can find your modules
sys.path.append(os.path.dirname(__file__))

from CS_agents import get_model, fetch_data, update_portfolios, get_signals
from RNN_train import load_lstm, train_lstm

# INITIALIZATION
st.set_page_config(page_title="Stock Market AI Agent Trackers", layout="wide")
model, scaler = get_model()


# Initialize Session State to track portfolio history across refreshes
if 'portfolio_history' not in st.session_state:
    st.session_state.portfolio_history = pd.DataFrame(columns=['Timestamp', 'SMA', 'RSI', 'RNN', 'Baseline'])
if 'trade_log' not in st.session_state:
    st.session_state.trade_log = pd.DataFrame(columns=['Timestamp', 'Agent', 'Action', 'Price', 'Shares', 'Total Value'])
if 'agent_assets' not in st.session_state:
    # Each agent starts with $10,000 cash and 0 shares
    start_cap = 10000
    st.session_state.agent_assets = {
        'SMA': {'cash': start_cap, 'shares': 0},
        'RSI': {'cash': start_cap, 'shares': 0},
        'RNN': {'cash': start_cap, 'shares': 0},
        'Baseline': {'cash': 0, 'shares': start_cap / 150} # Simplified baseline
    }


# SIDEBAR CONTROLS
st.sidebar.header("Live Trading Console")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_cap = st.sidebar.number_input("Agent's Starting Cash ($)", value=10000)
auto_refresh = st.sidebar.checkbox("Enable Live Updates", value=False)
live_data = st.sidebar.slider("Live Updates by minutes", 1, 30, 15)

if st.sidebar.button("Reset Portfolios"):
    st.session_state.portfolio_history = pd.DataFrame(columns=['Timestamp', 'SMA', 'RSI', 'RNN', 'Baseline'])
    del st.session_state.agent_assets
    del st.session_state.trade_log
    st.rerun()

if st.sidebar.button("Refresh RNN Model (Last 7 days)"):
    train_data = yf.download(ticker, period="7d", interval="1m")
    model, scaler = train_lstm(train_data)

st.sidebar.header("Risk Management")
pos_size_pct = st.sidebar.slider("Position Size (% of Cash)", 5, 100, 20) / 100


# UI LAYOUT
st.title("Real-Time Agent Performance")

data = fetch_data(ticker)

# Checks if market is closed/open
t = yf.Ticker(ticker)
status = t.info.get('marketState')
# Common states: 'REGULAR', 'CLOSED', 'PRE', 'POST', 'PREPRE'


if not data.empty:
    # Flatten Multi-Index columns so the chart can find 'Open', 'Close', etc.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    try:
        data.index = data.index.tz_convert('America/New_York')
    except TypeError:
        data.index = data.index.tz_localize('UTC').tz_convert('America/New_York')
    
    st.subheader(f"The market is currently in {status} trading hours.")

    current_p = data['Close'].iloc[-1].item()
    last_ts = data.index[-1]
    model, scaler = load_lstm()
    signals, rsi_val = get_signals(data, model, scaler)

    # Run logic update
    update_portfolios(current_p, signals, last_ts, pos_size_pct)
    
    # ROW 1: PRICE CHART
    st.subheader(f"Live Market: {ticker}")
    fig_price = go.Figure()
    fig_price.add_trace(go.Candlestick(x=data.index, open=data['Open'], high=data['High'], 
                                       low=data['Low'], close=data['Close'], name="Price"))
    fig_price.update_layout(height=400, template="plotly_dark", margin=dict(l=20, r=20, t=20, b=20), yaxis_title="Stock Value ($)")
    st.plotly_chart(fig_price, use_container_width=True)

    # Ledgers for each Agent
    with st.expander("Agent Internal Ledgers", expanded=True):
        ledger_data = []
        for agent, assets in st.session_state.agent_assets.items():
            # Calculate current total value for the table
            total_val = assets['cash'] + (assets['shares'] * current_p)
            ledger_data.append({
                "Agent": agent,
                "Cash": f"${assets['cash']:,.2f}",
                "Shares Owned": f"{assets['shares']:.4f}",
                "Total Value": f"${total_val:,.2f}"
            })
        st.table(ledger_data)

    # ROW 2: AGENT PORTFOLIO CHART
    st.subheader("Agent Performance Comparison (Portfolio Value)")
    hist = st.session_state.portfolio_history
    
    fig_port = go.Figure()
    colors = {'SMA': 'green', 'RSI': 'red', 'RNN': 'blue', 'Baseline': 'yellow'}
    
    for agent in ['SMA', 'RSI', 'RNN', 'Baseline']:
        fig_port.add_trace(go.Scatter(x=hist['Timestamp'], y=hist[agent], 
                                      mode='lines+markers', name=agent,
                                      line=dict(color=colors[agent], width=3)))
    
    fig_port.update_layout(height=400, template="plotly_dark", yaxis_title="Portfolio Value ($)")
    st.plotly_chart(fig_port, use_container_width=True)

    # ROW 3: LIVE SIGNALS
    cols = st.columns(4)
    for i, agent in enumerate(['SMA', 'RSI', 'RNN']):
        with cols[i]:
            sig_text = "BUY" if signals[agent] == 1 else ("SELL" if signals[agent] == -1 else "HOLD")
            st.metric(f"{agent} Action", sig_text)
            st.caption(f"Current Value: ${hist[agent].iloc[-1]:,.2f}")
    
    with cols[3]:
        st.metric("Market Price", f"${current_p:.2f}")
        st.caption(f"RSI Indicator: {rsi_val:.2f}")

    st.subheader("Recent Trade Activity")
    if not st.session_state.trade_log.empty:
        # Shows the last 10 trades, most recent first
        st.dataframe(st.session_state.trade_log.tail(10).sort_values(by='Timestamp', ascending=False), use_container_width=True)
    else:
        st.info("No trades executed yet.")

else:
    st.error("Failed to retrieve market data. Please check the ticker symbol.")

# NEW AUTO-REFRESH LOGIC
if auto_refresh:
    # interval is in milliseconds (60000ms = 1 minute)
    # The 'key' ensures Streamlit tracks this specific timer
    st_autorefresh(interval=live_data*60000, key="trading_update_timer")