import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import time

# SETUP
st.set_page_config(page_title="AI Trading Live Lab", layout="wide")

# Sidebar
st.sidebar.header("Live Controls")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
auto_refresh = st.sidebar.checkbox("Enable Live Mode (1m Refresh)", value=False)
initial_capital = st.sidebar.number_input("Starting Capital ($)", value=10000)

# DATA ENGINE
def fetch_live_data(symbol):
    # Fetches the most recent intraday data
    try:
        # Fetching 5 days of 15m data to ensure we have enough for moving averages
        #df = yf.download(symbol, period="1d", interval="1m")
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m")
        if df.empty:
            return None
        return df
    except Exception as e:
        st.error(f"Data Fetch Error: {e}")
        return None

# AGENT LOGIC (Internal processing)
def get_agent_signals(df):
    """Calculates the latest signal for all models"""
    results = {}
    
    # 1. SMA Signal
    sma_short = df['Close'].rolling(window=10).mean()
    sma_long = df['Close'].rolling(window=30).mean()
    if sma_short.iloc[-1].item() > sma_long.iloc[-1].item():
        results['SMA'] = "BUY"
    else:
        results['SMA'] = "SELL"
        
    # 2. RSI Signal
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1].item()
    if current_rsi < 35: results['RSI'] = "BUY"
    elif current_rsi > 65: results['RSI'] = "SELL"
    else: results['RSI'] = "HOLD"

    # 3. RNN
    #ERRORS FOR ANOTHER DAY
    
    #float(df['Close'].iloc[-1]) > float(df['Close'].iloc[-2]):
     #   results['RNN_Pred'] = "UPWARD TREND"
    #else:
     #   results['RNN_Pred'] = "DOWNWARD TREND"'''
        
    return results, current_rsi

# UI LAYOUT
tab_live, tab_backtest, tab_docs = st.tabs(["Live Dashboard", "Backtesting", "Project Info"])

with tab_live:
    live_data = fetch_live_data(ticker)
    
    if live_data is not None:
        last_price = live_data['Close'].iloc[-1].item()
        last_update = live_data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
        signals, current_rsi = get_agent_signals(live_data)
        
        st.subheader(f"Live Feed: {ticker} | Last Updated: {last_update}")
        
        # Metric Row
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${last_price:.2f}")
        m2.metric("Current RSI", f"{current_rsi:.2f}")
        m3.metric("Interval", "1 Minutes")

        st.divider()
        
        # Agent Signal Cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("SMA Agent")
            st.title(signals['SMA'])

        with col2:
            st.warning("RSI Agent")
            st.title(signals['RSI'])

        with col3:
            st.success("RNN Agent")
            st.title("ERROR FOR TODAY")

        # Live Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=live_data.index, y=live_data['Close'], name="Price", line=dict(color='green')))
        fig.update_layout(title="Intraday Price Action (1m)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Could not load live data. Check your ticker symbol.")

with tab_backtest:
    st.write("Backtesting functionality is available in the main simulation engine.")

# AUTO REFRESH LOGIC
if auto_refresh:
    # This countdown creates the "live" feel
    for i in range(60, 0, -1): # 60 seconds = 1 minutes
        if not auto_refresh: break
        time.sleep(1)
        # quick pause to not break
    st.rerun()