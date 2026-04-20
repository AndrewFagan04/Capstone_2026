import streamlit as st
import yfinance as yf
import pandas as pd
import time

# Use a fragment to refresh only the data/chart section every 15 minutes (900 seconds)
@st.fragment(run_every=90)
def live_market_monitor(ticker):
    # Fetch 15m interval data (yfinance supports '15m')
    data = yf.download(ticker, period="5d", interval="15m")
    
    if not data.empty:
        st.subheader(f"Latest {ticker} Market Data")
        # Functional-style processing to get the latest state
        
        latest_price = data['Close'].iloc[-1]
        latest_price = float(latest_price)
        st.metric("Current Price", f"${latest_price:.2f}")
        
        # Display the chart
        st.line_chart(data['Close'])
        
        # Trigger your baseline agent here
        run_baseline_agent(data)

def run_baseline_agent(df):
    # Your Moving Average logic
    pass

st.title("AI Stock Agent Capstone")
live_market_monitor("AAPL")