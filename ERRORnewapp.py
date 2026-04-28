import streamlit as st
import pandas as pd
import ERRORnewengine
from ERRORnewengine import get_market_data, run_backtest, get_movingaverage_signals, get_rsi_signals, get_rnn_signals, SimpleRNN, train_rnn, all_agents

# 1. Page Configuration
st.set_page_config(page_title="AI Trading Lab", layout="wide")
st.title("IT Sector AI Agent Analysis")

# 2. Sidebar for Configuration
st.sidebar.header("Simulation Settings")
ticker = st.sidebar.selectbox("Select Ticker", ["AAPL", "MSFT", "NVDA"])
agent_type = st.sidebar.radio("Select AI Agent", ["Moving Average", "RSI", "RNN", "All"])
test_period = st.sidebar.slider("Backtest Period (Days)", 30, 365, 180)

# 3. Data Ingestion
data = get_market_data(ticker, period=f"{test_period}d")

# if markets are closed
if data.empty:
    st.error("No market data returned. Try a different ticker or timeframe.")




# 4. Agent Execution
if agent_type == "Moving Average":
    agent = get_movingaverage_signals(data)
if agent_type == "RSI":
    agent = get_rsi_signals(data)
if agent_type == "RNN":
    model = SimpleRNN()
    model = train_rnn(model, data['Close'].values)
    agent = get_rnn_signals(data, model)
if agent_type == "All":
    agent = 0
    model = SimpleRNN()
    model = train_rnn(model, data['Close'].values)
    MA_results, RSI_results, RNN_results = all_agents(data, model)

    all_chart_data = pd.DataFrame({
        "price": data['Close'],
        "MA_equity": MA_results['equity_curve'],
        "RSI_equity": RSI_results['equity_curve'],
        "RNN_equity": RNN_results['equity_curve']
    })

    st.line_chart(all_chart_data)

results = run_backtest(data, agent)

# 5. Visual Representation (Deliverable 2 requirement)
col1, col2, col3 = st.columns(3)

col1.metric("Total Return", f"{float(results.get('return', 0)):.2f}%")
col2.metric("Sharpe Ratio", f"{float(results.get('sharpe', 0)):.2f}")
col3.metric("Max Drawdown", f"{float(results.get('drawdown', 0)):.2f}%")

chart_data = pd.DataFrame({
    "price": data['Close'],
    "equity": results['equity_curve']
})

st.line_chart(chart_data)



