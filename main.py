import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# ============================
# Streamlit App Title
# ============================
st.title("SPY Trading Strategy with Confidence, 200DMA & Signals")

# ============================
# Sidebar Parameters
# ============================
st.sidebar.header("Strategy Parameters")
high_conf_thresh = st.sidebar.slider("High Confidence Threshold", 0.5, 0.9, 0.65, 0.01)
low_conf_thresh = st.sidebar.slider("Low Confidence Threshold", 0.5, 0.9, 0.55, 0.01)
leverage_factor = st.sidebar.slider("Leverage Factor (when above 200DMA)", 1.0, 3.0, 2.0, 0.1)

# ============================
# Load Data
# ============================
spy = yf.Ticker("SPY")
data = spy.history(period="5y")

# Create Target (1 if tomorrow up, 0 otherwise)
data["Tomorrow"] = data["Close"].shift(-1)
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

# Add 200-day moving average
data["200DMA"] = data["Close"].rolling(200).mean()
data["Above200DMA"] = (data["Close"] > data["200DMA"]).astype(int)

# Drop NA
data.dropna(inplace=True)

# ============================
# Features & Training
# ============================
predictors = ["Close", "Volume", "Open", "High", "Low", "Above200DMA"]

train = data.iloc[:-200]
test = data.iloc[-200:]

model = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)
model.fit(train[predictors], train["Target"])

# ============================
# Predictions
# ============================
probs = model.predict_proba(test[predictors])[:, 1]  # probability of UP

test = test.copy()
test["Prob_UP"] = probs

# Signal classification function
def classify_signal(prob):
    if prob >= high_conf_thresh:
        return "Buy Signal"
    elif prob <= (1 - high_conf_thresh):
        return "Sell Signal"
    elif low_conf_thresh <= prob < high_conf_thresh:
        return "Low Confidence Buy"
    elif (1 - high_conf_thresh) < prob <= (1 - low_conf_thresh):
        return "Low Confidence Sell"
    else:
        return "Hold"

test["Signal"] = test["Prob_UP"].apply(classify_signal)

# ============================
# Backtest Simulation
# ============================
test["Market Return"] = test["Close"].pct_change()
test["Strategy Return"] = 0.0

for i in range(len(test)):
    signal = test.iloc[i]["Signal"]
    above200 = test.iloc[i]["Above200DMA"]

    if signal == "Buy Signal":
        if above200:
            test.iloc[i, test.columns.get_loc("Strategy Return")] = test.iloc[i]["Market Return"] * leverage_factor
        else:
            test.iloc[i, test.columns.get_loc("Strategy Return")] = test.iloc[i]["Market Return"]

    elif signal == "Sell Signal":
        test.iloc[i, test.columns.get_loc("Strategy Return")] = -test.iloc[i]["Market Return"]

# ============================
# Streamlit Output
# ============================
st.subheader("Latest Prediction")
latest = test.iloc[-1]
st.write(f"Signal: **{latest['Signal']}** (Prob UP = {latest['Prob_UP']:.2f})")
st.write(f"Above 200DMA: {bool(latest['Above200DMA'])}")

st.subheader("Backtest Performance")
st.write(f"Buy & Hold Return: {test['Market Return'].sum():.4f}")
st.write(f"Strategy Return: {test['Strategy Return'].sum():.4f}")

st.line_chart(test[["Market Return", "Strategy Return"]].cumsum())

st.subheader("Signals on Price Chart")

# ============================
# Plot Price with Buy/Sell Markers
# ============================
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test.index, test["Close"], label="SPY Close", color="blue")
ax.plot(test.index, test["200DMA"], label="200DMA", color="black", linestyle="--")

# Plot signals
buy_signals = test[test["Signal"] == "Buy Signal"]
sell_signals = test[test["Signal"] == "Sell Signal"]
low_conf_buy = test[test["Signal"] == "Low Confidence Buy"]
low_conf_sell = test[test["Signal"] == "Low Confidence Sell"]

ax.scatter(buy_signals.index, buy_signals["Close"], marker="^", color="green", s=100, label="Buy")
ax.scatter(sell_signals.index, sell_signals["Close"], marker="v", color="red", s=100, label="Sell")
ax.scatter(low_conf_buy.index, low_conf_buy["Close"], marker="^", color="orange", s=70, label="Low-Conf Buy")
ax.scatter(low_conf_sell.index, low_conf_sell["Close"], marker="v", color="orange", s=70, label="Low-Conf Sell")

ax.set_title("SPY Price with Trading Signals")
ax.set_ylabel("Price")
ax.legend()

st.pyplot(fig)

st.subheader("Recent Signals")
st.dataframe(test[["Close", "Prob_UP", "Signal", "Above200DMA"]].tail(15))
