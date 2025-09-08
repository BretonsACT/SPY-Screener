import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# ============================
# Streamlit App Title
# ============================
st.title("SPY Ensemble Strategy (Multi-Model, Confidence, 200DMA)")

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

# Target variable
data["Tomorrow"] = data["Close"].shift(-1)
data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

# Add 200DMA
data["200DMA"] = data["Close"].rolling(200).mean()
data["Above200DMA"] = (data["Close"] > data["200DMA"]).astype(int)

# Add derived features (technical indicators)
data["O-C"] = data["Close"] - data["Open"]
data["H-L"] = data["High"] - data["Low"]
data["Log_Return"] = np.log(data["Close"] / data["Close"].shift(1))
data["Momentum"] = data["Close"].diff(1)
data["SMA_5"] = data["Close"].rolling(window=5).mean()
data["SMA_10"] = data["Close"].rolling(window=10).mean()
data["SMA_20"] = data["Close"].rolling(window=20).mean()
data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()

# RSI
delta = data["Close"].diff(1)
gain = delta.where(delta > 0, 0).rolling(window=14).mean()
loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
rs = gain / loss
data["RSI"] = 100 - (100 / (1 + rs))

data.dropna(inplace=True)

# ============================
# Feature Sets
# ============================
basic_features = ["Close", "Volume", "Open", "High", "Low", "Above200DMA"]
derived_features = ["O-C", "H-L", "Log_Return", "Momentum", "SMA_5", "SMA_10", "SMA_20", "EMA_10", "EMA_20", "RSI"]

train = data.iloc[:-200]
test = data.iloc[-200:]

# ============================
# Train Models
# ============================
# Logistic Regression on basic features
scaler = StandardScaler()
X_train_lr = scaler.fit_transform(train[basic_features])
X_test_lr = scaler.transform(test[basic_features])

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_lr, train["Target"])
lr_preds = lr.predict(X_test_lr)
lr_probs = lr.predict_proba(X_test_lr)[:, 1]
lr_acc = accuracy_score(test["Target"], lr_preds)

# Random Forest on basic features
rf_basic = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)
rf_basic.fit(train[basic_features], train["Target"])
rf_preds = rf_basic.predict(test[basic_features])
rf_probs = rf_basic.predict_proba(test[basic_features])[:, 1]
rf_acc = accuracy_score(test["Target"], rf_preds)

# Random Forest on derived features
rf_derived = RandomForestClassifier(n_estimators=200, min_samples_split=100, random_state=1)
rf_derived.fit(train[derived_features], train["Target"])
rf2_preds = rf_derived.predict(test[derived_features])
rf2_probs = rf_derived.predict_proba(test[derived_features])[:, 1]
rf2_acc = accuracy_score(test["Target"], rf2_preds)

# ============================
# Weighted Ensemble
# ============================
total_acc = lr_acc + rf_acc + rf2_acc
lr_w = lr_acc / total_acc
rf_w = rf_acc / total_acc
rf2_w = rf2_acc / total_acc

test = test.copy()
test["Prob_UP"] = (lr_probs * lr_w) + (rf_probs * rf_w) + (rf2_probs * rf2_w)

# ============================
# Signal classification with confidence band
# ============================
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
# Backtest with 200DMA leverage
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
st.subheader("Model Accuracies & Weights")
st.write(f"Logistic Regression Accuracy: {lr_acc:.2%} → Weight: {lr_w:.2f}")
st.write(f"Random Forest (Basic) Accuracy: {rf_acc:.2%} → Weight: {rf_w:.2f}")
st.write(f"Random Forest (Derived) Accuracy: {rf2_acc:.2%} → Weight: {rf2_w:.2f}")

st.subheader("Latest Prediction")
latest = test.iloc[-1]
st.write(f"Signal: **{latest['Signal']}** (Prob UP = {latest['Prob_UP']:.2f})")
st.write(f"Above 200DMA: {bool(latest['Above200DMA'])}")

st.subheader("Backtest Performance")
st.write(f"Buy & Hold Return: {test['Market Return'].sum():.4f}")
st.write(f"Strategy Return: {test['Strategy Return'].sum():.4f}")

st.line_chart(test[["Market Return", "Strategy Return"]].cumsum())

st.subheader("Signals on Price Chart")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(test.index, test["Close"], label="SPY Close", color="blue")
ax.plot(test.index, test["200DMA"], label="200DMA", color="black", linestyle="--")

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
