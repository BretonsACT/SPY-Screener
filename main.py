Of course. To generate that table, you can insert the following block of code into your existing script.
A logical place to add this would be right after the "Final Consensus Signal" section and before the "SPY Backtest with Signals" chart. This new section will take the last 30 days from the validation set and display each model's prediction alongside the actual recorded outcome.
Here is the complete, modified code. The new section is clearly marked.
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score

# ---------------------------
# Streamlit App
# ---------------------------
st.title("📈 SPY Tomorrow Predictor")
st.write("Logistic Regression models on SPY data (3 years).")

# Step 1: Data Collection
# Use st.cache_data to avoid re-downloading data on every interaction
@st.cache_data
def load_data(ticker, period):
    data = yf.download(ticker, period=period)
    return data

data = load_data("SPY", "3y")

dataset = data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Step 2: Feature Engineering
dataset['short_mavg'] = dataset['Close'].rolling(window=10, min_periods=1).mean()
dataset['long_mavg'] = dataset['Close'].rolling(window=60, min_periods=1).mean()

# Target variables
dataset['Sign_1'] = np.where(dataset['short_mavg'] > dataset['long_mavg'], 1.0, 0.0)
dataset['Sign_1'] = dataset['Sign_1'].shift(-10)
dataset['Sign_2'] = (np.sign(np.log(dataset['Close'] / dataset['Close'].shift(1))) > 0).astype(int)
dataset['Sign_2'] = dataset['Sign_2'].shift(-1)

# Additional features
dataset['O-C'] = dataset['Close'] - dataset['Open']
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['Log_Return'] = np.log(dataset['Close'] / dataset['Close'].shift(1))
dataset['Momentum'] = dataset['Close'].diff(1)
dataset['SMA_5'] = dataset['Close'].rolling(window=5).mean()
dataset['SMA_10'] = dataset['Close'].rolling(window=10).mean()
dataset['SMA_20'] = dataset['Close'].rolling(window=20).mean()
dataset['SMA_50'] = dataset['Close'].rolling(window=50).mean()
dataset['SMA_100'] = dataset['Close'].rolling(window=100).mean()
dataset['EMA_5'] = dataset['Close'].ewm(span=5, adjust=False).mean()
dataset['EMA_10'] = dataset['Close'].ewm(span=10, adjust=False).mean()
dataset['EMA_20'] = dataset['Close'].ewm(span=20, adjust=False).mean()
dataset['EMA_50'] = dataset['Close'].ewm(span=50, adjust=False).mean()
dataset['EMA_100'] = dataset['Close'].ewm(span=100, adjust=False).mean()

# RSI
delta = dataset['Close'].diff(1)
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
dataset['RSI'] = 100 - (100 / (1 + rs))

dataset.dropna(inplace=True)

# Step 3: Train / Validation Split
validation_size = 0.2
split_index = int(len(dataset) * (1 - validation_size))
train_df = dataset.iloc[:split_index]
validation_df = dataset.iloc[split_index:]

features = [
    'O-C', 'H-L', 'Log_Return', 'Momentum',
    'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_100',
    'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50', 'EMA_100',
    'RSI'
]

X_train = train_df[features]
Y_train_1 = train_df['Sign_1']
Y_train_2 = train_df['Sign_2']

X_validation = validation_df[features]
Y_validation_1 = validation_df['Sign_1']
Y_validation_2 = validation_df['Sign_2']

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_validation_scaled = scaler.transform(X_validation)

# Step 4: Train Models
model1 = LogisticRegression(solver='liblinear')
model1.fit(X_train_scaled, Y_train_1)

model2 = LogisticRegression(solver='liblinear')
model2.fit(X_train_scaled, Y_train_2)

# Step 5: Validation Scores
predictions1 = model1.predict(X_validation_scaled)
predictions2 = model2.predict(X_validation_scaled)

acc1 = accuracy_score(Y_validation_1, predictions1)
acc2 = accuracy_score(Y_validation_2, predictions2)

st.subheader("📊 Model Accuracy")
st.write(f"Model 1 (MA crossover): **{acc1:.2%}**")
st.write(f"Model 2 (Daily returns): **{acc2:.2%}**")

# Step 6: Predict Tomorrow
latest_features = dataset[features].iloc[-1:].copy()
latest_scaled = scaler.transform(latest_features)

# Predictions & probabilities
tomorrow_pred1 = model1.predict(latest_scaled)[0]
tomorrow_prob1 = model1.predict_proba(latest_scaled)[0][1]

tomorrow_pred2 = model2.predict(latest_scaled)[0]
tomorrow_prob2 = model2.predict_proba(latest_scaled)[0][1]

signal_map = {0: "DOWN ⬇️", 1: "UP ✅"}

st.subheader("🔮 Tomorrow's Prediction")
st.write(f"Model 1 (MA Crossover): **{signal_map[tomorrow_pred1]}** (prob={tomorrow_prob1:.2f})")
st.write(f"Model 2 (Daily Returns): **{signal_map[tomorrow_pred2]}** (prob={tomorrow_prob2:.2f})")

# Step 7: Consensus Signal
votes = [tomorrow_pred1, tomorrow_pred2]
vote_sum = sum(votes)

if vote_sum == 2:
    final_signal = "STRONG UP ✅"
elif vote_sum == 0:
    final_signal = "STRONG DOWN ⬇️"
else:
    final_signal = "NEUTRAL ⚖️"

st.subheader("📌 Final Consensus Signal")
st.write(f"**{final_signal}**")


# ---------------------------------------------------------------------------------
# --------------------------- START: NEW TABLE SECTION ----------------------------
# ---------------------------------------------------------------------------------

st.subheader("🔍 Recent Performance (Last 30 Trading Days)")

# Get the last 30 days from the validation set
recent_df = validation_df.iloc[-30:].copy()
recent_predictions1 = predictions1[-30:]
recent_predictions2 = predictions2[-30:]

# Create a DataFrame for display
results_df = pd.DataFrame(index=recent_df.index)
results_df['Actual Daily Change'] = (recent_df['Close'].pct_change() * 100).map('{:.2f}%'.format).fillna("N/A")
results_df['Model 1 Pred (MA Cross)'] = pd.Series(recent_predictions1, index=recent_df.index).map(signal_map)
results_df['Model 1 Actual'] = recent_df['Sign_1'].map(signal_map)
results_df['Model 2 Pred (Daily)'] = pd.Series(recent_predictions2, index=recent_df.index).map(signal_map)
results_df['Model 2 Actual'] = recent_df['Sign_2'].map(signal_map)

# Reorder columns for clarity
results_df = results_df[['Actual Daily Change', 'Model 2 Pred (Daily)', 'Model 2 Actual', 'Model 1 Pred (MA Cross)', 'Model 1 Actual']]

# Display the table, sorting by date descending
st.dataframe(results_df.sort_index(ascending=False))

# ---------------------------------------------------------------------------------
# ---------------------------- END: NEW TABLE SECTION -----------------------------
# ---------------------------------------------------------------------------------


# Step 8: Validation Backtest Chart with Signals
st.subheader("📉 SPY Backtest with Signals")

# Consensus for validation
consensus = []
for p1, p2 in zip(predictions1, predictions2):
    if p1 + p2 == 2:
        consensus.append(1)
    elif p1 + p2 == 0:
        consensus.append(0)
    else:
        consensus.append(np.nan)  # Neutral

# Use .loc to avoid SettingWithCopyWarning
validation_df_copy = validation_df.copy()
validation_df_copy.loc[:, 'Consensus'] = consensus
validation_df_copy.loc[:, 'Market Returns'] = validation_df_copy['Close'].pct_change()
validation_df_copy.loc[:, 'Strategy Returns'] = validation_df_copy['Market Returns'] * validation_df_copy['Consensus'].shift(1)


# Plot signals on price
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(validation_df_copy.index, validation_df_copy['Close'], label='SPY Close', color='blue')

ax.scatter(validation_df_copy.index[validation_df_copy['Consensus'] == 1],
           validation_df_copy['Close'][validation_df_copy['Consensus'] == 1],
           marker='^', color='green', label='Buy Signal', alpha=0.8)

ax.scatter(validation_df_copy.index[validation_df_copy['Consensus'] == 0],
           validation_df_copy['Close'][validation_df_copy['Consensus'] == 0],
           marker='v', color='red', label='Sell Signal', alpha=0.8)

ax.set_title("SPY Validation Period with Consensus Buy/Sell Signals")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# Step 9: Backtest Performance
st.subheader("📈 Backtest Performance")

validation_df_copy.loc[:, 'Cumulative BuyHold'] = (1 + validation_df_copy['Market Returns']).cumprod()
validation_df_copy.loc[:, 'Cumulative Strategy'] = (1 + validation_df_copy['Strategy Returns'].fillna(0)).cumprod()

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(validation_df_copy.index, validation_df_copy['Cumulative BuyHold'], label="Buy & Hold", color='black')
ax2.plot(validation_df_copy.index, validation_df_copy['Cumulative Strategy'], label="Consensus Strategy", color='orange')
ax2.set_title("Cumulative Returns: Consensus vs Buy & Hold")
ax2.set_xlabel("Date")
ax2.set_ylabel("Cumulative Growth")
ax2.legend()
st.pyplot(fig2)

st.write("Final Returns:")
st.write(f"Buy & Hold: **{validation_df_copy['Cumulative BuyHold'].iloc[-1]-1:.2%}**")
st.write(f"Consensus Strategy: **{validation_df_copy['Cumulative Strategy'].iloc[-1]-1:.2%}**")

# Optional: ROC curve for Model 1
st.subheader("📐 ROC Curve (Model 1)")
y_pred_prob = model1.predict_proba(X_validation_scaled)[:, 1]
fpr, tpr, _ = roc_curve(Y_validation_1, y_pred_prob)
roc_auc = roc_auc_score(Y_validation_1, y_pred_prob)

fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
ax3.plot([0, 1], [0, 1], 'k--')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Receiver Operating Characteristic (Model 1)')
ax3.legend(loc="lower right")
st.pyplot(fig3)

