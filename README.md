# 📊 Calgary Crime Data Analysis & Neural Network Prediction 🔍

## 🏙️ Introduction
This project analyzes crime trends in **Calgary (2018-2024)** using data from the City of Calgary's open data portal. We leverage **data science, machine learning, and deep learning (LSTM)** to predict future crime occurrences, helping law enforcement and policymakers make informed decisions.

---

## 🚀 Project Workflow  
🔹 **Data Loading & Understanding** 📂  
🔹 **Data Preprocessing & Cleaning** 🛠️  
🔹 **Exploratory Data Analysis (EDA)** 📊  
🔹 **Neural Network Modeling (LSTM)** 🤖  
🔹 **Model Training & Optimization** 🔧  
🔹 **Crime Prediction & Visualization** 🔮  

---

## 🔍 Insights from EDA  

### 📍 **Community-Wise Crime Distribution**  
- **Most affected areas**: Beltline (11.4%), Forest Lawn (10.7%), Downtown Commercial Core (10.2%).  
- **Safest areas**: 13M (22.7%), 02K (13.6%), 02B (13.6%).  

### 🚔 **Crime Categories**  
- **Top crimes**:  
  - Theft from Vehicle **(21.7%)**  
  - Theft of Vehicle **(16.7%)**  
  - Break & Enter - Commercial **(13.8%)**  
- **Crime peaks**: 2019 recorded the highest crime rates.

### 📆 **Seasonal Trends**  
- Monthly crime patterns indicate **seasonal variations**, with certain months seeing a rise in crime.

---

## 🧠 Neural Network Model (LSTM)  

### 🔹 **Why LSTM?**  
Long Short-Term Memory (LSTM) networks are specialized for **time-series forecasting**, capturing long-term dependencies in crime trends.

### 🔹 **Model Architecture**
✅ **50 LSTM units** 🔄  
✅ **Dropout layer (prevents overfitting)** 🛑  
✅ **Dense output layer** 📉  
✅ **Optimizer: Adam** ⚡  
✅ **Loss Function: Mean Squared Error (MSE)** 🎯  
✅ **Training: 100 Epochs** 📈  

---

## 📝 Code Implementation  

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

# Load dataset
df = pd.read_csv('Calgary_Crime_Data.csv')

# Data Preprocessing
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Feature Scaling
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[['crime_count']])

# Creating sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 12  # 12 months look-back
X, y = create_sequences(df_scaled, seq_length)

# Train-Test Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(1)
])

# Compile Model
model.compile(optimizer='adam', loss='mse')

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16)

# Plot Loss Curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
