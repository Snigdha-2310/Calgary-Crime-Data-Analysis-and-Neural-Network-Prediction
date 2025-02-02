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
