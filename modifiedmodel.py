import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, fbeta_score
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from pickle import dump, load

# Disable TF v2 behavior if TF v1
if tf.__version__.startswith("1."):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# Load your datasets
stock_name = 'MSFT'
df = pd.read_csv('Dataset/stock_tweets.csv')
df = df[df['Stock Name'] == stock_name]
df['Date'] = pd.to_datetime(df['Date']).dt.date

# Sentiment Analysis (Using NLTK's SentimentIntensityAnalyzer)
from nltk.sentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['Tweet'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0)
df = df[['Date', 'sentiment_score']]
df = df.groupby('Date').mean().reset_index()

# Stock Data
stock_data = pd.read_csv('Dataset/stock_yfinance_data.csv')
stock_data = stock_data[stock_data['Stock Name'] == stock_name]
stock_data['Date'] = pd.to_datetime(stock_data['Date']).dt.date

# Merge datasets
final_df = stock_data.merge(df, on='Date', how='left').fillna(0)

# Normalize Data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(final_df.drop(columns=['Date', 'Stock Name']))
dump(scaler, open('newscaler.pkl', 'wb'))  # Save scaler

# Train/Test Split
train_size = int(len(scaled_data) * 0.8)
train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

# Prepare sequences for LSTM
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][0])  # Predict Close Price
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data)
X_test, y_test = create_sequences(test_data)

# Define LSTM Model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and Train the Model
model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Save the trained model
model.save('newstock_lstm_model.h5')

# Predictions
y_pred = model.predict(X_test)

# Rescale predictions and true values
y_pred_rescaled = scaler.inverse_transform(np.concatenate([y_pred, np.zeros((y_pred.shape[0], scaler.n_features_in_ - 1))], axis=1))[:, 0]
y_test_rescaled = scaler.inverse_transform(np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaler.n_features_in_ - 1))], axis=1))[:, 0]

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(y_test_rescaled, label='Actual Prices', color='blue')
plt.plot(y_pred_rescaled, label='Predicted Prices', color='red')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title(f'{stock_name} Stock Price Prediction')
plt.legend()
plt.show()

# Model Evaluation - Regression
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_rescaled)

print(f"\nRegression Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R-Squared: {r2:.4f}")

# Directional Accuracy
def directional_accuracy(y_true, y_pred):
    correct = np.sum((y_true[:-1] - y_true[1:]) * (y_pred[:-1] - y_pred[1:]) > 0)
    total = len(y_true) - 1
    return correct / total * 100

directional_acc = directional_accuracy(y_test_rescaled, y_pred_rescaled)
print(f"Directional Accuracy: {directional_acc:.2f}%")

# Classification Metrics (based on direction of price movement)
def get_direction(series):
    return np.where(np.diff(series, prepend=series[0]) > 0, 1, 0)

y_true_direction = get_direction(y_test_rescaled)
y_pred_direction = get_direction(y_pred_rescaled)

accuracy = accuracy_score(y_true_direction, y_pred_direction)
precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)
f_beta = fbeta_score(y_true_direction, y_pred_direction, beta=0.5, zero_division=0)

print(f"\nClassification Metrics (Based on Price Direction):")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1 Score:  {f1:.2f}")
print(f"F-beta Score (β=0.5): {f_beta:.2f}")
