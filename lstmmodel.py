import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score
)
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Sequential
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

# For Sentiment Analysis
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()

# Disable TF v2 behavior if using TF v1
if tf.__version__.startswith("1."):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

# Load Datasets
tweets_df = pd.read_csv('Dataset/stock_tweets.csv')
price_df = pd.read_csv('Dataset/stock_yfinance_data.csv')


# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)


# Convert Date
tweets_df['Date'] = pd.to_datetime(tweets_df['Date']).dt.date
price_df['Date'] = pd.to_datetime(price_df['Date']).dt.date

# Sentiment Scoring
tweets_df['sentiment_score'] = tweets_df['Tweet'].apply(
    lambda x: analyzer.polarity_scores(str(x))['compound'] if isinstance(x, str) else 0
)

# Unique Stocks
stock_list = price_df['Stock Name'].unique()

# Store metrics
metrics_list = []

# Utility Functions
def create_sequences(data, seq_length=5):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length][0])  # Close price
    return np.array(X), np.array(y)

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

def get_direction(series):
    return np.where(np.diff(series, prepend=series[0]) > 0, 1, 0)

# Main Loop
for stock_name in stock_list:
    print(f"\nProcessing stock: {stock_name}")

    # Filter Data
    sentiment_data = tweets_df[tweets_df['Stock Name'] == stock_name]
    sentiment_data = sentiment_data[['Date', 'sentiment_score']].groupby('Date').mean().reset_index()

    stock_data = price_df[price_df['Stock Name'] == stock_name]

    # Merge
    final_df = stock_data.merge(sentiment_data, on='Date', how='left').fillna(0)
    if final_df.shape[0] < 20:
        print(f"Skipping {stock_name} due to insufficient data.")
        continue

    # Normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(final_df.drop(columns=['Date', 'Stock Name']))
    dump(scaler, open(f'scalers/{stock_name}_scaler.pkl', 'wb'))

    # Split
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    if len(test_data) <= 6:
        print(f"Skipping {stock_name} due to small test set.")
        continue

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)

    # Train model
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=0)
    model.save(f'models/{stock_name}_lstm_model.h5')

    # Predict
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(
        np.concatenate([y_pred, np.zeros((y_pred.shape[0], scaler.n_features_in_ - 1))], axis=1)
    )[:, 0]

    y_test_rescaled = scaler.inverse_transform(
        np.concatenate([y_test.reshape(-1, 1), np.zeros((y_test.shape[0], scaler.n_features_in_ - 1))], axis=1)
    )[:, 0]

    # Regression Metrics
    rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
    mae = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    r2 = r2_score(y_test_rescaled, y_pred_rescaled)

    # Classification Metrics
    y_true_direction = get_direction(y_test_rescaled)
    y_pred_direction = get_direction(y_pred_rescaled)

    accuracy = accuracy_score(y_true_direction, y_pred_direction)
    precision = precision_score(y_true_direction, y_pred_direction, zero_division=0)
    recall = recall_score(y_true_direction, y_pred_direction, zero_division=0)
    f1 = f1_score(y_true_direction, y_pred_direction, zero_division=0)
    fbeta = fbeta_score(y_true_direction, y_pred_direction, beta=0.5, zero_division=0)

    metrics_list.append([
        stock_name, rmse, mae, r2,
        accuracy, precision, recall, f1, fbeta
    ])

# Results DataFrame
metrics_df = pd.DataFrame(metrics_list, columns=[
    'Stock', 'RMSE', 'MAE', 'R2', 'Accuracy', 'Precision', 'Recall', 'F1', 'F-Beta'
])

# Save to CSV
metrics_df.to_csv('all_stock_metrics.csv', index=False)

# Summary
print("\n\nOverall Summary Statistics Across All Stocks:")
print(metrics_df.describe())

# Optional: Display per-stock metrics
print("\nDetailed Per-Stock Metrics:")
print(metrics_df)
