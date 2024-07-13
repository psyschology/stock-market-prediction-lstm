import requests
import pandas as pd
import numpy as np

# Replace 'VM4W2GOSJ8QEXEME' with your actual Alpha Vantage API key
API_KEY = 'VM4W2GOSJ8QEXEME'
SYMBOL = 'AAPL'  # Example: Apple's ticker symbol

# Endpoint URL for Alpha Vantage
endpoint = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={SYMBOL}&apikey={API_KEY}'

# Make a GET request to the Alpha Vantage API
response = requests.get(endpoint)

if response.status_code == 200:
    data = response.json()  # Parse JSON response
    # Extract the time series data from the response
    time_series_data = data['Time Series (Daily)']
    
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(time_series_data).T  # Transpose to have dates as rows

    # Optional: Save the data to a CSV file
    df.to_csv(f'{SYMBOL}_daily_stock_data.csv', index_label='Date')

    print(f'Data successfully fetched and saved as {SYMBOL}_daily_stock_data.csv')
else:
    print('Error fetching data:', response.status_code)

# Load data from CSV
df = pd.read_csv('AAPL_daily_stock_data.csv', index_col='Date', parse_dates=True)

print(df.head())

# Check for missing values
print(df.isnull().sum())

# Fill missing values with forward fill (previous day's value)
df = df.ffill()

from sklearn.preprocessing import MinMaxScaler

# Initialize MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Normalize the data
scaled_data = scaler.fit_transform(df.values)

# Convert scaled data back to a DataFrame
df_scaled = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])  # Input sequence
        y.append(data[i + seq_length])   # Target value
    return np.array(X), np.array(y)

# Define sequence length (number of time steps to look back)
seq_length = 10  # Example: Look back 10 days

# Create sequences
X, y = create_sequences(df_scaled.values, seq_length)

from sklearn.model_selection import train_test_split

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print shapes to verify
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Reshape data for LSTM (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], seq_length, df.shape[1]))
X_test = X_test.reshape((X_test.shape[0], seq_length, df.shape[1]))

# Print reshaped shapes
print(f"X_train reshaped shape: {X_train.shape}")
print(f"X_test reshaped shape: {X_test.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()

# Example: Adding a single LSTM layer
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

model.add(Dropout(0.2))

model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")

predictions = model.predict(X_test)

# Inverse transform predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test.reshape(-1, 1))

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(actual_values, predictions)
rmse = mean_squared_error(actual_values, predictions, squared=False)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(actual_values, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Save the model
model.save('stock_prediction_model.h5')

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('stock_prediction_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Process input data and make predictions
    # Example: Predict from incoming data
    # predictions = model.predict(process_data(data))
    return jsonify({'prediction': 'prediction_value'})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict(data):
    try:
        # Make predictions
        predictions = model.predict(data)
        logging.info(f"Prediction successful: {predictions}")
        return predictions
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return None
