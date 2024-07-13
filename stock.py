import pandas as pd
import requests

# Function to fetch historical stock data from Yahoo Finance
def fetch_stock_data(symbol, start_date, end_date):
    base_url = 'https://query1.finance.yahoo.com/v7/finance/download/'
    params = {
        'symbol': symbol,
        'period1': int(pd.Timestamp(start_date).timestamp()),
        'period2': int(pd.Timestamp(end_date).timestamp()),
        'interval': '1d',
        'events': 'history',
        'includeAdjustedClose': 'true'
    }
    response = requests.get(base_url, params=params)
    return pd.read_csv(response.url)

# Example usage:
stock_data = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
print(stock_data.head())
