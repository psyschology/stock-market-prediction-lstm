import requests
from requests_oauthlib import OAuth1Session
import pandas as pd

# Replace with your own credentials
client_id = 'dj0yJmk9WmVKTWlQYUJiaVF1JmQ9WVdrOWJtNVVVamxDT1hnbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PWJh'
client_secret = 'your_client_secret'  # Not shown for security reasons

# OAuth endpoints
request_token_url = 'https://api.login.yahoo.com/oauth2/request_auth'
authorization_base_url = 'https://api.login.yahoo.com/oauth2/request_auth'
token_url = 'https://api.login.yahoo.com/oauth2/get_token'

# Create an OAuth session
yahoo_oauth = OAuth1Session(client_id, client_secret=client_secret)

# Fetch OAuth token
oauth_token = yahoo_oauth.fetch_access_token(token_url)

# Example function to fetch historical stock data
def fetch_stock_data(symbol, start_date, end_date):
    base_url = f'https://query1.finance.yahoo.com/v7/finance/download/{symbol}'
    params = {
        'period1': int(pd.Timestamp(start_date).timestamp()),
        'period2': int(pd.Timestamp(end_date).timestamp()),
        'interval': '1d',
        'events': 'history',
        'includeAdjustedClose': 'true'
    }
    response = yahoo_oauth.get(base_url, params=params)
    if response.status_code == 200:
        return pd.read_csv(pd.compat.StringIO(response.text))
    else:
        print(f"Failed to fetch data for symbol {symbol}. Status code: {response.status_code}")
        return None

# Example usage:
stock_data = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
if stock_data is not None:
    print(stock_data.head())
