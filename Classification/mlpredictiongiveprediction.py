import pandas as pd
from joblib import load
import yfinance as yf

# Fetching data from Yahoo Finance
nasdaq_info = yf.Ticker("^IXIC").info
nifty_info = yf.Ticker("^NSEI").info
interest_rate = 5.5  # Your interest rate value

# Creating a DataFrame with present-day data
present_data = pd.DataFrame({
    'Nifty 50 Open': [nifty_info["regularMarketOpen"]],
    'Nifty 50 High': [nifty_info["regularMarketDayHigh"]],
    'Nifty 50 Low': [nifty_info["regularMarketDayLow"]],
    'NASDAQ Open': [nasdaq_info["regularMarketOpen"]],
    'NASDAQ High': [nasdaq_info["regularMarketDayHigh"]],
    'NASDAQ Low': [nasdaq_info["regularMarketDayLow"]],
    'US Reporate': [interest_rate]
}, columns=['Nifty 50 Open', 'Nifty 50 High', 'Nifty 50 Low', 'NASDAQ Open', 'NASDAQ High', 'NASDAQ Low', 'US Reporate'])

# Load the saved model
model_path = "/Users/shakshamshubham/Desktop/RM Project/Classification/random_forest_model.joblib"
loaded_model = load(model_path)

# Make prediction on present-day data
present_prediction = loaded_model.predict(present_data)

# Print the prediction
print("Prediction for present-day data:", present_prediction)


