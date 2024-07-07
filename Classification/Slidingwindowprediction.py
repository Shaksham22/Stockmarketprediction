from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import joblib

# Fetching data from Yahoo Finance for the last one year
today = datetime.today().strftime('%Y-%m-%d')
one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
interest_rate=5.50
nasdaq_data = yf.Ticker("^IXIC").history(start=one_year_ago, end=today)
nifty_data = yf.Ticker("^NSEI").history(start=one_year_ago, end=today)

# Preprocess the data to match the format used during model training
present_data = pd.DataFrame({
    'NASDAQ Open': nasdaq_data["Open"].iloc[-1],
    'NASDAQ High': nasdaq_data["High"].iloc[-1],
    'NASDAQ Low': nasdaq_data["Low"].iloc[-1],
    'Nifty 50 Open': nifty_data["Open"].iloc[-1],
    'Nifty 50 High': nifty_data["High"].iloc[-1],
    'Nifty 50 Low': nifty_data["Low"].iloc[-1],
    'US Reporate': [interest_rate]
}, index=[0], columns=['Nifty 50 Open', 'Nifty 50 High', 'Nifty 50 Low', 'NASDAQ Open', 'NASDAQ High', 'NASDAQ Low', 'US Reporate'])

# Load the saved model
model_path = "/Users/shakshamshubham/Desktop/RM Project/Classification/random_forest_modelslidingwindow.joblib"
loaded_model = joblib.load(model_path)

# Make prediction on today's data
today_prediction = loaded_model.predict(present_data)

##print("Model prediction:", today_prediction)

# Calculate the moving averages
nifty_data['200MA'] = nifty_data['Close'].rolling(window=200).mean()
nifty_data['50MA'] = nifty_data['Close'].rolling(window=50).mean()
nifty_data['20MA'] = nifty_data['Close'].rolling(window=20).mean()

# Extract the last 200 days, 50 days, and 20 days moving averages
ma200 = nifty_data['200MA'].iloc[-1]
ma50 = nifty_data['50MA'].iloc[-1]
ma20 = nifty_data['20MA'].iloc[-1]
mares=0
if(ma20>ma50>ma200):
    mares=-1
elif(ma20<ma50<ma200):
    -1
finres=0
if(today_prediction==1):
    if(mares==1):
        finres=1
elif(today_prediction==-1):
    if(mares==-1):
        finres=-1

print(finres)
