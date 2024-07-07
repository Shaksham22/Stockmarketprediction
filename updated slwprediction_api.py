from flask import Flask, jsonify
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import joblib
from niftystocks import ns
from flask_cors import CORS
import socket
print("initiated...")
def get_local_ip():
    try:
        # Create a socket object
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        
        # Connect to a remote server (doesn't matter which)
        s.connect(("8.8.8.8", 80))
        
        # Get the local IP address
        local_ip = s.getsockname()[0]
        
        # Close the socket
        s.close()
        
        return local_ip
    except socket.error:
        return None


app = Flask(__name__)
CORS(app)

# Load the saved model
model_path = "/Users/shakshamshubham/Desktop/RM Project/Classification/random_forest_modelslidingwindow.joblib"
loaded_model = joblib.load(model_path)

# Fetching data from Yahoo Finance for the last one year
today = datetime.today().strftime('%Y-%m-%d')
one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
interest_rate = 5.50
nasdaq_data = yf.Ticker("^IXIC").history(start=one_year_ago, end=today)
nifty_data = yf.Ticker("^NSEI").history(start=one_year_ago, end=today)

def nifty50():
    def get_company_name(ticker_symbol):
        try:
            # Create a Ticker object for the given ticker symbol
            ticker = yf.Ticker(ticker_symbol)
            
            # Retrieve company name from the info dictionary
            company_name = ticker.info.get('longName', 'Company name not found')
            
            return company_name
        except Exception as e:
            print(f"Error occurred: {e}")
            return None

    stockname=['HDFCBANK', 'RELIANCE', 'ICICIBANK', 'INFY', 'LT', 'TCS', 'ITC', 'AXISBANK', 'BHARTIARTL', 'SBIN', 'KOTAKBANK', 'HINDUNILVR', 'M&M', 'BAJFINANCE', 'SUNPHARMA', 'HCLTECH', 'TATAMOTORS', 'MARUTI', 'NTPC', 'TITAN', 'TATASTEEL', 'POWERGRID', 'ASIANPAINT', 'ULTRACEMCO', 'ADANIPORTS', 'ONGC', 'BAJAJ-AUTO', 'COALINDIA', 'NESTLEIND', 'BAJAJFINSV', 'INDUSINDBK', 'ADANIENT', 'TECHM', 'HINDALCO', 'CIPLA', 'GRASIM', 'DRREDDY', 'TATACONSUM', 'JSWSTEEL', 'WIPRO', 'APOLLOHOSP', 'SBILIFE', 'HDFCLIFE', 'HEROMOTOCO', 'BRITANNIA', 'BPCL', 'EICHERMOT', 'LTIM', 'DIVISLAB', 'UPL']

    l={}
    for i,a in enumerate(stockname):
        try:
            l[i]={}
            l[i]['fullname']=get_company_name(a+".NS").replace("Limited", "Ltd")
            l[i]['tradingname']=a
            today = datetime.today().strftime('%Y-%m-%d')
            one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
            nifty_data = yf.Ticker(a+".NS").history(start=one_year_ago, end=today)
                # Calculate the moving averages
            nifty_data['200MA'] = nifty_data['Close'].rolling(window=200).mean()
            nifty_data['50MA'] = nifty_data['Close'].rolling(window=50).mean()
            nifty_data['20MA'] = nifty_data['Close'].rolling(window=20).mean()

            # Extract the last 200 days, 50 days, and 20 days moving averages
            ma200 = nifty_data['200MA'].iloc[-1]
            ma50 = nifty_data['50MA'].iloc[-1]
            ma20 = nifty_data['20MA'].iloc[-1]
            mares = 0
            if ma20 > ma50 > ma200:
                mares = -1
            elif ma20 < ma50 < ma200:
                mares = -1
            
            l[i]['prediction']=mares
        except:
            continue
    return(l)
resnifty50=nifty50()


@app.route('/predict', methods=['GET'])
def predict():
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

    # Make prediction on today's data
    today_prediction = int(loaded_model.predict(present_data)[0])

    # Calculate the moving averages
    nifty_data['200MA'] = nifty_data['Close'].rolling(window=200).mean()
    nifty_data['50MA'] = nifty_data['Close'].rolling(window=50).mean()
    nifty_data['20MA'] = nifty_data['Close'].rolling(window=20).mean()

    # Extract the last 200 days, 50 days, and 20 days moving averages
    ma200 = nifty_data['200MA'].iloc[-1]
    ma50 = nifty_data['50MA'].iloc[-1]
    ma20 = nifty_data['20MA'].iloc[-1]
    
    mares = 0
    if ma20 > ma50 > ma200:
        mares = -1
    elif ma20 < ma50 < ma200:
        mares = -1
    
    finres = 0
    if today_prediction == 1:
        if mares == 1:
            finres = 1
    elif today_prediction == -1:
        if mares == -1:
            finres = -1

    response = jsonify({'combinedprediction': finres, 'maprediction': mares, 'modelprediction': today_prediction})
    return response

    
@app.route('/stocklist', methods=['GET'])
def get_nifty50_stocks():
    # Get the Nifty 50 stocks using the niftystocks package
    stockname=['HDFCBANK', 'RELIANCE', 'ICICIBANK', 'INFY', 'LT', 'TCS', 'ITC', 'AXISBANK', 'BHARTIARTL', 'SBIN', 'KOTAKBANK', 'HINDUNILVR', 'M&M', 'BAJFINANCE', 'SUNPHARMA', 'HCLTECH', 'TATAMOTORS', 'MARUTI', 'NTPC', 'TITAN', 'TATASTEEL', 'POWERGRID', 'ASIANPAINT', 'ULTRACEMCO', 'ADANIPORTS', 'ONGC', 'BAJAJ-AUTO', 'COALINDIA', 'NESTLEIND', 'BAJAJFINSV', 'INDUSINDBK', 'ADANIENT', 'TECHM', 'HINDALCO', 'CIPLA', 'GRASIM', 'DRREDDY', 'TATACONSUM', 'JSWSTEEL', 'WIPRO', 'APOLLOHOSP', 'SBILIFE', 'HDFCLIFE', 'HEROMOTOCO', 'BRITANNIA', 'BPCL', 'EICHERMOT', 'LTIM', 'DIVISLAB', 'UPL']
    response = jsonify(resnifty50)
    return response

if __name__ == '__main__':
    print(get_local_ip())
    app.run(debug=True, host=get_local_ip(), port=8081)
