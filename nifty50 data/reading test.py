import pandas as pd
from sklearn.metrics import mean_absolute_error
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
pd.options.mode.chained_assignment = None  # default='warn'

data = pd.read_excel('/Users/shakshamshubham/Desktop/RM Project/nifty50 data/Nifty 50 data 2000 to 2023.xlsx', usecols=['Date', 'Close', 'Open', 'High', 'Low'])

data.columns = ['Date', 'Close', 'Open', 'High', 'Low']

data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

cols_to_convert = ['Close', 'Open', 'High', 'Low']
data[cols_to_convert] = data[cols_to_convert].astype(float)
data.set_index('Date', inplace=True)
print(data)

data.shape

start_date = '2000-01-01'
end_date = '2024-01-01'
temp=data
data = data.loc[start_date:end_date]

plt.figure(figsize=(16,8))
plt.title('Close price history')
plt.plot(data["Close"])
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close price of Nifty',fontsize=18)


close=data.filter(["Close"])
dataset=close.values

training_data_len=math.ceil(len(dataset)*0.85)
training_data_len

scaler=MinMaxScaler(feature_range=(-1,1))
scaled_data=scaler.fit_transform(dataset)

train_data=scaled_data[0:training_data_len,:]
x_train=[]
y_train=[]
for i in range(365,len(train_data)):
    x_train.append(train_data[i-365:i,0])
    y_train.append(train_data[i,0])
    if i<=61:
        print(x_train)
        print(y_train)
        print()

x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


x_train.shape

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(50,return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))


model.compile(optimizer="adam",loss="mean_squared_error")

model.fit(x_train,y_train,batch_size=32,epochs=50)

test_data=scaled_data[training_data_len-365:,:]

test_data=scaled_data[training_data_len-365:,:]
#create the data sets x_test,y_test
x_test=[]
y_test=dataset[training_data_len:,:]
for i in range(365,len(test_data)):
    x_test.append(test_data[i-365:i,0])
    
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
rmse=np.sqrt(np.mean(predictions-y_test)**2)
print("Root Meaned Square:",rmse)

mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error:", mae)

train=data[:training_data_len]
valid=data[training_data_len:]
valid["Predictions"]=predictions
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price of Nifty',fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[["Close","Predictions"]])
plt.legend(['Train',"Val","Predictions"],loc="lower right")

start_date = '2023-03-01'
end_date = '2024-03-01'
newa = temp.loc[start_date:end_date]
new_df=newa.filter(['Close'])

predicted_values = []

for _ in range(150):
    # Pop the last value from the DataFrame
    
    
    # Scale the last 365 days of data
    last_365_days_scaled = scaler.transform(new_df[-365:].values)
    
    # Reshape and predict the scaled price
    X_test = np.reshape(last_365_days_scaled, (1, last_365_days_scaled.shape[0], 1))
    pred_price = model.predict(X_test)
    
    # Invert the scaling
    pred_price = scaler.inverse_transform(pred_price)
    
    # Append the predicted value to the list
    predicted_values.append(pred_price[0, 0])
    new_df = new_df[1:]
    new_row = pd.DataFrame({'Close': [pred_price[0, 0]]}, index=[new_df.index[-1] + pd.DateOffset(days=1)])
    new_df = pd.concat([new_df, new_row])
print(predicted_values)

# Plot the predicted values
plt.figure(figsize=(16, 8))
plt.plot(predicted_values, label='Predicted Close Price')
plt.title('Predicted Close Price for Next 150 Days')
plt.xlabel('Days')
plt.ylabel('Close Price')
plt.legend()
plt.show()
