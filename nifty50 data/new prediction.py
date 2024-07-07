from nsepy import get_history as gh
import datetime as dt
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import tensorflow.keras as keras

nifty_df = pd.read_excel('/Users/shakshamshubham/Desktop/RM Project/nifty50 data/Nifty 50 data 2000 to 2023.xlsx', usecols=['Date', 'Close', 'Open', 'High', 'Low'])

# Rename the columns
nifty_df.columns = ['Date', 'Close', 'Open', 'High', 'Low']

# Print the DataFrame to check its contents and column names
print(nifty_df)

# Convert the 'Date' column to datetime format
nifty_df['Date'] = pd.to_datetime(nifty_df['Date'], format='%d-%m-%Y')

# Check if the 'Date' column exists in the DataFrame
if 'Date' not in nifty_df.columns:
    print("Error: 'Date' column does not exist in the DataFrame.")
else:
    # Convert certain columns to float
    cols_to_convert = ['Close', 'Open', 'High', 'Low']
    nifty_df[cols_to_convert] = nifty_df[cols_to_convert].astype(float)
    nifty_df.set_index('Date', inplace=True)

    # Divide the data into train and test sets based on the date
    test = nifty_df[nifty_df.index > pd.to_datetime('2019-01-01')]
    nifty_df = nifty_df[nifty_df.index <= pd.to_datetime('2019-01-01')]

    # Print the resulting DataFrames
    print("Train Data:")
    print(nifty_df)
    print("\nTest Data:")
    print(test)

# selecting only the opening price
train_set = nifty_df.iloc[:, 1:2].values
nifty_df.iloc[:, 1:2].tail()

# scaling
sc = MinMaxScaler(feature_range = (-1, 1))

training_set_scaled = sc.fit_transform(train_set)

# preparing data labels
# X - We use the previous 50 days data to create the training set
# Y - the 61st value serves as the Y value
# So standing at any point in time we are looking back 60 days to predict the 61st value

X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0]) 
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
y_train = y_train.reshape(-1,1)

# creating test dataset

testdata = test.copy()
real_stock_price = testdata.iloc[:, 1:2].values
dataset_total = pd.concat((nifty_df['Open'], testdata['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(testdata) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

y_test = sc.transform(real_stock_price)

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

mse = np.mean(keras.losses.mean_squared_error(real_stock_price, predicted_stock_price))

print(np.sqrt(mse))
plt.figure(figsize=(20,10))
plt.plot(real_stock_price, color = 'green', label = 'Nifty Stock Price')
plt.plot(predicted_stock_price, color = 'red', label = 'Predicted Nifty Stock Price')
plt.title('Nifty Stock Price Prediction')
plt.xlabel('Trading Day')
plt.ylabel('Nifty Stock Open Price')
#plt.ylim(bottom=0)

plt.legend()
plt.show()