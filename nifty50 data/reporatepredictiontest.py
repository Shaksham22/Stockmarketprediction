import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Read the data from Excel file
file_path = "/Users/shakshamshubham/Desktop/RM Project/API test/US reporate data.xlsx"
data_df = pd.read_excel(file_path)

# Extract date and value columns
date_column = 'date'  # Update with your actual column name for date
value_column = 'value'  # Update with your actual column name for value

# Filter data from 2000 to 2023
start_date = '2000-01-01'
end_date = '2023-12-31'
filtered_data = data_df.loc[(data_df[date_column] >= start_date) & (data_df[date_column] <= end_date)]

# Convert date column to datetime and set it as index
filtered_data[date_column] = pd.to_datetime(filtered_data[date_column])
filtered_data.set_index(date_column, inplace=True)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(filtered_data[value_column].values.reshape(-1, 1))

# Prepare the data
def prepare_data(data, n_steps):
    X, y = [], []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data)-1:
            break
        seq_x, seq_y = data[i:end_ix], data[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

n_steps = 30  # number of time steps to look back
X, y = prepare_data(scaled_data, n_steps)

# Reshape input data to fit into LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
split_ratio = 0.8
split_index = int(len(X) * split_ratio)
X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# Predict future two years
future_steps = 730  # 2 years
future_data = []
current_batch = X_test[-1].reshape((1, n_steps, 1))

for i in range(future_steps):
    future_pred = model.predict(current_batch)[0,0]
    future_data.append(future_pred)
    current_batch = np.append(current_batch[:,1:,:],[[future_pred]],axis=1)

# Inverse scale the predictions
future_data = scaler.inverse_transform(np.array(future_data).reshape(-1, 1))

# Plot the results
plt.plot(filtered_data.index, filtered_data[value_column], label='Original Data')
plt.plot(pd.date_range(start=filtered_data.index[-1], periods=future_steps+1, freq='D')[1:], future_data, label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Repo Rate')
plt.legend()
plt.show()
