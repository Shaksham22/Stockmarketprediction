import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import time
# Load data from Excel file
file_path = "/Users/shakshamshubham/Desktop/RM Project/API test/US reporate data.xlsx"
df = pd.read_excel(file_path)

# Filter data for the desired range
start_year = 2007
end_year = 2000
filtered_df = df[(df['date'].dt.year >= start_year) & (df['date'].dt.year <= end_year)]

# Extract 'value' column
data = filtered_df['value'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

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
X, y = prepare_data(data_scaled, n_steps)

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
model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1, validation_data=(X_test, y_test))

# Predict future two years
future_steps = 730  # 2 years
future_data = []
current_batch = X_test[-1].reshape((1, n_steps, 1))

for i in range(future_steps):
    future_pred = model.predict(current_batch)[0,0]
    future_data.append(future_pred)
    current_batch = np.append(current_batch[:,1:,:], future_pred.reshape(1,1,1), axis=1)
    
# Inverse scale the predictions
future_data = scaler.inverse_transform(np.array(future_data).reshape(-1, 1))

# Plot the results
import matplotlib.pyplot as plt

plt.plot(filtered_df['date'], filtered_df['value'], label='Original Data')
plt.plot(filtered_df['date'].iloc[-1] + pd.to_timedelta(np.arange(1, future_steps+1), unit='D'), future_data, label='Predicted Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()
