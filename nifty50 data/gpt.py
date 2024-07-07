import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load data from Excel
df = pd.read_excel('/Users/shakshamshubham/Desktop/RM Project/nifty50 data/Nifty 50 data 2000 to 2023.xlsx', usecols=['Date', 'Open'])

# Preprocess the data
data = df['Open'].values.reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Define function to create dataset for LSTM
def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

# Create the dataset
time_step = 100
X, y = create_dataset(scaled_data, time_step)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=64)

# Evaluate the model
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)
print('Train Loss:', train_loss)
print('Test Loss:', test_loss)

# Make predictions for the future period (2021 to 2023)
prediction_preiod = 365 * 2  # 2 years
future_data = scaled_data[-time_step:].tolist()

for i in range(prediction_preiod):
    x_input = np.array(future_data[-time_step:]).reshape(1, time_step, 1)
    prediction = model.predict(x_input, verbose=0)
    future_data.append(prediction[0].tolist())

# Inverse scaling for predicted values
predicted_values = scaler.inverse_transform(np.array(future_data[-prediction_preiod:]).reshape(-1, 1))

# Plotting the results
plt.plot(df['Date'][-prediction_preiod:], predicted_values, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Opening Price')
plt.title('Nifty 50 Opening Price Prediction')
plt.legend()
plt.show()
