import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate some sample data
# Let's say we have a sequence of numbers
data = np.array([[i] for i in range(100)])
target = np.array([[i+1] for i in range(100)])  # Predict the next number in the sequence
# Reshape the data for LSTM input [samples, time steps, features]
data = np.reshape(data, (data.shape[0], 1, 1))
# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(1, 1)))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(data, target, epochs=100, batch_size=1, verbose=0)

# Now, let's predict the next 5 numbers in the sequence
next_number = 100  # The last number in the sequence
predictions = []
n=int(input("Input the length of numbers you want to predict:-"))
i=0
while(i<n):
    next_number_input = np.reshape(next_number, (1, 1, 1))  # Reshape for LSTM input
    prediction = model.predict(next_number_input,verbose=0)
    predictions.append(round(prediction[0][0]))
    print(round(prediction[0][0]))
    next_number=round(prediction[0][0])
    if(i!=0 and i%5==0):
        ndata = np.array(predictions)
        ntarget = np.array(predictions)  # Predict the next number in the sequence

        # Reshape the data for LSTM input [samples, time steps, features]
        ndata = np.reshape(ndata, (ndata.shape[0], 1, 1))
        extended_data = np.append(data, ndata, axis=0)
        extended_target = np.append(target, ntarget, axis=0)
        # Retrain the model with the extended dataset
        model.fit(extended_data, extended_target, epochs=100, batch_size=1, verbose=0)
        prediction=[]
    i+=1
