import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load the historical Nifty 50 data
data = pd.read_excel('/Users/shakshamshubham/Desktop/RM Project/nifty50 data/Nifty 50 data 2000 to 2023.xlsx')

# Prepare the data for Prophet
# Rename columns to 'ds' and 'y' as required by Prophet
data.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

# Initialize Prophet model
model = Prophet()

# Fit the model to the data
model.fit(data)

# Make future predictions for the next one year
future = model.make_future_dataframe(periods=365)

# Forecast future values
forecast = model.predict(future)

# Visualize the forecast
fig = model.plot(forecast)
plt.title('Forecasted Nifty 50 Stock Price')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.show()
