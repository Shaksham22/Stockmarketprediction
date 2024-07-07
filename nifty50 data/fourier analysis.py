import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Read data from Excel sheet and parse the 'Date' column as datetime
file_path = "/Users/shakshamshubham/Desktop/RM Project/nifty50 data/Nifty 50 data 2000 to 2023.xlsx"
data = pd.read_excel(file_path, parse_dates=['Date'])

# Step 2: Extract the 'Date' and 'Open' columns
date_open = data[['Date', 'Open']]

# Step 3: Set the 'Date' column as the index
date_open.set_index('Date', inplace=True)

# Step 4: Perform seasonal decomposition
result = seasonal_decompose(date_open, model='additive', period=252//2)  # Assuming yearly seasonality for stock market data

# Step 5: Plot the decomposed components
plt.figure(figsize=(12, 8))

# Original data
plt.subplot(4, 1, 1)
plt.plot(date_open, label='Original')
plt.legend(loc='best')
plt.title('Original Data')

# Trend component
plt.subplot(4, 1, 2)
plt.plot(result.trend, label='Trend')
plt.legend(loc='best')
plt.title('Trend')

# Seasonal component
plt.subplot(4, 1, 3)
plt.plot(result.seasonal, label='Seasonal')
plt.legend(loc='best')
plt.title('Seasonal')

# Residual component
plt.subplot(4, 1, 4)
plt.plot(result.resid, label='Residual')
plt.legend(loc='best')
plt.title('Residual')

plt.tight_layout()
plt.show()
