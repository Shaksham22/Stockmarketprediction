import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = '/Users/shakshamshubham/Desktop/RM Project/nifty50 data/Nifty 50 data 2000 to 2023.xlsx'
df = pd.read_excel(file_path)

# Convert 'Date' column to datetime type
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# Extract year from the 'Date' column
df['Year'] = df['Date'].dt.year

# Group by year and get the first and lowest opening price for each year
first_opening_prices = df.groupby('Year')['High'].max()
lowest_opening_prices = df.groupby('Year')['Low'].min()

# Create DataFrames with year and opening prices
first_opening_df = pd.DataFrame(first_opening_prices).reset_index()
lowest_opening_df = pd.DataFrame(lowest_opening_prices).reset_index()

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(first_opening_df['Year'], first_opening_df['High'], label='Lowest High Price of the Year', marker='o')
plt.plot(lowest_opening_df['Year'], lowest_opening_df['Low'], label='Lowest Opening Price of the Year', marker='o')
plt.title('Highest Low and Lowest Opening Prices of Nifty 50')
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.xticks(first_opening_df['Year'])
plt.tight_layout()
plt.show()
