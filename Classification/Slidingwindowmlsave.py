import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load data from Excel file
file_path = "/Users/shakshamshubham/Desktop/RM Project/combined data prediction.xlsx"
df = pd.read_excel(file_path)

# Drop Date column as it's not needed for modeling
df = df.drop(columns=['Date'])

# Define features (X) and target (y)
X = df.drop(columns=['Status'])
y = df['Status']

# Define window size (1 year in this case)
window_size = 252  # Assuming 252 trading days in a year

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Initialize empty lists to store evaluation metrics
confusion_matrices = []
all_actual = []
all_predictions = []

# Iterate over the dataset in sliding window
for i in range(0, len(df) - window_size + 1, window_size):
    # Get the current window of data
    X_window = X.iloc[i:i+window_size]
    y_window = y.iloc[i:i+window_size]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_window, y_window, test_size=0.3, random_state=42)
    
    # Train the Random Forest Classifier
    rf_classifier.fit(X_train, y_train)
    
    # Store all actual and predicted values for the time series plot
    all_actual.extend(y_test)
    all_predictions.extend(rf_classifier.predict(X_test))

# Save the trained model
model_path = "/Users/shakshamshubham/Desktop/RM Project/Classification/random_forest_modelslidingwindow.joblib"
joblib.dump(rf_classifier, model_path)

print("Model trained and saved successfully!")
