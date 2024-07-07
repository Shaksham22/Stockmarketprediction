import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns

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
    
    # Make predictions
    y_pred = rf_classifier.predict(X_test)
    
    # Store confusion matrix
    confusion_matrices.append(confusion_matrix(y_test, y_pred, labels=[-1, 0, 1]))
    
    # Store all actual and predicted values for the time series plot
    all_actual.extend(y_test)
    all_predictions.extend(y_pred)

# Pad or trim the confusion matrices to ensure they all have the same shape
max_rows = max(cm.shape[0] for cm in confusion_matrices)
max_cols = max(cm.shape[1] for cm in confusion_matrices)
confusion_matrices_padded = [np.pad(cm, ((0, max_rows - cm.shape[0]), (0, max_cols - cm.shape[1])), mode='constant') for cm in confusion_matrices]

# Sum all confusion matrices to get the total counts
total_confusion_matrix = np.sum(confusion_matrices_padded, axis=0)

# Plot confusion matrix with counts
plt.figure(figsize=(8, 6))
sns.heatmap(total_confusion_matrix, annot=True, fmt='g', cmap='Blues', 
            xticklabels=['-1', '0', '1'], yticklabels=['-1', '0', '1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Total Confusion Matrix over Windows')
plt.show()

# Plot Actual vs Predicted time series
plt.figure(figsize=(10, 6))
plt.plot(all_actual, 'bo-', label='Actual')
plt.plot(all_predictions, 'ro-', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Status')
plt.title('Actual vs Predicted (Time Series)')
plt.legend()
plt.show()
