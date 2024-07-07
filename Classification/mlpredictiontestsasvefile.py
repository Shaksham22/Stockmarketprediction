import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

# Load data from Excel file
file_path = "/Users/shakshamshubham/Desktop/RM Project/combined data prediction.xlsx"
df = pd.read_excel(file_path)

# Drop Date column as it's not needed for modeling
df = df.drop(columns=['Date'])

# Define features (X) and target (y)
X = df.drop(columns=['Status'])
y = df['Status']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Save the trained model
model_path = "/Users/shakshamshubham/Desktop/RM Project/Classification/random_forest_model.joblib"
dump(rf_classifier, model_path)
