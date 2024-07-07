
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

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

# Make predictions
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Plot Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['-1', '0', '1'], yticklabels=['-1', '0', '1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, 'bo-', label='Actual')
plt.plot(y_test.index, y_pred, 'ro-', label='Predicted')
plt.xlabel('Index')
plt.ylabel('Status')
plt.title('Actual vs Predicted')
plt.legend()
plt.show()
