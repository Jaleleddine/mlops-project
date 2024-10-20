import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer  # Importing the dataset directly


# Load dataset
data = load_breast_cancer()  # Load the dataset
X = pd.DataFrame(data.data, columns=data.feature_names)  # Convert to DataFrame
y = pd.Series(data.target)  # Target values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy: {accuracy}')

# Save model
import joblib
joblib.dump(model, 'model.joblib')




