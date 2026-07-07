# Import Libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load Dataset

data = pd.read_csv("fraudTrain.csv", nrows=50000)

print("Dataset Loaded Successfully!")

# Keep only useful columns

data = data[["amt", "lat", "long", "city_pop", "merch_lat", "merch_long", "is_fraud"]]

# Split features and target

X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Split data into training and testing

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scale the data

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression Model

model = LogisticRegression(class_weight="balanced", max_iter=1000)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

# Make Predictions

y_pred = model.predict(X_test)

# Model Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))