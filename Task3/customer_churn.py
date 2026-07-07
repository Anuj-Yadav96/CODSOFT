import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("Churn_Modelling.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Remove unnecessary columns
df.drop(["RowNumber", "CustomerId", "Surname"], axis=1, inplace=True)

# Encode categorical columns
label_encoder = LabelEncoder()

df["Gender"] = label_encoder.fit_transform(df["Gender"])
df["Geography"] = label_encoder.fit_transform(df["Geography"])

# Features and Target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Testing New Customer Prediction
sample_customer = [[
    650,   # CreditScore
    0,     # Geography
    1,     # Gender
    40,    # Age
    5,     # Tenure
    70000, # Balance
    2,     # NumOfProducts
    1,     # HasCrCard
    1,     # IsActiveMember
    50000  # EstimatedSalary
]]

sample_customer = scaler.transform(sample_customer)

prediction = model.predict(sample_customer)

if prediction[0] == 1:
    print("\nCustomer will leave the bank (Churn)")
else:
    print("\nCustomer will stay with the bank")