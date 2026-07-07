import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Select required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Convert labels
data['label'] = data['label'].map({'ham': 0, 'spam': 1})


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    data['message'],
    data['label'],
    test_size=0.2,
    random_state=42
)


# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Train model
model = MultinomialNB()
model.fit(X_train, y_train)


# Prediction
prediction = model.predict(X_test)


# Accuracy in percentage
accuracy = accuracy_score(y_test, prediction)

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, prediction))


# Test custom message
msg = ["Congratulations! You won a free prize"]

msg_vector = vectorizer.transform(msg)

result = model.predict(msg_vector)


if result[0] == 1:
    print("Prediction: Spam Message")
else:
    print("Prediction: Normal Message")