import pandas as pd
from sklearn.model_selection import train_test_splitH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[["v1", "v2"]]
data.columns = ["label", "message"]

# Convert labels to numbers
data["label"] = data["label"].map({"ham": 0, "spam": 1})

# Features and target
X = data["message"]
y = data["label"]

# Convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# User input for prediction
print("\nSpam SMS Detection System")
message = input("Enter an SMS message: ")

# Convert message into TF-IDF features
message_vector = vectorizer.transform([message])

# Predict
prediction = model.predict(message_vector)

# Display result
if prediction[0] == 1:
    print("Result: Spam Message")
else:
    print("Result: Ham (Normal) Message")