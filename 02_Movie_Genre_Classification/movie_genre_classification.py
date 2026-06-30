import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Training Data
train = pd.read_csv(
    "Genre Classification Dataset/train_data.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

# Load Test Data
test = pd.read_csv(
    "Genre Classification Dataset/test_data.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["ID", "TITLE", "DESCRIPTION"]
)

# Load Test Solution
test_solution = pd.read_csv(
    "Genre Classification Dataset/test_data_solution.txt",
    sep=" ::: ",
    engine="python",
    header=None,
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"]
)

# Use a smaller dataset for faster execution
train = train.head(10000)
test = test.head(2000)
test_solution = test_solution.head(2000)

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)

X_train = vectorizer.fit_transform(train["DESCRIPTION"])
X_test = vectorizer.transform(test["DESCRIPTION"])

y_train = train["GENRE"]
y_test = test_solution["GENRE"]

# Train Model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print("MOVIE GENRE CLASSIFICATION")
print("=" * 50)
print("Accuracy:", round(accuracy * 100, 2), "%")

# User Prediction
print("\nPredict Movie Genre")
movie = input("Enter movie description: ")

movie_vector = vectorizer.transform([movie])

genre = model.predict(movie_vector)

print("Predicted Genre:", genre[0])