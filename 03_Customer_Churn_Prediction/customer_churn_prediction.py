import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Churn_Modelling.csv")

# Remove unnecessary columns
data = data.drop(["RowNumber", "CustomerId", "Surname"], axis=1)

# Create separate encoders
geo_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

# Encode categorical columns
data["Geography"] = geo_encoder.fit_transform(data["Geography"])
data["Gender"] = gender_encoder.fit_transform(data["Gender"])

# Features and Target
X = data.drop("Exited", axis=1)
y = data["Exited"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("=" * 50)
print("CUSTOMER CHURN PREDICTION")
print("=" * 50)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# -----------------------------
# User Prediction
# -----------------------------
print("\nEnter Customer Details")

credit_score = int(input("Credit Score: "))

print("\nAvailable Geography:")
print("1. France")
print("2. Germany")
print("3. Spain")

geography = input("Enter Geography: ").strip().title()

while geography not in ["France", "Germany", "Spain"]:
    geography = input("Please enter France, Germany or Spain: ").strip().title()

gender = input("Gender (Male/Female): ").strip().title()

while gender not in ["Male", "Female"]:
    gender = input("Please enter Male or Female: ").strip().title()

age = int(input("Age: "))
tenure = int(input("Tenure: "))
balance = float(input("Balance: "))
products = int(input("Number of Products: "))
credit_card = int(input("Has Credit Card? (1=Yes, 0=No): "))
active = int(input("Is Active Member? (1=Yes, 0=No): "))
salary = float(input("Estimated Salary: "))

# Encode user input
geography = geo_encoder.transform([geography])[0]
gender = gender_encoder.transform([gender])[0]

# Customer data
customer = pd.DataFrame([{
    "CreditScore": credit_score,
    "Geography": geography,
    "Gender": gender,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": products,
    "HasCrCard": credit_card,
    "IsActiveMember": active,
    "EstimatedSalary": salary
}])

prediction = model.predict(customer)
# Prediction
prediction = model.predict(customer)

print("\n" + "=" * 50)

if prediction[0] == 1:
    print("Prediction: Customer is likely to CHURN.")
else:
    print("Prediction: Customer is likely to STAY.")

print("=" * 50)