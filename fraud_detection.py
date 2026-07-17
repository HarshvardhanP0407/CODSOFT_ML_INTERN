import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# ============================================
# LOAD DATASET
# ============================================

print("Loading datasets...")

train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")

print("Training Dataset Shape:", train_df.shape)
print("Testing Dataset Shape:", test_df.shape)

# ============================================
# DROP UNNECESSARY COLUMNS
# ============================================

drop_columns = [
    "Unnamed: 0",
    "trans_date_trans_time",
    "cc_num",
    "merchant",
    "first",
    "last",
    "street",
    "city",
    "state",
    "zip",
    "dob",
    "trans_num"
]

train_df.drop(columns=drop_columns, inplace=True)
test_df.drop(columns=drop_columns, inplace=True)

# ============================================
# HANDLE MISSING VALUES
# ============================================

train_df = train_df.ffill()
test_df = test_df.ffill()

# ============================================
# ENCODE CATEGORICAL COLUMNS
# ============================================

categorical_columns = train_df.select_dtypes(include=["object", "string"]).columns

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

train_df[categorical_columns] = encoder.fit_transform(train_df[categorical_columns])
test_df[categorical_columns] = encoder.transform(test_df[categorical_columns])

# ============================================
# PREPARE TRAINING AND TESTING DATA
# ============================================

X_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]

X_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]

print("\nData Preprocessing Completed!")

# ============================================
# LOGISTIC REGRESSION
# ============================================

print("\n==============================")
print("LOGISTIC REGRESSION")
print("==============================")

lr = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("Accuracy:", accuracy_score(y_test, lr_pred))
print(classification_report(y_test, lr_pred))
print(confusion_matrix(y_test, lr_pred))

# ============================================
# DECISION TREE
# ============================================

print("\n==============================")
print("DECISION TREE")
print("==============================")

dt = DecisionTreeClassifier(
    random_state=42,
    class_weight="balanced"
)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
print(confusion_matrix(y_test, dt_pred))

# ============================================
# RANDOM FOREST
# ============================================

print("\n==============================")
print("RANDOM FOREST")
print("==============================")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))
print(confusion_matrix(y_test, rf_pred))

print("\n===================================")
print("ALL MODELS TRAINED SUCCESSFULLY")
print("===================================")