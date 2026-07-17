# Credit Card Fraud Detection using Machine Learning

## Project Overview

This project detects fraudulent credit card transactions using machine learning techniques. It compares Logistic Regression, Decision Tree, and Random Forest classifiers to determine the most effective model for fraud detection.

---

## Dataset

Dataset Files:

- fraudTrain.csv
- fraudTest.csv

Target Column:

- is_fraud

0 → Legitimate Transaction

1 → Fraudulent Transaction

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

---

## Algorithms Used

- Logistic Regression
- Decision Tree
- Random Forest

---

## Results

| Model | Accuracy |
|--------|----------|
| Logistic Regression | 95.58% |
| Decision Tree | 99.66% |
| Random Forest | 99.72% |

Random Forest achieved the best overall performance.

---

## Project Structure

CreditCardFraudDetection/

- fraudTrain.csv
- fraudTest.csv
- fraud_detection.py
- model.pkl
- README.md
- requirements.txt

---

## How to Run

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python fraud_detection.py
```

---

## Future Improvements

- Hyperparameter tuning
- Real-time fraud detection
- Deep Learning models
- XGBoost implementation

---

## Author

Harshvardhan P