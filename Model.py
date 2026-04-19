import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

# -----------------------------
# 2. Clean data
# -----------------------------
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

# -----------------------------
# 3. Encode target
# -----------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# -----------------------------
# 4. Encode categorical columns
# -----------------------------
categorical_cols = df.select_dtypes(include=["object"]).columns

le = LabelEncoder()

for col in categorical_cols:
    if col != "customerID":
        df[col] = le.fit_transform(df[col])

# -----------------------------
# 5. Drop ID column
# -----------------------------
df = df.drop("customerID", axis=1)

# -----------------------------
# 6. Split data
# -----------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 8. Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 9. Evaluation
# -----------------------------
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
