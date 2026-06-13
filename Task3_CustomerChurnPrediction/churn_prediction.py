"""
Customer Churn Prediction using Gradient Boosting
CodSoft ML Internship - Task 3
Author: Parth Singh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ─── Synthetic Telecom Churn Dataset ──────────────────────────────────────────
# Real dataset: Kaggle "Telco Customer Churn" CSV
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "tenure":            np.random.randint(1, 72, n),
    "monthly_charges":   np.random.uniform(20, 120, n),
    "total_charges":     np.random.uniform(100, 8000, n),
    "contract":          np.random.choice(["Month-to-month", "One year", "Two year"], n),
    "internet_service":  np.random.choice(["DSL", "Fiber optic", "No"], n),
    "tech_support":      np.random.choice(["Yes", "No"], n),
    "payment_method":    np.random.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n),
    "num_complaints":    np.random.randint(0, 6, n),
    "senior_citizen":    np.random.randint(0, 2, n),
})

# Simulate churn: short tenure + month-to-month + high charges = more churn
churn_prob = (
    (df["tenure"] < 12).astype(float) * 0.35 +
    (df["contract"] == "Month-to-month").astype(float) * 0.25 +
    (df["monthly_charges"] > 80).astype(float) * 0.20 +
    (df["num_complaints"] > 2).astype(float) * 0.15 +
    np.random.uniform(0, 0.1, n)
)
df["churn"] = (churn_prob > 0.45).astype(int)

print("Dataset Overview:")
print(df.head(5).to_string())
print(f"\nChurn Rate: {df['churn'].mean()*100:.1f}%")
print(f"Churned: {df['churn'].sum()} | Retained: {(df['churn']==0).sum()}\n")

# ─── Encoding Categorical Variables ───────────────────────────────────────────
le = LabelEncoder()
categorical_cols = ["contract", "internet_service", "tech_support", "payment_method"]
for col in categorical_cols:
    df[col + "_enc"] = le.fit_transform(df[col])

feature_cols = [
    "tenure", "monthly_charges", "total_charges", "num_complaints", "senior_citizen",
    "contract_enc", "internet_service_enc", "tech_support_enc", "payment_method_enc"
]

X = df[feature_cols]
y = df["churn"]

# ─── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Gradient Boosting Model ───────────────────────────────────────────────────
gb_model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)
gb_model.fit(X_train_sc, y_train)

# ─── Evaluation ────────────────────────────────────────────────────────────────
y_pred  = gb_model.predict(X_test_sc)
y_proba = gb_model.predict_proba(X_test_sc)[:, 1]

print("─── Gradient Boosting Results ───────────────────────")
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

# Cross-validation score
cv_scores = cross_val_score(gb_model, X_train_sc, y_train, cv=5, scoring="roc_auc")
print(f"5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

# ─── Feature Importance ───────────────────────────────────────────────────────
print("Top Features Driving Churn:")
importances = pd.Series(gb_model.feature_importances_, index=feature_cols)
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 60)
    print(f"  {feat:25s}: {bar} {imp:.4f}")

# ─── Predict Churn for a Single Customer ──────────────────────────────────────
def predict_churn(tenure, monthly_charges, total_charges, num_complaints,
                  senior_citizen, contract, internet_service, tech_support, payment_method):

    contract_enc      = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
    internet_enc      = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service]
    support_enc       = {"No": 0, "Yes": 1}[tech_support]
    payment_enc       = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}[payment_method]

    row = [[tenure, monthly_charges, total_charges, num_complaints, senior_citizen,
            contract_enc, internet_enc, support_enc, payment_enc]]
    row_sc = scaler.transform(row)

    pred = gb_model.predict(row_sc)[0]
    prob = gb_model.predict_proba(row_sc)[0][1]
    label = "⚠️  LIKELY TO CHURN" if pred == 1 else "✅  LIKELY TO STAY"

    print(f"\nCustomer Profile:")
    print(f"  Tenure: {tenure} months | Monthly: ${monthly_charges} | Contract: {contract}")
    print(f"Prediction  → {label}")
    print(f"Churn Risk  → {prob*100:.1f}%")

# Demo predictions
predict_churn(3, 95.0, 285.0, 3, 0, "Month-to-month", "Fiber optic", "No", "Electronic check")
predict_churn(48, 45.0, 2160.0, 0, 0, "Two year", "DSL", "Yes", "Bank transfer")
