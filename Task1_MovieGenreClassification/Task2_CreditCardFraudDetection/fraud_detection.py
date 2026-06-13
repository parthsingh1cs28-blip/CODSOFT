"""
Credit Card Fraud Detection using Random Forest + SMOTE
CodSoft ML Internship - Task 2
Author: Parth Singh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.preprocessing import StandardScaler

# ─── Generate Synthetic Imbalanced Dataset ────────────────────────────────────
# In real use: df = pd.read_csv("creditcard.csv")  [Kaggle dataset]
np.random.seed(42)

n_legit = 950
n_fraud = 50

legit = pd.DataFrame({
    "amount":    np.random.normal(100, 50, n_legit).clip(1),
    "hour":      np.random.randint(0, 24, n_legit),
    "v1":        np.random.normal(0, 1, n_legit),
    "v2":        np.random.normal(0, 1, n_legit),
    "v3":        np.random.normal(0, 1, n_legit),
    "v4":        np.random.normal(0, 1, n_legit),
    "class":     0
})

fraud = pd.DataFrame({
    "amount":    np.random.normal(400, 100, n_fraud).clip(1),
    "hour":      np.random.choice([1, 2, 3, 23], n_fraud),   # odd hours
    "v1":        np.random.normal(-3, 1, n_fraud),
    "v2":        np.random.normal(3, 1, n_fraud),
    "v3":        np.random.normal(-2, 1, n_fraud),
    "v4":        np.random.normal(2, 1, n_fraud),
    "class":     1
})

df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)

print("Dataset Overview:")
print(f"Total transactions : {len(df)}")
print(f"Legitimate         : {(df['class']==0).sum()}")
print(f"Fraudulent         : {(df['class']==1).sum()}")
print(f"Fraud %            : {df['class'].mean()*100:.2f}%\n")

# ─── Features & Labels ────────────────────────────────────────────────────────
feature_cols = ["amount", "hour", "v1", "v2", "v3", "v4"]
X = df[feature_cols]
y = df["class"]

# ─── Train / Test Split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─── Scale Features ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── Handle Class Imbalance via class_weight ───────────────────────────────────
# This tells the model to penalise missing a fraud case more heavily
rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",   # handles imbalance without SMOTE dependency
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sc, y_train)

# ─── Evaluation ────────────────────────────────────────────────────────────────
y_pred     = rf_model.predict(X_test_sc)
y_proba    = rf_model.predict_proba(X_test_sc)[:, 1]

print("─── Random Forest Results ───────────────────────────")
print(f"Accuracy : {accuracy_score(y_test, y_pred)*100:.2f}%")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"  True Negative  (Legit correctly identified) : {cm[0][0]}")
print(f"  False Positive (Legit flagged as Fraud)     : {cm[0][1]}")
print(f"  False Negative (Fraud missed)               : {cm[1][0]}")
print(f"  True Positive  (Fraud correctly caught)     : {cm[1][1]}\n")

# ─── Feature Importance ───────────────────────────────────────────────────────
print("Feature Importances:")
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
for feat, imp in importances.sort_values(ascending=False).items():
    bar = "█" * int(imp * 50)
    print(f"  {feat:8s}: {bar} {imp:.4f}")

# ─── Predict on a Single Transaction ──────────────────────────────────────────
def predict_transaction(amount, hour, v1, v2, v3, v4):
    features = scaler.transform([[amount, hour, v1, v2, v3, v4]])
    pred  = rf_model.predict(features)[0]
    prob  = rf_model.predict_proba(features)[0][1]
    label = "🚨 FRAUD" if pred == 1 else "✅ LEGITIMATE"
    print(f"\nTransaction  → Amount: ${amount}, Hour: {hour}")
    print(f"Prediction   → {label}")
    print(f"Fraud Prob   → {prob*100:.1f}%")

predict_transaction(amount=85, hour=14, v1=0.1, v2=-0.2, v3=0.3, v4=-0.1)
predict_transaction(amount=950, hour=2, v1=-3.5, v2=4.1, v3=-2.8, v4=3.2)
predict_transaction(amount=200, hour=11, v1=0.5, v2=-0.8, v3=0.2, v4=-0.3)
predict_transaction(amount=200, hour=11, v1=0.5, v2=-0.8, v3=0.2, v4=-0.3)
