"""
Credit Card Fraud Detection - Streamlit Web App
CodSoft ML Internship - Task 2
Author: Parth Singh
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                              confusion_matrix, roc_curve)

st.set_page_config(page_title="Fraud Detector", page_icon="🔍", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #f093fb, #f5576c);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .fraud-box {
        background: linear-gradient(135deg, #f5576c, #f093fb);
        padding: 2rem; border-radius: 15px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: bold; margin: 1rem 0;
    }
    .safe-box {
        background: linear-gradient(135deg, #4facfe, #00f2fe);
        padding: 2rem; border-radius: 15px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: bold; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ─── Train Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def train_fraud_model():
    np.random.seed(42)
    n_legit, n_fraud = 2000, 200

    legit = pd.DataFrame({
        "amount":   np.random.normal(85, 40, n_legit).clip(1, 500),
        "hour":     np.random.randint(6, 23, n_legit),
        "v1":       np.random.normal(0, 1, n_legit),
        "v2":       np.random.normal(0, 1, n_legit),
        "v3":       np.random.normal(0, 1, n_legit),
        "v4":       np.random.normal(0, 1, n_legit),
        "class":    0
    })

    fraud = pd.DataFrame({
        "amount":   np.random.normal(450, 150, n_fraud).clip(1),
        "hour":     np.random.choice([0, 1, 2, 3, 23], n_fraud),
        "v1":       np.random.normal(-3.5, 1, n_fraud),
        "v2":       np.random.normal(3.5, 1, n_fraud),
        "v3":       np.random.normal(-2.5, 1, n_fraud),
        "v4":       np.random.normal(2.5, 1, n_fraud),
        "class":    1
    })

    df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42)
    feature_cols = ["amount", "hour", "v1", "v2", "v3", "v4"]
    X, y = df[feature_cols], df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    fraud_classifier = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    fraud_classifier.fit(X_train_sc, y_train)

    y_pred  = fraud_classifier.predict(X_test_sc)
    y_proba = fraud_classifier.predict_proba(X_test_sc)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    return fraud_classifier, scaler, acc, auc, cm, fpr, tpr, feature_cols

fraud_classifier, scaler, acc, auc, cm, fpr, tpr, feature_cols = train_fraud_model()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🔍 Credit Card Fraud Detector</div>', unsafe_allow_html=True)
st.markdown("### Powered by Random Forest | Real-time Transaction Analysis")
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 About This App")
    st.markdown("""
    Detects fraudulent credit card transactions using Machine Learning.
    
    **Model:** Random Forest Classifier  
    **Imbalance Handling:** class_weight=balanced  
    **Features:** Amount, Hour, Transaction Patterns
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc*100:.1f}%")
    with col2:
        st.metric("ROC-AUC", f"{auc:.4f}")
    st.markdown("---")
    st.markdown("**CodSoft ML Internship**")
    st.markdown("Task 2 — Fraud Detection")
    st.markdown("**Author:** Parth Singh")

# ─── Main Layout ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Predict Transaction", "📊 Model Performance", "📈 Feature Analysis"])

with tab1:
    st.markdown("### Enter Transaction Details")
    col1, col2 = st.columns(2)

    with col1:
        amount = st.slider("Transaction Amount ($)", 1, 2000, 100)
        hour   = st.slider("Transaction Hour (0-23)", 0, 23, 14)
        v1     = st.slider("Pattern V1 (velocity score)", -6.0, 6.0, 0.0, 0.1)
        v2     = st.slider("Pattern V2 (location score)", -6.0, 6.0, 0.0, 0.1)

    with col2:
        v3 = st.slider("Pattern V3 (merchant score)", -6.0, 6.0, 0.0, 0.1)
        v4 = st.slider("Pattern V4 (device score)", -6.0, 6.0, 0.0, 0.1)

        st.markdown("**Quick Presets:**")
        preset = st.selectbox("Load example:", [
            "Custom",
            "Normal daytime purchase",
            "Suspicious late night high amount",
            "Typical online shopping",
            "Likely fraud pattern"
        ])

        presets = {
            "Normal daytime purchase":          (85,  14,  0.2, -0.3,  0.1, -0.2),
            "Suspicious late night high amount": (950,  2, -3.5,  4.1, -2.8,  3.2),
            "Typical online shopping":          (120,  11,  0.5, -0.5,  0.3, -0.4),
            "Likely fraud pattern":             (780,   1, -4.0,  4.5, -3.0,  3.5),
        }

        if preset != "Custom":
            amount, hour, v1, v2, v3, v4 = presets[preset]

    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        features = scaler.transform([[amount, hour, v1, v2, v3, v4]])
        pred  = fraud_classifier.predict(features)[0]
        prob  = fraud_classifier.predict_proba(features)[0][1]

        if pred == 1:
            st.markdown(f'<div class="fraud-box">🚨 FRAUDULENT TRANSACTION DETECTED<br>Fraud Probability: {prob*100:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="safe-box">✅ LEGITIMATE TRANSACTION<br>Fraud Probability: {prob*100:.1f}%</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Amount", f"${amount}")
        with col2:
            st.metric("Hour", f"{hour}:00")
        with col3:
            st.metric("Risk Level", "HIGH 🔴" if prob > 0.6 else "MEDIUM 🟡" if prob > 0.3 else "LOW 🟢")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Fraud Risk Score"},
            delta={"reference": 50},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if prob > 0.5 else "darkgreen"},
                "steps": [
                    {"range": [0, 30],  "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100],"color": "lightcoral"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    st.markdown("### Model Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy",  f"{acc*100:.2f}%")
    with col2:
        st.metric("ROC-AUC",   f"{auc:.4f}")
    with col3:
        st.metric("True Positives",  str(cm[1][1]))
    with col4:
        st.metric("False Negatives", str(cm[1][0]))

    col1, col2 = st.columns(2)

    with col1:
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legitimate", "Fraud"],
            y=["Legitimate", "Fraud"],
            color_continuous_scale="RdYlGn_r",
            title="Confusion Matrix"
        )
        fig_cm.update_traces(text=cm, texttemplate="%{text}")
        fig_cm.update_layout(height=350)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC Curve (AUC = {auc:.4f})",
            line=dict(color="royalblue", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=2, dash="dash")
        ))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=350
        )
        st.plotly_chart(fig_roc, use_container_width=True)

with tab3:
    st.markdown("### Feature Importance Analysis")

    importances = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": fraud_classifier.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(
        importances, x="Importance", y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title="What Drives Fraud Detection?",
        labels={"Importance": "Feature Importance Score"}
    )
    fig_imp.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    **Feature Explanations:**
    - **Amount** — Unusually high transaction amounts are a strong fraud indicator
    - **V1-V4** — Transaction pattern scores (velocity, location, merchant, device)
    - **Hour** — Fraudulent transactions often happen late at night (0-3 AM)
    """)

st.markdown("---")
st.markdown("Built with ❤️ by Parth Singh | CodSoft ML Internship 2026")
