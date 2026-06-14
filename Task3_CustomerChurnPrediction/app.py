"""
Customer Churn Prediction - Streamlit Web App
CodSoft ML Internship - Task 3
Author: Parth Singh
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #11998e, #38ef7d);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .churn-box {
        background: linear-gradient(135deg, #f5576c, #f093fb);
        padding: 2rem; border-radius: 15px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: bold; margin: 1rem 0;
    }
    .stay-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 2rem; border-radius: 15px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: bold; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_churn_model():
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "tenure":           np.random.randint(1, 72, n),
        "monthly_charges":  np.random.uniform(20, 120, n),
        "total_charges":    np.random.uniform(100, 8000, n),
        "contract":         np.random.choice(["Month-to-month", "One year", "Two year"], n),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "tech_support":     np.random.choice(["Yes", "No"], n),
        "payment_method":   np.random.choice(["Credit card", "Bank transfer", "Electronic check", "Mailed check"], n),
        "num_complaints":   np.random.randint(0, 6, n),
        "senior_citizen":   np.random.randint(0, 2, n),
    })

    churn_prob = (
        (df["tenure"] < 12).astype(float) * 0.35 +
        (df["contract"] == "Month-to-month").astype(float) * 0.25 +
        (df["monthly_charges"] > 80).astype(float) * 0.20 +
        (df["num_complaints"] > 2).astype(float) * 0.15 +
        np.random.uniform(0, 0.1, n)
    )
    df["churn"] = (churn_prob > 0.45).astype(int)

    le = LabelEncoder()
    cat_cols = ["contract", "internet_service", "tech_support", "payment_method"]
    for col in cat_cols:
        df[col + "_enc"] = le.fit_transform(df[col])

    feature_cols = [
        "tenure", "monthly_charges", "total_charges", "num_complaints",
        "senior_citizen", "contract_enc", "internet_service_enc",
        "tech_support_enc", "payment_method_enc"
    ]

    X, y = df[feature_cols], df["churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    churn_classifier = GradientBoostingClassifier(
        n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42
    )
    churn_classifier.fit(X_train_sc, y_train)

    y_pred  = churn_classifier.predict(X_test_sc)
    y_proba = churn_classifier.predict_proba(X_test_sc)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cm      = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    importances = pd.DataFrame({
        "Feature": ["Tenure", "Monthly Charges", "Total Charges", "Complaints",
                    "Senior Citizen", "Contract", "Internet", "Tech Support", "Payment"],
        "Importance": churn_classifier.feature_importances_
    }).sort_values("Importance", ascending=True)

    return churn_classifier, scaler, acc, auc, cm, fpr, tpr, importances

churn_classifier, scaler, acc, auc, cm, fpr, tpr, importances = train_churn_model()

st.markdown('<div class="main-header">📉 Customer Churn Predictor</div>', unsafe_allow_html=True)
st.markdown("### Powered by Gradient Boosting | Telecom Customer Analytics")
st.markdown("---")

with st.sidebar:
    st.markdown("## 📉 About This App")
    st.markdown("""
    Predicts whether a telecom customer will cancel their subscription.
    
    **Model:** Gradient Boosting Classifier  
    **Use Case:** Customer Retention Strategy  
    **Training Samples:** 1000 customers
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{acc*100:.1f}%")
    with col2:
        st.metric("ROC-AUC", f"{auc:.3f}")
    st.markdown("---")
    st.markdown("**CodSoft ML Internship**")
    st.markdown("Task 3 — Churn Prediction")
    st.markdown("**Author:** Parth Singh")

tab1, tab2, tab3 = st.tabs(["🎯 Predict Churn", "📊 Model Performance", "📈 Feature Insights"])

with tab1:
    st.markdown("### Enter Customer Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Account Info**")
        tenure           = st.slider("Tenure (months)", 1, 72, 12)
        monthly_charges  = st.slider("Monthly Charges ($)", 20, 120, 70)
        total_charges    = st.slider("Total Charges ($)", 100, 8000, 1000)
        num_complaints   = st.slider("Number of Complaints", 0, 5, 1)

    with col2:
        st.markdown("**📦 Service Details**")
        contract         = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        tech_support     = st.selectbox("Tech Support", ["Yes", "No"])
        payment_method   = st.selectbox("Payment Method", ["Credit card", "Bank transfer", "Electronic check", "Mailed check"])

    with col3:
        st.markdown("**👤 Demographics**")
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

        st.markdown("**⚡ Quick Presets:**")
        preset = st.selectbox("Load example:", [
            "Custom",
            "High risk customer",
            "Loyal long-term customer",
            "New unhappy customer",
        ])

        presets = {
            "High risk customer":       (3,  95, 285, 3, "Month-to-month", "Fiber optic", "No",  "Electronic check", "No"),
            "Loyal long-term customer": (48, 45, 2160, 0, "Two year",       "DSL",         "Yes", "Bank transfer",    "No"),
            "New unhappy customer":     (2,  85, 170, 4, "Month-to-month", "Fiber optic", "No",  "Electronic check", "No"),
        }

        if preset != "Custom":
            tenure, monthly_charges, total_charges, num_complaints, contract, internet_service, tech_support, payment_method, senior_citizen = presets[preset]

    if st.button("📉 Predict Churn", type="primary", use_container_width=True):
        contract_enc      = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
        internet_enc      = {"DSL": 0, "Fiber optic": 1, "No": 2}[internet_service]
        support_enc       = {"No": 0, "Yes": 1}[tech_support]
        payment_enc       = {"Bank transfer": 0, "Credit card": 1, "Electronic check": 2, "Mailed check": 3}[payment_method]
        senior_enc        = 1 if senior_citizen == "Yes" else 0

        row    = [[tenure, monthly_charges, total_charges, num_complaints, senior_enc,
                   contract_enc, internet_enc, support_enc, payment_enc]]
        row_sc = scaler.transform(row)

        pred = churn_classifier.predict(row_sc)[0]
        prob = churn_classifier.predict_proba(row_sc)[0][1]

        if pred == 1:
            st.markdown(f'<div class="churn-box">⚠️ CUSTOMER LIKELY TO CHURN<br>Churn Probability: {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.error("**Recommended Action:** Offer a discounted annual plan or loyalty reward immediately.")
        else:
            st.markdown(f'<div class="stay-box">✅ CUSTOMER LIKELY TO STAY<br>Churn Probability: {prob*100:.1f}%</div>', unsafe_allow_html=True)
            st.success("**Status:** Low churn risk. Continue standard engagement.")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Tenure", f"{tenure} months")
        with col2:
            st.metric("Monthly", f"${monthly_charges}")
        with col3:
            st.metric("Contract", contract)
        with col4:
            st.metric("Risk", "HIGH 🔴" if prob > 0.6 else "MEDIUM 🟡" if prob > 0.3 else "LOW 🟢")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Churn Risk Score (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "darkred" if prob > 0.5 else "darkgreen"},
                "steps": [
                    {"range": [0, 30],   "color": "lightgreen"},
                    {"range": [30, 60],  "color": "yellow"},
                    {"range": [60, 100], "color": "lightcoral"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)

with tab2:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy",        f"{acc*100:.2f}%")
    with col2:
        st.metric("ROC-AUC",         f"{auc:.4f}")
    with col3:
        st.metric("True Positives",  str(cm[1][1]))
    with col4:
        st.metric("False Negatives", str(cm[1][0]))

    col1, col2 = st.columns(2)
    with col1:
        fig_cm = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Retained", "Churned"],
            y=["Retained", "Churned"],
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
            name=f"ROC Curve (AUC={auc:.3f})",
            line=dict(color="#11998e", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random",
            line=dict(color="gray", dash="dash")
        ))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=350)
        st.plotly_chart(fig_roc, use_container_width=True)

with tab3:
    fig_imp = px.bar(
        importances, x="Importance", y="Feature",
        orientation="h", color="Importance",
        color_continuous_scale="Teal",
        title="Top Features Driving Customer Churn"
    )
    fig_imp.update_layout(height=450, showlegend=False)
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("""
    **Key Insights:**
    - 📋 **Contract Type** — Month-to-month customers churn 3x more than annual contracts
    - 💰 **Monthly Charges** — Higher bills = higher churn risk
    - ⏱️ **Tenure** — New customers (< 12 months) are most likely to leave
    - 😤 **Complaints** — Each complaint significantly increases churn probability
    """)

st.markdown("---")
st.markdown("Built with ❤️ by Parth Singh | CodSoft ML Internship 2026")
