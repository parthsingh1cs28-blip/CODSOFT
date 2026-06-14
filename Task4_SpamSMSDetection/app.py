"""
Spam SMS Detection - Streamlit Web App
CodSoft ML Internship - Task 4
Author: Parth Singh
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Spam Detector", page_icon="📱", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .spam-box {
        background: linear-gradient(135deg, #f5576c, #f093fb);
        padding: 2rem; border-radius: 15px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: bold; margin: 1rem 0;
    }
    .ham-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        padding: 2rem; border-radius: 15px; text-align: center;
        color: white; font-size: 1.8rem; font-weight: bold; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_spam_model():
    spam_messages = [
        "WINNER!! You have been selected to receive a $1000 prize. Call now to claim!",
        "FREE entry in 2 a weekly competition to win FA Cup final tickets! Text FA to 87121",
        "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
        "URGENT: Your account has been compromised. Call this number immediately to secure it.",
        "You have won 500000 dollars. Send your bank details to claim your reward today.",
        "Get rich quick! Work from home and earn $5000 per week. No experience needed!",
        "Your mobile number has won £1,000,000 in our lottery. Claim within 24 hours!",
        "FREE SMS! Reply STOP to unsubscribe. You have been pre-approved for a loan!",
        "Click here for a FREE gift card worth $500. Limited time offer. Act now!",
        "Urgent! Your bank account will be suspended. Verify your details immediately.",
        "You are a lucky winner of our grand draw. Call 09061743386 to collect prize.",
        "Claim your free ringtone now! Text RING to 85069. Only £3 per week.",
        "Hot singles in your area want to meet you tonight. Click here to see them!",
        "Your PayPal account has been limited. Click here to restore access now.",
        "Win a holiday for 2 to USA or Europe. To claim call 09050000327 now.",
        "CASH PRIZE of $3000 waiting for you! Reply YES to claim. T&C apply.",
        "You've been selected for a special offer. Buy now and get 90% discount!",
        "Congratulations you have won a Nokia phone. To claim text NOKIA to 82277.",
        "IMPORTANT: Your subscription will expire. Renew now to avoid service disruption.",
        "Make money fast! Invest in crypto and earn 500% returns guaranteed!",
        "Free msg: Txt CALL to 86688 and you could win a weekends getaway for two!",
        "SIX chances to WIN CASH! From 100 to 20,000 pounds txt> CSH11 to 87575.",
        "Did you know you can now get your insurance cheaper? Call free on 0800 505 060.",
        "Todays Vodafone numbers ending with 2603 are selected to receive a 350 award.",
        "Your free ringtones are waiting. To get them txt INFO to 83383.",
        "Claim your prize now! You have been randomly selected to win $10,000 cash.",
        "APPLY NOW for a credit card with 0% interest for 12 months. Bad credit OK!",
        "You won! To collect your winnings of 1000 pounds call 0906 260 1234 now.",
        "FREE CASH! No catches, no gimmicks. You have been pre-selected to receive cash.",
        "Urgent! Reply with your credit card number to claim your exclusive reward.",
    ]

    ham_messages = [
        "Hey, are you coming to the party tonight?",
        "I'll be home by 7pm. Can you please start dinner?",
        "Thanks for the help yesterday. Really appreciated it!",
        "Can we reschedule our meeting to Thursday afternoon?",
        "The exam results are out. I passed with distinction!",
        "Mom wants to know if you're coming for dinner on Sunday.",
        "I'm running 10 minutes late. Traffic is terrible today.",
        "Did you see the match last night? It was incredible!",
        "Happy birthday! Hope you have a wonderful day today.",
        "I left my keys at your place. Can I pick them up tomorrow?",
        "The project deadline has been extended to next Friday.",
        "Can you send me the report before the meeting at 3pm?",
        "I'll pick up the kids from school. Don't worry about it.",
        "We need to buy groceries. I'll stop at the store tonight.",
        "The doctor confirmed everything is fine. No issues found.",
        "Great game today! You played really well out there.",
        "I'm at the library. Will be back around 6pm tonight.",
        "Can you help me move this weekend? I'll buy you lunch!",
        "The flight is delayed by 2 hours. Will update you soon.",
        "Just finished the gym. Feeling great! See you at home.",
        "Remember we have a parent teacher meeting on Wednesday.",
        "The baby just said her first word! We are so excited.",
        "I got the job! Starting next Monday. So happy right now.",
        "Don't forget to take your medicine before going to bed.",
        "Can you call me when you get a chance? It's important.",
        "The wifi password is written on the back of the router.",
        "I sent you the document. Please review and send feedback.",
        "We're planning a surprise party for dad. Keep it secret!",
        "The train is on time today. I'll be there by 5:30pm.",
        "Lunch was amazing! We should go to that restaurant again.",
        "Your package has been delivered and left at the front door.",
        "I finished reading that book you recommended. Loved it!",
        "The meeting has been moved to the conference room on floor 3.",
        "Can you water my plants while I'm away on vacation?",
        "I just saw the news. Are you okay? Please call me back.",
    ]

    messages = spam_messages + ham_messages
    labels   = ["spam"] * len(spam_messages) + ["ham"] * len(ham_messages)

    df = pd.DataFrame({"message": messages, "label": labels})

    def clean_sms(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s!?£$]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["clean_msg"] = df["message"].apply(clean_sms)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_msg"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    spam_detector = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=3000)),
        ("clf",   LogisticRegression(max_iter=300, random_state=42, C=1.5))
    ])

    spam_detector.fit(X_train, y_train)
    y_pred  = spam_detector.predict(X_test)
    y_proba = spam_detector.predict_proba(X_test)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, (y_test == "spam").astype(int),
                             sample_weight=None)
    cm      = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])

    return spam_detector, acc, cm, clean_sms, df

spam_detector, acc, cm, clean_sms, df = train_spam_model()

st.markdown('<div class="main-header">📱 Spam SMS Detector</div>', unsafe_allow_html=True)
st.markdown("### Powered by TF-IDF + Logistic Regression | Real-time SMS Classification")
st.markdown("---")

with st.sidebar:
    st.markdown("## 📱 About This App")
    st.markdown("""
    Classifies SMS messages as Spam or Legitimate using NLP.
    
    **Model:** Logistic Regression  
    **Features:** TF-IDF (unigrams + bigrams)  
    **Training:** 65 real-world SMS examples
    """)
    st.metric("Model Accuracy", f"{acc*100:.1f}%")
    st.markdown("---")
    st.markdown("**CodSoft ML Internship**")
    st.markdown("Task 4 — Spam SMS Detection")
    st.markdown("**Author:** Parth Singh")

tab1, tab2, tab3 = st.tabs(["📱 Detect Spam", "📊 Model Performance", "🔍 SMS Analysis"])

with tab1:
    st.markdown("### Paste any SMS message below")

    example_msgs = [
        "",
        "WINNER!! You have been selected to receive $1000 prize. Call now!",
        "Hey, are you coming to the party tonight?",
        "Congratulations! You won a free iPhone. Click here to claim!",
        "Can we reschedule our meeting to Thursday afternoon?",
        "Your account has been compromised. Call immediately to secure it.",
        "I'll be home by 7pm. Can you please start dinner?",
        "FREE CASH! No catches. You have been pre-selected to receive money.",
        "Don't forget to take your medicine before going to bed.",
    ]

    selected = st.selectbox("Try an example SMS:", example_msgs)
    user_sms = st.text_area(
        "Or type your own SMS:",
        value=selected,
        height=120,
        placeholder="Paste any SMS message here..."
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Check SMS", type="primary", use_container_width=True):
            if user_sms.strip():
                cleaned  = clean_sms(user_sms)
                pred     = spam_detector.predict([cleaned])[0]
                proba    = spam_detector.predict_proba([cleaned])[0]
                classes  = spam_detector.classes_
                spam_idx = list(classes).index("spam")
                spam_prob = proba[spam_idx]

                if pred == "spam":
                    st.markdown(f'<div class="spam-box">🚨 SPAM DETECTED!<br>Confidence: {spam_prob*100:.1f}%</div>', unsafe_allow_html=True)
                    st.error("⚠️ Warning: This message shows strong spam indicators. Do not click any links or call any numbers.")
                else:
                    st.markdown(f'<div class="ham-box">✅ LEGITIMATE SMS<br>Spam Probability: {spam_prob*100:.1f}%</div>', unsafe_allow_html=True)
                    st.success("This message appears to be a genuine SMS from a real person.")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Classification", "🚨 SPAM" if pred == "spam" else "✅ HAM")
                with col2:
                    st.metric("Spam Prob", f"{spam_prob*100:.1f}%")
                with col3:
                    st.metric("Risk Level", "HIGH 🔴" if spam_prob > 0.7 else "MEDIUM 🟡" if spam_prob > 0.4 else "LOW 🟢")

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=spam_prob * 100,
                    title={"text": "Spam Risk Score (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar":  {"color": "darkred" if spam_prob > 0.5 else "darkgreen"},
                        "steps": [
                            {"range": [0, 30],   "color": "lightgreen"},
                            {"range": [30, 60],  "color": "yellow"},
                            {"range": [60, 100], "color": "lightcoral"},
                        ],
                        "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 50}
                    }
                ))
                fig_gauge.update_layout(height=280)
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.warning("Please enter an SMS message first!")

    with col2:
        st.markdown("### 🚩 Spam Red Flags")
        st.markdown("""
        Common spam indicators:
        - 🏆 **Prize/Winner** language
        - 💰 **Free money** offers
        - ⚡ **Urgent** action required
        - 📞 **Call now** instructions
        - 🔗 **Click here** links
        - 💳 **Bank/account** alerts
        - 🎁 **Free gift** offers
        """)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{acc*100:.1f}%")
    with col2:
        st.metric("True Spam Caught", str(cm[1][1]))
    with col3:
        st.metric("Spam Missed", str(cm[1][0]))

    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["Ham (Legit)", "Spam"],
        y=["Ham (Legit)", "Spam"],
        color_continuous_scale="RdYlGn_r",
        title="Confusion Matrix — Spam vs Ham"
    )
    fig_cm.update_traces(text=cm, texttemplate="%{text}")
    fig_cm.update_layout(height=400)
    st.plotly_chart(fig_cm, use_container_width=True)

with tab3:
    st.markdown("### Dataset Overview")
    label_counts = df["label"].value_counts()
    fig_pie = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Training Data Distribution",
        color_discrete_map={"spam": "#f5576c", "ham": "#11998e"}
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### Sample Training Messages")
    st.dataframe(
        df[["message", "label"]].sample(10, random_state=42).reset_index(drop=True),
        use_container_width=True
    )

st.markdown("---")
st.markdown("Built with ❤️ by Parth Singh | CodSoft ML Internship 2026")
