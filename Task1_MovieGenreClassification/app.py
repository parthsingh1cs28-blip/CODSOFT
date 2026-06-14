"""
Movie Genre Classification - Streamlit Web App
CodSoft ML Internship - Task 1
Author: Parth Singh
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    data = {
        "description": [
            # ACTION (15)
            "A group of superheroes battle an alien invasion threatening Earth",
            "Secret agents must stop a terrorist plot to destroy the world",
            "An ex-soldier goes on a dangerous mission to rescue hostages",
            "A superhero loses his powers and must rediscover his true strength",
            "Two rivals compete in a deadly martial arts tournament for glory",
            "A retired hitman is forced back into action to avenge his family",
            "A brave warrior fights to protect his kingdom from an invading army",
            "A police officer goes undercover to infiltrate a dangerous gang",
            "A mercenary team is hired to overthrow a ruthless dictator",
            "An elite soldier battles terrorists who have taken over a building",
            "A fighter pilot must stop a missile attack from destroying a city",
            "A special forces team rescues prisoners from a jungle war camp",
            "A vigilante takes justice into his own hands after his son is killed",
            "An army ranger leads his squad through enemy territory to safety",
            "A martial arts champion fights to save his kidnapped daughter",
            # ROMANCE (15)
            "Two strangers fall in love during a summer vacation in Paris",
            "A couple navigates heartbreak and rediscovers themselves after loss",
            "A forbidden romance blossoms between enemies during wartime chaos",
            "Two childhood friends reunite and fall deeply in love after years apart",
            "A chef opens a restaurant and falls in love with their sous-chef",
            "A writer falls in love with his muse who may not be real",
            "Two people from different worlds find love against all odds",
            "A young couple must fight to keep their love alive across distance",
            "A wedding planner falls for the groom she is supposed to help",
            "Two rivals in the workplace slowly fall deeply in love",
            "A soldier returns home and reconnects with his childhood sweetheart",
            "A woman falls in love with a mysterious stranger in a small town",
            "An arranged marriage slowly turns into a beautiful true love story",
            "Two people meet on a train and fall in love during the journey",
            "A blind date goes wrong but eventually leads to true love",
            # THRILLER (15)
            "A detective hunts a serial killer through dark city streets at night",
            "Scientists race to stop a deadly virus from wiping out humanity",
            "A hacker uncovers a massive government conspiracy threatening freedom",
            "A killer leaves cryptic clues and the detective must solve them fast",
            "A woman discovers her husband has been living a double secret life",
            "A witness to a murder must survive while being hunted by killers",
            "A forensic expert uncovers a massive corruption scandal in politics",
            "A journalist investigates a powerful criminal organization alone",
            "A psychologist realizes her patient is planning a horrific crime",
            "A missing child case leads a detective to a dark underground network",
            "An innocent man is framed for murder and must prove his innocence",
            "A spy must find a mole inside his own intelligence agency fast",
            "A bomb threat forces a detective to race against the clock to defuse",
            "A hostage negotiator faces his most dangerous case of his career",
            "A woman is stalked by an obsessive ex who will stop at nothing",
            # COMEDY (15)
            "Friends go on a hilarious road trip full of unexpected mishaps",
            "A comedian stand-up tour goes hilariously wrong on live television",
            "A family vacation turns into a series of non-stop comedic disasters",
            "Three friends accidentally stumble into a wildly absurd adventure",
            "A mistaken identity leads to a series of increasingly funny situations",
            "A man pretends to be someone else to impress a girl he likes",
            "Two complete strangers must pretend to be married for a weekend",
            "A clumsy secret agent accidentally solves the biggest case ever",
            "A group of office workers plan an elaborate prank on their mean boss",
            "A serious politician accidentally goes viral for all the wrong reasons",
            "A man babysits his nephews and everything goes completely wrong",
            "Two rivals must work together and hate every single second of it",
            "A shy man accidentally becomes the most popular person in his town",
            "A wedding goes completely off the rails in the funniest possible way",
            "A time traveler keeps accidentally making history even more chaotic",
            # SCI-FI (10)
            "Astronauts struggle to survive after their spacecraft malfunctions",
            "Robots gain consciousness and rebel violently against their human creators",
            "A space explorer discovers a new planet inhabited by strange beings",
            "Humans discover time travel and accidentally change course of history",
            "Scientists discover an alien signal from deep space and investigate",
            "A colony on Mars fights for survival after their oxygen system fails",
            "A soldier wakes up in the future and must find his way back home",
            "An AI becomes sentient and decides humans are a threat to be eliminated",
            "Earth faces extinction as an asteroid approaches at alarming speed",
            "Scientists create a portal to a parallel universe with dangerous results",
            # FANTASY (10)
            "A young wizard discovers his magical powers and fights evil sorcerers",
            "A dark wizard threatens to plunge the entire magical kingdom into chaos",
            "A young girl discovers a magical world hidden beneath her small town",
            "A chosen hero must collect ancient relics to defeat an immortal demon",
            "A dragon and a knight form an unlikely bond to save their kingdom",
            "An ordinary boy discovers he is the last heir to a magical throne",
            "A sorceress must break an ancient curse before the next full moon",
            "Warriors from a mystical land must unite to fight an ancient evil god",
            "A young elf embarks on a quest to find the lost crown of her ancestors",
            "A magician discovers a prophecy that says he will destroy the world",
            # HORROR (10)
            "A ghost haunts the family living in her old haunted mansion",
            "Monsters terrorize a small town with absolutely no way to escape",
            "A group of friends go camping and encounter terrifying supernatural forces",
            "A vampire hunter tracks an ancient evil through the streets of London",
            "A family moves into a new house and strange things start happening",
            "A scientist experiments on himself and transforms into a monster",
            "Survivors of a zombie outbreak must find safety across a dead city",
            "A demonic possession threatens a young girl and her desperate family",
            "A creature lurks in the deep dark woods hunting anyone who enters",
            "Children start disappearing in a small town and nobody knows why",
            # SPORTS (10)
            "An underdog sports team fights their way to the championship finals",
            "A retired boxer makes one last comeback to reclaim his lost glory",
            "A young cricket player from a village dreams of playing for his country",
            "A coach transforms a losing team into unlikely national champions",
            "A swimmer overcomes a career-ending injury to compete in Olympics",
            "India wins the world cup cricket after a nail-biting final match",
            "A football team from a poor neighborhood beats the richest club",
            "A tennis prodigy must overcome his fear to win the grand slam title",
            "A basketball player fights racism to become the greatest of all time",
            "A young sprinter breaks the world record against all expectations",
        ],
        "genre": (
            ["Action"] * 15 +
            ["Romance"] * 15 +
            ["Thriller"] * 15 +
            ["Comedy"] * 15 +
            ["Sci-Fi"] * 10 +
            ["Fantasy"] * 10 +
            ["Horror"] * 10 +
            ["Sports"] * 10
        )
    }

    df = pd.DataFrame(data)

    def clean_text(text):
        text = text.lower()
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    df["clean_desc"] = df["description"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_desc"], df["genre"], test_size=0.2, random_state=42
    )

    genre_model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
        ("clf", LogisticRegression(max_iter=500, random_state=42, C=2.0))
    ])

    genre_model.fit(X_train, y_train)
    y_pred = genre_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return genre_model, acc, df, clean_text

# ─── Main App ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🎬 Movie Genre Classifier</div>', unsafe_allow_html=True)
st.markdown("### Powered by TF-IDF + Logistic Regression")
st.markdown("---")

genre_model, acc, df, clean_text = train_model()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 About This App")
    st.markdown("""
    This ML model classifies movies into genres based on their plot description.
    
    **Model:** Logistic Regression  
    **Features:** TF-IDF Vectorizer  
    **Training Samples:** 100  
    **Genres:** Action, Romance, Thriller, Comedy, Sci-Fi, Fantasy, Horror, Sports
    """)
    st.metric("Model Accuracy", f"{acc*100:.1f}%")
    st.markdown("---")
    st.markdown("**CodSoft ML Internship**")
    st.markdown("Task 1 — Movie Genre Classification")
    st.markdown("**Author:** Parth Singh")

# ─── Input Section ────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Movie Description")
    user_input = st.text_area(
        "Describe the movie plot:",
        placeholder="e.g. A detective investigates a mysterious murder in a small town...",
        height=150
    )

    example_descriptions = [
        "A brave soldier fights to protect his nation from a powerful enemy army",
        "Two childhood sweethearts reunite after ten years and fall in love again",
        "A detective discovers the killer is someone very close to him all along",
        "India wins the world cup cricket in a thrilling last over finish",
        "An astronaut stranded on Mars must find a way to survive and return home",
        "A young boy discovers he has magical powers and enters a school for wizards",
        "A family moves into a haunted house and terror begins immediately",
        "Three best friends plan a ridiculous heist that goes completely wrong",
    ]

    st.markdown("**Or try an example:**")
    selected_example = st.selectbox("Choose example:", [""] + example_descriptions)
    if selected_example and not user_input.strip():
        user_input = selected_example

with col2:
    st.markdown("### 📊 Dataset Stats")
    genre_counts = df["genre"].value_counts()
    fig_pie = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        title="Training Data Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_pie.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

# ─── Prediction ────────────────────────────────────────────────────────────────
if st.button("🎯 Classify Genre", type="primary", use_container_width=True):
    if user_input.strip():
        cleaned = clean_text(user_input)
        prediction = genre_model.predict([cleaned])[0]
        proba = genre_model.predict_proba([cleaned])[0]
        classes = genre_model.classes_

        genre_emojis = {
            "Action": "💥", "Romance": "❤️", "Thriller": "😱",
            "Comedy": "😂", "Sci-Fi": "🚀", "Fantasy": "🧙",
            "Horror": "👻", "Sports": "🏆"
        }
        emoji = genre_emojis.get(prediction, "🎬")

        st.markdown(f"""
        <div class="prediction-box">
            {emoji} Predicted Genre: {prediction}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📈 Confidence Scores")
        prob_df = pd.DataFrame({
            "Genre": classes,
            "Confidence": proba * 100
        }).sort_values("Confidence", ascending=True)

        fig_bar = px.bar(
            prob_df,
            x="Confidence",
            y="Genre",
            orientation="h",
            color="Confidence",
            color_continuous_scale="Viridis",
            title="Genre Probability Distribution",
            labels={"Confidence": "Confidence (%)"}
        )
        fig_bar.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

        top3 = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)[:3]
        cols = st.columns(3)
        for i, (genre, prob) in enumerate(top3):
            with cols[i]:
                st.metric(f"#{i+1} {genre_emojis.get(genre,'🎬')} {genre}", f"{prob*100:.1f}%")
    else:
        st.warning("Please enter a movie description first!")

st.markdown("---")
st.markdown("Built with ❤️ by Parth Singh | CodSoft ML Internship 2026")
