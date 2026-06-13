"""
Movie Genre Classification using TF-IDF + Logistic Regression
CodSoft ML Internship - Task 1
Author: Parth Singh
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import re

# Sample Dataset 
data = {
    "description": [
        "A group of superheroes battle an alien invasion threatening Earth",
        "Two strangers fall in love during a summer vacation in Paris",
        "A detective hunts a serial killer through dark city streets at night",
        "Friends go on a hilarious road trip full of unexpected mishaps",
        "Astronauts struggle to survive after their spacecraft malfunctions",
        "A young wizard discovers his magical powers and fights evil",
        "Secret agents must stop a terrorist plot to destroy the world",
        "A couple navigates heartbreak and rediscovers themselves",
        "Scientists race to stop a deadly virus from wiping out humanity",
        "A comedian's stand-up tour goes hilariously wrong",
        "Two rivals compete in a deadly tournament for ultimate glory",
        "A ghost haunts the family living in her old mansion",
        "An underdog team fights their way to the championship",
        "A forbidden romance blossoms between enemies during wartime",
        "A hacker uncovers a government conspiracy threatening freedom",
        "Monsters terrorize a small town with no way to escape",
        "A chef opens a new restaurant and falls in love with their sous-chef",
        "An ex-soldier goes on a dangerous mission to rescue hostages",
        "Robots gain consciousness and rebel against their human creators",
        "A family vacation turns into a series of comedic disasters",
    ],
    "genre": [
        "Action", "Romance", "Thriller", "Comedy", "Sci-Fi",
        "Fantasy", "Action", "Romance", "Thriller", "Comedy",
        "Action", "Horror", "Sports", "Romance", "Thriller",
        "Horror", "Romance", "Action", "Sci-Fi", "Comedy",
    ]
}

df = pd.DataFrame(data)

#  Text Preprocessing 
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_desc"] = df["description"].apply(clean_text)

print("Dataset Overview:")
print(df[["description", "genre"]].head(10))
print(f"\nGenre Distribution:\n{df['genre'].value_counts()}\n")

# Train / Test Split 
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_desc"], df["genre"], test_size=0.2, random_state=42
)

# Model Pipeline: TF-IDF + Logistic Regression 
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=200, random_state=42))
])

pipeline.fit(X_train, y_train)

# Evaluation 
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Predict on New Descriptions 
def predict_genre(description):
    cleaned = clean_text(description)
    prediction = pipeline.predict([cleaned])[0]
    proba = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    top_idx = np.argsort(proba)[::-1][:3]
    print(f"\nInput: {description}")
    print(f"Predicted Genre: {prediction}")
    print("Top 3 probabilities:")
    for i in top_idx:
        print(f"  {classes[i]}: {proba[i]*100:.1f}%")

# Demo predictions
predict_genre("A brave soldier fights to save his country from invasion")
predict_genre("Two people meet unexpectedly and slowly fall in love")
predict_genre("Scientists discover an alien signal from deep space")
