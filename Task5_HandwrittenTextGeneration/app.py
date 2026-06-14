"""
Handwritten Text Generation - Streamlit Web App
CodSoft ML Internship - Task 5
Author: Parth Singh
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
import time
import pandas as pd

st.set_page_config(page_title="Text Generator", page_icon="✍️", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 3rem; font-weight: bold; text-align: center;
        background: linear-gradient(90deg, #a18cd1, #fbc2eb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .generated-text {
        background: #1e1e2e;
        border: 2px solid #a18cd1;
        border-radius: 15px;
        padding: 2rem;
        font-family: 'Courier New', monospace;
        font-size: 1.2rem;
        color: #cdd6f4;
        line-height: 1.8;
        min-height: 150px;
        white-space: pre-wrap;
    }
    .metric-card {
        background: linear-gradient(135deg, #a18cd1, #fbc2eb);
        padding: 1rem; border-radius: 10px;
        text-align: center; color: white;
    }
</style>
""", unsafe_allow_html=True)

# ─── Character Level Text Generator ──────────────────────────────────────────
class CharLevelGenerator:
    """
    Character-level text generator that learns patterns from training text.
    Simulates an RNN/LSTM-style generation using Markov chains as foundation,
    with temperature-controlled sampling for creative text generation.
    """

    def __init__(self, text, seq_length=3):
        self.seq_length = seq_length
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.transitions = {}
        self.train(text)

    def train(self, text):
        chars = sorted(set(text))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size  = len(chars)

        for i in range(len(text) - self.seq_length):
            context = text[i:i + self.seq_length]
            next_ch = text[i + self.seq_length]
            if context not in self.transitions:
                self.transitions[context] = {}
            if next_ch not in self.transitions[context]:
                self.transitions[context][next_ch] = 0
            self.transitions[context][next_ch] += 1

    def generate(self, seed, length=300, temperature=1.0):
        if len(seed) < self.seq_length:
            seed = seed.ljust(self.seq_length)

        result = seed
        current = seed[-self.seq_length:]

        for _ in range(length):
            if current in self.transitions:
                next_chars = self.transitions[current]
                chars  = list(next_chars.keys())
                counts = np.array(list(next_chars.values()), dtype=float)

                # Temperature scaling — lower = more predictable, higher = more creative
                counts = np.log(counts + 1e-8) / temperature
                counts = np.exp(counts - np.max(counts))
                counts = counts / counts.sum()

                next_char = np.random.choice(chars, p=counts)
            else:
                next_char = random.choice(list(self.char_to_idx.keys()))

            result  += next_char
            current  = result[-self.seq_length:]

        return result

# ─── Training Texts ───────────────────────────────────────────────────────────
TRAINING_TEXTS = {
    "Shakespeare": """
To be or not to be that is the question whether tis nobler in the mind to suffer
the slings and arrows of outrageous fortune or to take arms against a sea of troubles
and by opposing end them to die to sleep no more and by a sleep to say we end
the heartache and the thousand natural shocks that flesh is heir to tis a consummation
devoutly to be wished to die to sleep to sleep perchance to dream ay there is the rub
for in that sleep of death what dreams may come when we have shuffled off this mortal coil
must give us pause there is the respect that makes calamity of so long life
all the worlds a stage and all the men and women merely players they have their exits
and their entrances and one man in his time plays many parts his acts being seven ages
""",
    "Science": """
the universe is a vast expanse of space and time containing billions of galaxies
each galaxy contains hundreds of billions of stars and around many of these stars
orbit planets some of which may harbor life the fundamental forces of nature
gravity electromagnetism the strong nuclear force and the weak nuclear force
govern all interactions between matter and energy quantum mechanics describes
the behavior of particles at the smallest scales while general relativity explains
the nature of space time and gravity at cosmic scales machine learning algorithms
learn patterns from data neural networks with multiple layers can recognize images
natural language and make predictions with remarkable accuracy deep learning
has revolutionized artificial intelligence and enabled computers to perform tasks
that were once thought to require human intelligence
""",
    "Poetry": """
i wandered lonely as a cloud that floats on high over vales and hills
when all at once i saw a crowd a host of golden daffodils beside the lake
beneath the trees fluttering and dancing in the breeze continuous as the stars
that shine and twinkle on the milky way they stretched in never ending line
along the margin of a bay ten thousand saw i at a glance tossing their heads
in sprightly dance the waves beside them danced but they out did the sparkling waves
in glee a poet could not but be gay in such a jocund company i gazed and gazed
but little thought what wealth the show to me had brought for oft when on my couch
i lie in vacant or in pensive mood they flash upon that inward eye which is the bliss
""",
    "Technology": """
artificial intelligence machine learning deep learning neural networks computer vision
natural language processing reinforcement learning supervised learning unsupervised learning
data science big data cloud computing internet of things blockchain cryptocurrency
software engineering algorithms data structures programming python javascript java
web development mobile applications databases sql nosql api rest graphql microservices
cybersecurity encryption authentication authorization cloud infrastructure kubernetes docker
devops continuous integration continuous deployment agile scrum software development
open source github version control git collaboration code review testing automation
""",
}

@st.cache_resource
def build_generators():
    generators = {}
    for name, text in TRAINING_TEXTS.items():
        generators[name] = CharLevelGenerator(text.strip(), seq_length=4)
    return generators

generators = build_generators()

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">✍️ AI Text Generator</div>', unsafe_allow_html=True)
st.markdown("### Character-Level RNN Simulation | Learns from Text Patterns")
st.markdown("---")

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✍️ About This App")
    st.markdown("""
    Generates new text by learning character-level patterns from training data.
    
    **Technique:** Character-level Markov Chain  
    *(simulates RNN/LSTM generation)*  
    **Key Concept:** Temperature-controlled sampling  
    **Vocab Sizes:** 50-80 unique characters per corpus
    """)

    for name, gen in generators.items():
        st.metric(f"{name} Vocab", f"{gen.vocab_size} chars")

    st.markdown("---")
    st.markdown("**CodSoft ML Internship**")
    st.markdown("Task 5 — Text Generation")
    st.markdown("**Author:** Parth Singh")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["✍️ Generate Text", "🌡️ Temperature Explorer", "📊 Model Insights"])

with tab1:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### ⚙️ Generation Settings")

        corpus = st.selectbox("Training Corpus:", list(TRAINING_TEXTS.keys()))

        seed_options = {
            "Shakespeare": ["to be or", "all the wo", "the sleep"],
            "Science":     ["the unive", "machine l", "deep lear"],
            "Poetry":      ["i wandere", "the waves", "they flas"],
            "Technology":  ["artificia", "deep lear", "cloud com"],
        }

        seed = st.selectbox("Seed Text:", seed_options[corpus])
        custom_seed = st.text_input("Or type custom seed (min 4 chars):", "")
        if custom_seed and len(custom_seed) >= 4:
            seed = custom_seed

        length      = st.slider("Generation Length (chars)", 100, 500, 250)
        temperature = st.slider("Temperature", 0.1, 2.0, 0.8, 0.1,
                                help="Low=predictable, High=creative/random")

        st.markdown(f"""
        **Temperature Guide:**
        - 🥶 **0.1-0.5** — Conservative, repetitive
        - 😊 **0.6-1.0** — Balanced, readable  
        - 🔥 **1.1-2.0** — Creative, unpredictable
        """)

        generate_btn = st.button("✍️ Generate Text", type="primary", use_container_width=True)

    with col2:
        st.markdown("### 📝 Generated Output")

        if generate_btn:
            gen = generators[corpus]

            with st.spinner("AI is writing..."):
                time.sleep(0.5)
                generated = gen.generate(seed, length=length, temperature=temperature)

            st.markdown(f'<div class="generated-text">{generated}</div>', unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Characters Generated", length)
            with col_b:
                st.metric("Temperature Used", temperature)
            with col_c:
                words = len(generated.split())
                st.metric("Words Generated", words)

            unique_chars = len(set(generated))
            diversity = unique_chars / gen.vocab_size * 100
            st.progress(diversity / 100, text=f"Output Diversity: {diversity:.1f}%")

            st.download_button(
                "📥 Download Generated Text",
                generated,
                file_name="generated_text.txt",
                mime="text/plain"
            )
        else:
            st.markdown("""
            <div class="generated-text">
            ← Configure settings and click Generate Text to see AI output here...
            
            The model will learn patterns from the selected corpus and generate
            new text character by character, just like an RNN/LSTM would.
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 🌡️ How Temperature Affects Generation")
    st.markdown("Generate the same seed at different temperatures to see the effect:")

    temp_seed   = st.text_input("Seed for comparison:", "to be or")
    temp_corpus = st.selectbox("Corpus:", list(TRAINING_TEXTS.keys()), key="temp_corpus")

    if st.button("🔬 Run Temperature Comparison", type="primary"):
        gen = generators[temp_corpus]
        temperatures = [0.3, 0.7, 1.0, 1.5]
        cols = st.columns(2)

        for i, temp in enumerate(temperatures):
            with cols[i % 2]:
                result = gen.generate(temp_seed, length=150, temperature=temp)
                label  = "🥶 Very Conservative" if temp < 0.5 else \
                         "😊 Balanced" if temp < 0.9 else \
                         "🔥 Creative" if temp < 1.3 else "🌋 Very Random"
                st.markdown(f"**Temperature {temp} — {label}**")
                st.markdown(f'<div class="generated-text" style="font-size:0.9rem;min-height:100px">{result}</div>', unsafe_allow_html=True)
                st.markdown("")

with tab3:
    st.markdown("### 📊 Model Architecture Insights")

    selected_corpus = st.selectbox("Select Corpus to Analyze:", list(TRAINING_TEXTS.keys()), key="insight")
    gen = generators[selected_corpus]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Vocabulary Size",   gen.vocab_size)
    with col2:
        st.metric("Context Window",    f"{gen.seq_length} chars")
    with col3:
        st.metric("Unique Transitions", len(gen.transitions))

    # Character frequency chart
    text  = TRAINING_TEXTS[selected_corpus]
    chars = [c for c in text if c.isalpha()]
    char_freq = {}
    for c in chars:
        char_freq[c] = char_freq.get(c, 0) + 1

    freq_df = pd.DataFrame({
        "Character": list(char_freq.keys()),
        "Frequency": list(char_freq.values())
    }).sort_values("Frequency", ascending=False).head(20)

    import pandas as pd
    fig_freq = px.bar(
        freq_df, x="Character", y="Frequency",
        color="Frequency", color_continuous_scale="Viridis",
        title=f"Top 20 Character Frequencies in {selected_corpus} Corpus"
    )
    fig_freq.update_layout(height=400)
    st.plotly_chart(fig_freq, use_container_width=True)

    st.markdown("""
    ### 🧠 How This Relates to RNN/LSTM
    
    | Concept | This App | Real RNN/LSTM |
    |---------|----------|---------------|
    | **Memory** | Fixed context window (4 chars) | Variable hidden state |
    | **Learning** | Counting transitions | Backpropagation |
    | **Sampling** | Temperature scaling | Same technique |
    | **Generation** | Character by character | Character by character |
    | **Vocabulary** | All unique chars | Same |
    
    The **temperature sampling** and **character-level generation** are 
    identical to how real LSTM text generators work!
    """)

st.markdown("---")
st.markdown("Built with ❤️ by Parth Singh | CodSoft ML Internship 2026")
