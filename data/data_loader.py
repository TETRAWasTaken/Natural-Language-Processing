import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "dataset.csv")

def get_default_dataset():
    data = [
        {"id": 1, "category": "technology", "text": "Artificial intelligence and machine learning models are revolutionizing modern computer science and software development."},
        {"id": 2, "category": "technology", "text": "Deep learning neural networks require massive computing power and graphics processing units GPU for training."},
        {"id": 3, "category": "technology", "text": "Natural language processing allows computers to understand, analyze, and generate human languages effectively."},
        {"id": 4, "category": "technology", "text": "Cloud computing providers offer scalable virtual servers, storage databases, and AI machine learning APIs."},
        {"id": 5, "category": "technology", "text": "Python is a popular programming language for data science, artificial intelligence, and web development."},
        {"id": 6, "category": "space", "text": "Astronomers discover new planets and distant galaxies using powerful optical and infrared space telescopes."},
        {"id": 7, "category": "space", "text": "NASA and SpaceX launch rockets and spacecraft into outer space to explore Mars and the Moon."},
        {"id": 8, "category": "space", "text": "The solar system consists of the Sun, eight planets, moons, asteroids, comets, and cosmic dust particles."},
        {"id": 9, "category": "space", "text": "Astronauts aboard the international space station conduct scientific experiments in zero gravity orbit."},
        {"id": 10, "category": "space", "text": "Black holes possess immense gravitational pull such that even light cannot escape their event horizon."},
        {"id": 11, "category": "health", "text": "Regular physical exercise, balanced nutrition, and sufficient sleep promote long term cardiovascular health."},
        {"id": 12, "category": "health", "text": "Medical researchers develop novel vaccines, antibiotics, and therapeutic treatments to fight infectious viral diseases."},
        {"id": 13, "category": "health", "text": "Doctors and healthcare professionals recommend a diet rich in fruits, vegetables, proteins, and essential vitamins."},
        {"id": 14, "category": "health", "text": "Mental health awareness encourages mindfulness meditation, stress reduction techniques, and emotional wellness."},
        {"id": 15, "category": "health", "text": "Hospitals utilize advanced medical diagnostics, magnetic resonance imaging MRI, and genetic sequencing."},
        {"id": 16, "category": "sports", "text": "Football players train rigorously on tactical formations, sprint speed, passing accuracy, and teamwork."},
        {"id": 17, "category": "sports", "text": "The Olympic games bring together global athletes competing in athletics, swimming, gymnastics, and cycling."},
        {"id": 18, "category": "sports", "text": "Basketball teams focus on fast breaks, three point shooting defense, rebounds, and tactical coaching."},
        {"id": 19, "category": "sports", "text": "Tennis champions master powerful serves, baseline forehands, backhand slices, and mental endurance."},
        {"id": 20, "category": "sports", "text": "Marathon runners require intense cardiovascular endurance, proper hydration, and strategic pacing."},
        {"id": 21, "category": "finance", "text": "Stock market investors analyze corporate financial statements, earnings reports, interest rates, and revenue growth."},
        {"id": 22, "category": "finance", "text": "Central banks manage monetary policy, inflation targets, interest rates, and economic stability."},
        {"id": 23, "category": "finance", "text": "Venture capital funds invest capital into early stage technology startups with high growth potential."},
        {"id": 24, "category": "finance", "text": "Cryptocurrency assets, digital tokens, and blockchain ledgers facilitate decentralized financial transactions."},
        {"id": 25, "category": "finance", "text": "Global trade, supply chain logistics, and currency exchange rates affect international business commerce."}
    ]
    return pd.DataFrame(data)

def load_dataset(filepath=CSV_PATH):
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    df = get_default_dataset()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return df

def clean_text(text, remove_stopwords=True):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    if remove_stopwords:
        try:
            stop_words = set(stopwords.words("english"))
        except Exception:
            stop_words = {"and", "the", "is", "in", "to", "for", "with", "on", "of", "a", "an"}
        tokens = [w for w in text.split() if w not in stop_words and len(w) > 1]
        text = " ".join(tokens)
    return text

def tokenize_corpus(texts):
    return [text.split() for text in texts if len(text.split()) > 0]
