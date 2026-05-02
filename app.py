"""
app.py - Flask backend for the Fake News Detector
"""

import os
import pickle
import re
import string

import nltk
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data silently on first run
for pkg in ("stopwords", "wordnet", "punkt", "omw-1.4"):
    nltk.download(pkg, quiet=True)

app = Flask(__name__)

# ── Load ML artefacts ──────────────────────────────────────────────────────────
MODEL_PATH      = os.path.join(os.path.dirname(__file__), "model.pkl")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.pkl")

model      = None
vectorizer = None

def load_artefacts():
    global model, vectorizer
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return False
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return True


loaded = load_artefacts()
if loaded:
    print("✅  Model and vectorizer loaded successfully.")
else:
    print("⚠️  model.pkl / vectorizer.pkl not found.")
    print("   Run  python train_model.py  first, then restart the server.")


# ── Text preprocessing (must match train_model.py) ─────────────────────────────
lemmatizer = WordNetLemmatizer()
stop_words  = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded. Run python train_model.py first."}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please provide news text to analyse."}), 400

    if len(text) < 10:
        return jsonify({"error": "Text is too short. Please provide a headline or article."}), 400

    cleaned   = clean_text(text)
    vectorised = vectorizer.transform([cleaned])

    prediction  = model.predict(vectorised)[0]
    proba       = model.predict_proba(vectorised)[0]

    label       = "Real News" if prediction == 1 else "Fake News"
    confidence  = float(max(proba)) * 100
    real_prob   = float(proba[1]) * 100
    fake_prob   = float(proba[0]) * 100

    return jsonify({
        "label":      label,
        "prediction": int(prediction),
        "confidence": round(confidence, 1),
        "real_prob":  round(real_prob, 1),
        "fake_prob":  round(fake_prob, 1),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)