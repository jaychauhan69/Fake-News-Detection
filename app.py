"""
app.py - TruthLens: Fake News + Image Authenticity Detection
"""

import os
import pickle
import re
import json
import uuid
import tempfile
import base64

import nltk
from flask import Flask, jsonify, render_template, request, send_file
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

for pkg in ("stopwords", "wordnet", "punkt", "omw-1.4"):
    nltk.download(pkg, quiet=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

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
    print("⚠️  model.pkl / vectorizer.pkl not found. Run train_model_2.py first.")

# ── Text preprocessing ─────────────────────────────────────────────────────────
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
        return jsonify({"error": "Model not loaded. Run train_model_2.py first."}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Please provide news text to analyse."}), 400
    if len(text) < 10:
        return jsonify({"error": "Text is too short. Please provide a headline or article."}), 400

    cleaned    = clean_text(text)
    vectorised = vectorizer.transform([cleaned])

    prediction = model.predict(vectorised)[0]
    proba      = model.predict_proba(vectorised)[0]

    label      = "Real News" if prediction == 1 else "Fake News"
    confidence = float(max(proba)) * 100
    real_prob  = float(proba[1]) * 100
    fake_prob  = float(proba[0]) * 100

    return jsonify({
        "label":      label,
        "prediction": int(prediction),
        "confidence": round(confidence, 1),
        "real_prob":  round(real_prob, 1),
        "fake_prob":  round(fake_prob, 1),
    })


@app.route("/analyze-image", methods=["POST"])
def analyze_image_route():
    try:
        from analyze import analyze_image
    except ImportError:
        return jsonify({"error": "Image analysis module not found. Make sure analyze.py and forensics_core.py are in the project folder."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected."}), 400

    allowed = {".jpg", ".jpeg", ".png", ".webp"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format '{ext}'. Use JPG, PNG, or WEBP."}), 400

    try:
        # Save uploaded file to temp directory
        tmp_dir = tempfile.mkdtemp()
        tmp_input = os.path.join(tmp_dir, f"upload{ext}")
        file.save(tmp_input)

        # Run analysis
        report, report_path, heatmap_path = analyze_image(tmp_input, tmp_dir)

        # Read heatmap as base64 to send to frontend
        heatmap_b64 = None
        if os.path.exists(heatmap_path):
            with open(heatmap_path, "rb") as f:
                heatmap_b64 = base64.b64encode(f.read()).decode("utf-8")

        return jsonify({
            "classification":      report["classification"],
            "confidence":          report["confidence_percentages"],
            "metadata":            report["metadata_summary"],
            "editing_detected":    report["editing_software_detected"],
            "ai_detected":         report["ai_generator_signature_detected"],
            "indicators":          report["forensic_indicators"][:5],
            "measurements":        report["technical_measurements"],
            "explanation":         report["explanation"],
            "heatmap_base64":      heatmap_b64,
            "suspicious_regions":  report["suspicious_regions_bbox_xywh"],
        })

    except Exception as e:
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
