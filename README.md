# 🔍 TruthLens — Fake News Detector

An AI-powered web application that detects fake news using Machine Learning. Built with Flask, scikit-learn, and NLTK.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Accuracy](https://img.shields.io/badge/Accuracy-96.24%25-brightgreen)
![Dataset](https://img.shields.io/badge/Dataset-72134%20Articles-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📸 Preview

> Paste any news headline or article and instantly find out if it is Real or Fake!

---

## ✨ Features

- ✅ Detects fake news with **96.24% accuracy**
- ✅ Shows **confidence percentage** for predictions
- ✅ Beautiful modern **dark editorial UI**
- ✅ Trained on **72,134 real news articles**
- ✅ Fast predictions in **milliseconds**
- ✅ Simple and easy to use

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript |
| Backend | Python, Flask |
| Machine Learning | scikit-learn, NLTK |
| Vectorization | TF-IDF (15,000 features) |
| Model | Logistic Regression |
| Dataset | WELFake (72,134 articles) |

---

## 📁 Project Structure

```
Fake News Detection/
├── app.py              # Flask web server & API
├── train_model.py      # Model training script
├── model.pkl           # Trained ML model
├── vectorizer.pkl      # TF-IDF vectorizer
├── templates/
│   └── index.html      # Frontend UI
└── static/
    └── style.css       # Stylesheet
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/jaychauhan69/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Install dependencies
```bash
pip install flask nltk scikit-learn pandas numpy
```

### 3. Train the model

**Option A — Built-in dataset (no download needed):**
```bash
python train_model.py
```

**Option B — WELFake dataset (best accuracy - 96.24%):**
1. Download WELFake dataset from [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification)
2. Place `WELFake_Dataset.csv` in the project folder
3. Run:
```bash
python train_model.py --welfake
```

**Option C — Kaggle Fake/Real dataset:**
1. Download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Place `Fake.csv` and `True.csv` in project folder
3. Run:
```bash
python train_model.py --kaggle
```

### 4. Start the app
```bash
python app.py
```

### 5. Open browser
```
http://127.0.0.1:5000
```

---

## 🤖 Machine Learning Pipeline

```
Input Text
    ↓
Text Cleaning (NLTK)
Remove URLs, HTML, stopwords, lemmatize
    ↓
TF-IDF Vectorization
15,000 features, unigrams + bigrams + trigrams
    ↓
Logistic Regression Model
Trained on 72,134 articles
    ↓
Prediction + Confidence Score
Fake News or Real News
```

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 96.24% |
| Dataset Size | 72,134 articles |
| Fake Articles | 35,028 |
| Real Articles | 37,106 |
| TF-IDF Features | 15,000 |
| Train/Test Split | 80% / 20% |

---

## 🌐 API Reference

### POST /predict

**Request:**
```json
{
  "text": "Your news headline or article here"
}
```

**Response:**
```json
{
  "label": "Fake News",
  "prediction": 0,
  "confidence": 94.5,
  "fake_prob": 94.5,
  "real_prob": 5.5
}
```

### GET /health
```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## 📦 Datasets Used

| Dataset | Articles | Link |
|---|---|---|
| WELFake | 72,134 | [Kaggle](https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification) |
| Kaggle Fake/Real | 44,898 | [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |

> Note: Datasets are not included in this repository due to large file size. Download from Kaggle links above.

---

## ⚠️ Disclaimer

This tool is for educational and research purposes only. The machine learning model can make errors. Always verify news with trusted primary sources.

---

## 👨‍💻 Author

**Jay Chauhan**
- GitHub: [@jaychauhan69](https://github.com/jaychauhan69)

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ If you found this project helpful, please give it a star on GitHub!
"# TruthLens" 
