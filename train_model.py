"""
train_model.py - Train the Fake News Detection model

Usage:
  Option 1 (Kaggle dataset): Place 'Fake.csv' and 'True.csv' from Kaggle in this directory, then run:
      python train_model.py --kaggle

  Option 2 (built-in demo dataset): Run without arguments:
      python train_model.py
"""

import argparse
import pickle
import re
import string

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """Clean and normalize text for vectorization."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)           # remove URLs
    text = re.sub(r"<.*?>", "", text)                      # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)                  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


def load_kaggle_data() -> pd.DataFrame:
    """Load the Kaggle Fake/True news CSVs."""
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 0   # 0 = fake
    true["label"] = 1   # 1 = real
    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
    return df[["text", "label"]]


def load_demo_data() -> pd.DataFrame:
    """Built-in sample dataset for demonstration purposes."""
    fake_headlines = [
        "BREAKING: Scientists discover drinking bleach cures all diseases instantly",
        "Shocking: The moon is actually a giant disco ball installed by NASA in 1969",
        "EXPOSED: All doctors secretly paid by Big Pharma to make you sick",
        "EXCLUSIVE: Aliens have been living in the White House basement since 1947",
        "URGENT: 5G towers confirmed to control human minds, leaked documents show",
        "BOMBSHELL: Bill Gates microchipping the population through flu shots",
        "REVEALED: The Earth is flat and space agencies have been lying for decades",
        "SECRET: Chemtrails contain mind-control chemicals, whistleblower claims",
        "CONFIRMED: Eating chocolate daily cures cancer, suppressed by medical establishment",
        "SHOCKING: Mainstream media covering up alien invasion happening right now",
        "Democrat politician caught selling children to underground sex ring, sources say",
        "COVID vaccine makes people magnetic, nurse demonstrates with spoon on forehead",
        "The government is putting fluoride in water to make people obedient, expert warns",
        "PROOF: Obama was born in Kenya according to secret document just revealed",
        "Scientists find evidence Noah's ark landed on Mount Ararat, Bible confirmed true",
        "LEAKED: Major celebrities are actually lizard people, Hollywood insider reveals",
        "President signs secret order to confiscate all guns by end of the year, sources say",
        "Miracle cure for diabetes discovered by native tribe, Big Pharma trying to suppress it",
        "New study proves vaccines cause autism, was hidden by CDC for decades",
        "Billionaires plan to depopulate Earth by 90% by 2030, leaked document shows",
        "Election was stolen through Dominion voting machines, expert analysis confirms",
        "Deep state operatives assassinated president's top advisor last night",
        "Soros funding antifa terrorists to overthrow US government this summer",
        "QAnon prediction confirmed: major arrest of global elite happening tomorrow",
        "Hollywood celebrities drinking adrenochrome harvested from children, report says",
        "Pentagon admits using weather control technology to cause recent hurricanes",
        "Microplastics in drinking water proven to change sexual orientation, new study",
        "FEMA concentration camps ready to house political dissidents, journalist reveals",
        "AI robots secretly taking over government positions, insiders confirm",
        "New world order plan to eliminate middle class accelerating, economist warns",
    ]

    real_headlines = [
        "Federal Reserve raises interest rates by 25 basis points to combat inflation",
        "NASA's James Webb Telescope captures stunning images of distant galaxies",
        "Scientists develop new vaccine candidate showing promise against malaria",
        "EU parliament passes landmark data privacy legislation affecting tech companies",
        "Researchers find correlation between sleep deprivation and cognitive decline",
        "Stock markets fall amid concerns over rising inflation and interest rates",
        "New archaeological dig in Egypt uncovers previously unknown pharaoh's tomb",
        "Climate scientists report Arctic sea ice at record low this decade",
        "UN peacekeeping mission deployed to conflict zone following ceasefire agreement",
        "Major pharmaceutical company reports successful phase 3 trial for new drug",
        "Global chip shortage expected to ease as manufacturers expand production",
        "Wildfires force thousands to evacuate as firefighters battle blazes in California",
        "WHO reports measles cases rising globally due to declining vaccination rates",
        "Congress passes bipartisan infrastructure bill allocating billions for roads and bridges",
        "New study links ultra-processed food consumption to increased mortality risk",
        "SpaceX successfully launches 60 more Starlink satellites into low Earth orbit",
        "International climate summit reaches agreement on carbon emission reductions",
        "Supreme Court issues ruling on landmark case involving digital privacy rights",
        "Economists warn of recession risk as consumer spending slows significantly",
        "Breakthrough in quantum computing achieved by team at leading research university",
        "Tropical storm strengthens to Category 3 hurricane ahead of Gulf Coast landfall",
        "Federal indictment names three executives in corporate fraud scheme",
        "New housing data shows prices cooling in major metropolitan areas",
        "Bipartisan Senate bill proposes reforms to campaign finance regulations",
        "Medical researchers identify genetic markers linked to higher Alzheimer's risk",
        "Former president makes first public appearance since leaving office",
        "Tech giant announces layoffs of 5,000 workers amid restructuring plans",
        "International trade negotiations resume after months-long diplomatic standoff",
        "Solar panel efficiency record broken by research team at MIT laboratory",
        "Flood waters recede in coastal city after hurricane causes widespread damage",
    ]

    data = (
        [(h, 0) for h in fake_headlines] +
        [(h, 1) for h in real_headlines]
    )
    # Augment with slight variations for a larger training set
    augmented = []
    for text, label in data:
        for _ in range(4):   # 5x augmentation
            words = text.split()
            np.random.shuffle(words)
            augmented.append((" ".join(words), label))
    data += augmented

    df = pd.DataFrame(data, columns=["text", "label"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def train(use_kaggle: bool = False):
    print("=" * 55)
    print("  Fake News Detector — Model Training")
    print("=" * 55)

    # ── Load data ──────────────────────────────────────────
    if use_kaggle:
        print("\n[1/5] Loading Kaggle dataset …")
        df = load_kaggle_data()
    else:
        print("\n[1/5] Loading built-in demo dataset …")
        df = load_demo_data()

    print(f"      Total samples : {len(df)}")
    print(f"      Fake (0)      : {(df.label == 0).sum()}")
    print(f"      Real (1)      : {(df.label == 1).sum()}")

    # ── Clean text ─────────────────────────────────────────
    print("\n[2/5] Cleaning text …")
    df["clean_text"] = df["text"].apply(clean_text)

    # ── TF-IDF vectorization ───────────────────────────────
    print("\n[3/5] Vectorizing with TF-IDF …")
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values

    # ── Train / test split ─────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Train model ────────────────────────────────────────
    print("\n[4/5] Training Logistic Regression …")
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n      Accuracy : {acc:.2%}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    # ── Save artefacts ─────────────────────────────────────
    print("[5/5] Saving model and vectorizer …")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n✅  model.pkl and vectorizer.pkl saved successfully.")
    print("    You can now start the app with:  python app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle", action="store_true",
                        help="Use Kaggle Fake.csv / True.csv instead of built-in demo data")
    args = parser.parse_args()
    train(use_kaggle=args.kaggle)
