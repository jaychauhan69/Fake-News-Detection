"""
train_model.py - Train the Fake News Detection model

Usage:
  Option 1 (built-in dataset - works without any download):
      python train_model.py

  Option 2 (WELFake dataset - 72,134 articles):
      python train_model.py --welfake

  Option 3 (Kaggle dataset):
      python train_model.py --kaggle

  Option 4 (ISOT dataset):
      python train_model.py --isot

  Option 5 (WELFake + ISOT combined - BEST 116,134 articles):
      python train_model.py --combined
"""

import argparse
import pickle
import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

for pkg in ("stopwords", "wordnet", "punkt", "omw-1.4"):
    nltk.download(pkg, quiet=True)

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


def balance_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Balance dataset so Fake and Real news have EQUAL counts."""
    fake_df = df[df["label"] == 0]
    real_df = df[df["label"] == 1]

    print(f"      Before balancing -> Fake: {len(fake_df)}, Real: {len(real_df)}")

    # Take the minimum count so both classes are equal
    min_count = min(len(fake_df), len(real_df))

    fake_balanced = fake_df.sample(n=min_count, random_state=42)
    real_balanced = real_df.sample(n=min_count, random_state=42)

    balanced_df = pd.concat([fake_balanced, real_balanced], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"      After balancing  -> Fake: {min_count}, Real: {min_count} (Equal)")
    return balanced_df


def load_welfake_data() -> pd.DataFrame:
    """Load WELFake dataset - 72,134 articles"""
    print("      Loading WELFake_Dataset.csv ...")
    df = pd.read_csv("WELFake_Dataset.csv")
    df = df.dropna(subset=["label"])
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
    df["label"] = df["label"].astype(int)
    print(f"      WELFake articles loaded : {len(df)}")
    return df[["text", "label"]]


def load_isot_data() -> pd.DataFrame:
    """Load ISOT dataset - 44,000 articles"""
    print("      Loading ISOT_Fake.csv and ISOT_True.csv ...")
    fake = pd.read_csv("ISOT_Fake.csv")
    true = pd.read_csv("ISOT_True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
    print(f"      ISOT articles loaded : {len(df)}")
    return df[["text", "label"]]


def load_kaggle_data() -> pd.DataFrame:
    """Load Kaggle Fake/Real dataset - 44,898 articles"""
    print("      Loading Fake.csv and True.csv ...")
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
    print(f"      Kaggle articles loaded : {len(df)}")
    return df[["text", "label"]]


def load_combined_data() -> pd.DataFrame:
    """Load WELFake + ISOT combined - 116,134 articles"""
    welfake = load_welfake_data()
    isot    = load_isot_data()
    df = pd.concat([welfake, isot], ignore_index=True)
    df = df.drop_duplicates(subset=["text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"      Combined total articles : {len(df)}")
    return df[["text", "label"]]


def load_demo_data() -> pd.DataFrame:
    fake_news = [
        "BREAKING: Scientists discover drinking bleach cures all diseases instantly no side effects whatsoever",
        "Shocking revelation the moon is actually a giant hologram projected by secret government shadow agency",
        "EXPOSED: All doctors secretly paid by Big Pharma to keep patients sick and dependent on expensive pills",
        "EXCLUSIVE: Aliens living in White House basement have been controlling world leaders since nineteen forty seven",
        "URGENT: 5G towers confirmed to remotely control human minds leaked documents from brave whistleblower show proof",
        "BOMBSHELL: Bill Gates microchipping entire population through COVID flu shots shocking secret plan finally revealed",
        "REVEALED: Earth is completely flat and all space agencies worldwide have been lying to humanity for decades",
        "SECRET EXPOSED: Chemtrails contain powerful mind control chemicals sprayed daily on population by shadow government",
        "CONFIRMED: Eating dark chocolate daily completely cures cancer suppressed by medical establishment purely for profit",
        "SHOCKING: Mainstream media deliberately covering up massive alien invasion happening right now in multiple major cities",
        "Democrat politician caught red handed selling children to underground satanic sex trafficking ring anonymous sources reveal",
        "COVID vaccine proven to make people magnetic nurse demonstrates metal spoon sticking to forehead on video",
        "Government secretly putting fluoride in water supply to make entire population docile and easier to control expert warns",
        "ABSOLUTE PROOF: Obama secretly born in Kenya according to newly discovered classified document just leaked online",
        "Scientists find conclusive undeniable evidence Noah ark landed on Mount Ararat completely confirms Bible as literal truth",
        "LEAKED VIDEO: Major Hollywood celebrities exposed as reptilian lizard people Hollywood insider bravely reveals dark secret",
        "President signs secret executive order to confiscate all civilian firearms by end of month anonymous White House sources",
        "Miracle diabetes cure discovered by ancient isolated native tribe Big Pharma paying millions to desperately suppress it",
        "Bombshell new study definitively proves vaccines directly cause autism in children was deliberately hidden by CDC decades",
        "Secret billionaire meeting to depopulate Earth by ninety percent before year two thousand thirty leaked document confirms",
        "Presidential election completely stolen through Dominion voting machines expert forensic analysis definitively confirms widespread fraud",
        "Deep state operatives brutally assassinated top presidential advisor staged to look like natural causes last night sources say",
        "George Soros secretly funneling billions to fund antifa terrorists plotting violent overthrow of US government this summer",
        "QAnon prediction finally fully confirmed massive global arrest of entire elite cabal definitely happening tomorrow morning",
        "Hollywood elite secretly harvesting adrenochrome from trafficked children for youth drug insider witness comes forward bravely",
        "Pentagon finally forced to admit using advanced weather control technology to deliberately cause recent catastrophic hurricanes",
        "Microplastics secretly added to public drinking water scientifically proven to permanently alter human DNA and control behavior",
        "FEMA concentration camps fully built and operational ready to immediately house political dissidents brave journalist reveals truth",
        "AI robots secretly replacing senior government officials in Washington insiders at highest levels of White House confirm",
        "New world order globalist depopulation plan to completely eliminate middle class and enslave all humanity accelerating rapidly now",
    ]

    real_news = [
        "Federal Reserve raises benchmark interest rates by twenty five basis points to combat persistent above target inflation",
        "Stock markets decline sharply amid growing investor concerns over rising inflation and tightening global monetary policy",
        "National unemployment rate falls to three point five percent as economy adds two hundred thousand jobs in November",
        "Congress passes major bipartisan infrastructure investment bill allocating one point two trillion for roads bridges and broadband",
        "International Monetary Fund warns of significant global recession risk as central banks raise rates simultaneously worldwide",
        "Treasury Department issues comprehensive new regulations targeting cryptocurrency exchanges to prevent money laundering and tax evasion",
        "Consumer price index rises four point two percent year over year as energy food and housing costs continue surging",
        "Federal budget deficit widens to one point nine trillion dollars amid increased government spending and lower corporate tax revenues",
        "World Bank approves fifteen billion dollar emergency loan package to help developing nations recover from prolonged economic crisis",
        "Major technology sector layoffs continue as leading companies cut workforce to reduce costs amid significant slowing revenue growth",
        "NASA James Webb Space Telescope captures unprecedented detailed images of distant galaxy clusters never observed before by astronomers",
        "Scientists develop highly promising new vaccine candidate showing strong efficacy against malaria parasite in early phase clinical trial",
        "Researchers find significant statistical correlation between chronic sleep deprivation and accelerated cognitive decline in aging adult population",
        "Landmark peer reviewed study published in Nature journal links ultra processed food consumption to significantly elevated mortality risk",
        "Major breakthrough in practical quantum computing achieved by research team at MIT using novel error correction technique approach",
        "World Health Organization reports measles cases rising sharply across multiple continents due to declining childhood vaccination coverage rates",
        "Medical research team identifies seventeen specific genetic markers strongly associated with significantly higher Alzheimer disease development risk",
        "SpaceX Falcon rocket successfully launches sixty additional Starlink internet satellites into planned low Earth orbit expanding global coverage",
        "Large scale clinical trial demonstrates new diabetes medication reduces major cardiovascular event risk by thirty percent in patients",
        "Marine scientists discover deep ocean microorganism capable of efficiently breaking down plastic waste opening promising new recycling possibilities",
        "European Union parliament passes landmark comprehensive digital data privacy legislation significantly affecting operations of major technology companies",
        "United Nations peacekeeping forces deployed to conflict zone following successful internationally brokered ceasefire agreement between warring factions",
        "Historic international climate conference reaches binding multilateral agreement on significant carbon emission reduction targets for participating member nations",
        "Supreme Court issues major landmark ruling on important case involving digital privacy rights and scope of government surveillance authority",
        "Senate committee approves comprehensive new cybersecurity legislation requiring critical infrastructure operators to promptly report significant security breaches",
        "Climate scientists report Arctic sea ice annual extent reaches alarming record low for third consecutive year deeply concerning researchers",
        "Wildfires force evacuation orders for tens of thousands of residents as firefighters battle massive blazes across drought stricken California",
        "New comprehensive government climate report warns rising sea levels may displace hundreds of millions of coastal residents by century end",
        "Major semiconductor manufacturer announces fifteen billion dollar capital investment to construct new advanced chip fabrication facility on domestic soil",
        "University research team publishes large scale longitudinal study demonstrating strong causal link between heavy social media use and adolescent anxiety",
    ]

    data = [(t, 0) for t in fake_news] + [(t, 1) for t in real_news]
    augmented = []
    rng = np.random.default_rng(42)
    for text, label in data:
        words = text.split()
        for _ in range(9):
            w = words.copy()
            n_drop = rng.integers(1, 3)
            for _ in range(n_drop):
                if len(w) > 6:
                    idx = rng.integers(0, len(w))
                    w.pop(int(idx))
            augmented.append((" ".join(w), label))

    all_data = data + augmented
    df = pd.DataFrame(all_data, columns=["text", "label"])
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def load_ultimate_data() -> pd.DataFrame:
    """Load WELFake + ISOT + Pattern-reinforced demo data (BEST for catching obvious fake news)"""
    welfake = load_welfake_data()
    isot    = load_isot_data()
    demo    = load_demo_data()  # includes augmented sensational fake-news patterns

    df = pd.concat([welfake, isot, demo], ignore_index=True)
    df = df.drop_duplicates(subset=["text"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"      Ultimate combined total articles : {len(df)}")
    return df[["text", "label"]]


def train(use_kaggle=False, use_welfake=False, use_isot=False, use_combined=False, use_ultimate=False, balance=False):
    print("=" * 55)
    print("  Fake News Detector — Model Training")
    print("=" * 55)

    if use_ultimate:
        print("\n[1/5] Loading WELFake + ISOT + Pattern-reinforced dataset (ULTIMATE) ...")
        df = load_ultimate_data()
    elif use_combined:
        print("\n[1/5] Loading WELFake + ISOT combined dataset ...")
        df = load_combined_data()
    elif use_welfake:
        print("\n[1/5] Loading WELFake dataset ...")
        df = load_welfake_data()
    elif use_isot:
        print("\n[1/5] Loading ISOT dataset ...")
        df = load_isot_data()
    elif use_kaggle:
        print("\n[1/5] Loading Kaggle dataset ...")
        df = load_kaggle_data()
    else:
        print("\n[1/5] Loading built-in dataset ...")
        df = load_demo_data()

    print(f"\n      Total samples : {len(df)}")
    print(f"      Fake (0)      : {(df.label == 0).sum()}")
    print(f"      Real (1)      : {(df.label == 1).sum()}")

    if balance:
        print("\n      Balancing dataset (Fake = Real count) ...")
        df = balance_dataset(df)
        print(f"\n      Total samples after balancing : {len(df)}")
        print(f"      Fake (0)      : {(df.label == 0).sum()}")
        print(f"      Real (1)      : {(df.label == 1).sum()}")

    print("\n[2/5] Cleaning text ...")
    df["clean_text"] = df["text"].apply(clean_text)

    print("\n[3/5] Vectorizing with TF-IDF ...")
    vectorizer = TfidfVectorizer(
        max_features=15_000,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=2,
    )
    X = vectorizer.fit_transform(df["clean_text"])
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[4/5] Training Logistic Regression ...")
    model = LogisticRegression(max_iter=2000, C=2.0, solver="lbfgs")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n      Accuracy : {acc:.2%}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

    print("[5/5] Saving model and vectorizer ...")
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n✅  model.pkl and vectorizer.pkl saved successfully.")
    print("    Start the app with:  python app.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle",   action="store_true", help="Use Kaggle Fake.csv / True.csv")
    parser.add_argument("--welfake",  action="store_true", help="Use WELFake_Dataset.csv")
    parser.add_argument("--isot",     action="store_true", help="Use ISOT_Fake.csv / ISOT_True.csv")
    parser.add_argument("--combined", action="store_true", help="Use WELFake + ISOT combined")
    parser.add_argument("--ultimate", action="store_true", help="Use WELFake + ISOT + Pattern data (BEST)")
    parser.add_argument("--balance", action="store_true", help="Balance dataset so Fake = Real count exactly")
    args = parser.parse_args()
    train(
        use_kaggle=args.kaggle,
        use_welfake=args.welfake,
        use_isot=args.isot,
        use_combined=args.combined,
        use_ultimate=args.ultimate,
        balance=args.balance,
    )
