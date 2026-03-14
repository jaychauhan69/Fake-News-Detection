"""
train_model.py - Train the Fake News Detection model

Usage:
  Option 1 (built-in dataset - works without Kaggle):
      python train_model.py

  Option 2 (Kaggle dataset - highest accuracy):
      Place 'Fake.csv' and 'True.csv' from Kaggle in this directory, then run:
      python train_model.py --kaggle
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


def load_kaggle_data() -> pd.DataFrame:
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 0
    true["label"] = 1
    df = pd.concat([fake, true], ignore_index=True)
    df["text"] = (df.get("title", "").fillna("") + " " + df.get("text", "").fillna("")).str.strip()
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
        "SHOCKING CONFESSION: Famous actor caught on hidden camera admitting global warming is complete hoax invented by global elites",
        "EXPOSED: Major hospitals deliberately killing patients to harvest and sell organs to wealthy buyers worldwide whistleblower reveals",
        "Massive secret underground tunnel network beneath every major American city used by elites for child trafficking finally exposed",
        "BREAKING ADMISSION: Government secretly admits spraying toxic biological chemicals on unknowing population from planes for thirty years",
        "Princess Diana was deliberately murdered by British royal family because she discovered their darkest most disturbing secrets",
        "Climate scientists bribed millions of dollars annually to completely fabricate global warming data entire narrative is elaborate lie",
        "URGENT TECH WARNING: Latest smartphone software update secretly activates front camera and microphone to spy on users delete now",
        "Suppressed miracle cancer cure hidden from public for over fifty years finally leaked online Big Pharma furiously tries to censor",
        "All world leaders secretly replaced by identical clones at recent Davos meeting top body language expert reveals irrefutable proof",
        "NASA finally secretly admits never actually landed on moon entire Apollo program elaborately filmed on Hollywood soundstage by Kubrick",
        "BREAKING EXCLUSIVE: Famous senior politician arrested for massive international child trafficking ring but entire media completely refuses to report",
        "Ancient secret society has controlled all world governments central banks and mainstream media their sinister plan for humanity exposed",
        "BOMBSHELL ADMISSION: Top virologist secretly admits COVID nineteen was deliberately engineered as bioweapon against civilian population revealed",
        "Drinking raw unfiltered morning urine completely cures every known disease ancient suppressed remedy Big Pharma desperately hiding from public",
        "URGENT ALERT: Five major American cities scheduled to be hit by staged government false flag terror attack this weekend",
        "Hidden Illuminati symbols discovered throughout new dollar bill design absolute irrefutable proof they control entire global financial system",
        "EXPOSED FINALLY: Every major American election for past thirty years has been completely rigged by unelected shadow government",
        "Groundbreaking discovery proves human beings are actually genetically engineered alien hybrid species truth deliberately hidden by world governments",
        "BREAKING INSIDER: Whistleblower from deep inside CDC admits all childhood vaccines are completely ineffective and deliberately harmful to children",
        "Giant underwater pyramid structures discovered off coast definitively prove lost continent of Atlantis was real government covering up discovery",
        "Politician caught on hidden camera laughing and admitting to personally rigging votes across multiple swing states footage leaked online",
        "EXCLUSIVE BOMBSHELL: Senior senator arrested in massive bribery corruption scandal involving hostile foreign government mainstream media completely silent",
        "Leaked internal emails prove entire major political party secretly working to systematically destroy America from within shocking new revelations",
        "Famous celebrity publicly announces open support for designated terrorist organization mainstream media desperately tries to completely cover it up",
        "BOMBSHELL REPORT: Senior four star general confirms military actively planning armed coup against democratically elected government sources confirm",
        "Politician secretly owns large financial stake in defense contractor that profits enormously from war explaining hawkish foreign policy positions",
        "LEAKED MEMO: Internal network document shows major television news network ordered journalists to deliberately lie about election to help candidate",
        "Famous award winning journalist secretly paid tens of millions by hostile foreign government to spread anti American propaganda confirmed",
        "URGENT BREAKING: Martial law being secretly planned and prepared for implementation in major American cities rollout begins next week sources",
        "Senior politician diagnosed with serious terminal illness being actively hidden from voting public in massive cover up by inner circle",
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
        "Major longitudinal study confirms Mediterranean dietary pattern significantly reduces risk of heart disease stroke and type two diabetes",
        "Research team announces successful gene therapy trial that partially restores functional sight to patients with severe inherited blindness",
        "Novel antibiotic compound discovered by university researchers effectively eliminates drug resistant bacteria strains including deadly hospital MRSA",
        "International team successfully sequences complete genome of preserved woolly mammoth specimen raising future de extinction scientific discussions",
        "Large neuroimaging study reveals distinct measurable brain activity patterns uniquely associated with treatment resistant chronic clinical depression",
        "European Union parliament passes landmark comprehensive digital data privacy legislation significantly affecting operations of major technology companies",
        "United Nations peacekeeping forces deployed to conflict zone following successful internationally brokered ceasefire agreement between warring factions",
        "Historic international climate conference reaches binding multilateral agreement on significant carbon emission reduction targets for participating member nations",
        "Supreme Court issues major landmark ruling on important case involving digital privacy rights and scope of government surveillance authority",
        "Senate committee approves comprehensive new cybersecurity legislation requiring critical infrastructure operators to promptly report significant security breaches",
        "Foreign ministers from twelve nations convene in Geneva to negotiate updated framework for nuclear nonproliferation treaty compliance verification",
        "Presidential administration formally announces new executive order significantly expanding federal environmental protections for national forest wilderness lands",
        "House of Representatives passes sweeping comprehensive immigration reform legislation with meaningful bipartisan support sending measure to Senate",
        "International Criminal Court formally issues arrest warrant for senior military commander accused of systematic war crimes and atrocities",
        "State department announces new multilateral diplomatic initiative specifically aimed at meaningfully reducing dangerous tensions between nuclear armed nations",
        "Climate scientists report Arctic sea ice annual extent reaches alarming record low for third consecutive year deeply concerning researchers",
        "Wildfires force evacuation orders for tens of thousands of residents as firefighters battle massive blazes across drought stricken California",
        "Flood waters slowly begin receding in devastated coastal communities after powerful Category four hurricane causes widespread catastrophic infrastructure damage",
        "Tropical depression rapidly intensifies to dangerous Category four hurricane ahead of predicted destructive Gulf Coast landfall tomorrow afternoon",
        "New comprehensive government climate report warns rising sea levels may displace hundreds of millions of coastal residents by century end",
        "Environmental protection agency announces strict comprehensive new regulations significantly limiting industrial carbon dioxide emissions from coal power plants",
        "Record breaking prolonged heat wave grips southern Europe as temperatures exceed forty seven degrees Celsius across multiple major cities",
        "Scientists issue urgent warning that Amazon rainforest is rapidly approaching critical ecological tipping point that could trigger irreversible collapse",
        "Ocean surface temperatures reach unprecedented record highs triggering massive widespread coral bleaching event across entire Great Barrier Reef system",
        "Major metropolitan areas announce ambitious coordinated plans to achieve full carbon neutrality by year two thousand forty accelerating renewable energy",
        "Major semiconductor manufacturer announces fifteen billion dollar capital investment to construct new advanced chip fabrication facility on domestic soil",
        "Global semiconductor chip shortage expected to gradually ease as manufacturers significantly expand production capacity throughout next calendar year",
        "Leading technology company announces major corporate restructuring plan including significant layoffs of five thousand workers to reduce operating overhead",
        "Prominent artificial intelligence startup raises record breaking three billion dollar Series C funding round to develop advanced language model",
        "Critical cybersecurity vulnerability discovered affecting hundreds of millions of connected devices worldwide prompting urgent emergency software patch release",
        "Major social media platform announces comprehensive sweeping content moderation policy reforms aimed at meaningfully reducing harmful misinformation spread",
        "Electric vehicle unit sales surpass traditional internal combustion engine car sales for first time in significant major European market",
        "Dominant technology company reaches regulatory settlement paying record breaking two billion dollar financial penalty for serious antitrust violations",
        "University research laboratory demonstrates revolutionary new solid state battery technology offering double the energy density of current lithium ion",
        "Large scale coordinated internet infrastructure cyberattack temporarily disrupts essential online services for millions of users across multiple continents Tuesday",
        "New significant archaeological excavation in Egypt uncovers previously completely unknown ancient royal pharaoh burial tomb filled with priceless artifacts",
        "Former head of state makes first significant public appearance since departing office delivering keynote address at major economic policy conference",
        "Federal grand jury formal indictment names three senior corporate executives in large scale sophisticated accounting fraud conspiracy scheme",
        "New comprehensive housing market data confirms home prices cooling measurably in major metropolitan areas after sustained years of appreciation",
        "Bipartisan Senate working group formally proposes comprehensive meaningful reforms to significantly outdated campaign finance disclosure and contribution regulations",
        "National health service publishes report documenting substantial surge in mental health treatment service demand following prolonged pandemic related stress",
        "Major international airline formally announces significant expansion of long haul international route network as global travel demand recovers strongly",
        "City council formally approves ambitious fifteen billion dollar comprehensive public mass transportation network expansion plan after years of deliberation",
        "New comprehensive economic analysis finds persistent significant gender pay gap across most major industries despite decades of equal pay law",
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


def train(use_kaggle: bool = False):
    print("=" * 55)
    print("  Fake News Detector — Model Training")
    print("=" * 55)

    if use_kaggle:
        print("\n[1/5] Loading Kaggle dataset ...")
        df = load_kaggle_data()
    else:
        print("\n[1/5] Loading built-in dataset ...")
        df = load_demo_data()

    print(f"      Total samples : {len(df)}")
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
    parser.add_argument("--kaggle", action="store_true",
                        help="Use Kaggle Fake.csv / True.csv instead of built-in data")
    args = parser.parse_args()
    train(use_kaggle=args.kaggle)
