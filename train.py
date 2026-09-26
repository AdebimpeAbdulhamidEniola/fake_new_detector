"""
Automated Fake News Detection using NLP (Group 5)

Pipeline (as specified in the project proposal):
    raw text -> preprocessing -> TF-IDF (stop_words='english', max_df=0.7)
             -> 80/20 train/test split (random_state=7)
             -> PassiveAggressiveClassifier(max_iter=50)
             -> Accuracy, Precision, Recall, F1, 2x2 confusion matrix

Dataset note:
    The proposal describes the DataFlair `news.csv` (7,796 rows). The dataset
    supplied is the ISOT Fake News dataset (Fake.csv + True.csv, ~44.9k rows),
    so the two files are merged into the same schema: text + label (REAL/FAKE).

    Almost every article in True.csv starts with a Reuters dateline such as
    "WASHINGTON (Reuters) - ". Fake.csv never does. Left in, the classifier
    learns "Reuters => REAL" instead of anything about the content, so the
    dateline and the word "Reuters" are removed during preprocessing.
"""

import json
import re
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
REPORTS = BASE / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

RANDOM_STATE = 7
LABELS = ["REAL", "FAKE"]  # REAL is the positive class, matching the proposal's TP definition

# "WASHINGTON (Reuters) - ", "LONDON/PARIS (Reuters) - ", "(Reuters) - " etc.
DATELINE_RE = re.compile(r"^.{0,120}?\(reuters\)\s*-\s*", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+|pic\.twitter\.com/\S+")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
SPACES_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, drop Reuters datelines, URLs, punctuation and digits."""
    text = DATELINE_RE.sub("", str(text))
    text = text.lower()
    text = text.replace("reuters", " ")
    text = URL_RE.sub(" ", text)
    text = NON_ALPHA_RE.sub(" ", text)
    return SPACES_RE.sub(" ", text).strip()


def load_dataset() -> pd.DataFrame:
    fake = pd.read_csv(DATA / "Fake.csv")
    true = pd.read_csv(DATA / "True.csv")
    fake["label"] = "FAKE"
    true["label"] = "REAL"
    df = pd.concat([fake, true], ignore_index=True)
    return df


def eda(df: pd.DataFrame) -> dict:
    print("\n=== 1. Data acquisition & exploration ===")
    print(f"Raw shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Missing values per column:\n" + df.isna().sum().to_string())
    print("Class distribution:\n" + df["label"].value_counts().to_string())

    df["word_count"] = df["text"].astype(str).str.split().str.len()
    print("Word count per article by class:")
    print(df.groupby("label")["word_count"].describe().round(1).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    df["label"].value_counts().reindex(LABELS).plot.bar(
        ax=axes[0], color=["#2a9d8f", "#e76f51"]
    )
    axes[0].set_title("Class distribution")
    axes[0].set_ylabel("Articles")
    axes[0].tick_params(axis="x", rotation=0)

    for label, colour in zip(LABELS, ["#2a9d8f", "#e76f51"]):
        wc = df.loc[df["label"] == label, "word_count"].clip(upper=2000)
        axes[1].hist(wc, bins=60, alpha=0.6, label=label, color=colour)
    axes[1].set_title("Article length (words, clipped at 2000)")
    axes[1].legend()

    pd.crosstab(df["subject"], df["label"]).plot.barh(
        ax=axes[2], color=["#e76f51", "#2a9d8f"]
    )
    axes[2].set_title("Articles per subject")
    fig.tight_layout()
    fig.savefig(REPORTS / "eda.png", dpi=130)
    plt.close(fig)

    return {
        "raw_rows": int(len(df)),
        "class_counts_raw": df["label"].value_counts().to_dict(),
        "mean_words": df.groupby("label")["word_count"].mean().round(1).to_dict(),
    }


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== 2. Text cleaning & preprocessing ===")
    df = df.copy()
    # Title + body gives the model the headline as well as the article text
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)

    before = len(df)
    df = df[df["content"].str.split().str.len() >= 5]
    print(f"Dropped {before - len(df)} empty / near-empty articles")

    before = len(df)
    df = df.drop_duplicates(subset="content")
    print(f"Dropped {before - len(df)} duplicate articles")
    print(f"Clean shape: {df.shape}")
    print("Clean class distribution:\n" + df["label"].value_counts().to_string())
    return df.reset_index(drop=True)


def evaluate(name, y_test, y_pred) -> dict:
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)
    tp, fn, fp, tn = cm.ravel()  # rows = actual [REAL, FAKE], cols = predicted
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label="REAL"),
        "recall": recall_score(y_test, y_pred, pos_label="REAL"),
        "f1": f1_score(y_test, y_pred, pos_label="REAL"),
        "confusion_matrix": {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn)},
    }
    print(f"\n--- {name} ---")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print(f"Confusion matrix (rows=actual, cols=predicted, order {LABELS}):\n{cm}")
    print(classification_report(y_test, y_pred, labels=LABELS, digits=4))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def plot_confusion(y_test, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, labels=LABELS, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_top_terms(vectorizer, model, path, n=20):
    names = np.array(vectorizer.get_feature_names_out())
    coef = model.coef_[0]
    # classes_ are sorted alphabetically: ['FAKE', 'REAL'] -> positive coef => REAL
    real_idx = np.argsort(coef)[-n:]
    fake_idx = np.argsort(coef)[:n]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].barh(names[fake_idx][::-1], -coef[fake_idx][::-1], color="#e76f51")
    axes[0].set_title(f"Top {n} terms pushing towards FAKE")
    axes[1].barh(names[real_idx], coef[real_idx], color="#2a9d8f")
    axes[1].set_title(f"Top {n} terms pushing towards REAL")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"fake": names[fake_idx].tolist(), "real": names[real_idx][::-1].tolist()}


def main():
    df = load_dataset()
    results = {"dataset": eda(df)}
    df = preprocess(df)
    results["dataset"]["clean_rows"] = int(len(df))
    results["dataset"]["class_counts_clean"] = df["label"].value_counts().to_dict()

    print("\n=== 3. Dataset partitioning (80/20) ===")
    x_train, x_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=RANDOM_STATE,
        stratify=df["label"],
    )
    print(f"Train: {len(x_train)}  Test: {len(x_test)}")
    results["split"] = {"train": len(x_train), "test": len(x_test)}

    print("\n=== 4. TF-IDF feature extraction ===")
    t0 = time.perf_counter()
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    tfidf_train = vectorizer.fit_transform(x_train)  # fit on training data only
    tfidf_test = vectorizer.transform(x_test)        # no leakage from test set
    print(f"Vocabulary size: {len(vectorizer.vocabulary_):,}  "
          f"({time.perf_counter() - t0:.1f}s)")
    results["vocabulary_size"] = len(vectorizer.vocabulary_)

    print("\n=== 5. Passive-Aggressive Classifier training ===")
    t0 = time.perf_counter()
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=RANDOM_STATE)
    pac.fit(tfidf_train, y_train)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = pac.predict(tfidf_test)
    infer_ms = (time.perf_counter() - t0) / len(x_test) * 1000
    print(f"Training time: {train_s:.2f}s  |  inference: {infer_ms:.4f} ms/article")

    print("\n=== 6. Performance evaluation ===")
    results["passive_aggressive"] = evaluate("Passive-Aggressive (proposed)", y_test, y_pred)
    results["passive_aggressive"]["train_seconds"] = round(train_s, 3)
    results["passive_aggressive"]["inference_ms_per_article"] = round(infer_ms, 5)
    plot_confusion(y_test, y_pred, "Passive-Aggressive - confusion matrix",
                   REPORTS / "confusion_matrix_pac.png")

    # Baseline from the proposal's comparative analysis
    t0 = time.perf_counter()
    nb = MultinomialNB().fit(tfidf_train, y_train)
    nb_train_s = time.perf_counter() - t0
    y_pred_nb = nb.predict(tfidf_test)
    results["naive_bayes"] = evaluate("Multinomial Naive Bayes (baseline)", y_test, y_pred_nb)
    results["naive_bayes"]["train_seconds"] = round(nb_train_s, 3)
    plot_confusion(y_test, y_pred_nb, "Naive Bayes - confusion matrix",
                   REPORTS / "confusion_matrix_nb.png")

    results["top_terms"] = plot_top_terms(vectorizer, pac, REPORTS / "top_terms.png")

    joblib.dump(vectorizer, MODELS / "tfidf_vectorizer.joblib")
    joblib.dump(pac, MODELS / "passive_aggressive_model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved model to {MODELS}  |  reports to {REPORTS}")


if __name__ == "__main__":
    main()
