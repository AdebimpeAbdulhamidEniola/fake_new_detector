"""
External (cross-dataset) evaluation.

Part A - Generalisation test:
    The model trained on ISOT (train.py) is tested on a completely different
    dataset it has never seen: data/external/news.csv (the DataFlair /
    "fake_or_real_news" dataset named in the proposal). Articles that also
    appear in ISOT are removed first so the test is truly unseen.

Part B - Proposal replication:
    The exact experiment described in the proposal is run on news.csv itself
    (TF-IDF max_df=0.7, 80/20 split, random_state=7, PAC max_iter=50) to check
    the ~92.8% accuracy figure quoted in the proposal.

Usage:
    python evaluate_external.py
    python evaluate_external.py --file path/to/other.csv --text-col text --label-col label
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split

from train import DATA, MODELS, RANDOM_STATE, REPORTS, clean_text, evaluate, plot_confusion

warnings.filterwarnings("ignore", category=FutureWarning)


def load_external(path, text_col, label_col, title_col):
    df = pd.read_csv(path)
    df = df.dropna(subset=[text_col, label_col])
    title = df[title_col].fillna("") if title_col in df.columns else ""
    df["content"] = (title + " " + df[text_col].astype(str)).map(clean_text)
    df["label"] = df[label_col].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(["REAL", "FAKE"])]
    df = df[df["content"].str.split().str.len() >= 5].drop_duplicates(subset="content")
    return df.reset_index(drop=True)


def remove_training_overlap(df):
    fake = pd.read_csv(DATA / "Fake.csv")
    true = pd.read_csv(DATA / "True.csv")
    isot = pd.concat([fake, true])
    seen = set((isot["title"].fillna("") + " " + isot["text"].fillna("")).map(clean_text))
    mask = df["content"].isin(seen)
    print(f"Removed {mask.sum()} articles that also appear in the training data (ISOT)")
    return df[~mask].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(DATA / "external" / "news.csv"))
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--title-col", default="title")
    args = parser.parse_args()

    df = load_external(args.file, args.text_col, args.label_col, args.title_col)
    print(f"External dataset: {args.file}")
    print(f"Rows after cleaning: {len(df)}")
    print(df["label"].value_counts().to_string())

    results = {}

    print("\n=== Part A: ISOT-trained model on unseen external data ===")
    df_unseen = remove_training_overlap(df)
    vectorizer = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS / "passive_aggressive_model.joblib")
    y_pred = model.predict(vectorizer.transform(df_unseen["content"]))
    results["cross_dataset"] = evaluate("ISOT model -> external dataset", df_unseen["label"], y_pred)
    results["cross_dataset"]["rows"] = int(len(df_unseen))
    plot_confusion(df_unseen["label"], y_pred, "ISOT model on external data",
                   REPORTS / "confusion_matrix_external.png")

    print("\n=== Part B: proposal experiment reproduced on the external dataset ===")
    x_train, x_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(x_train)}  Test: {len(x_test)}")
    vec = TfidfVectorizer(stop_words="english", max_df=0.7)
    pac = PassiveAggressiveClassifier(max_iter=50, random_state=RANDOM_STATE)
    pac.fit(vec.fit_transform(x_train), y_train)
    y_pred_b = pac.predict(vec.transform(x_test))
    results["proposal_replication"] = evaluate("PAC trained & tested on external dataset", y_test, y_pred_b)
    results["proposal_replication"]["train"] = len(x_train)
    results["proposal_replication"]["test"] = len(x_test)
    plot_confusion(y_test, y_pred_b, "PAC trained on news.csv",
                   REPORTS / "confusion_matrix_newscsv.png")

    print("\n=== Part C: combined training (ISOT + external) to improve generalisation ===")
    isot = pd.concat([
        pd.read_csv(DATA / "Fake.csv").assign(label="FAKE"),
        pd.read_csv(DATA / "True.csv").assign(label="REAL"),
    ])
    isot["content"] = (isot["title"].fillna("") + " " + isot["text"].fillna("")).map(clean_text)
    isot = isot[isot["content"].str.split().str.len() >= 5].drop_duplicates(subset="content")
    xi_train, xi_test, yi_train, yi_test = train_test_split(
        isot["content"], isot["label"], test_size=0.2, random_state=RANDOM_STATE,
        stratify=isot["label"],
    )
    vec_c = TfidfVectorizer(stop_words="english", max_df=0.7)
    pac_c = PassiveAggressiveClassifier(max_iter=50, random_state=RANDOM_STATE)
    pac_c.fit(vec_c.fit_transform(pd.concat([xi_train, x_train])), pd.concat([yi_train, y_train]))
    results["combined_on_isot_test"] = evaluate(
        "Combined model -> ISOT test set", yi_test, pac_c.predict(vec_c.transform(xi_test)))
    results["combined_on_external_test"] = evaluate(
        "Combined model -> external test set", y_test, pac_c.predict(vec_c.transform(x_test)))

    (REPORTS / "metrics_external.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {REPORTS / 'metrics_external.json'}")


if __name__ == "__main__":
    main()
