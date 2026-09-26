"""
Classify a news article as REAL or FAKE with the trained model.

Usage:
    python predict.py "Article text here..."
    python predict.py --file article.txt
    python predict.py                  # interactive mode
"""

import argparse
import sys
from pathlib import Path

import joblib

from train import clean_text

MODELS = Path(__file__).resolve().parent / "models"


def load():
    vectorizer = joblib.load(MODELS / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS / "passive_aggressive_model.joblib")
    return vectorizer, model


def predict(text, vectorizer, model):
    features = vectorizer.transform([clean_text(text)])
    label = model.predict(features)[0]
    # Signed distance from the decision boundary; larger magnitude = more confident
    score = float(model.decision_function(features)[0])
    return label, abs(score)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("text", nargs="?", help="article text")
    parser.add_argument("--file", help="path to a text file containing the article")
    args = parser.parse_args()

    vectorizer, model = load()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.text:
        text = args.text
    else:
        print("Paste an article and press Enter (empty line to quit):")
        for line in sys.stdin:
            if not line.strip():
                break
            label, margin = predict(line, vectorizer, model)
            print(f"  -> {label}  (margin {margin:.3f})\n")
        return

    label, margin = predict(text, vectorizer, model)
    print(f"Prediction: {label}  (margin {margin:.3f})")


if __name__ == "__main__":
    main()
