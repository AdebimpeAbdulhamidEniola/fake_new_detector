# Automated Fake News Detection using NLP (Group 5)

TF-IDF + Passive-Aggressive Classifier, as specified in the project proposal.

## Setup and run

```powershell
cd fake-news-detection
.venv\Scripts\activate              # or: python -m venv .venv; pip install -r requirements.txt
python train.py                     # trains, evaluates and saves the model and reports
python predict.py "Article text..." # classify one article
python predict.py --file article.txt
```

## Files

| Path | Purpose |
|---|---|
| `data/Fake.csv`, `data/True.csv` | ISOT Fake News dataset |
| `train.py` | Full pipeline: EDA → cleaning → 80/20 split → TF-IDF → PAC → evaluation |
| `predict.py` | Loads the saved model and classifies new text |
| `models/*.joblib` | Trained TF-IDF vectorizer and Passive-Aggressive model |
| `reports/metrics.json` | All metrics, confusion matrices and timings |
| `reports/eda.png` | Class distribution, article lengths, subjects |
| `reports/confusion_matrix_pac.png`, `confusion_matrix_nb.png` | 2×2 confusion matrices |
| `reports/top_terms.png` | Words the model relies on most for each class |

## Results (test set, 7,764 articles)

| Model | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|
| **TF-IDF + Passive-Aggressive (proposed)** | **98.38%** | 98.17% | 98.83% | 98.50% | 0.75 s |
| TF-IDF + Multinomial Naive Bayes (baseline) | 93.29% | 92.37% | 95.43% | 93.88% | < 0.1 s |

Passive-Aggressive confusion matrix (REAL = positive class):

|  | Predicted REAL | Predicted FAKE |
|---|---|---|
| **Actual REAL** | TP = 4135 | FN = 49 |
| **Actual FAKE** | FP = 77 | TN = 3503 |

Inference: ~0.0014 ms per article on CPU.

## External (unseen) dataset test

`python evaluate_external.py` tests on `data/external/news.csv` (DataFlair / fake_or_real_news, 6,335 articles):

| Experiment | Accuracy |
|---|---|
| ISOT-trained model → news.csv (never seen) | 57.87% |
| Proposal pipeline trained & tested on news.csv | 94.61% |
| Combined ISOT + news.csv training → news.csv test | 88.18% |
| Combined ISOT + news.csv training → ISOT test | 97.04% |

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the full run guide, proposal corrections and defense Q&A.

## Differences from the proposal

1. **Dataset.** The proposal describes DataFlair's `news.csv` (7,796 rows). The supplied dataset is ISOT
   (44,898 rows in `Fake.csv` + `True.csv`, columns `title, text, subject, date`). The files are
   merged and labelled `FAKE` / `REAL` to match the proposal's schema. The proposal's 7,796-row,
   6,236/1,560 split figures should be updated to the numbers above.
2. **Reuters leakage removed.** Almost every `True.csv` article begins with a dateline such as
   `WASHINGTON (Reuters) -`. Left in, the model classifies on that alone. The dateline and the word
   "Reuters" are stripped during preprocessing.
3. **Duplicates removed.** 6,067 duplicate articles (mostly in `Fake.csv`) were dropped so the
   same article cannot appear in both the training and test sets. 38,820 articles remain.
4. **Title + text** are combined as the input feature, and the split is stratified by label.

All other settings match the proposal: `stop_words='english'`, `max_df=0.7`, `test_size=0.2`,
`random_state=7`, `PassiveAggressiveClassifier(max_iter=50)`.

## Limitations (for the report)

- The top terms show the model partly learns **publisher style**, not truthfulness: Reuters wire
  style (weekdays, "spokesman", "statement") points to REAL, and blog/clickbait artefacts
  ("video", "featured image", "getty", "watch") point to FAKE. Accuracy on articles from
  other sources or time periods (ISOT covers 2016–2017) will likely be lower.
- scikit-learn 1.8+ deprecates `PassiveAggressiveClassifier` (removal planned for 1.10). The
  equivalent is `SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0)`.
  Pin `scikit-learn<1.10` or switch to that call if you upgrade.
