# Fake News Detection — Project Guide (Group 5)

This guide covers:
1. How to run the model, step by step
2. Testing on a second, unseen dataset, and what the results mean
3. Every place in the proposal that must be changed
4. Likely lecturer questions, with answers

---

## 1. How to run the model

### 1.1 What you need
- Windows 10/11 (the steps also work on Mac/Linux with small path changes)
- **Python 3.10–3.12**. It is already installed on this laptop at
  `C:\Users\ABDULHAMID\AppData\Local\Programs\Python\Python312`.
  On another computer, download it from https://www.python.org/downloads/ and **tick "Add Python to PATH"** during install.
- About 1 GB of free disk space and 4 GB of RAM

### 1.2 Folder layout
```
fake-news-detection/
├── data/
│   ├── Fake.csv                 ← training data (ISOT, fake articles)
│   ├── True.csv                 ← training data (ISOT, real articles)
│   └── external/news.csv        ← second, unseen test dataset
├── models/                      ← saved model (created by train.py)
├── reports/                     ← charts + metrics (created by the scripts)
├── train.py                     ← trains and evaluates the model
├── evaluate_external.py         ← tests on the unseen dataset
├── predict.py                   ← classifies any new article
└── requirements.txt             ← list of Python libraries
```

### 1.3 First-time setup (only once per computer)
Open **PowerShell** (or the VS Code terminal) and run:

```powershell
cd "C:\Users\ABDULHAMID\Desktop\Final Year Project\fake-news-detection"
python -m venv .venv                      # create an isolated environment
.venv\Scripts\activate                    # switch it on; (.venv) appears in the prompt
pip install -r requirements.txt           # install pandas, scikit-learn, matplotlib...
```

> On this laptop the `.venv` folder already exists, so only `activate` is needed.
> If PowerShell says *"running scripts is disabled"*, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or skip activation and call
> `.venv\Scripts\python.exe` directly instead of `python`.

### 1.4 Train the model
```powershell
python train.py
```
It takes about 1 minute. The script prints each stage of the proposal's pipeline:

| Stage printed | What happens |
|---|---|
| 1. Data acquisition & exploration | Loads Fake.csv + True.csv (44,898 articles), shows missing values, class balance, article lengths → `reports/eda.png` |
| 2. Text cleaning & preprocessing | Lowercasing; removes Reuters datelines, URLs, punctuation and digits; drops 11 empty and 6,067 duplicate articles → 38,820 remain |
| 3. Dataset partitioning | 80/20 split, `random_state=7` → 31,056 train / 7,764 test |
| 4. TF-IDF feature extraction | `TfidfVectorizer(stop_words='english', max_df=0.7)` fitted on the **training data only** → 95,819-word vocabulary |
| 5. Passive-Aggressive training | `PassiveAggressiveClassifier(max_iter=50)`, about 0.75 s |
| 6. Performance evaluation | Accuracy, Precision, Recall, F1 and confusion matrix for the PAC and for a Naive Bayes baseline |

Outputs:
- `models/tfidf_vectorizer.joblib`, `models/passive_aggressive_model.joblib` — the trained model
- `reports/metrics.json` — all numbers
- `reports/confusion_matrix_pac.png`, `confusion_matrix_nb.png`, `top_terms.png`, `eda.png` — charts for the report and slides

### 1.5 Classify your own article
```powershell
python predict.py "Paste the full article text here"
python predict.py --file my_article.txt
python predict.py                 # interactive: paste articles one per line, blank line to quit
```
Output example: `Prediction: FAKE  (margin 8.780)`. The *margin* is the distance from the decision
boundary: the bigger it is, the more confident the model. A margin near 0 means it is unsure.

**Live demo tip:** use a full article (at least 2–3 paragraphs). One sentence gives the model very few words to work with.

### 1.6 Test on the unseen dataset
```powershell
python evaluate_external.py
```
To test any other CSV file with a text column and a REAL/FAKE label column:
```powershell
python evaluate_external.py --file path\to\file.csv --text-col text --label-col label --title-col title
```

### 1.7 Common errors
| Error | Fix |
|---|---|
| `python is not recognized` | Python is not on PATH. Reinstall with "Add to PATH" ticked, or use the full path to `python.exe` |
| `ModuleNotFoundError: No module named 'sklearn'` | The venv is not activated. Run `.venv\Scripts\activate`, then `pip install -r requirements.txt` |
| `FileNotFoundError: ...tfidf_vectorizer.joblib` | Run `python train.py` before `predict.py` / `evaluate_external.py` |
| `FutureWarning: PassiveAggressiveClassifier is deprecated` | Harmless in scikit-learn 1.8/1.9. See Q30 |

---

## 2. Testing on a second (unseen) dataset

### 2.1 The second dataset
`data/external/news.csv` is the **DataFlair / "fake_or_real_news"** dataset, the one the proposal
actually describes. It has 6,335 articles (3,171 REAL, 3,164 FAKE) from different outlets
(NYT, CNN, Breitbart, fringe blogs, …) and is mostly from the 2016 US election period.
The script checked that **none** of its articles also appear in the training data.

### 2.2 Results

| Experiment | Trained on | Tested on | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| A. Main model (in-domain) | ISOT (80%) | ISOT (20%) | **98.38%** | 98.17% | 98.83% | 98.50% |
| A. Naive Bayes baseline | ISOT (80%) | ISOT (20%) | 93.29% | 92.37% | 95.43% | 93.88% |
| **B. Cross-dataset** | ISOT (all) | news.csv (all 6,302, never seen) | **57.87%** | 66.49% | 31.84% | 43.06% |
| C. Proposal replication | news.csv (80%) | news.csv (20%) | **94.61%** | 93.83% | 95.07% | 94.44% |
| D. Combined training | ISOT + news.csv (80%) | ISOT test | 97.04% | 96.30% | 98.28% | 97.28% |
| D. Combined training | ISOT + news.csv (80%) | news.csv test | **88.18%** | 89.77% | 85.20% | 87.43% |

(Precision/Recall/F1 treat REAL as the positive class, as in the proposal.)

### 2.3 What the results mean (say this in the defense)
1. **In-domain the model is excellent (98.4%)** and beats Naive Bayes by about 5 points, as the proposal predicted.
2. **The proposal's accuracy claim holds.** Run exactly as the proposal describes on news.csv, the pipeline gets 94.6%, above the 92.8% quoted.
3. **Cross-dataset accuracy falls to 57.9%, close to guessing.** The model trained on ISOT labelled 68% of the REAL
   news.csv articles as FAKE. In ISOT every real article comes from Reuters, so the model learned
   *"Reuters wire style = real"* (weekdays, "spokesman", "statement"). Real articles from the NYT or CNN
   don't look like Reuters, so they get labelled FAKE. This is called **dataset bias / domain shift**, and it is a
   well-known weakness of text-based fake news detectors.
4. **The fix is more diverse training data.** Training on both datasets together keeps ISOT at 97% and
   raises the news.csv score from 57.9% to 88.2%.
5. **Conclusion:** TF-IDF + PAC is fast and accurate *within the kind of news it was trained on*. It
   detects **writing style and source patterns, not factual truth**. Deployment needs training data from many
   sources, plus regular retraining. The PAC's online-learning ability (`partial_fit`) helps with that.

---

## 3. Everything to change in the proposal

| # | Section | Current text | Change to | Why |
|---|---|---|---|---|
| 1 | Executive Summary, 2nd paragraph | "…feature extraction via Termend content moderation systems. Frequency - Inverse Document Frequency (TF-IDF)…" | "…feature extraction via **Term Frequency – Inverse Document Frequency (TF-IDF)** and a specialized online learning algorithm known as the Passive-Aggressive Classifier… making it ideal for scalable, real-time integration into web platforms **and content moderation systems**." | Sentence is scrambled: "…and content moderation systems" was pasted into the middle of "Term Frequency" |
| 2 | Executive Summary | "Tested on a benchmark political news dataset containing 7,796 records… (~92.8%)" | "Trained and tested on the ISOT Fake News dataset (44,898 articles; 38,820 after cleaning), achieving **98.4%** accuracy, and validated on the independent news.csv dataset (6,335 articles)" | Wrong dataset and wrong row count |
| 3 | Objective 1 | "…on the target dataset (news.csv)…" | "…on the ISOT Fake News dataset (Fake.csv and True.csv)…" | The training data is ISOT |
| 4 | Objective 2 | "lowercasing, punctuation handling, and filtering out non-informative English stop words" | Add: "**removal of source-identifying artefacts (Reuters datelines), URLs, digits, and duplicate articles**" | This is what the code does, and it is a strength worth claiming |
| 5 | Objectives (new) | — | Add **Objective 8: "Cross-dataset validation — evaluate generalisation on an independent dataset (news.csv) not used for training."** | You now have this result |
| 6 | Dataset Description — whole section | news.csv, 29.2 MB, 7,796 rows, 4 columns (`Unnamed: 0, title, text, label`) | **Primary dataset:** ISOT Fake News dataset, University of Victoria (Ahmed, Traore & Saad, 2017). Two files: Fake.csv (23,481 rows, 62.8 MB) and True.csv (21,417 rows, 53.6 MB). Columns: `title, text, subject, date`. Label added from file name (FAKE/REAL). **Secondary (validation) dataset:** news.csv, 6,335 rows, columns `id/Unnamed: 0, title, text, label` | Wrong dataset; also news.csv really has 6,335 rows, not 7,796 |
| 7 | Dataset → Class Distribution | "balanced (≈50% REAL / ≈50% FAKE)" | "Raw: 52.3% FAKE / 47.7% REAL. After removing duplicates: 53.9% REAL (20,921) / 46.1% FAKE (17,899), roughly balanced. A stratified split keeps the same ratio in train and test." | Real figures |
| 8 | Dataset (new paragraph) | — | Add "Data quality issues found: 6,067 duplicate articles; 11 empty articles; every real article carries a 'CITY (Reuters) –' prefix that would leak the label; these were removed." | Shows critical thinking |
| 9 | TF-IDF formulas (Section A.1–A.3) | TF = f/Σf; IDF = log(\|D\|/df) | Keep the textbook formulas, but add: "scikit-learn's implementation uses raw counts for TF, **smoothed IDF = ln((1+n)/(1+df)) + 1**, and L2-normalises each document vector." | So the maths matches the code (see Q11) |
| 10 | Passive-Aggressive formula | τₜ = ℓ / ‖xₜ‖² | "τₜ = **min(C, ℓ / ‖xₜ‖²)** with C = 1.0 (PA-I variant, scikit-learn's default)" | scikit-learn uses PA-I, not the basic PA (see Q15) |
| 11 | System Architecture → Raw Data layer | "inspects the dataset shape (7,796 × 4)" | "loads Fake.csv and True.csv, assigns labels, merges them (44,898 × 5)" | Wrong shape |
| 12 | System Architecture → Feature extraction | "isolates df['text']" | "combines `title + text` into one cleaned `content` field" | Matches the code |
| 13 | System Architecture → Splitter | "test_size=0.2, random_state=7 → 6,236 training / 1,560 testing" | "test_size=0.2, random_state=7, **stratify=label** → **31,056 training / 7,764 testing**" | 6,236/1,560 would come from 7,796 rows, which is itself wrong |
| 14 | Evaluation layer | Accuracy + confusion matrix only | Add "Precision, Recall, F1 (classification report), Naive Bayes baseline comparison, cross-dataset evaluation" | Matches Objective 6 and the code |
| 15 | Comparison table — accuracy row | PA ≈92.8%, NB ≈84–88%, BERT ≈94–96% | PA **98.4%**, NB **93.3%** (both measured on ISOT); BERT: cite a paper or mark as "literature value" | Replace guesses with your own results |
| 16 | Summary of Advantages & final Summary | "≈92.8% accuracy", "news.csv (7,796 labeled articles)" | "98.4% accuracy on ISOT, 94.6% on news.csv, 88.2% cross-domain with combined training" | Consistency |
| 17 | New section: Limitations | — | Add: (a) learns style/source not truth (cross-dataset 57.9%); (b) English-only, 2016–2017 US politics; (c) no fact-checking against a knowledge base; (d) PAC deprecated in scikit-learn ≥1.8 | Examiners expect a limitations section |
| 18 | New section: Tools | — | "Python 3.12, pandas, NumPy, scikit-learn 1.9, matplotlib, joblib; runs on a standard CPU" | Usually required |
| 19 | References | none shown | Add: Crammer et al. (2006) *Online Passive-Aggressive Algorithms*, JMLR 7; Ahmed, Traore & Saad (2017) ISOT dataset paper; Salton & Buckley (1988) on TF-IDF; Pedregosa et al. (2011) scikit-learn; DataFlair tutorial for news.csv | A proposal needs references |

---

## 4. Likely lecturer questions, with answers

### A. Problem & motivation
**Q1. What problem are you solving?**
Automatically classifying a news article as REAL or FAKE from its text alone, quickly and cheaply, because there is far too much content for human fact-checkers to review.

**Q2. Why not just use human fact-checkers?**
They can't keep up: millions of posts appear daily, and fake news spreads before a check is finished. Our system screens articles in about 0.001 ms each and flags suspicious ones for humans to review. It supports fact-checkers rather than replacing them.

**Q3. What is your definition of fake news?**
Fabricated or deliberately misleading content presented as news. In practice our definition is the dataset's labels: articles from sites flagged as unreliable by PolitiFact/Wikipedia are FAKE, and Reuters articles are REAL.

### B. Dataset
**Q4. Which dataset did you use and where is it from?**
The ISOT Fake News dataset from the University of Victoria, Canada: 44,898 articles (23,481 fake, 21,417 real) from 2016–2017. Real articles are from Reuters; fake ones come from sites flagged by PolitiFact. We used news.csv (6,335 articles) as an independent test set.

**Q5. Your proposal says news.csv with 7,796 rows. Why is it different?**
The dataset we received was ISOT, which is larger (44,898 vs 6,335), so we used it for training. We still used news.csv, both to reproduce the proposal's experiment (94.6%, above the 92.8% quoted) and as an unseen test set. We also found that news.csv really has 6,335 rows, so the 7,796 figure in the proposal was wrong.

**Q6. Is the dataset balanced? Does it matter?**
Roughly: 53.9% REAL / 46.1% FAKE after cleaning. Imbalance matters because a model can score high accuracy just by predicting the majority class. Here the majority baseline is only 53.9%, and we used a stratified split and report precision, recall and F1 as well as accuracy.

**Q7. What data quality problems did you find?**
(1) 6,067 duplicate articles; if kept, the same article could land in both train and test and inflate the score. (2) 11 empty articles. (3) **Label leakage**: nearly every real article starts with "WASHINGTON (Reuters) –" and fake ones never do, so a model could reach about 99% just by spotting "Reuters". We removed all three.

**Q8. What is data leakage and how did you prevent it?**
Leakage is when information that wouldn't be available at prediction time, or that comes from the test set, gets into training, so the score looks better than it really is. We (a) fitted TF-IDF on the training set only and used `transform()` on the test set; (b) removed duplicates before splitting; (c) removed the Reuters dateline that gives the label away; (d) checked that no news.csv article appears in ISOT.

### C. Preprocessing & TF-IDF
**Q9. What preprocessing did you do?**
Joined title and text, lowercased, removed Reuters datelines and the word "reuters", removed URLs, punctuation and digits, collapsed whitespace, then removed English stop words and very common words inside the TF-IDF step.

**Q10. What is TF-IDF, in simple terms?**
A score for each word in each article: high if the word appears often in *this* article (TF) but rarely across *all* articles (IDF). Words like "the" score near zero; distinctive words like "hoax" score high. Each article becomes a vector of these scores.

**Q11. Is the formula in your proposal what the code computes?**
Nearly. scikit-learn uses raw counts for TF, a smoothed IDF = ln((1+n)/(1+df)) + 1 (which avoids division by zero and never gives a word zero weight), and L2-normalises each article vector so long and short articles are comparable.

**Q12. What does max_df = 0.7 do? Why 0.7?**
It ignores any word that appears in more than 70% of articles, because such words appear in both classes and don't help tell them apart. 0.7 is the value from the reference study. A better approach would be to tune it with cross-validation (for example, try 0.5–0.9).

**Q13. What does stop_words='english' do?**
Removes about 318 common English words ("the", "is", "and") that carry no information about whether an article is real or fake.

**Q14. How many features does your model have?**
95,819: one per unique word in the training vocabulary. The matrix is **sparse** (each article uses only a few hundred words), so it is stored efficiently.

### D. Passive-Aggressive Classifier
**Q15. Explain the Passive-Aggressive algorithm.**
It is an online linear classifier. For each training example it checks the prediction:
- **Passive:** if the prediction is correct with margin ≥ 1 (hinge loss = 0), it does nothing.
- **Aggressive:** otherwise it updates the weights just enough to fix the mistake: w ← w + τ·y·x, with τ = min(C, loss/‖x‖²).

C (default 1.0) caps the step size so a single noisy example can't move the weights too far (this is the PA-I variant).

**Q16. Why the name "Passive-Aggressive"?**
It is passive (no change) when it is right and aggressive (a correcting update) when it is wrong or not confident enough.

**Q17. What is hinge loss?**
ℓ = max(0, 1 − y·(w·x)), with y ∈ {−1, +1}. It is 0 when the prediction is correct with margin ≥ 1, and grows linearly as the prediction gets more wrong. SVMs use the same loss.

**Q18. What does max_iter=50 mean?**
At most 50 passes (epochs) over the training data. Training stops earlier if the loss stops improving (tolerance 1e-3). Ours converged well within 50.

**Q19. Why PAC rather than Logistic Regression or SVM?**
All three are linear models and perform similarly on TF-IDF data. PAC is very fast, and it supports **online learning** (`partial_fit`), so it can learn from new labelled articles without retraining from scratch. That suits news, which changes every day.

**Q20. What is online learning, and did you use it?**
Learning one example (or small batch) at a time instead of all data at once. Our experiment trains in batch mode with `fit()` for reproducibility. In deployment, `model.partial_fit(new_X, new_y)` would update the model as fact-checkers label new articles.

**Q21. Is PAC a linear model? What are the limits of that?**
Yes: it learns one weight per word and adds them up. It can't understand word order, sarcasm or context ("not true" vs "true"). Using bigrams (`ngram_range=(1,2)`) or transformer models would help with that.

### E. Evaluation & results
**Q22. What accuracy did you get?**
98.38% on 7,764 unseen ISOT test articles. Precision 98.17%, recall 98.83%, F1 98.50%. Only 126 errors: 49 real articles labelled fake and 77 fake articles labelled real.

**Q23. Explain your confusion matrix.**
With REAL as positive: TP = 4,135 (real → real), TN = 3,503 (fake → fake), FP = 77 (fake → real, the dangerous error because fake news gets through), FN = 49 (real → fake, which would wrongly flag genuine journalism).

**Q24. Precision vs recall: which matters more here?**
It depends on who uses the system. A platform blocking content should avoid flagging real news (high precision for FAKE). A fact-checking triage tool should catch as much fake news as possible (high recall for FAKE). F1 balances the two. You can move this trade-off by adjusting the decision threshold on `decision_function`.

**Q25. 98% sounds too good. Is it real?**
In-domain, yes, after we removed the leaks. But we tested it on a *different* dataset and it dropped to 57.9%, because it partly learned Reuters writing style rather than truthfulness. Trained on both datasets it reaches 88.2% on the new one. So 98% is the in-domain figure; the realistic cross-source figure is 88%.

**Q26. Why did cross-dataset performance drop so much?**
Domain shift. All the real ISOT articles come from Reuters. The top REAL words were weekdays, "spokesman" and "statement", which are Reuters habits. Real NYT/CNN articles in news.csv don't read like Reuters, so the model called 68% of them fake. It learned "does this look like Reuters?", not "is this true?"

**Q27. How did you compare against other algorithms?**
Multinomial Naive Bayes on the same TF-IDF features got 93.3% vs PAC's 98.4%. The BERT figures in the proposal come from the literature; we didn't train BERT because it needs a GPU.

**Q28. Did you do cross-validation or hyperparameter tuning?**
No. We used the proposal's fixed settings and a single stratified 80/20 split with a fixed seed so results can be reproduced. Future work: 5-fold `GridSearchCV` over `max_df`, `ngram_range` and `C`.

**Q29. Why random_state=7?**
It fixes the random shuffle, so anyone running the code gets exactly the same split and results. The value 7 itself is arbitrary; it matches the reference study.

### F. Implementation & deployment
**Q30. The PAC is deprecated in scikit-learn. Is that a problem?**
It still works in 1.8/1.9 and will be removed in 1.10. We pinned `scikit-learn<1.10`. The replacement, `SGDClassifier(loss='hinge', penalty=None, learning_rate='pa1', eta0=1.0)`, runs the same algorithm.

**Q31. How is the model saved and reused?**
With `joblib`: the fitted vectorizer and classifier are saved to `models/`. `predict.py` loads both and applies the same `clean_text()` function, so new text is processed exactly like the training data.

**Q32. How would you deploy it?**
Wrap `predict()` in a FastAPI/Flask endpoint (`POST /predict` with article text, returning label + confidence), load the model once at startup, and run it on any CPU server. Each prediction takes microseconds, so one server could handle thousands of requests per second.

**Q33. What hardware did you use? How long does training take?**
An ordinary laptop CPU with no GPU. TF-IDF takes about 27 s and PAC training about 0.75 s.

**Q34. How would the system stay up to date?**
Fact-checkers label new articles → `partial_fit()` updates the model incrementally → evaluate it regularly on a recent held-out set → do a full retrain periodically to rebuild the vocabulary, since `partial_fit` cannot add new words to TF-IDF.

### G. Critical thinking
**Q35. What are the limitations of your project?**
(1) It detects style and source patterns, not facts. (2) English, US politics, 2016–2017 only. (3) Easy to fool: a fake article written in Reuters style may pass. (4) No word order or context. (5) The labels depend on the source site rather than checking each article.

**Q36. Can an adversary fool your model?**
Yes. Words like "said on Tuesday" and "spokesman" push towards REAL, so a fake story written in wire-service style could be classified REAL. That's why it should be used as a screening tool with a human in the loop.

**Q37. What would you do with more time?**
Train on more diverse data (WELFake, LIAR, Nigerian/African news sources); add bigrams; tune hyperparameters with cross-validation; compare with Logistic Regression, a linear SVM and DistilBERT; add explanations of each prediction (show the words that drove it, e.g. with LIME); build the FastAPI service and a web front end.

**Q38. Would this work on Nigerian news?**
Probably not well without retraining. It was trained on US political news, so vocabulary, names and writing style differ. The cross-dataset result (57.9%) shows how much accuracy falls on a new domain. We would need labelled Nigerian articles (e.g. from Dubawa or Africa Check) and retraining, which the online-learning design makes easy.

**Q39. What is your contribution, beyond following a tutorial?**
(1) We found and removed a label leak (the Reuters dateline) and 6,067 duplicates. (2) We ran a cross-dataset evaluation showing the known domain-shift weakness (98.4% → 57.9%). (3) We showed that training on combined data recovers most of it (88.2%). (4) We compared against a Naive Bayes baseline and built a reusable prediction tool.

**Q40. Ethical concerns?**
False positives could censor legitimate journalism; the dataset may carry political bias (most fake examples are anti-Clinton/pro-Trump US content); automated labelling needs transparency and human review. The system should flag articles, not delete them.

---

## 5. Two-minute defense summary
> "We built a fake news detector using TF-IDF features and a Passive-Aggressive Classifier. We trained it on
> the ISOT dataset of 44,898 articles and cleaned the data by removing 6,067 duplicates and a Reuters dateline
> that gave the label away. It reaches 98.4% accuracy, beating Naive Bayes at 93.3%, trains in under a second on
> a laptop CPU, and classifies an article in microseconds. We also tested it on a completely separate dataset.
> Accuracy fell to 58%, which showed that the model partly learns the source's writing style rather than truth.
> Training on both datasets raised that to 88%. The approach is fast and accurate within one news domain, but a
> real deployment needs diverse training data and ongoing online updates, which the Passive-Aggressive algorithm supports."
