Title: Disaster Tweets Classification (Kaggle)

Summary: Binary classification of tweets to detect real disaster reports. Approach uses simple text cleaning, TF-IDF features (1–2 grams), and Logistic Regression. Fast, reproducible, CPU-only baseline.

Data: Kaggle competition “Natural Language Processing with Disaster Tweets”. Files: train.csv, test.csv, sample_submission.csv. Columns in train.csv: id, keyword, location, text, target. Do not redistribute raw files.

Exploratory Data Analysis (what I examined):

Word/tweet length distributions (tokens and characters) with histograms.

Class balance histogram for target.

Missing-value heatmap or counts for keyword and location.

Top tokens/hashtags/bigrams tables for each class.

Data Cleaning (what and why):

Lowercasing; URL normalization; punctuation normalization; collapse whitespace.

Stop-word removal to reduce uninformative tokens.

Optional stemming/lemmatization note: kept vocabulary stable for n-grams, so I DID_OR_DID_NOT apply it because REASON.

Kept # and @ to preserve hashtag/mention signal.

Plan of Analysis (based on EDA + embedding methods explained):

Word embeddings considered:

TF-IDF (bag-of-words 1–2 grams; explains term weighting and sparsity).

Character n-grams for misspellings/hashtags.

Contextual embeddings via a small transformer (e.g., DistilBERT) to capture semantics.

Plan: start with TF-IDF baselines (fast, interpretable), then try Linear SVM and Complement NB, then optionally a small transformer. Use 5-fold stratified CV with F1. Use EDA findings to set max_features, n-gram ranges, and consider class weights due to CLASS_IMBALANCE_RATE.

Model Architecture (what, why, tuning, comparisons):

Models evaluated:
A) Logistic Regression (TF-IDF) — chosen for strong linear baseline on sparse text.
B) Linear SVM (TF-IDF) — margin-based classifier often strong for text.
C) Complement Naive Bayes (TF-IDF) — robust on imbalanced text.
D) Optional: DistilBERT classifier (transformer) — contextual understanding.

Hyperparameters tuned (examples):
TF-IDF: ngram_range {(1,1),(1,2)}, max_features {20k,40k}, min_df {1,2}.
LR: C {0.5,1,2,4}, class_weight {None, balanced}.
LinearSVC: C {0.5,1,2}, loss {hinge, squared_hinge}.
NB: alpha {0.5,1,2}.
DistilBERT: learning_rate, epochs, max_len, batch_size.

Why suitable: short tweets favor sparse linear models; hashtags/bigrams capture event context; transformer adds semantics if compute allows. I compare architectures using identical CV splits.

Results and Analysis (tables/figures + reasoning + HPO summary + troubleshooting):

Main metric: F1 on 5-fold CV.

Results table (example columns): Model | Embedding | Key Hyperparams | CV F1 Mean | CV F1 Std | Kaggle Public F1.

Hyperparameter optimization summary: WHAT_WORKED and WHY; WHAT_DIDN’T and WHY.

Troubleshooting notes: e.g., overfitting with too many features, class imbalance mitigated by class_weight, URL/hashtag handling improved recall, threshold tuning if using probabilistic outputs.

Conclusion (restate results + learnings + why some things failed + improvements):

Best model: BEST_MODEL with CV F1 = CV_MEAN ± CV_STD; Kaggle public = LB_SCORE.

Key takeaways: WHAT_I_LEARNED.

What didn’t work and why: FAILED_ATTEMPTS + REASONS.

Improvements: add character n-grams, tune C and max_features further, try calibrated probabilities, experiment with a small transformer and cleaner domain-specific stop list.

Reproducibility (how to run):

Kaggle: Add Data → Competitions → “nlp-getting-started”; run notebook; outputs /kaggle/working/submission.csv.

Local: pip install -r requirements.txt; place CSVs; run notebook to produce submission.csv.

Random seeds and versions noted in the notebook header.

Repository structure and code quality:

Files: disaster_tweets.ipynb, README.md, requirements.txt, submission.csv (optional).

Code commented to explain why each block exists (cleaning choices, vectorizer settings, model/tuning rationale).

Notebook includes the public GitHub URL in the first markdown cell.

Kaggle participation:

I submitted submission.csv and captured a leaderboard screenshot as verification (score > 0).

Links for graders:

Public GitHub repository: GITHUB_URL

Competition page: https://www.kaggle.com/c/nlp-getting-started
