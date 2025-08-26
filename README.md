Title: Disaster Tweets Classification (Kaggle)

Summary: Binary classification of tweets to detect real disaster reports. Approach uses simple text cleaning, TF-IDF features (1–2 grams), and Logistic Regression. Fast, reproducible, CPU-only baseline.

Data: Kaggle competition “Natural Language Processing with Disaster Tweets”. Files: train.csv, test.csv, sample_submission.csv. Columns in train.csv: id, keyword, location, text, target. Do not redistribute raw files.

Requirements: pandas, numpy, scikit-learn.

Reproduce (Kaggle Notebook):

Add Data -> Competitions -> “nlp-getting-started”.

Run disaster_tweets.ipynb.

Reads from /kaggle/input/nlp-getting-started/train.csv, /kaggle/input/nlp-getting-started/test.csv, /kaggle/input/nlp-getting-started/sample_submission.csv.

Writes /kaggle/working/submission_baseline.csv.

Reproduce (Local):

Install requirements.

Place train.csv, test.csv, sample_submission.csv in project root (or update paths).

Run disaster_tweets.ipynb to create submission_baseline.csv.

Method:

Preprocessing: lowercase, replace URLs with token URL, keep basic characters, collapse whitespace.

Features: TF-IDF, ngram_range (1,2), min_df 2, max_features 20000, sublinear_tf True.

Model: LogisticRegression, solver liblinear, C 4.0, class_weight balanced, max_iter 200, random_state 42.

Evaluation: 5-fold Stratified CV, metric F1.

Results:
CV F1 (mean ± std): REPLACE_WITH_VALUE
Kaggle public leaderboard score: REPLACE_WITH_VALUE

Next Steps:
Try character n-grams, tune hyperparameters, try LinearSVC or Complement NB, consider pretrained transformers.

Repo Files:
disaster_tweets.ipynb
README.md
requirements.txt
