"""
=============================================================================
BASELINE 1: SVM + TF-IDF
=============================================================================
Paper   : Canedo, E.D.; Mendes, B.C.
Title   : Software Requirements Classification using Machine Learning Algorithms
Journal : Entropy, MDPI, 2020, Vol. 22, Article 1057
URL     : https://doi.org/10.3390/e22101057
-----------------------------------------------------------------------------
Exact methodology:
  - TF-IDF vectorization with ngram_range=(1,2), sublinear_tf=True
  - SVM with RBF kernel, C=1.0, class_weight='balanced'
  - Evaluated on Project-Level Group Split (P1-P10 Train, P11-P15 Test)
    instead of their original 10-Fold CV, to ensure apple-to-apple fairness
=============================================================================
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}

def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(base, "data")

    train_df = pd.read_csv(os.path.join(data, "train_p1_p10.csv"))
    test_df  = pd.read_csv(os.path.join(data, "test_p11_p15.csv"))

    X_train = train_df["RequirementTextClean"].fillna("").tolist()
    y_train = train_df["Class"].map(LABEL_TO_IDX).values
    X_test  = test_df["RequirementTextClean"].fillna("").tolist()
    y_test  = test_df["Class"].map(LABEL_TO_IDX).values

    print("=" * 70)
    print("BASELINE 1 | Canedo & Mendes (Entropy 2020)")
    print("MODEL      : SVM with RBF kernel + TF-IDF ngram(1,2)")
    print(f"Train      : {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print("=" * 70)

    # --- Exact parameters from Canedo & Mendes (2020) ---
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        max_features=None      # no cap — as in original
    )
    X_tr_vec = vectorizer.fit_transform(X_train)
    X_te_vec = vectorizer.transform(X_test)

    clf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced",
        random_state=42
    )
    clf.fit(X_tr_vec, y_train)
    y_pred = clf.predict(X_te_vec)

    _report(y_test, y_pred)

def _report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _  = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    wp, wr, wf, _  = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _,  _,  cf, _  = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3],
                                                      average=None, zero_division=0)
    print(f"\nAccuracy    : {acc*100:.2f}%")
    print(f"Macro-F1    : {mf*100:.2f}%")
    print(f"Weighted-F1 : {wf*100:.2f}%")
    print(f"Macro-Prec  : {mp*100:.2f}%  |  Macro-Rec : {mr*100:.2f}%")
    print("\nPer-class F1:")
    for c, f in zip(TARGET_CLASSES, cf):
        print(f"  {c:3s}: {f*100:.2f}%")

if __name__ == "__main__":
    run()
