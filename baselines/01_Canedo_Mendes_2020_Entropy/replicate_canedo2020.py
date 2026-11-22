import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def clean_text(text):
    text = str(text)
    # Remove leading quotes or IDs like 1,'
    text = re.sub(r'^\d+,[\'\"\s]*', '', text).rstrip('\'\"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_task(X, y, task_name, class_names):
    print("\n" + "=" * 80)
    print(f"RUNNING 10-FOLD CV IN PARALLEL (16 CORES): {task_name}")
    print(f"Total samples: {len(y)} | Classes: {class_names}")
    print("=" * 80)

    classifiers = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Linear SVM (LinearSVC)": LinearSVC(C=1.0, random_state=42),
        "RBF SVM": SVC(C=2.0, kernel='rbf', random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SGD Classifier (SVM loss)": SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

    results = []
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    scoring = {
        'accuracy': 'accuracy',
        'precision_macro': 'precision_macro',
        'recall_macro': 'recall_macro',
        'f1_macro': 'f1_macro',
        'f1_weighted': 'f1_weighted'
    }

    for name, clf in classifiers.items():
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)),
            ('clf', clf)
        ])

        scores = cross_validate(
            pipeline, X, y, cv=skf, scoring=scoring, n_jobs=-1, return_train_score=False
        )

        acc = np.mean(scores['test_accuracy'])
        p_mac = np.mean(scores['test_precision_macro'])
        r_mac = np.mean(scores['test_recall_macro'])
        f1_mac = np.mean(scores['test_f1_macro'])
        f1_wt = np.mean(scores['test_f1_weighted'])

        results.append({
            "Task": task_name,
            "Classifier": name,
            "Accuracy": acc,
            "Macro_Precision": p_mac,
            "Macro_Recall": r_mac,
            "Macro_F1": f1_mac,
            "Weighted_F1": f1_wt
        })

        print(f"  [{name:<28}] Acc: {acc*100:.2f}% | Macro-F1: {f1_mac*100:.2f}% | Weighted-F1: {f1_wt*100:.2f}%")

    return results

def main():
    data_path = r"B:\Dataset\nfr.csv"
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("PAPER 1 REPLICATION: Canedo & Mendes (Entropy 2020)")
    print(f"Loading exact dataset from: {data_path}")
    print("=" * 80)

    df = pd.read_csv(data_path, encoding='latin-1')
    df['CleanText'] = df['RequirementText'].apply(clean_text)

    print(f"Total rows in nfr.csv: {len(df)}")
    print(f"Class distribution:\n{df['class'].value_counts().to_string()}")

    all_results = []

    # =========================================================================
    # TASK 1: Binary Classification (Functional vs. Non-Functional) - 625 items
    # In Canedo & Mendes (2020): F is class 'F', all others are NFR
    # =========================================================================
    df['BinaryLabel'] = df['class'].apply(lambda c: 'FR' if c == 'F' else 'NFR')
    X_bin = df['CleanText'].values
    y_bin = df['BinaryLabel'].values

    bin_results = run_task(
        X_bin, y_bin, 
        task_name="Task 1: Binary Classification (FR vs. NFR on 625 requirements)",
        class_names=["FR (255)", "NFR (370)"]
    )
    all_results.extend(bin_results)

    # =========================================================================
    # TASK 2: 4-Class Quality Requirement Classification (PE, SE, US, A) - 208 items
    # In Canedo & Mendes (2020): Evaluating quality sub-classes
    # =========================================================================
    target_4 = ['PE', 'SE', 'US', 'A']
    df_4 = df[df['class'].isin(target_4)].copy().reset_index(drop=True)
    X_4 = df_4['CleanText'].values
    y_4 = df_4['class'].values

    task2_results = run_task(
        X_4, y_4,
        task_name="Task 2: 4 Quality Requirement Classes (PE, SE, US, A)",
        class_names=target_4
    )
    all_results.extend(task2_results)

    # =========================================================================
    # TASK 3: Most Frequent NFR Sub-classes (US, SE, O, PE, LF) - 287 items
    # Top 5 most frequent NFR classes reported in Canedo & Mendes Table 6
    # =========================================================================
    target_top5 = ['US', 'SE', 'O', 'PE', 'LF']
    df_top5 = df[df['class'].isin(target_top5)].copy().reset_index(drop=True)
    X_top5 = df_top5['CleanText'].values
    y_top5 = df_top5['class'].values

    task3_results = run_task(
        X_top5, y_top5,
        task_name="Task 3: Top 5 Most Frequent NFR Classes (US, SE, O, PE, LF)",
        class_names=target_top5
    )
    all_results.extend(task3_results)

    # Save Results
    res_df = pd.DataFrame(all_results)
    csv_out = os.path.join(out_dir, "canedo2020_results.csv")
    json_out = os.path.join(out_dir, "canedo2020_results.json")

    res_df.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n" + "=" * 80)
    print("PAPER 1 (Canedo & Mendes 2020) REPLICATION COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  - {csv_out}\n  - {json_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
