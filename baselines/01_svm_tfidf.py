"""
Baseline 1: Classical Machine Learning with TF-IDF + Support Vector Machine (SVM)
Reference: Canedo & Mendes (Entropy 2020), Kurtanovic & Maalej (IEEE RE 2017)
Evaluates on: Project-Level Split (Train: P1-P10, Test: P11-P15)
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

np.random.seed(42)

def run_svm_baseline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    train_df = pd.read_csv(os.path.join(data_dir, "train_p1_p10.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_p11_p15.csv"))
    
    target_classes = ['PE', 'SE', 'US', 'A']
    label_to_idx = {c: i for i, c in enumerate(target_classes)}
    
    X_train_raw = train_df['RequirementTextClean'].tolist()
    y_train = train_df['Class'].map(label_to_idx).values
    
    X_test_raw = test_df['RequirementTextClean'].tolist()
    y_test = test_df['Class'].map(label_to_idx).values
    
    print("=" * 70)
    print("BASELINE 1: SVM (TF-IDF + Linear Support Vector Classifier)")
    print(f"Train samples (P1-P10): {len(X_train_raw)} | Test samples (P11-P15): {len(X_test_raw)}")
    print("=" * 70)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=1500, sublinear_tf=True)
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)
    
    clf = LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=2000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    
    print(f"\n[RESULTS] Accuracy:    {acc * 100:.2f}%")
    print(f"[RESULTS] Macro-F1:    {macro_f1 * 100:.2f}%")
    print(f"[RESULTS] Weighted-F1: {weighted_f1 * 100:.2f}%")
    print(f"[RESULTS] Macro-Prec:  {macro_p * 100:.2f}%")
    print(f"[RESULTS] Macro-Rec:   {macro_r * 100:.2f}%")
    print("\nPer-Category F1 Scores:")
    for c, f1 in zip(target_classes, class_f1s):
        print(f"  - {c} F1: {f1 * 100:.2f}%")
        
    return {
        'Model': 'SVM (TF-IDF Baseline)',
        'Accuracy': round(acc * 100, 2),
        'Macro_F1': round(macro_f1 * 100, 2),
        'Weighted_F1': round(weighted_f1 * 100, 2),
        'PE_F1': round(class_f1s[0] * 100, 2),
        'SE_F1': round(class_f1s[1] * 100, 2),
        'US_F1': round(class_f1s[2] * 100, 2),
        'A_F1': round(class_f1s[3] * 100, 2)
    }

if __name__ == "__main__":
    run_svm_baseline()
