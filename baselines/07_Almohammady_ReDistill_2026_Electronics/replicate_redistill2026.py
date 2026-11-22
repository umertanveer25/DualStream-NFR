import sys
sys.modules['tensorflow'] = None
sys.modules['keras'] = None

import os
import re
import json
import warnings
warnings.filterwarnings("ignore")

os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.set_num_threads(16)
torch.set_num_interop_threads(4)
torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_LABEL   = {i: c for i, c in enumerate(TARGET_CLASSES)}

MAX_LEN_CONCAT = 192   # FR (128) + NFR (64)
MAX_LEN_NFR    = 96    # NFR isolated
BATCH_SIZE     = 16
EPOCHS         = 12
LR             = 2e-5

def clean_text(text):
    text = str(text)
    text = re.sub(r'^\d+,[\'\"\s]*', '', text).rstrip('\'\"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class SingleStreamDataset(Dataset):
    """
    Implements the EXACT concatenation format from Almohammady et al. (2026):
      [CLS] FR_text [SEP] NFR_text [SEP]
    """
    def __init__(self, nfr_texts, fr_texts, labels, tokenizer, max_len=MAX_LEN_CONCAT):
        if fr_texts is not None:
            self.encodings = tokenizer(
                fr_texts,
                nfr_texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
        else:
            self.encodings = tokenizer(
                nfr_texts,
                padding="max_length",
                truncation=True,
                max_length=max_len,
                return_tensors="pt"
            )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        return item, self.labels[i]

class SingleStreamLinearHead(nn.Module):
    """Linear Classification Head on final [CLS] representation"""
    def __init__(self, hidden_size=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, cls_rep):
        return self.classifier(self.dropout(cls_rep))

class SingleStreamMLPHead(nn.Module):
    """Deep 2-Layer MLP Classification Head on final [CLS] representation"""
    def __init__(self, hidden_size=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, cls_rep):
        return self.mlp(cls_rep)

def precompute_bert_cls_features(nfr_texts, fr_texts, tokenizer, bert_model, max_len=MAX_LEN_CONCAT, desc=""):
    print(f"Precomputing BERT [CLS] representations for {desc} (16 cores active)...", flush=True)
    bert_model.eval()
    if fr_texts is not None:
        encodings = tokenizer(fr_texts, nfr_texts, padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt")
    else:
        encodings = tokenizer(nfr_texts, padding="max_length", truncation=True,
                              max_length=max_len, return_tensors="pt")

    dataset = torch.utils.data.TensorDataset(encodings["input_ids"], encodings["attention_mask"], encodings["token_type_ids"])
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_cls = []
    with torch.no_grad():
        for b_ids, b_mask, b_type in loader:
            outputs = bert_model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type)
            cls_rep = outputs.last_hidden_state[:, 0, :]
            all_cls.append(cls_rep.cpu())

    return torch.cat(all_cls, dim=0).numpy()

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, cf, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "Accuracy": float(acc),
        "Macro_Precision": float(mp),
        "Macro_Recall": float(mr),
        "Macro_F1": float(mf),
        "Weighted_F1": float(wf),
        "PE_F1": float(cf[0]),
        "SE_F1": float(cf[1]),
        "US_F1": float(cf[2]),
        "A_F1": float(cf[3])
    }

def train_neural_head(head_cls, X_train_emb, y_train, X_test_emb, y_test, epochs=EPOCHS, lr=LR):
    torch.manual_seed(42)
    device = torch.device("cpu")
    head = head_cls(hidden_size=X_train_emb.shape[1], num_classes=4).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(torch.tensor(X_train_emb, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    head.train()
    for _ in range(epochs):
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            out = head(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

    head.eval()
    with torch.no_grad():
        test_x = torch.tensor(X_test_emb, dtype=torch.float32)
        logits = head(test_x)
        preds = torch.argmax(logits, dim=1).numpy()

    return preds

def run_project_split_experiment(train_df, test_df, tokenizer, bert_model):
    print("\n" + "=" * 80)
    print("PROTOCOL 1: Published Project-Level Holdout Split (P1-P10 Train vs. P11-P15 Test)")
    print(f"Train: {len(train_df)} samples | Test: {len(test_df)} samples")
    print("Exact protocol from Almohammady et al. (Electronics 2026)")
    print("=" * 80)

    X_nfr_tr = [clean_text(t) for t in train_df["RequirementTextClean"]]
    X_fr_tr  = [clean_text(t) for t in train_df["FR_Context"]]
    y_tr     = train_df["Class"].map(LABEL_TO_IDX).values

    X_nfr_te = [clean_text(t) for t in test_df["RequirementTextClean"]]
    X_fr_te  = [clean_text(t) for t in test_df["FR_Context"]]
    y_te     = test_df["Class"].map(LABEL_TO_IDX).values

    # Precompute Contextual [CLS] embeddings
    X_tr_concat_cls = precompute_bert_cls_features(X_nfr_tr, X_fr_tr, tokenizer, bert_model, MAX_LEN_CONCAT, "Train (Contextual)")
    X_te_concat_cls = precompute_bert_cls_features(X_nfr_te, X_fr_te, tokenizer, bert_model, MAX_LEN_CONCAT, "Test (Contextual)")

    # Precompute Isolated NFR [CLS] embeddings
    X_tr_iso_cls = precompute_bert_cls_features(X_nfr_tr, None, tokenizer, bert_model, MAX_LEN_NFR, "Train (Isolated NFR)")
    X_te_iso_cls = precompute_bert_cls_features(X_nfr_te, None, tokenizer, bert_model, MAX_LEN_NFR, "Test (Isolated NFR)")

    # Precompute TF-IDF
    concat_tr_texts = [f"{fr} {nfr}" for fr, nfr in zip(X_fr_tr, X_nfr_tr)]
    concat_te_texts = [f"{fr} {nfr}" for fr, nfr in zip(X_fr_te, X_nfr_te)]

    tfidf_concat = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    X_tr_tfidf_concat = tfidf_concat.fit_transform(concat_tr_texts).toarray()
    X_te_tfidf_concat = tfidf_concat.transform(concat_te_texts).toarray()

    tfidf_iso = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    X_tr_tfidf_iso = tfidf_iso.fit_transform(X_nfr_tr).toarray()
    X_te_tfidf_iso = tfidf_iso.transform(X_nfr_te).toarray()

    results = []

    # 1. Single-Stream BERT (Linear Head) - Author's exact architecture
    preds_lin = train_neural_head(SingleStreamLinearHead, X_tr_concat_cls, y_tr, X_te_concat_cls, y_te, epochs=15, lr=5e-4)
    m = evaluate_metrics(y_te, preds_lin)
    m["Technique"] = "Single-Stream BERT (Linear Head)"
    m["Input_Type"] = "Contextual (FR + NFR)"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    # 2. Single-Stream BERT (MLP Head)
    preds_mlp = train_neural_head(SingleStreamMLPHead, X_tr_concat_cls, y_tr, X_te_concat_cls, y_te, epochs=15, lr=5e-4)
    m = evaluate_metrics(y_te, preds_mlp)
    m["Technique"] = "Single-Stream BERT (MLP Head)"
    m["Input_Type"] = "Contextual (FR + NFR)"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    # 3. Isolated BERT (Linear Head - No FR Context)
    preds_iso = train_neural_head(SingleStreamLinearHead, X_tr_iso_cls, y_tr, X_te_iso_cls, y_te, epochs=15, lr=5e-4)
    m = evaluate_metrics(y_te, preds_iso)
    m["Technique"] = "Isolated NFR BERT (Linear Head)"
    m["Input_Type"] = "Isolated (NFR Only)"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    # 4. Classical ML Techniques on Contextual TF-IDF
    ml_models = {
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=16),
        "SGD Classifier": SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=16),
        "RBF SVM": SVC(kernel='rbf', C=1.0, random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    for name, clf in ml_models.items():
        clf.fit(X_tr_tfidf_concat, y_tr)
        preds = clf.predict(X_te_tfidf_concat)
        m = evaluate_metrics(y_te, preds)
        m["Technique"] = f"TF-IDF + {name}"
        m["Input_Type"] = "Contextual (FR + NFR)"
        m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
        results.append(m)

    return pd.DataFrame(results)

def run_10fold_cv_experiment(all_df, tokenizer, bert_model):
    print("\n" + "=" * 80)
    print("PROTOCOL 2: 10-Fold Stratified Cross-Validation on Full Contextual NFR Dataset (208 samples)")
    print("Consistent with 10-fold benchmark protocol across Papers 1, 2, 3, 5, 6")
    print("=" * 80)

    X_nfr = [clean_text(t) for t in all_df["RequirementTextClean"]]
    X_fr  = [clean_text(t) for t in all_df["FR_Context"]]
    y     = all_df["Class"].map(LABEL_TO_IDX).values

    # Precompute Contextual [CLS] embeddings
    X_concat_cls = precompute_bert_cls_features(X_nfr, X_fr, tokenizer, bert_model, MAX_LEN_CONCAT, "All 208 Samples (Contextual)")
    X_iso_cls    = precompute_bert_cls_features(X_nfr, None, tokenizer, bert_model, MAX_LEN_NFR, "All 208 Samples (Isolated NFR)")

    concat_texts = [f"{fr} {nfr}" for fr, nfr in zip(X_fr, X_nfr)]
    tfidf_concat = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    X_tfidf_concat = tfidf_concat.fit_transform(concat_texts).toarray()

    tfidf_iso = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    X_tfidf_iso = tfidf_iso.fit_transform(X_nfr).toarray()

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    techniques = {
        "Single-Stream BERT (Linear Head)": "neural_lin",
        "Single-Stream BERT (MLP Head)": "neural_mlp",
        "Isolated NFR BERT (Linear Head)": "neural_iso",
        "TF-IDF (Contextual) + Linear SVM": "svm_lin",
        "TF-IDF (Contextual) + Extra Trees": "et",
        "TF-IDF (Contextual) + SGD Classifier": "sgd",
        "TF-IDF (Contextual) + Logistic Regression": "lr",
        "TF-IDF (Contextual) + Naive Bayes": "nb",
        "TF-IDF (Contextual) + Random Forest": "rf",
        "TF-IDF (Contextual) + RBF SVM": "rbf",
        "TF-IDF (Contextual) + MLP Classifier": "mlp",
        "TF-IDF (Contextual) + Decision Tree": "dt"
    }

    all_fold_preds = {k: np.zeros(len(y), dtype=int) for k in techniques}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_concat_cls, y)):
        y_tr, y_va = y[train_idx], y[val_idx]

        # Single-Stream BERT (Linear)
        preds_lin = train_neural_head(SingleStreamLinearHead, X_concat_cls[train_idx], y_tr, X_concat_cls[val_idx], y_va, epochs=15, lr=5e-4)
        all_fold_preds["Single-Stream BERT (Linear Head)"][val_idx] = preds_lin

        # Single-Stream BERT (MLP)
        preds_mlp = train_neural_head(SingleStreamMLPHead, X_concat_cls[train_idx], y_tr, X_concat_cls[val_idx], y_va, epochs=15, lr=5e-4)
        all_fold_preds["Single-Stream BERT (MLP Head)"][val_idx] = preds_mlp

        # Isolated BERT
        preds_iso = train_neural_head(SingleStreamLinearHead, X_iso_cls[train_idx], y_tr, X_iso_cls[val_idx], y_va, epochs=15, lr=5e-4)
        all_fold_preds["Isolated NFR BERT (Linear Head)"][val_idx] = preds_iso

        # Classical ML
        ml_inst = {
            "TF-IDF (Contextual) + Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
            "TF-IDF (Contextual) + Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=16),
            "TF-IDF (Contextual) + SGD Classifier": SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42),
            "TF-IDF (Contextual) + Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "TF-IDF (Contextual) + Naive Bayes": MultinomialNB(),
            "TF-IDF (Contextual) + Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=16),
            "TF-IDF (Contextual) + RBF SVM": SVC(kernel='rbf', C=1.0, random_state=42),
            "TF-IDF (Contextual) + MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            "TF-IDF (Contextual) + Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        for name, clf in ml_inst.items():
            clf.fit(X_tfidf_concat[train_idx], y_tr)
            all_fold_preds[name][val_idx] = clf.predict(X_tfidf_concat[val_idx])

    results = []
    for name in techniques:
        m = evaluate_metrics(y, all_fold_preds[name])
        m["Technique"] = name
        m["Input_Type"] = "Isolated" if "Isolated" in name else "Contextual"
        m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
        results.append(m)

    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("PAPER 7 REPLICATION: Almohammady et al. (MDPI Electronics 2026)")
    print("Context-Aware NFR Classification Using Single-Stream Concatenation & Re-Distill")
    print("Dataset: DualStream-NFR/data/nfr_all_with_context.csv (PROMISE NFR + FR traces)")
    print("Allocated CPU Threads: 16 (all 16 cores active)")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    out_dir  = os.path.join(base_dir, "baselines", "07_Almohammady_ReDistill_2026_Electronics")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_p1_p10.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test_p11_p15.csv"))
    all_df   = pd.read_csv(os.path.join(data_dir, "nfr_all_with_context.csv"))

    print(f"Loaded train_p1_p10: {len(train_df)} rows")
    print(f"Loaded test_p11_p15: {len(test_df)} rows")
    print(f"Loaded nfr_all_with_context: {len(all_df)} rows")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # Protocol 1: Project Split
    res_split = run_project_split_experiment(train_df, test_df, tokenizer, bert_model)

    # Protocol 2: 10-Fold CV
    res_10cv  = run_10fold_cv_experiment(all_df, tokenizer, bert_model)

    combined_df = pd.concat([res_split, res_10cv], ignore_index=True)

    csv_path  = os.path.join(out_dir, "redistill2026_results.csv")
    json_path = os.path.join(out_dir, "redistill2026_results.json")

    combined_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(combined_df.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 80)
    print("PAPER 7 REPLICATION COMPLETE!")
    print(f"Saved CSV  : {csv_path}")
    print(f"Saved JSON : {json_path}")
    print("=" * 80)

    print("\n[SUMMARY RESULTS: PROJECT-LEVEL SPLIT (P1-P10 -> P11-P15)]")
    cols_display = ["Technique", "Accuracy", "Macro_F1", "Weighted_F1", "PE_F1", "SE_F1", "US_F1"]
    split_view = res_split[cols_display].copy()
    for col in cols_display[1:]:
        split_view[col] = (split_view[col] * 100).round(2)
    print(split_view.to_string(index=False))

    print("\n[SUMMARY RESULTS: 10-FOLD STRATIFIED CV (208 SAMPLES)]")
    cv_view = res_10cv[cols_display].copy()
    for col in cols_display[1:]:
        cv_view[col] = (cv_view[col] * 100).round(2)
    print(cv_view.to_string(index=False))

if __name__ == "__main__":
    main()
