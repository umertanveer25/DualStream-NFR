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
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.set_num_threads(16)
torch.set_num_interop_threads(4)
torch.manual_seed(42)
np.random.seed(42)

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3

def clean_text(text):
    text = str(text)
    text = re.sub(r'^\d+,[\'\"\s]*', '', text).rstrip('\'\"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class BertCLSClassifier(nn.Module):
    """Exact Devlin / Ayed 2024 Sequence Classification Head on [CLS] token"""
    def __init__(self, emb_dim=768, num_classes=4, dropout=0.1):
        super(BertCLSClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, cls_embs):
        return self.classifier(self.dropout(cls_embs))

class BertMLPClassifier(nn.Module):
    """BERT [CLS] + 2-layer MLP Head"""
    def __init__(self, emb_dim=768, num_classes=4, dropout=0.2):
        super(BertMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(emb_dim, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, cls_embs):
        h = self.dropout(self.relu(self.fc1(cls_embs)))
        return self.fc2(h)

def precompute_cls_embeddings(texts, tokenizer, bert_model):
    print("Precomputing BERT [CLS] contextual embeddings (all 16 cores active)...", flush=True)
    bert_model.eval()
    cls_list = []
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch = texts[i:i+32]
            enc = tokenizer(batch, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
            out = bert_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            # [CLS] token representation
            cls_rep = out.last_hidden_state[:, 0, :].cpu()
            cls_list.append(cls_rep)
    return torch.cat(cls_list, dim=0)

def train_and_eval_neural(model_fn, X_embs, y_arr, num_classes, task_name, class_names, model_name):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_preds = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_embs, y_arr)):
        X_tr, y_tr = X_embs[train_idx], torch.tensor(y_arr[train_idx], dtype=torch.long)
        X_te, y_te = X_embs[test_idx], torch.tensor(y_arr[test_idx], dtype=torch.long)

        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = model_fn()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for epoch in range(EPOCHS):
            for bx, by in train_loader:
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            preds = torch.argmax(model(X_te), dim=1).numpy()

        f_acc = accuracy_score(y_te.numpy(), preds)
        _, _, f_f1, _ = precision_recall_fscore_support(y_te.numpy(), preds, average="macro", zero_division=0)
        print(f"  [{model_name:<30}] Fold {fold+1:2d}/10 | Acc: {f_acc*100:.2f}% | Macro-F1: {f_f1*100:.2f}%", flush=True)

        y_trues.extend(y_te.numpy())
        y_preds.extend(preds)

    acc = accuracy_score(y_trues, y_preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_trues, y_preds, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_trues, y_preds, labels=list(range(num_classes)), average=None, zero_division=0)

    result = {
        'Task': task_name,
        'Model': model_name,
        'Accuracy': float(acc),
        'Macro_Precision': float(p_m),
        'Macro_Recall': float(r_m),
        'Macro_F1': float(f1_m),
        'Weighted_F1': float(f1_w),
        'Class_F1s': {class_names[i]: float(class_f1s[i]) for i in range(num_classes)}
    }
    return result

def train_and_eval_ml(clf, texts, y_arr, num_classes, task_name, class_names, model_name):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_preds = [], []
    texts_arr = np.array(texts)

    for train_idx, test_idx in skf.split(texts_arr, y_arr):
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_tr = tfidf.fit_transform(texts_arr[train_idx])
        X_te = tfidf.transform(texts_arr[test_idx])
        clf.fit(X_tr, y_arr[train_idx])
        preds = clf.predict(X_te)
        y_trues.extend(y_arr[test_idx])
        y_preds.extend(preds)

    acc = accuracy_score(y_trues, y_preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_trues, y_preds, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_trues, y_preds, labels=list(range(num_classes)), average=None, zero_division=0)

    result = {
        'Task': task_name,
        'Model': model_name,
        'Accuracy': float(acc),
        'Macro_Precision': float(p_m),
        'Macro_Recall': float(r_m),
        'Macro_F1': float(f1_m),
        'Weighted_F1': float(f1_w),
        'Class_F1s': {class_names[i]: float(class_f1s[i]) for i in range(num_classes)}
    }
    print(f"  [{model_name:<30}] Mean Acc: {acc*100:.2f}% | Macro-F1: {f1_m*100:.2f}% | Weighted-F1: {f1_w*100:.2f}%", flush=True)
    return result

def main():
    data_path = r"B:\Dataset\Promise_Exp.csv"
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("PAPER 3 REPLICATION: Devlin et al. (2019) / Ayed & Alrammal (COMPSAC 2024)")
    print(f"Loading exact dataset from: {data_path}")
    print(f"Allocated CPU Threads: {torch.get_num_threads()} (all 16 cores active)")
    print("=" * 80)

    df = pd.read_csv(data_path, encoding='latin-1')
    req_col = 'Requirement' if 'Requirement' in df.columns else 'RequirementText'
    type_col = 'Type' if 'Type' in df.columns else 'class'
    df['CleanText'] = df[req_col].apply(clean_text)

    print(f"Total rows in Promise_Exp.csv: {len(df)}")
    print(f"Class distribution:\n{df[type_col].value_counts().to_string()}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    all_texts = df['CleanText'].tolist()
    all_cls_embs = precompute_cls_embeddings(all_texts, tokenizer, bert_model)

    all_results = []

    # =========================================================================
    # Task 1: 4 Quality Requirement Classes (PE, SE, US, A) - 308 items
    # The primary benchmark in Ayed & Alrammal / DualStream-NFR
    # =========================================================================
    target_4 = ['PE', 'SE', 'US', 'A']
    mask_4 = df[type_col].isin(target_4)
    idx_4 = np.where(mask_4)[0]
    X_4_embs = all_cls_embs[idx_4]
    texts_4 = [all_texts[i] for i in idx_4]
    map_4 = {c: i for i, c in enumerate(target_4)}
    y_4 = np.array([map_4[c] for c in df.loc[mask_4, type_col]])

    print("\n" + "=" * 80)
    print("BENCHMARKING ALL TECHNIQUES: Task 1: 4 Quality Classes (PE, SE, US, A - 308 items)")
    print("=" * 80)

    # 1. BERT ([CLS] + Linear Head - Devlin / Ayed 2024)
    res_bert_cls = train_and_eval_neural(
        lambda: BertCLSClassifier(emb_dim=768, num_classes=4),
        X_4_embs, y_4, 4,
        task_name="Task 1: 4 Quality Classes (PE, SE, US, A on 308 requirements)",
        class_names=target_4,
        model_name="BERT [CLS] + Linear (Ayed 2024)"
    )
    all_results.append(res_bert_cls)

    # 2. BERT ([CLS] + MLP Head)
    res_bert_mlp = train_and_eval_neural(
        lambda: BertMLPClassifier(emb_dim=768, num_classes=4),
        X_4_embs, y_4, 4,
        task_name="Task 1: 4 Quality Classes (PE, SE, US, A on 308 requirements)",
        class_names=target_4,
        model_name="BERT [CLS] + MLP Head"
    )
    all_results.append(res_bert_mlp)

    # 3. Linear SVM Baseline
    res_svm = train_and_eval_ml(
        LinearSVC(C=1.0, random_state=42),
        texts_4, y_4, 4,
        task_name="Task 1: 4 Quality Classes (PE, SE, US, A on 308 requirements)",
        class_names=target_4,
        model_name="Linear SVM Baseline"
    )
    all_results.append(res_svm)

    # 4. Multinomial Naive Bayes Baseline
    res_nb = train_and_eval_ml(
        MultinomialNB(alpha=0.5),
        texts_4, y_4, 4,
        task_name="Task 1: 4 Quality Classes (PE, SE, US, A on 308 requirements)",
        class_names=target_4,
        model_name="Multinomial Naive Bayes"
    )
    all_results.append(res_nb)

    # 5. Random Forest Baseline
    res_rf = train_and_eval_ml(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        texts_4, y_4, 4,
        task_name="Task 1: 4 Quality Classes (PE, SE, US, A on 308 requirements)",
        class_names=target_4,
        model_name="Random Forest Baseline"
    )
    all_results.append(res_rf)

    # =========================================================================
    # Task 2: Binary Classification (FR vs NFR - 969 items)
    # =========================================================================
    df['BinaryLabel'] = df[type_col].apply(lambda x: 'FR' if x == 'F' else 'NFR')
    bin_classes = ['FR', 'NFR']
    bin_map = {'FR': 0, 'NFR': 1}
    y_bin = np.array([bin_map[c] for c in df['BinaryLabel']])

    print("\n" + "=" * 80)
    print("BENCHMARKING ALL TECHNIQUES: Task 2: Binary Classification (FR vs NFR - 969 items)")
    print("=" * 80)

    # 1. BERT ([CLS] + Linear Head)
    res_bin_bert = train_and_eval_neural(
        lambda: BertCLSClassifier(emb_dim=768, num_classes=2),
        all_cls_embs, y_bin, 2,
        task_name="Task 2: Binary Classification (FR vs NFR on 969 requirements)",
        class_names=bin_classes,
        model_name="BERT [CLS] + Linear (Ayed 2024)"
    )
    all_results.append(res_bin_bert)

    # 2. Linear SVM Baseline
    res_bin_svm = train_and_eval_ml(
        LinearSVC(C=1.0, random_state=42),
        all_texts, y_bin, 2,
        task_name="Task 2: Binary Classification (FR vs NFR on 969 requirements)",
        class_names=bin_classes,
        model_name="Linear SVM Baseline"
    )
    all_results.append(res_bin_svm)

    # 3. Random Forest Baseline
    res_bin_rf = train_and_eval_ml(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        all_texts, y_bin, 2,
        task_name="Task 2: Binary Classification (FR vs NFR on 969 requirements)",
        class_names=bin_classes,
        model_name="Random Forest Baseline"
    )
    all_results.append(res_bin_rf)

    # Save to files
    json_path = os.path.join(out_dir, "ayed2024_results.json")
    csv_path = os.path.join(out_dir, "ayed2024_results.csv")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    rows = []
    for r in all_results:
        row = {
            'Task': r['Task'],
            'Model': r['Model'],
            'Accuracy': r['Accuracy'],
            'Macro_Precision': r['Macro_Precision'],
            'Macro_Recall': r['Macro_Recall'],
            'Macro_F1': r['Macro_F1'],
            'Weighted_F1': r['Weighted_F1'],
        }
        for k, v in r['Class_F1s'].items():
            row[f"{k}_F1"] = v
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("PAPER 3 (ALL TECHNIQUES) REPLICATION COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  - {csv_path}\n  - {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
