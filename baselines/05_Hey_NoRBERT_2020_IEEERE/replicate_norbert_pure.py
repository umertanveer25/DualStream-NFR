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
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.set_num_threads(16)
torch.set_num_interop_threads(4)
torch.manual_seed(42)
np.random.seed(42)

MAX_LEN = 128
BATCH_SIZE = 32
FT_EPOCHS = 3
LR = 2e-5

def clean_text(text):
    text = str(text)
    text = re.sub(r'^\d+,[\'\"\s]*', '', text).rstrip('\'\"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class PUREDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "PURE_PROMISE_FR_NFR_Dataset.csv")
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 80)
    print("PAPER 5 (NoRBERT) REPLICATION ON PURE DATASET")
    print(f"Allocated CPU Threads: {torch.get_num_threads()} (all cores active)")
    print("=" * 80)

    print("Loading authentic PURE dataset from:", data_path)
    df = pd.read_csv(data_path).dropna(subset=['Requirement Text', 'Type'])
    df['CleanText'] = df['Requirement Text'].apply(clean_text)

    # Filter to FR and NFR
    df = df[df['Type'].isin(['FR', 'NFR'])].copy().reset_index(drop=True)
    label_map = {'FR': 0, 'NFR': 1}
    df['Label'] = df['Type'].map(label_map)

    print(f"Total PURE Samples: {len(df)}")
    print(f"Class Distribution: {df['Type'].value_counts().to_dict()}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # 5-Fold Stratified Cross-Validation on PURE for NoRBERT
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    texts_arr = df['CleanText'].values
    labels_arr = df['Label'].values

    y_trues, y_preds = [], []
    fold_metrics = []

    print("\n" + "=" * 80)
    print("RUNNING 5-FOLD CV: NoRBERT on PURE Requirements (FR vs NFR)")
    print("=" * 80)

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        X_tr, y_tr = texts_arr[train_idx].tolist(), labels_arr[train_idx].tolist()
        X_te, y_te = texts_arr[test_idx].tolist(), labels_arr[test_idx].tolist()

        train_ds = PUREDataset(X_tr, y_tr, tokenizer)
        test_ds = PUREDataset(X_te, y_te, tokenizer)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * FT_EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.10 * total_steps), total_steps)

        model.train()
        for epoch in range(FT_EPOCHS):
            total_loss = 0
            for b_inputs, b_labels in train_loader:
                optimizer.zero_grad()
                out = model(**b_inputs, labels=b_labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()

        # Evaluate Fold
        model.eval()
        f_preds, f_trues = [], []
        with torch.no_grad():
            for b_inputs, b_labels in test_loader:
                out = model(**b_inputs)
                preds = torch.argmax(out.logits, dim=1).cpu().tolist()
                f_preds.extend(preds)
                f_trues.extend(b_labels.tolist())

        y_preds.extend(f_preds)
        y_trues.extend(f_trues)

        f_acc = accuracy_score(f_trues, f_preds)
        f_prec, f_rec, f_f1, _ = precision_recall_fscore_support(f_trues, f_preds, average="macro", zero_division=0)
        fold_metrics.append({
            "Fold": fold + 1,
            "Accuracy": f_acc,
            "Macro_Precision": f_prec,
            "Macro_Recall": f_rec,
            "Macro_F1": f_f1
        })
        print(f"  Fold {fold+1}/5 | Accuracy: {f_acc*100:.2f}% | Macro-F1: {f_f1*100:.2f}%", flush=True)

    # Overall Summary
    overall_acc = accuracy_score(y_trues, y_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_trues, y_preds, average="macro", zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(y_trues, y_preds, average="weighted", zero_division=0)
    p_cls, r_cls, f1_cls, _ = precision_recall_fscore_support(y_trues, y_preds, average=None, zero_division=0)

    print("\n" + "-" * 70)
    print("OVERALL 5-FOLD RESULTS FOR NoRBERT ON PURE DATASET:")
    print(f"Accuracy    : {overall_acc*100:.2f}%")
    print(f"Macro-F1    : {f1_macro*100:.2f}%")
    print(f"Weighted-F1 : {f1_wt*100:.2f}%")
    print("Per-class F1 breakdown:")
    print(f"  FR  : {f1_cls[0]*100:.2f}%")
    print(f"  NFR : {f1_cls[1]*100:.2f}%")
    print("-" * 70)

    # Save to CSV and JSON
    res_df = pd.DataFrame(fold_metrics)
    csv_out = os.path.join(out_dir, "norbert_pure_results.csv")
    res_df.to_csv(csv_out, index=False)

    summary_dict = {
        "Paper": "05_Hey_NoRBERT_2020_IEEERE",
        "Dataset": "PURE (6086 requirements: FR vs NFR)",
        "Overall_Accuracy": overall_acc,
        "Macro_Precision": p_macro,
        "Macro_Recall": r_macro,
        "Macro_F1": f1_macro,
        "Weighted_F1": f1_wt,
        "FR_F1": f1_cls[0],
        "NFR_F1": f1_cls[1],
        "Folds": fold_metrics
    }
    json_out = os.path.join(out_dir, "norbert_pure_results.json")
    with open(json_out, "w") as f:
        json.dump(summary_dict, f, indent=4)

    print("\n" + "=" * 80)
    print("PAPER 5 (NoRBERT ON PURE) COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  - {csv_out}\n  - {json_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
