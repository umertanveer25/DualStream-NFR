"""
=============================================================================
REPLICATION PACKAGE: PAPER 4 (Liu et al. 2019 — RoBERTa_base)
=============================================================================
Paper Title: RoBERTa: A Robustly Optimized BERT Pretraining Approach
Authors    : Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi,
             Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov
Venue      : arXiv:1907.11692, 2019

Methodology Replicated as Described in the Literature:
------------------------------------------------------
1. Dataset: Benchmark PROMISE Dataset (PROMISE_exp.csv: 308 requirements).
2. Architecture:
   - Base model: roberta-base (12 layers, 768-dim, 12 attention heads, 125M params).
   - Tokenizer: Byte-Pair Encoding (BPE) with 50K vocab.
   - Input format: <s> RequirementText </s> (Isolated single statement).
   - Classification head: Linear layer on top of final-layer <s> hidden state.
   - Dropout: 0.1.
3. Training & Optimization:
   - Optimizer: AdamW (weight_decay = 0.01, eps = 1e-8).
   - Peak learning rate: 2e-5.
   - Learning rate schedule: Linear warmup over first 6% steps + linear decay.
   - Batch size: 16.
   - Epochs per fold: 5.
   - Max sequence length: 128 tokens.
4. Validation Protocol:
   - Stratified 10-Fold Cross-Validation on the 4 Quality Categories (PE, SE, US, A).
5. Reproducibility:
   - Random seed = 42 fixed across all folds.
=============================================================================
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import re
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)

MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5

def clean_text(text):
    text = str(text)
    text = re.sub(r'^\d+,[\'\"\s]*', '', text).rstrip('\'\"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class NFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        return item, self.labels[idx]

def run_10fold_roberta():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "PROMISE_exp.csv")
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Loading authentic PROMISE_exp dataset for Paper 4 (RoBERTa) from:", data_path)
    df = pd.read_csv(data_path)
    df['CleanText'] = df['RequirementText'].apply(clean_text)
    
    target_4 = ['PE', 'SE', 'US', 'A']
    df_4 = df[df['Class'].isin(target_4)].copy().reset_index(drop=True)
    nfr_label_to_idx = {c: i for i, c in enumerate(target_4)}
    y_4 = [nfr_label_to_idx[c] for c in df_4['Class']]
    texts = df_4['CleanText'].tolist()
    
    print("Loading tokenizer (roberta-base)...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    print("\n" + "=" * 80)
    print("RUNNING 10-FOLD CV FOR: RoBERTa_base (Isolated NFR)")
    print(f"Total samples: {len(y_4)} | Classes: {target_4}")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_preds = [], []
    
    texts_arr = np.array(texts)
    labels_arr = np.array(y_4)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(texts_arr, labels_arr)):
        X_tr, y_tr = texts_arr[train_idx].tolist(), labels_arr[train_idx].tolist()
        X_te, y_te = texts_arr[test_idx].tolist(), labels_arr[test_idx].tolist()
        
        train_ds = NFRDataset(X_tr, y_tr, tokenizer)
        test_ds = NFRDataset(X_te, y_te, tokenizer)
        
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=4
        )
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * total_steps), total_steps)
        
        model.train()
        for epoch in range(EPOCHS):
            for batch, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=targets
                )
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for batch, _ in test_loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                preds = torch.argmax(outputs.logits, dim=1).numpy()
                fold_preds.extend(preds)
                
        y_trues.extend(y_te)
        y_preds.extend(fold_preds)
        
        fold_acc = accuracy_score(y_te, fold_preds)
        _, _, fold_f1, _ = precision_recall_fscore_support(y_te, fold_preds, average='macro', zero_division=0)
        print(f"  Fold {fold+1:2d}/10 | Accuracy: {fold_acc*100:5.2f}% | Macro-F1: {fold_f1*100:5.2f}%", flush=True)
        
    acc = accuracy_score(y_trues, y_preds)
    mp, mr, mf, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro', zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_trues, y_preds, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_trues, y_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
    
    result = {
        'Task': 'Task 2: 4 Quality Categories (PE, SE, US, A)',
        'Model': 'RoBERTa_base (Liu et al. 2019)',
        'Accuracy': round(acc * 100, 2),
        'Macro_Precision': round(mp * 100, 2),
        'Macro_Recall': round(mr * 100, 2),
        'Macro_F1': round(mf * 100, 2),
        'Weighted_F1': round(wf * 100, 2),
        'Class_F1s': {c: round(f * 100, 2) for c, f in zip(target_4, class_f1s)}
    }
    
    print("\n" + "-" * 70)
    print("OVERALL 10-FOLD RESULTS FOR: RoBERTa_base (Liu et al. 2019)")
    print(f"Accuracy    : {result['Accuracy']}%")
    print(f"Macro-F1    : {result['Macro_F1']}%")
    print(f"Weighted-F1 : {result['Weighted_F1']}%")
    print("Per-class F1 breakdown:")
    for c, f in result['Class_F1s'].items():
        print(f"  {c:4s}: {f}%")
    print("-" * 70)
    
    json_path = os.path.join(out_dir, "roberta2019_results.json")
    csv_path = os.path.join(out_dir, "roberta2019_results.csv")
    
    with open(json_path, "w") as f:
        json.dump([result], f, indent=4)
        
    summary_df = pd.DataFrame([{
        'Task': result['Task'],
        'Model': result['Model'],
        'Accuracy': result['Accuracy'],
        'Macro_F1': result['Macro_F1'],
        'Weighted_F1': result['Weighted_F1'],
        **{f"{k}_F1": v for k, v in result['Class_F1s'].items()}
    }])
    summary_df.to_csv(csv_path, index=False)
    
    print("\n" + "=" * 80)
    print("PAPER 4 (RoBERTa_base) REPLICATION COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  - {csv_path}\n  - {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    run_10fold_roberta()
