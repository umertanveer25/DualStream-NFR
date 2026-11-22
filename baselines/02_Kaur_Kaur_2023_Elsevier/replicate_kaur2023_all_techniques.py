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
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.set_num_threads(16)
torch.set_num_interop_threads(4)
torch.manual_seed(42)
np.random.seed(42)

MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-3

def clean_text(text):
    text = str(text)
    text = re.sub(r'^\d+,[\'\"\s]*', '', text).rstrip('\'\"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# =============================================================================
# NEURAL ARCHITECTURES COMPARED IN KAUR & KAUR (2023)
# =============================================================================

# 1. Proposed Model: BERT + BiLSTM + CNN
class BERT_BiLSTM_CNN(nn.Module):
    def __init__(self, emb_dim=768, lstm_hidden=128, cnn_filters=64, num_classes=4, dropout=0.3):
        super(BERT_BiLSTM_CNN, self).__init__()
        self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        lstm_out_dim = lstm_hidden * 2
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=lstm_out_dim, out_channels=cnn_filters, kernel_size=k)
            for k in [2, 3, 4]
        ])
        self.dropout = nn.Dropout(dropout)
        total_cnn_dim = cnn_filters * len(self.convs)
        self.fc = nn.Linear(total_cnn_dim, num_classes)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        x_conv = lstm_out.permute(0, 2, 1)
        pooled = [F.adaptive_max_pool1d(F.relu(conv(x_conv)), 1).squeeze(-1) for conv in self.convs]
        fused = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(fused))

# 2. Ablation: BERT + BiLSTM
class BERT_BiLSTM(nn.Module):
    def __init__(self, emb_dim=768, lstm_hidden=128, num_classes=4, dropout=0.3):
        super(BERT_BiLSTM, self).__init__()
        self.bilstm = nn.LSTM(input_size=emb_dim, hidden_size=lstm_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        # Max-pool across time dimension
        pooled = F.adaptive_max_pool1d(lstm_out.permute(0, 2, 1), 1).squeeze(-1)
        return self.fc(self.dropout(pooled))

# 3. Ablation: BERT + CNN
class BERT_CNN(nn.Module):
    def __init__(self, emb_dim=768, cnn_filters=64, num_classes=4, dropout=0.3):
        super(BERT_CNN, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=emb_dim, out_channels=cnn_filters, kernel_size=k)
            for k in [2, 3, 4]
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_filters * len(self.convs), num_classes)

    def forward(self, x):
        x_conv = x.permute(0, 2, 1)
        pooled = [F.adaptive_max_pool1d(F.relu(conv(x_conv)), 1).squeeze(-1) for conv in self.convs]
        fused = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(fused))

# 4. Ablation: BERT Alone (Mean Pooling + Dense)
class BERT_Alone(nn.Module):
    def __init__(self, emb_dim=768, num_classes=4, dropout=0.3):
        super(BERT_Alone, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.fc(self.dropout(pooled))

def train_and_eval_neural(model_fn, X_embs, y_arr, num_classes):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_preds = [], []
    for train_idx, test_idx in skf.split(X_embs, y_arr):
        X_tr, y_tr = X_embs[train_idx], torch.tensor(y_arr[train_idx], dtype=torch.long)
        X_te, y_te = X_embs[test_idx], torch.tensor(y_arr[test_idx], dtype=torch.long)
        
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model = model_fn()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
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
        y_trues.extend(y_te.numpy())
        y_preds.extend(preds)
        
    acc = accuracy_score(y_trues, y_preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_trues, y_preds, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_trues, y_preds, labels=list(range(num_classes)), average=None, zero_division=0)
    return acc, p_m, r_m, f1_m, f1_w, class_f1s

def train_and_eval_ml(clf, texts, y_arr, num_classes):
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
    return acc, p_m, r_m, f1_m, f1_w, class_f1s

def main():
    data_path = r"B:\Dataset\nfr.csv"
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("PAPER 2 COMPLETE REPLICATION: ALL TECHNIQUES IN KAUR & KAUR (2023)")
    print(f"Dataset: {data_path} (PROMISE NFR repository)")
    print(f"Allocated CPU Threads: {torch.get_num_threads()} (all 16 cores active)")
    print("=" * 80)
    
    df = pd.read_csv(data_path, encoding='latin-1')
    class_col = 'class' if 'class' in df.columns else 'Class'
    df['CleanText'] = df['RequirementText'].apply(clean_text)
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    
    print("Extracting contextual representations using 16 threads...")
    all_texts = df['CleanText'].tolist()
    all_embs = []
    with torch.no_grad():
        for i in range(0, len(all_texts), 32):
            batch = all_texts[i:i+32]
            enc = tokenizer(batch, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt")
            out = bert_model(**enc)
            all_embs.append(out.last_hidden_state.cpu())
    all_embs = torch.cat(all_embs, dim=0)
    
    # Define Tasks
    # Task 1: Binary (FR vs NFR)
    df['BinaryLabel'] = df[class_col].apply(lambda x: 'FR' if x == 'F' else 'NFR')
    bin_map = {'FR': 0, 'NFR': 1}
    y_bin = np.array([bin_map[c] for c in df['BinaryLabel']])
    
    # Task 2: 4 Quality Categories (PE, SE, US, A)
    target_4 = ['PE', 'SE', 'US', 'A']
    mask_4 = df[class_col].isin(target_4)
    idx_4 = np.where(mask_4)[0]
    X_4_embs = all_embs[idx_4]
    texts_4 = [all_texts[i] for i in idx_4]
    map_4 = {c: i for i, c in enumerate(target_4)}
    y_4 = np.array([map_4[c] for c in df.loc[mask_4, class_col]])
    
    tasks = [
        ("Task 1: Binary Classification (FR vs NFR - 625 items)", all_embs, all_texts, y_bin, 2, ['FR', 'NFR']),
        ("Task 2: 4 Quality Categories (PE, SE, US, A - 208 items)", X_4_embs, texts_4, y_4, 4, target_4)
    ]
    
    all_technique_results = []
    
    for task_name, X_embs_task, texts_task, y_task, num_cls, cls_names in tasks:
        print("\n" + "=" * 80)
        print(f"BENCHMARKING ALL TECHNIQUES: {task_name}")
        print("=" * 80)
        
        # 1. Proposed: BERT + BiLSTM + CNN
        acc, pm, rm, f1m, f1w, cf1 = train_and_eval_neural(
            lambda: BERT_BiLSTM_CNN(num_classes=num_cls), X_embs_task, y_task, num_cls
        )
        print(f"  [1. BERT + BiLSTM + CNN (Proposed)] Acc: {acc*100:.2f}% | Macro-F1: {f1m*100:.2f}% | Weighted-F1: {f1w*100:.2f}%")
        all_technique_results.append({
            "Task": task_name, "Technique": "BERT + BiLSTM + CNN (Proposed)",
            "Accuracy": acc, "Macro_Precision": pm, "Macro_Recall": rm, "Macro_F1": f1m, "Weighted_F1": f1w,
            **{f"{cls_names[i]}_F1": cf1[i] for i in range(num_cls)}
        })
        
        # 2. Ablation: BERT + BiLSTM
        acc, pm, rm, f1m, f1w, cf1 = train_and_eval_neural(
            lambda: BERT_BiLSTM(num_classes=num_cls), X_embs_task, y_task, num_cls
        )
        print(f"  [2. BERT + BiLSTM (Ablation)     ] Acc: {acc*100:.2f}% | Macro-F1: {f1m*100:.2f}% | Weighted-F1: {f1w*100:.2f}%")
        all_technique_results.append({
            "Task": task_name, "Technique": "BERT + BiLSTM (Ablation)",
            "Accuracy": acc, "Macro_Precision": pm, "Macro_Recall": rm, "Macro_F1": f1m, "Weighted_F1": f1w,
            **{f"{cls_names[i]}_F1": cf1[i] for i in range(num_cls)}
        })
        
        # 3. Ablation: BERT + CNN
        acc, pm, rm, f1m, f1w, cf1 = train_and_eval_neural(
            lambda: BERT_CNN(num_classes=num_cls), X_embs_task, y_task, num_cls
        )
        print(f"  [3. BERT + CNN (Ablation)        ] Acc: {acc*100:.2f}% | Macro-F1: {f1m*100:.2f}% | Weighted-F1: {f1w*100:.2f}%")
        all_technique_results.append({
            "Task": task_name, "Technique": "BERT + CNN (Ablation)",
            "Accuracy": acc, "Macro_Precision": pm, "Macro_Recall": rm, "Macro_F1": f1m, "Weighted_F1": f1w,
            **{f"{cls_names[i]}_F1": cf1[i] for i in range(num_cls)}
        })
        
        # 4. Ablation: BERT Alone
        acc, pm, rm, f1m, f1w, cf1 = train_and_eval_neural(
            lambda: BERT_Alone(num_classes=num_cls), X_embs_task, y_task, num_cls
        )
        print(f"  [4. BERT Alone (Ablation)        ] Acc: {acc*100:.2f}% | Macro-F1: {f1m*100:.2f}% | Weighted-F1: {f1w*100:.2f}%")
        all_technique_results.append({
            "Task": task_name, "Technique": "BERT Alone (Ablation)",
            "Accuracy": acc, "Macro_Precision": pm, "Macro_Recall": rm, "Macro_F1": f1m, "Weighted_F1": f1w,
            **{f"{cls_names[i]}_F1": cf1[i] for i in range(num_cls)}
        })
        
        # 5. Baseline: Linear SVM (TF-IDF)
        acc, pm, rm, f1m, f1w, cf1 = train_and_eval_ml(
            LinearSVC(C=1.0, random_state=42), texts_task, y_task, num_cls
        )
        print(f"  [5. Linear SVM Baseline          ] Acc: {acc*100:.2f}% | Macro-F1: {f1m*100:.2f}% | Weighted-F1: {f1w*100:.2f}%")
        all_technique_results.append({
            "Task": task_name, "Technique": "Linear SVM (TF-IDF Baseline)",
            "Accuracy": acc, "Macro_Precision": pm, "Macro_Recall": rm, "Macro_F1": f1m, "Weighted_F1": f1w,
            **{f"{cls_names[i]}_F1": cf1[i] for i in range(num_cls)}
        })
        
        # 6. Baseline: Random Forest (TF-IDF)
        acc, pm, rm, f1m, f1w, cf1 = train_and_eval_ml(
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1), texts_task, y_task, num_cls
        )
        print(f"  [6. Random Forest Baseline       ] Acc: {acc*100:.2f}% | Macro-F1: {f1m*100:.2f}% | Weighted-F1: {f1w*100:.2f}%")
        all_technique_results.append({
            "Task": task_name, "Technique": "Random Forest Baseline",
            "Accuracy": acc, "Macro_Precision": pm, "Macro_Recall": rm, "Macro_F1": f1m, "Weighted_F1": f1w,
            **{f"{cls_names[i]}_F1": cf1[i] for i in range(num_cls)}
        })

    # Save to files
    csv_out = os.path.join(out_dir, "kaur2023_all_techniques_results.csv")
    json_out = os.path.join(out_dir, "kaur2023_all_techniques_results.json")
    
    df_out = pd.DataFrame(all_technique_results)
    df_out.to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump(all_technique_results, f, indent=4)
        
    print("\n" + "=" * 80)
    print("ALL TECHNIQUES REPLICATION FOR PAPER 2 COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  - {csv_out}\n  - {json_out}")
    print("=" * 80)

if __name__ == "__main__":
    main()
