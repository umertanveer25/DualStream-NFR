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

class BERT_BiLSTM_CNN(nn.Module):
    """
    Kaur & Kaur (2023) Architecture:
    Input (Sequence of BERT embeddings) -> BiLSTM -> Parallel Multi-Kernel CNN -> MaxPool -> Dense
    """
    def __init__(self, emb_dim=768, lstm_hidden=128, cnn_filters=64, num_classes=4, dropout=0.3):
        super(BERT_BiLSTM_CNN, self).__init__()
        
        self.bilstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True
        )
        
        lstm_out_dim = lstm_hidden * 2
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=lstm_out_dim, out_channels=cnn_filters, kernel_size=k)
            for k in [2, 3, 4]
        ])
        
        self.dropout = nn.Dropout(dropout)
        total_cnn_dim = cnn_filters * len(self.convs)
        self.fc = nn.Linear(total_cnn_dim, num_classes)

    def forward(self, x):
        # x: (Batch, Seq_Len, 768)
        lstm_out, _ = self.bilstm(x)  # (Batch, Seq_Len, 256)
        x_conv = lstm_out.permute(0, 2, 1)  # (Batch, 256, Seq_Len)
        
        pooled_outputs = []
        for conv in self.convs:
            c = F.relu(conv(x_conv))
            pooled = F.adaptive_max_pool1d(c, 1).squeeze(-1)
            pooled_outputs.append(pooled)
            
        fused = torch.cat(pooled_outputs, dim=1)
        logits = self.fc(self.dropout(fused))
        return logits

def precompute_bert_embeddings(texts, tokenizer, bert_model):
    print("Precomputing BERT contextual sequence representations for all texts (using 16 threads)...")
    bert_model.eval()
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), 32):
            batch_texts = texts[i:i+32]
            enc = tokenizer(batch_texts, padding="max_length", truncation=True,
                            max_length=MAX_LEN, return_tensors="pt")
            out = bert_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            seq_reps = out.last_hidden_state.cpu()
            all_embeddings.append(seq_reps)
            
    return torch.cat(all_embeddings, dim=0)

def evaluate_10fold(X_embeddings, y_indices, num_classes, task_name, class_names):
    print("\n" + "=" * 80)
    print(f"RUNNING 10-FOLD CV FOR: {task_name}")
    print(f"Total samples: {len(y_indices)} | Classes: {class_names}")
    print("=" * 80)
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_trues, y_preds = [], []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_embeddings, y_indices)):
        X_tr, y_tr = X_embeddings[train_idx], torch.tensor(y_indices[train_idx], dtype=torch.long)
        X_te, y_te = X_embeddings[test_idx], torch.tensor(y_indices[test_idx], dtype=torch.long)
        
        train_ds = TensorDataset(X_tr, y_tr)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        model = BERT_BiLSTM_CNN(
            emb_dim=768,
            lstm_hidden=128,
            cnn_filters=64,
            num_classes=num_classes,
            dropout=0.3
        )
        
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
            test_logits = model(X_te)
            preds = torch.argmax(test_logits, dim=1).numpy()
            
        fold_acc = accuracy_score(y_te.numpy(), preds)
        _, _, fold_f1, _ = precision_recall_fscore_support(y_te.numpy(), preds, average="macro", zero_division=0)
        print(f"  Fold {fold+1:2d}/10 | Accuracy: {fold_acc*100:.2f}% | Macro-F1: {fold_f1*100:.2f}%", flush=True)
        
        y_trues.extend(y_te.numpy())
        y_preds.extend(preds)
        
    acc = accuracy_score(y_trues, y_preds)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro', zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_trues, y_preds, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_trues, y_preds, labels=list(range(num_classes)), average=None, zero_division=0)
    
    result = {
        'Task': task_name,
        'Model': 'BERT + BiLSTM + CNN (Kaur & Kaur 2023)',
        'Accuracy': float(acc),
        'Macro_Precision': float(p_m),
        'Macro_Recall': float(r_m),
        'Macro_F1': float(f1_m),
        'Weighted_F1': float(f1_w),
        'Class_F1s': {class_names[i]: float(class_f1s[i]) for i in range(num_classes)}
    }
    
    print("\n" + "-" * 70)
    print(f"SUMMARY FOR: {task_name}")
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print(f"Macro-F1        : {f1_m*100:.2f}%")
    print(f"Weighted-F1     : {f1_w*100:.2f}%")
    print("Class F1 Breakdown:")
    for cname, cf1 in result['Class_F1s'].items():
        print(f"  {cname:<12}: {cf1*100:.2f}%")
    print("-" * 70)
    
    return result

def main():
    data_path = r"B:\Dataset\nfr.csv"
    out_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("=" * 80)
    print("PAPER 2 REPLICATION: Kaur & Kaur (Computers & Electrical Engineering 2023)")
    print("Model: BERT + BiLSTM + CNN Hybrid Deep Learning")
    print(f"Loading exact dataset from: {data_path}")
    print(f"Allocated CPU Threads: {torch.get_num_threads()} (all 16 cores active)")
    print("=" * 80)
    
    df = pd.read_csv(data_path, encoding='latin-1')
    class_col = 'class' if 'class' in df.columns else 'Class'
    df['CleanText'] = df['RequirementText'].apply(clean_text)
    
    print(f"Total rows in nfr.csv: {len(df)}")
    print(f"Class distribution:\n{df[class_col].value_counts().to_string()}")
    
    print("\nInitializing BERT Tokenizer and Model (bert-base-uncased)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    
    all_results = []
    
    # Precompute BERT embeddings once for the entire dataset
    all_texts = df['CleanText'].tolist()
    all_embs = precompute_bert_embeddings(all_texts, tokenizer, bert_model)
    
    # -------------------------------------------------------------
    # Task 1: Binary Classification (FR vs NFR) - 625 items
    # -------------------------------------------------------------
    df['BinaryLabel'] = df[class_col].apply(lambda x: 'FR' if x == 'F' else 'NFR')
    binary_classes = ['FR', 'NFR']
    bin_label_to_idx = {c: i for i, c in enumerate(binary_classes)}
    y_binary = np.array([bin_label_to_idx[c] for c in df['BinaryLabel']])
    
    res_binary = evaluate_10fold(
        all_embs, y_binary, num_classes=2,
        task_name="Task 1: Binary Classification (FR vs NFR on 625 requirements)",
        class_names=binary_classes
    )
    all_results.append(res_binary)
    
    # -------------------------------------------------------------
    # Task 2: 4 Core Architectural Categories (PE, SE, US, A) - 208 items
    # -------------------------------------------------------------
    target_4 = ['PE', 'SE', 'US', 'A']
    mask_4 = df[class_col].isin(target_4)
    df_4 = df[mask_4].copy().reset_index(drop=True)
    idx_4 = np.where(mask_4)[0]
    X_4_embs = all_embs[idx_4]
    
    nfr_label_to_idx = {c: i for i, c in enumerate(target_4)}
    y_4 = np.array([nfr_label_to_idx[c] for c in df_4[class_col]])
    
    res_4 = evaluate_10fold(
        X_4_embs, y_4, num_classes=4,
        task_name="Task 2: 4 Quality Categories (PE, SE, US, A on 208 requirements)",
        class_names=target_4
    )
    all_results.append(res_4)
    
    # -------------------------------------------------------------
    # Task 3: Top 5 Most Frequent NFR Classes (US, SE, O, PE, LF) - 287 items
    # -------------------------------------------------------------
    target_top5 = ['US', 'SE', 'O', 'PE', 'LF']
    mask_top5 = df[class_col].isin(target_top5)
    df_top5 = df[mask_top5].copy().reset_index(drop=True)
    idx_top5 = np.where(mask_top5)[0]
    X_top5_embs = all_embs[idx_top5]
    
    top5_label_to_idx = {c: i for i, c in enumerate(target_top5)}
    y_top5 = np.array([top5_label_to_idx[c] for c in df_top5[class_col]])
    
    res_top5 = evaluate_10fold(
        X_top5_embs, y_top5, num_classes=5,
        task_name="Task 3: Top 5 NFR Classes (US, SE, O, PE, LF on 287 requirements)",
        class_names=target_top5
    )
    all_results.append(res_top5)
    
    # Save results to JSON and CSV
    json_path = os.path.join(out_dir, "kaur2023_results.json")
    csv_path = os.path.join(out_dir, "kaur2023_results.csv")
    
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
    print("PAPER 2 (Kaur & Kaur 2023) REPLICATION COMPLETED SUCCESSFULLY!")
    print(f"Results saved to:\n  - {csv_path}\n  - {json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
