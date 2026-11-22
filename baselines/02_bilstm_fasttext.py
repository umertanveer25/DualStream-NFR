"""
Baseline 2: Bidirectional LSTM (BiLSTM) with Word Representations
Reference: Kaur & Kaur (Computers & Electrical Engineering 2023), Baskoro et al. (ICoSNIKOM 2021)
Evaluates on: Project-Level Split (Train: P1-P10, Test: P11-P15)
"""
import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)

def tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

class Vocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        
    def build_vocab(self, texts, max_size=3000):
        freq = {}
        for t in texts:
            for w in tokenize(t):
                freq[w] = freq.get(w, 0) + 1
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:max_size]
        for w, _ in sorted_words:
            if w not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[w] = idx
                self.idx2word[idx] = w
                
    def encode(self, text, max_len=64):
        tokens = tokenize(text)
        ids = [self.word2idx.get(w, 1) for w in tokens[:max_len]]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=4, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (hn, _) = self.lstm(emb)
        # hn shape: (2, batch, hidden_dim)
        feat = torch.cat([hn[0], hn[1]], dim=1)
        logits = self.fc(self.dropout(feat))
        return logits

def run_bilstm_baseline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    train_df = pd.read_csv(os.path.join(data_dir, "train_p1_p10.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_p11_p15.csv"))
    
    target_classes = ['PE', 'SE', 'US', 'A']
    label_to_idx = {c: i for i, c in enumerate(target_classes)}
    
    vocab = Vocabulary()
    vocab.build_vocab(train_df['RequirementTextClean'].tolist())
    
    X_tr = torch.tensor([vocab.encode(t) for t in train_df['RequirementTextClean']], dtype=torch.long)
    y_tr = torch.tensor(train_df['Class'].map(label_to_idx).values, dtype=torch.long)
    
    X_te = torch.tensor([vocab.encode(t) for t in test_df['RequirementTextClean']], dtype=torch.long)
    y_te = np.array(test_df['Class'].map(label_to_idx).values)
    
    print("=" * 70)
    print("BASELINE 2: BiLSTM Neural Network")
    print(f"Vocab size: {len(vocab.word2idx)} | Train: {len(X_tr)} | Test: {len(X_te)}")
    print("=" * 70)
    
    model = BiLSTMClassifier(len(vocab.word2idx), embed_dim=128, hidden_dim=128, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(35):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), 16):
            idxs = perm[i:i+16]
            bx, by = X_tr[idxs], y_tr[idxs]
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_logits = model(X_te)
        y_pred = torch.argmax(test_logits, dim=1).numpy()
        
    acc = accuracy_score(y_te, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_te, y_pred, average='macro', zero_division=0)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(y_te, y_pred, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_te, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    
    print(f"\n[RESULTS] Accuracy:    {acc * 100:.2f}%")
    print(f"[RESULTS] Macro-F1:    {macro_f1 * 100:.2f}%")
    print(f"[RESULTS] Weighted-F1: {weighted_f1 * 100:.2f}%")
    print("\nPer-Category F1 Scores:")
    for c, f1 in zip(target_classes, class_f1s):
        print(f"  - {c} F1: {f1 * 100:.2f}%")
        
    return {
        'Model': 'BiLSTM + FastText',
        'Accuracy': round(acc * 100, 2),
        'Macro_F1': round(macro_f1 * 100, 2),
        'Weighted_F1': round(weighted_f1 * 100, 2),
        'PE_F1': round(class_f1s[0] * 100, 2),
        'SE_F1': round(class_f1s[1] * 100, 2),
        'US_F1': round(class_f1s[2] * 100, 2),
        'A_F1': round(class_f1s[3] * 100, 2)
    }

if __name__ == "__main__":
    run_bilstm_baseline()
