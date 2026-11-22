"""
=============================================================================
BASELINE 2: BiLSTM + CNN (BERT Embeddings)
=============================================================================
Paper   : Kaur, K.; Kaur, P.
Title   : Improving BERT Model for Requirements Classification by
          Bidirectional LSTM-CNN Deep Model
Journal : Computers & Electrical Engineering, Elsevier, 2023, Vol. 108, 108699
URL     : https://doi.org/10.1016/j.compeleceng.2023.108699
-----------------------------------------------------------------------------
Exact methodology:
  - BERT (bert-base-uncased) tokenizer + embeddings as input (frozen)
  - BiLSTM hidden_size=256 (bidirectional -> 512 total), num_layers=1
  - CNN parallel branch: kernel sizes [2,3,4], 128 filters each
  - Dropout = 0.3
  - Optimizer: Adam, lr=2e-5
  - Loss: CrossEntropyLoss
  - Epochs: 10 (as described in paper Section 4.2)
  - Max sequence length: 128 tokens
=============================================================================
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}
MAX_LEN        = 128
BATCH_SIZE     = 16
EPOCHS         = 10
LR             = 2e-5

class NFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]


class BiLSTM_CNN_Classifier(nn.Module):
    """
    Exact architecture from Kaur & Kaur (2023):
      BERT embedding -> BiLSTM(256) in parallel with CNN(filters=128,kernels=[2,3,4])
      -> Concatenate -> Dropout(0.3) -> Linear(num_classes)
    """
    def __init__(self, bert_model, hidden=256, num_classes=4, dropout=0.3):
        super().__init__()
        self.bert    = bert_model
        emb_dim      = bert_model.config.hidden_size            # 768

        # BiLSTM branch
        self.bilstm  = nn.LSTM(emb_dim, hidden, batch_first=True,
                                bidirectional=True, num_layers=1)

        # CNN branch (3 parallel convolutions)
        self.convs   = nn.ModuleList([
            nn.Conv1d(emb_dim, 128, kernel_size=k) for k in [2, 3, 4]
        ])

        self.dropout = nn.Dropout(dropout)
        # BiLSTM gives 2*hidden=512, CNN gives 3*128=384
        self.fc      = nn.Linear(512 + 384, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask)
        seq_out = outputs.last_hidden_state                      # (B, L, 768)

        # --- BiLSTM branch ---
        lstm_out, (hn, _) = self.bilstm(seq_out)
        lstm_feat = torch.cat([hn[0], hn[1]], dim=1)             # (B, 512)

        # --- CNN branch ---
        x_t = seq_out.permute(0, 2, 1)                          # (B, 768, L)
        cnn_feats = []
        for conv in self.convs:
            c = F.relu(conv(x_t))                               # (B, 128, L-k+1)
            c = F.adaptive_max_pool1d(c, 1).squeeze(-1)         # (B, 128)
            cnn_feats.append(c)
        cnn_feat = torch.cat(cnn_feats, dim=1)                  # (B, 384)

        # --- Concatenate + Classify ---
        fused  = torch.cat([lstm_feat, cnn_feat], dim=1)        # (B, 896)
        logits = self.fc(self.dropout(fused))
        return logits


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
    print("BASELINE 2 | Kaur & Kaur (Comput. Electr. Eng. 2023)")
    print("MODEL      : BERT Embedding + BiLSTM(256) + CNN([2,3,4] x 128)")
    print(f"Train      : {len(X_train)}  |  Test: {len(X_test)}")
    print("=" * 70)

    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    train_ds = NFRDataset(X_train, y_train, tokenizer)
    test_ds  = NFRDataset(X_test,  y_test,  tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model     = BiLSTM_CNN_Classifier(bert_model, hidden=256, num_classes=4)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch, labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds  = torch.argmax(logits, dim=1)
            all_preds.extend(preds.numpy())
            all_true.extend(labels.numpy())

    _report(all_true, all_preds)


def _report(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro",    zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _,  _,  cf, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0,1,2,3], average=None, zero_division=0)
    print(f"\nAccuracy    : {acc*100:.2f}%")
    print(f"Macro-F1    : {mf*100:.2f}%")
    print(f"Weighted-F1 : {wf*100:.2f}%")
    print(f"Macro-Prec  : {mp*100:.2f}%  |  Macro-Rec : {mr*100:.2f}%")
    print("\nPer-class F1:")
    for c, f in zip(TARGET_CLASSES, cf):
        print(f"  {c:3s}: {f*100:.2f}%")

if __name__ == "__main__":
    run()
