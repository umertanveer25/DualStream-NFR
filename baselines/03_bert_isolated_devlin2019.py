"""
=============================================================================
BASELINE 3: BERT_base Fine-Tuned (Isolated NFR — No Context)
=============================================================================
Paper   : Devlin, J.; Chang, M.W.; Lee, K.; Toutanova, K.
Title   : BERT: Pre-training of Deep Bidirectional Transformers for
          Language Understanding
Venue   : NAACL-HLT 2019, pp. 4171–4186
Also:   : Ayed, N.E.B.; Alrammal, M.
Title   : Benchmarking BERT for Software Requirements Classification
          Across Diverse Datasets
Venue   : IEEE COMPSAC 2024, pp. 1120–1129
-----------------------------------------------------------------------------
Exact methodology:
  - Model: bert-base-uncased (12 layers, 768-dim, 12 heads, 110M params)
  - Input: [CLS] NFR_text [SEP]  (NO functional context — isolated)
  - Classification head: Linear(768, num_classes) on [CLS] final hidden state
  - Optimizer: AdamW, lr=2e-5, weight_decay=0.01
  - Scheduler: Linear warmup (10% steps) + linear decay
  - Batch size: 16
  - Epochs: 5 (as in Ayed & Alrammal 2024, Section 3.2)
  - Max sequence length: 128 tokens
  - Dropout: 0.1
=============================================================================
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}
MAX_LEN        = 128
BATCH_SIZE     = 16
EPOCHS         = 5
LR             = 2e-5

class NFRDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=MAX_LEN, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]

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
    print("BASELINE 3 | Devlin et al. (NAACL-HLT 2019) + Ayed & Alrammal (COMPSAC 2024)")
    print("MODEL      : bert-base-uncased, Isolated NFR (no FR context)")
    print(f"Train      : {len(X_train)}  |  Test: {len(X_test)}")
    print("=" * 70)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertForSequenceClassification.from_pretrained(
                    "bert-base-uncased",
                    num_labels=4,
                    hidden_dropout_prob=0.1,
                    attention_probs_dropout_prob=0.1
                )

    train_ds     = NFRDataset(X_train, y_train, tokenizer)
    test_ds      = NFRDataset(X_test,  y_test,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"],
                            labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            token_type_ids=batch["token_type_ids"])
            preds = torch.argmax(outputs.logits, dim=1)
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
