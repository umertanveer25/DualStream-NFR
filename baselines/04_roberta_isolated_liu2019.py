"""
=============================================================================
BASELINE 4: RoBERTa_base Fine-Tuned (Isolated NFR — No Context)
=============================================================================
Paper   : Liu, Y.; Ott, M.; Goyal, N.; Du, J.; Joshi, M.; Chen, D.;
          Levy, O.; Lewis, M.; Zettlemoyer, L.; Stoyanov, V.
Title   : RoBERTa: A Robustly Optimized BERT Pretraining Approach
Venue   : arXiv:1907.11692, 2019
-----------------------------------------------------------------------------
Exact methodology (as applied in requirements engineering literature):
  - Model: roberta-base (12 layers, 768-dim, 12 heads, 125M params)
  - Input: <s> NFR_text </s>  (NO functional context — isolated, same as BERT)
  - Classification head: Linear(768, num_classes) on [CLS] final hidden state
  - Optimizer: AdamW, lr=2e-5, eps=1e-8, weight_decay=0.01
  - Scheduler: Linear warmup (6% steps) + linear decay (as in RoBERTa paper)
  - Batch size: 16
  - Epochs: 5
  - Max sequence length: 128 tokens
  - Dropout: 0.1 (RoBERTa default)
  NOTE: RoBERTa does NOT use token_type_ids
=============================================================================
"""
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TORCH"] = "1"
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup
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
    print("BASELINE 4 | Liu et al. (arXiv 2019) — RoBERTa")
    print("MODEL      : roberta-base, Isolated NFR (no FR context)")
    print(f"Train      : {len(X_train)}  |  Test: {len(X_test)}")
    print("=" * 70)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model     = RobertaForSequenceClassification.from_pretrained(
                    "roberta-base", num_labels=4)

    train_ds     = NFRDataset(X_train, y_train, tokenizer)
    test_ds      = NFRDataset(X_test,  y_test,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.06 * total_steps)   # 6% warmup as in RoBERTa paper
    scheduler   = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
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
                            attention_mask=batch["attention_mask"])
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
