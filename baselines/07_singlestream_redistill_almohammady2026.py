"""
=============================================================================
BASELINE 7: Single-Stream Re-Distill (The Most Important SOTA — Base Paper)
=============================================================================
Paper   : Almohammady, A.; Alnanih, R.; Alowidi, N.
Title   : Context-Aware Non-Functional Requirement Classification Using
          Functional Requirement Guidance and Domain-Specialized Representations
Journal : Electronics, MDPI, 2026, Vol. 15, Article 4299
URL     : https://doi.org/10.3390/electronics15214299
-----------------------------------------------------------------------------
Exact methodology (Sections 3.1–3.3 of Almohammady et al. 2026):
  - Base model: bert-base-uncased (Re-Distill Stage-3 ≈ BERT domain-adapted
    on RE corpus; we use bert-base-uncased as the backbone since the exact
    Re-Distill checkpoint is not publicly released)
  - Input format (EXACT single-stream concatenation from their paper):
      [CLS] FR_text [SEP] NFR_text [SEP]
  - Feature extraction: ONLY the final-layer [CLS] hidden state h^(L)_{[CLS]}
    (this is their key design choice — no multi-layer aggregation)
  - Classification head: Dropout(0.1) -> Linear(768, num_classes)
  - Optimizer: AdamW, lr=2e-5, weight_decay=0.01
  - Scheduler: Linear warmup 10% + cosine decay
  - Batch size: 16, Epochs: 10
  - Max sequence length: FR=128 tokens, NFR=64 tokens (concat max=192 -> truncated to 192)
  - Loss: Standard CrossEntropyLoss (no focal loss, no contrastive — that is OURS)
  - Split: Project-Level Group Split P1-P10 / P11-P15 (same as their paper)

  IMPORTANT: This baseline demonstrates the CONTEXT NOISE DILUTION problem:
  the dense FR token sequence overwhelms sparse quality-attribute tokens in
  the concatenated single-stream input under unconstrained self-attention.
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
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}
MAX_LEN_CONCAT = 192   # FR(128) + NFR(64) + special tokens
BATCH_SIZE     = 16
EPOCHS         = 10
LR             = 2e-5


class SingleStreamDataset(Dataset):
    """
    Implements the EXACT concatenation format from Almohammady et al. (2026):
      [CLS] FR_text [SEP] NFR_text [SEP]
    Both FR and NFR are encoded into a SINGLE input sequence.
    """
    def __init__(self, nfr_texts, fr_texts, labels, tokenizer):
        # Use tokenizer's built-in sentence-pair encoding
        # text_a = FR (functional context), text_b = NFR (target)
        self.encodings = tokenizer(
            fr_texts,                   # text_a = FR (goes first after [CLS])
            nfr_texts,                  # text_b = NFR (goes second after [SEP])
            padding="max_length",
            truncation="longest_first",
            max_length=MAX_LEN_CONCAT,
            return_tensors="pt"
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.encodings.items()}, self.labels[i]


class SingleStreamReDist(nn.Module):
    """
    Single-stream classifier using ONLY the final [CLS] token (Layer L).
    No multi-layer aggregation. This is Almohammady et al.'s exact design.
    """
    def __init__(self, bert_model, num_classes=4, dropout=0.1):
        super().__init__()
        self.bert    = bert_model
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False    # Only final layer — Almohammady et al.
        )
        # Use ONLY the final [CLS] token hidden state (position 0)
        cls_output = outputs.last_hidden_state[:, 0, :]   # h^(L)_{[CLS]}
        logits     = self.classifier(self.dropout(cls_output))
        return logits


def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(base, "data")

    train_df = pd.read_csv(os.path.join(data, "train_p1_p10.csv"))
    test_df  = pd.read_csv(os.path.join(data, "test_p11_p15.csv"))

    # NFR text (target) + FR context (from traced traceability links)
    X_nfr_train = train_df["RequirementTextClean"].fillna("").tolist()
    X_fr_train  = train_df["FR_Context"].fillna("").tolist()
    y_train     = train_df["Class"].map(LABEL_TO_IDX).values

    X_nfr_test  = test_df["RequirementTextClean"].fillna("").tolist()
    X_fr_test   = test_df["FR_Context"].fillna("").tolist()
    y_test      = test_df["Class"].map(LABEL_TO_IDX).values

    print("=" * 70)
    print("BASELINE 7 | Almohammady et al. (Electronics MDPI 2026) — Single-Stream Re-Distill")
    print("MODEL      : bert-base-uncased, [CLS] FR [SEP] NFR [SEP], Final [CLS] only")
    print(f"Train      : {len(X_nfr_train)}  |  Test: {len(X_nfr_test)}")
    print("NOTE       : This is the base SOTA that our model improves upon")
    print("=" * 70)

    tokenizer  = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    train_ds     = SingleStreamDataset(X_nfr_train, X_fr_train, y_train, tokenizer)
    test_ds      = SingleStreamDataset(X_nfr_test,  X_fr_test,  y_test,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model     = SingleStreamReDist(bert_model, num_classes=4, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.10 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss()   # Standard CE — NO focal loss (that's ours)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch, labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["token_type_ids"])
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            logits = model(batch["input_ids"], batch["attention_mask"],
                           batch["token_type_ids"])
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
    print("\n[KEY OBSERVATION] If Availability F1 is significantly lower than")
    print("other classes, this confirms the context noise dilution anomaly")
    print("reported in Almohammady et al. (2026).")

if __name__ == "__main__":
    run()
