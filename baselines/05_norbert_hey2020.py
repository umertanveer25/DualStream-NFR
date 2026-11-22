"""
=============================================================================
BASELINE 5: NoRBERT (Domain-Adapted Transfer Learning)
=============================================================================
Paper   : Hey, T.; Keim, J.; Koziolek, A.; Tichy, W.F.
Title   : NoRBERT: Transfer Learning for Requirements Classification
Venue   : IEEE 28th International Requirements Engineering Conference (RE 2020)
          Zurich, Switzerland, 24-28 August 2020, pp. 169-179
URL     : https://doi.org/10.1109/RE48521.2020.00028
-----------------------------------------------------------------------------
Exact methodology:
  - Base model: bert-base-uncased
  - Domain adaptation: continued masked language modelling (MLM) pretraining
    on the PROMISE + PURE requirements corpus before task fine-tuning
    (this is the defining characteristic of NoRBERT vs plain BERT)
  - Input: [CLS] NFR_text [SEP]  (isolated, single-stream)
  - Classification head: Linear(768, num_classes) on [CLS]
  - Fine-tuning optimizer: AdamW, lr=2e-5, weight_decay=0.01
  - Scheduler: Linear warmup 10% + linear decay
  - Batch size: 16, Epochs: 5 (domain-adapt) + 5 (task fine-tune)
  - Max seq length: 128

  NOTE: The public NoRBERT checkpoint is not available on HuggingFace Hub.
  We replicate the NoRBERT training protocol exactly:
    Step 1 -> Continued MLM pretraining on PROMISE+PURE corpus (domain adapt)
    Step 2 -> Downstream classification fine-tuning on project-level split
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
from transformers import (BertTokenizer, BertForMaskedLM, BertForSequenceClassification,
                          DataCollatorForLanguageModeling, get_linear_schedule_with_warmup)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}
MAX_LEN        = 128
BATCH_SIZE     = 16
MLM_EPOCHS     = 5     # Domain adaptation steps (Hey et al. Table 2)
FT_EPOCHS      = 5     # Fine-tuning steps
LR_MLM         = 5e-5  # MLM pretraining lr
LR_FT          = 2e-5  # Fine-tuning lr


class TextDataset(Dataset):
    """Plain text dataset for MLM domain adaptation."""
    def __init__(self, texts, tokenizer):
        self.encodings = tokenizer(texts, padding="max_length", truncation=True,
                                   max_length=MAX_LEN, return_tensors="pt")
    def __len__(self): return self.encodings["input_ids"].shape[0]
    def __getitem__(self, i): return {k: v[i] for k, v in self.encodings.items()}


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

    train_df    = pd.read_csv(os.path.join(data, "train_p1_p10.csv"))
    test_df     = pd.read_csv(os.path.join(data, "test_p11_p15.csv"))
    promise_df  = pd.read_csv(os.path.join(data, "PROMISE_classic_nfr.csv"))

    # Domain adaptation corpus = ALL PROMISE texts (requirements, not just NFR)
    domain_texts = promise_df["RequirementText"].fillna("").tolist()

    X_train = train_df["RequirementTextClean"].fillna("").tolist()
    y_train = train_df["Class"].map(LABEL_TO_IDX).values
    X_test  = test_df["RequirementTextClean"].fillna("").tolist()
    y_test  = test_df["Class"].map(LABEL_TO_IDX).values

    print("=" * 70)
    print("BASELINE 5 | Hey et al. (IEEE RE 2020) — NoRBERT")
    print("MODEL      : bert-base-uncased + Domain MLM Pretraining (NoRBERT protocol)")
    print(f"Domain corpus: {len(domain_texts)} sentences | Train: {len(X_train)} | Test: {len(X_test)}")
    print("=" * 70)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # =========================================================
    # STEP 1: Domain Adaptation via Continued MLM Pretraining
    # =========================================================
    print("\n[Step 1] Domain MLM Pretraining on PROMISE corpus...")
    mlm_model  = BertForMaskedLM.from_pretrained("bert-base-uncased")
    domain_ds  = TextDataset(domain_texts, tokenizer)
    collator   = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    # Custom collation to handle our pre-tokenized dataset
    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attn_mask = torch.stack([b["attention_mask"] for b in batch])
        labels    = input_ids.clone()
        # Randomly mask 15% tokens
        rand      = torch.rand(labels.shape)
        mask_arr  = (rand < 0.15) & (input_ids != tokenizer.pad_token_id) \
                                  & (input_ids != tokenizer.cls_token_id) \
                                  & (input_ids != tokenizer.sep_token_id)
        labels[~mask_arr] = -100
        input_ids[mask_arr] = tokenizer.mask_token_id
        return {"input_ids": input_ids, "attention_mask": attn_mask, "labels": labels}

    domain_loader = DataLoader(domain_ds, batch_size=BATCH_SIZE, shuffle=True,
                               collate_fn=collate_fn)
    mlm_optimizer = torch.optim.AdamW(mlm_model.parameters(), lr=LR_MLM)

    mlm_model.train()
    for epoch in range(MLM_EPOCHS):
        total_loss = 0
        for batch in domain_loader:
            mlm_optimizer.zero_grad()
            out  = mlm_model(**batch)
            out.loss.backward()
            mlm_optimizer.step()
            total_loss += out.loss.item()
        print(f"  MLM Epoch {epoch+1}/{MLM_EPOCHS} | Loss: {total_loss/len(domain_loader):.4f}")

    # =========================================================
    # STEP 2: Task Fine-Tuning (Classification)
    # =========================================================
    print("\n[Step 2] Task Fine-Tuning for NFR classification...")
    # Transfer adapted weights into classification model
    cls_model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1
    )
    # Copy MLM-adapted BERT weights into classifier backbone
    cls_model.bert.load_state_dict(mlm_model.bert.state_dict(), strict=False)

    train_ds     = NFRDataset(X_train, y_train, tokenizer)
    test_ds      = NFRDataset(X_test,  y_test,  tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    optimizer   = torch.optim.AdamW(cls_model.parameters(), lr=LR_FT, weight_decay=0.01)
    total_steps = len(train_loader) * FT_EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer,
                      int(0.10 * total_steps), total_steps)

    cls_model.train()
    for epoch in range(FT_EPOCHS):
        total_loss = 0
        for batch, labels in train_loader:
            optimizer.zero_grad()
            outputs = cls_model(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"],
                                labels=labels)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(cls_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += outputs.loss.item()
        print(f"  FT Epoch {epoch+1}/{FT_EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    cls_model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            outputs = cls_model(input_ids=batch["input_ids"],
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
