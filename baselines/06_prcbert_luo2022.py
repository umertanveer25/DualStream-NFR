"""
=============================================================================
BASELINE 6: PRCBERT — Prompt Learning for Requirement Classification
=============================================================================
Paper   : Luo, X.; Xue, Y.; Xing, Z.; Sun, J.
Title   : PRCBERT: Prompt Learning for Requirement Classification Using
          BERT-based Pretrained Language Models
Venue   : IEEE/ACM 37th Int. Conf. on Automated Software Engineering (ASE 2022)
          Rochester, MI, USA, 10-14 Oct. 2022, pp. 1-13
URL     : https://doi.org/10.1145/3551349.3556898
-----------------------------------------------------------------------------
Exact methodology (Section 3 of Luo et al. 2022):
  - Model: bert-base-uncased with Masked Language Model (MLM) head
  - Prompt template (from Luo et al. Table 1):
      "[CLS] {requirement_text}. This is a [MASK] requirement. [SEP]"
  - The [MASK] token's logit scores are mapped to class labels via
    a verbalizer: { "performance": PE, "security": SE,
                    "usability": US, "availability": A }
  - Training: The model learns to predict the correct class word at [MASK]
    (this is "prompt-based fine-tuning", not standard sequence classification)
  - Optimizer: AdamW, lr=2e-5, weight_decay=0.01
  - Scheduler: Linear warmup 10% + linear decay
  - Batch size: 16, Epochs: 10 (as in Luo et al. Section 4.1)
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
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}

# Verbalizer: maps each class to its natural language label word (Luo et al. Table 1)
VERBALIZER = {
    'PE': 'performance',
    'SE': 'security',
    'US': 'usability',
    'A':  'availability'
}

PROMPT_TEMPLATE = ". This is a [MASK] requirement."
MAX_LEN    = 128
BATCH_SIZE = 16
EPOCHS     = 10
LR         = 2e-5


class PRCBERTDataset(Dataset):
    """
    Builds prompt-augmented inputs:
      text + ". This is a [MASK] requirement."
    Stores the token-level label for the [MASK] position.
    """
    def __init__(self, texts, labels, tokenizer):
        self.tokenizer = tokenizer
        self.prompts   = [t.strip() + PROMPT_TEMPLATE for t in texts]
        self.labels    = labels
        # Pre-tokenize
        self.encodings = tokenizer(self.prompts, padding="max_length",
                                   truncation=True, max_length=MAX_LEN,
                                   return_tensors="pt")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        item   = {k: v[i] for k, v in self.encodings.items()}
        # Find [MASK] position
        input_ids = item["input_ids"]
        mask_pos  = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
        mask_pos  = mask_pos[0].item() if len(mask_pos) > 0 else 0
        return item, self.labels[i], mask_pos


def run():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(base, "data")

    train_df = pd.read_csv(os.path.join(data, "train_p1_p10.csv"))
    test_df  = pd.read_csv(os.path.join(data, "test_p11_p15.csv"))

    X_train  = train_df["RequirementTextClean"].fillna("").tolist()
    y_train  = train_df["Class"].map(LABEL_TO_IDX).values
    X_test   = test_df["RequirementTextClean"].fillna("").tolist()
    y_test   = test_df["Class"].map(LABEL_TO_IDX).values

    print("=" * 70)
    print("BASELINE 6 | Luo et al. (ASE 2022) — PRCBERT")
    print("MODEL      : bert-base-uncased + Prompt Template + Verbalizer")
    print(f"Train      : {len(X_train)}  |  Test: {len(X_test)}")
    print(f"Template   : '{{req}}{PROMPT_TEMPLATE}'")
    print(f"Verbalizer : {VERBALIZER}")
    print("=" * 70)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model     = BertForMaskedLM.from_pretrained("bert-base-uncased")

    # Get token IDs of verbalizer words
    verb_token_ids = {
        cls_name: tokenizer.convert_tokens_to_ids(word)
        for cls_name, word in VERBALIZER.items()
    }
    # Ordered by LABEL_TO_IDX: PE=0, SE=1, US=2, A=3
    verb_ids = [verb_token_ids[c] for c in TARGET_CLASSES]
    print(f"Verbalizer token IDs: {dict(zip(TARGET_CLASSES, verb_ids))}")

    train_ds = PRCBERTDataset(X_train, y_train, tokenizer)
    test_ds  = PRCBERTDataset(X_test,  y_test,  tokenizer)

    def collate(batch):
        items, labels, mask_positions = zip(*batch)
        input_ids      = torch.stack([x["input_ids"]      for x in items])
        attention_mask = torch.stack([x["attention_mask"] for x in items])
        token_type_ids = torch.stack([x["token_type_ids"] for x in items])
        return ({"input_ids": input_ids,
                 "attention_mask": attention_mask,
                 "token_type_ids": token_type_ids},
                torch.tensor(labels, dtype=torch.long),
                list(mask_positions))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(optimizer, int(0.10 * total_steps), total_steps)
    ce_loss     = nn.CrossEntropyLoss()
    verb_ids_t  = torch.tensor(verb_ids, dtype=torch.long)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch, labels, mask_positions in train_loader:
            optimizer.zero_grad()
            outputs = model(**batch)
            # Extract logits at [MASK] position for each sample in batch
            batch_logits = []
            for b_idx, mp in enumerate(mask_positions):
                logits_at_mask = outputs.logits[b_idx, mp, :]   # (vocab_size,)
                # Project to verbalizer: pick logits only for the 4 label words
                verb_logits = logits_at_mask[verb_ids_t]        # (4,)
                batch_logits.append(verb_logits)
            batch_logits = torch.stack(batch_logits, dim=0)     # (B, 4)
            loss = ce_loss(batch_logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch, labels, mask_positions in test_loader:
            outputs = model(**batch)
            for b_idx, mp in enumerate(mask_positions):
                logits_at_mask = outputs.logits[b_idx, mp, :]
                verb_logits    = logits_at_mask[verb_ids_t]
                pred           = torch.argmax(verb_logits).item()
                all_preds.append(pred)
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
