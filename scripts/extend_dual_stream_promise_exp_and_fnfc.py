import sys
sys.modules['tensorflow'] = None
sys.modules['keras'] = None

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
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.set_num_threads(16)
torch.set_num_interop_threads(4)
torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_LABEL   = {i: c for i, c in enumerate(TARGET_CLASSES)}

MAX_LEN_CONCAT = 192
MAX_LEN_SINGLE = 128
BATCH_SIZE     = 16
EPOCHS         = 35
LR             = 1e-3

def clean_text(text):
    text = str(text)
    text = re.sub(r'^\d+,[\'\"\s]*', '', text).rstrip('\'\"')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==============================================================================
# LOSS FUNCTIONS & ARCHITECTURES
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()

class SingleStreamLinearHead(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.dropout(x))

class SingleStreamMLPHead(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.mlp(x)

class ProposedDualStream(nn.Module):
    def __init__(self, hidden_dim=768, num_classes=4, dropout=0.20):
        super().__init__()
        self.nfr_attn = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fr_attn  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        self.temp = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, nfr_layers, fr_layers):
        b, l, d = nfr_layers.shape
        w_nfr = F.softmax(self.nfr_attn(nfr_layers.view(-1, d)).view(b, l, 1), dim=1)
        h_nfr = (nfr_layers * w_nfr).sum(dim=1)

        w_fr = F.softmax(self.fr_attn(fr_layers.view(-1, d)).view(b, l, 1), dim=1)
        h_fr = (fr_layers * w_fr).sum(dim=1)

        alpha = self.context_gate(torch.cat([h_nfr, h_fr], dim=1))
        h_gated_fr = alpha * h_fr
        fused = torch.cat([h_nfr, h_gated_fr], dim=1)

        out1 = self.gelu(self.bn1(self.fc1(fused)))
        out2 = self.gelu(self.bn2(self.fc2(self.dropout(out1))))
        logits = self.classifier(out2) / torch.clamp(self.temp, min=0.01, max=1.0)
        return logits

# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================
def extract_representations(nfr_texts, fr_texts, tokenizer, bert_model):
    bert_model.eval()

    # Single-Stream: [CLS] FR [SEP] NFR [SEP]
    enc_ss = tokenizer(fr_texts, nfr_texts, padding="max_length", truncation=True,
                       max_length=MAX_LEN_CONCAT, return_tensors="pt")
    ds_ss = TensorDataset(enc_ss["input_ids"], enc_ss["attention_mask"], enc_ss["token_type_ids"])
    loader_ss = DataLoader(ds_ss, batch_size=32, shuffle=False)
    ss_cls = []
    with torch.no_grad():
        for b_ids, b_mask, b_type in loader_ss:
            out = bert_model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type)
            ss_cls.append(out.last_hidden_state[:, 0, :].cpu())
    ss_cls_t = torch.cat(ss_cls, dim=0)

    # Dual-Stream: NFR top4 layers & FR top4 layers
    def get_top4(texts):
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN_SINGLE, return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        top4_list = []
        with torch.no_grad():
            for b_ids, b_mask, b_type in loader:
                out = bert_model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type, output_hidden_states=True)
                top4 = torch.stack([out.hidden_states[k][:, 0, :] for k in range(-4, 0)], dim=1).cpu()
                top4_list.append(top4)
        return torch.cat(top4_list, dim=0)

    nfr_top4 = get_top4(nfr_texts)
    fr_top4  = get_top4(fr_texts)

    return ss_cls_t, nfr_top4, fr_top4

def evaluate_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    mp, mr, mf, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    _, _, cf, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {
        "Accuracy": float(acc),
        "Macro_Precision": float(mp),
        "Macro_Recall": float(mr),
        "Macro_F1": float(mf),
        "Weighted_F1": float(wf),
        "PE_F1": float(cf[0]),
        "SE_F1": float(cf[1]),
        "US_F1": float(cf[2]),
        "A_F1": float(cf[3])
    }

def train_and_eval_fold(model, tr_x, tr_y, val_x, is_dual=False, use_focal=False, class_weights=None):
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    criterion = FocalLoss(gamma=2.0, alpha=class_weights) if use_focal else nn.CrossEntropyLoss(weight=class_weights)

    if is_dual:
        ds = TensorDataset(tr_x[0], tr_x[1], tr_y)
    else:
        ds = TensorDataset(tr_x, tr_y)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for batch in loader:
            optimizer.zero_grad()
            if is_dual:
                logits = model(batch[0], batch[1])
                loss = criterion(logits, batch[2])
            else:
                logits = model(batch[0])
                loss = criterion(logits, batch[1])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        if is_dual:
            logits = model(val_x[0], val_x[1])
        else:
            logits = model(val_x)
        preds = torch.argmax(logits, dim=1).numpy()
    return preds

def run_evaluation_on_dataset(dataset_name, nfr_texts, fr_texts, labels, tokenizer, bert_model):
    print("\n" + "=" * 80)
    print(f"BENCHMARKING: {dataset_name} ({len(nfr_texts)} Quality Requirements, 4 Classes)")
    print("=" * 80)

    y_tensor = torch.tensor(labels, dtype=torch.long)
    counts = np.bincount(labels, minlength=4)
    weights = len(labels) / (4.0 * counts)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    print(f"Precomputing representations on 16 cores for {dataset_name}...", flush=True)
    ss_cls, nfr_top4, fr_top4 = extract_representations(nfr_texts, fr_texts, tokenizer, bert_model)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    preds_ss_lin = np.zeros(len(labels), dtype=int)
    preds_ss_mlp = np.zeros(len(labels), dtype=int)
    preds_dual   = np.zeros(len(labels), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(ss_cls, labels)):
        y_tr = y_tensor[tr_idx]

        # 1. Single-Stream Linear (Paper 7 Base Architecture)
        m1 = SingleStreamLinearHead()
        preds_ss_lin[val_idx] = train_and_eval_fold(
            m1, ss_cls[tr_idx], y_tr, ss_cls[val_idx], is_dual=False, use_focal=False, class_weights=class_weights_t
        )

        # 2. Single-Stream MLP
        m2 = SingleStreamMLPHead()
        preds_ss_mlp[val_idx] = train_and_eval_fold(
            m2, ss_cls[tr_idx], y_tr, ss_cls[val_idx], is_dual=False, use_focal=False, class_weights=class_weights_t
        )

        # 3. Proposed Dual-Stream
        m3 = ProposedDualStream()
        preds_dual[val_idx] = train_and_eval_fold(
            m3, (nfr_top4[tr_idx], fr_top4[tr_idx]), y_tr,
            (nfr_top4[val_idx], fr_top4[val_idx]), is_dual=True, use_focal=True, class_weights=class_weights_t
        )

    res = []
    for model_name, p in [
        ("Single-Stream BERT (Linear Head - Base SOTA)", preds_ss_lin),
        ("Single-Stream BERT (MLP Head)", preds_ss_mlp),
        ("Proposed Dual-Stream (Gated Attention + Focal)", preds_dual)
    ]:
        m = evaluate_metrics(labels, p)
        m["Dataset"] = dataset_name
        m["Model"] = model_name
        m["Samples"] = len(labels)
        res.append(m)

    return res

def main():
    print("=" * 80)
    print("CROSS-CORPUS EXPANSION: EXTENDING DUAL-STREAM TO PROMISE-EXP AND FNFC")
    print("Backbone: bert-base-uncased | Evaluation: 10-Fold Stratified Cross-Validation")
    print("Allocated CPU Threads: 16 (all 16 cores active)")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir  = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    all_results = []

    # ==========================================================================
    # CORPUS 1: PROMISE Extended (B:\Dataset\Promise_Exp.csv)
    # ==========================================================================
    df_pexp = pd.read_csv('B:/Dataset/Promise_Exp.csv', encoding='latin-1')
    fr_pool_pexp = df_pexp[df_pexp['Type'] == 'F']['Requirement'].dropna().tolist()
    df_pexp_nfr = df_pexp[df_pexp['Type'].isin(TARGET_CLASSES)].copy().reset_index(drop=True)

    nfr_pexp = [clean_text(t) for t in df_pexp_nfr['Requirement']]
    fr_pexp  = [clean_text(fr_pool_pexp[i % len(fr_pool_pexp)]) for i in range(len(df_pexp_nfr))]
    y_pexp   = df_pexp_nfr['Type'].map(LABEL_TO_IDX).values

    print(f"Loaded PROMISE Extended: {len(df_pexp_nfr)} Quality Reqs across {df_pexp_nfr['Type'].value_counts().to_dict()}")
    res_pexp = run_evaluation_on_dataset("PROMISE Extended (Promise_Exp.csv)", nfr_pexp, fr_pexp, y_pexp, tokenizer, bert_model)
    all_results.extend(res_pexp)

    # ==========================================================================
    # CORPUS 2: Large-Scale FNFC (B:\Dataset\FNFC.csv)
    # ==========================================================================
    df_fnfc = pd.read_csv('B:/Dataset/FNFC.csv', encoding='latin-1')
    fr_pool_fnfc = df_fnfc[df_fnfc['class'] == 'F']['text'].dropna().tolist()
    df_fnfc_nfr = df_fnfc[df_fnfc['class'].isin(TARGET_CLASSES)].copy().reset_index(drop=True)

    nfr_fnfc = [clean_text(t) for t in df_fnfc_nfr['text']]
    fr_fnfc  = [clean_text(fr_pool_fnfc[i % len(fr_pool_fnfc)]) for i in range(len(df_fnfc_nfr))]
    y_fnfc   = df_fnfc_nfr['class'].map(LABEL_TO_IDX).values

    print(f"Loaded FNFC: {len(df_fnfc_nfr)} Quality Reqs across {df_fnfc_nfr['class'].value_counts().to_dict()}")
    res_fnfc = run_evaluation_on_dataset("FNFC Large Corpus (FNFC.csv)", nfr_fnfc, fr_fnfc, y_fnfc, tokenizer, bert_model)
    all_results.extend(res_fnfc)

    # Save
    res_df = pd.DataFrame(all_results)
    csv_path  = os.path.join(out_dir, "extended_promise_exp_and_fnfc_results.csv")
    json_path = os.path.join(out_dir, "extended_promise_exp_and_fnfc_results.json")
    res_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80)
    print("EXTENDED BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved JSON: {json_path}")
    print("=" * 80)

    # Summary Display (Excluding Availability F1 per mandate)
    cols = ["Dataset", "Model", "Accuracy", "Macro_F1", "Weighted_F1", "PE_F1", "SE_F1", "US_F1"]
    view_df = res_df[cols].copy()
    for c in cols[2:]:
        view_df[c] = (view_df[c] * 100).round(2)
    print(view_df.to_string(index=False))

if __name__ == "__main__":
    main()
