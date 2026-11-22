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

MAX_LEN_CONCAT = 192   # FR (128) + NFR (64)
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
# LOSS FUNCTIONS
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

# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================
class SingleStreamLinear(nn.Module):
    """Paper 7 (Almohammady et al., 2026): Linear head over single-stream [CLS]"""
    def __init__(self, hidden_dim=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.dropout(x))

class SingleStreamMLP(nn.Module):
    """Single-Stream with 2-layer MLP head"""
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
    """
    Proposed Dual-Stream Model:
    1. Cross-Layer Attention over Stream A (NFR) and Stream B (FR Context)
    2. Adaptive Context Gating Unit
    3. Hierarchical Cross-Attention Fusion with Temperature Scaling
    """
    def __init__(self, hidden_dim=768, num_classes=4, dropout=0.20):
        super().__init__()
        # Cross-layer attention for NFR stream
        self.nfr_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Cross-layer attention for FR stream
        self.fr_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Adaptive Context Gate: alpha in [0, 1]
        self.context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        # Fusion projection
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

class DualStreamNoGate(nn.Module):
    """Ablation: Dual-stream without adaptive gating (direct concat)"""
    def __init__(self, hidden_dim=768, num_classes=4, dropout=0.20):
        super().__init__()
        self.nfr_attn = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fr_attn  = nn.Sequential(nn.Linear(hidden_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, nfr_layers, fr_layers):
        b, l, d = nfr_layers.shape
        w_nfr = F.softmax(self.nfr_attn(nfr_layers.view(-1, d)).view(b, l, 1), dim=1)
        h_nfr = (nfr_layers * w_nfr).sum(dim=1)

        w_fr = F.softmax(self.fr_attn(fr_layers.view(-1, d)).view(b, l, 1), dim=1)
        h_fr = (fr_layers * w_fr).sum(dim=1)

        fused = torch.cat([h_nfr, h_fr], dim=1)
        out1 = self.gelu(self.bn1(self.fc1(fused)))
        out2 = self.gelu(self.bn2(self.fc2(self.dropout(out1))))
        return self.classifier(out2)

# ==============================================================================
# FEATURE EXTRACTION
# ==============================================================================
def extract_single_stream_features(nfr_texts, fr_texts, tokenizer, bert_model):
    """Extracts single-stream [CLS] representation: [CLS] FR [SEP] NFR [SEP]"""
    bert_model.eval()
    encodings = tokenizer(fr_texts, nfr_texts, padding="max_length", truncation=True,
                          max_length=MAX_LEN_CONCAT, return_tensors="pt")
    ds = TensorDataset(encodings["input_ids"], encodings["attention_mask"], encodings["token_type_ids"])
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    cls_list = []
    with torch.no_grad():
        for b_ids, b_mask, b_type in loader:
            outputs = bert_model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type)
            cls_list.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(cls_list, dim=0)

def extract_dual_stream_features(texts, tokenizer, bert_model, max_len=MAX_LEN_SINGLE):
    """Extracts top 4 hidden layers of [CLS] for one stream"""
    bert_model.eval()
    encodings = tokenizer(texts, padding="max_length", truncation=True,
                          max_length=max_len, return_tensors="pt")
    ds = TensorDataset(encodings["input_ids"], encodings["attention_mask"], encodings["token_type_ids"])
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    layers_list = []
    with torch.no_grad():
        for b_ids, b_mask, b_type in loader:
            outputs = bert_model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type, output_hidden_states=True)
            top4 = torch.stack([outputs.hidden_states[k][:, 0, :] for k in range(-4, 0)], dim=1).cpu()
            layers_list.append(top4)
    return torch.cat(layers_list, dim=0)

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

# ==============================================================================
# BENCHMARK RUNNER
# ==============================================================================
def train_model(model, train_data, y_train, test_data, is_dual=False, use_focal=False, class_weights=None):
    torch.manual_seed(42)
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    if use_focal:
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    if is_dual:
        train_ds = TensorDataset(train_data[0], train_data[1], y_train)
    else:
        train_ds = TensorDataset(train_data, y_train)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for batch in train_loader:
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
            logits = model(test_data[0], test_data[1])
        else:
            logits = model(test_data)
        preds = torch.argmax(logits, dim=1).numpy()
    return preds

def main():
    print("=" * 80)
    print("SIDE-BY-SIDE BENCHMARK: SINGLE-STREAM vs. PROPOSED DUAL-STREAM")
    print("Same Backbone: bert-base-uncased | Same Hyperparameters & Data Splits")
    print("Allocated CPU Threads: 16 (all 16 cores active)")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir  = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_p1_p10.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test_p11_p15.csv"))
    all_df   = pd.read_csv(os.path.join(data_dir, "nfr_all_with_context.csv"))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")

    # Clean strings
    tr_nfr = [clean_text(t) for t in train_df["RequirementTextClean"]]
    tr_fr  = [clean_text(t) for t in train_df["FR_Context"]]
    y_tr   = torch.tensor(train_df["Class"].map(LABEL_TO_IDX).values, dtype=torch.long)

    te_nfr = [clean_text(t) for t in test_df["RequirementTextClean"]]
    te_fr  = [clean_text(t) for t in test_df["FR_Context"]]
    y_te   = torch.tensor(test_df["Class"].map(LABEL_TO_IDX).values, dtype=torch.long)

    all_nfr = [clean_text(t) for t in all_df["RequirementTextClean"]]
    all_fr  = [clean_text(t) for t in all_df["FR_Context"]]
    y_all   = torch.tensor(all_df["Class"].map(LABEL_TO_IDX).values, dtype=torch.long)

    # Class weights for focal loss
    counts = np.bincount(y_all.numpy(), minlength=4)
    weights = len(y_all) / (4.0 * counts)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    print("\n--- 1. Precomputing Representations ---")
    print("Single-stream features (P1-P10 train / P11-P15 test)...")
    ss_tr = extract_single_stream_features(tr_nfr, tr_fr, tokenizer, bert_model)
    ss_te = extract_single_stream_features(te_nfr, te_fr, tokenizer, bert_model)

    print("Dual-stream features (P1-P10 train / P11-P15 test)...")
    ds_tr_nfr = extract_dual_stream_features(tr_nfr, tokenizer, bert_model)
    ds_tr_fr  = extract_dual_stream_features(tr_fr, tokenizer, bert_model)
    ds_te_nfr = extract_dual_stream_features(te_nfr, tokenizer, bert_model)
    ds_te_fr  = extract_dual_stream_features(te_fr, tokenizer, bert_model)

    print("Single-stream & Dual-stream features (All 208 for 10-Fold CV)...")
    ss_all    = extract_single_stream_features(all_nfr, all_fr, tokenizer, bert_model)
    ds_all_nfr = extract_dual_stream_features(all_nfr, tokenizer, bert_model)
    ds_all_fr  = extract_dual_stream_features(all_fr, tokenizer, bert_model)

    # ==============================================================================
    # EXPERIMENT 1: PROJECT-LEVEL SPLIT (P1-P10 -> P11-P15)
    # ==============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: PROJECT-LEVEL SPLIT (P1-P10 -> P11-P15)")
    print("=" * 80)

    exp1_models = [
        ("Single-Stream BERT (Linear Head)", SingleStreamLinear(), ss_tr, ss_te, False, False),
        ("Single-Stream BERT (MLP Head)", SingleStreamMLP(), ss_tr, ss_te, False, False),
        ("Single-Stream BERT (Focal Loss)", SingleStreamMLP(), ss_tr, ss_te, False, True),
        ("Dual-Stream (Ablation: No Gate)", DualStreamNoGate(), (ds_tr_nfr, ds_tr_fr), (ds_te_nfr, ds_te_fr), True, False),
        ("Proposed Dual-Stream (Gated Attention + Focal)", ProposedDualStream(), (ds_tr_nfr, ds_tr_fr), (ds_te_nfr, ds_te_fr), True, True),
    ]

    exp1_results = []
    for name, model, tr_x, te_x, is_dual, use_focal in exp1_models:
        preds = train_model(model, tr_x, y_tr, te_x, is_dual=is_dual, use_focal=use_focal, class_weights=class_weights_t)
        m = evaluate_metrics(y_te.numpy(), preds)
        m["Model"] = name
        m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
        exp1_results.append(m)

    # ==============================================================================
    # EXPERIMENT 2: 10-FOLD STRATIFIED CROSS-VALIDATION
    # ==============================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: 10-FOLD STRATIFIED CROSS-VALIDATION (208 SAMPLES)")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    model_keys = [
        "Single-Stream BERT (Linear Head)",
        "Single-Stream BERT (MLP Head)",
        "Single-Stream BERT (Focal Loss)",
        "Dual-Stream (Ablation: No Gate)",
        "Proposed Dual-Stream (Gated Attention + Focal)"
    ]

    cv_preds = {k: np.zeros(len(y_all), dtype=int) for k in model_keys}

    for fold, (tr_idx, val_idx) in enumerate(skf.split(ss_all, y_all.numpy())):
        y_tr_fold = y_all[tr_idx]
        y_val_fold = y_all[val_idx]

        # 1. Single-Stream Linear
        m1 = SingleStreamLinear()
        cv_preds["Single-Stream BERT (Linear Head)"][val_idx] = train_model(
            m1, ss_all[tr_idx], y_tr_fold, ss_all[val_idx], is_dual=False, use_focal=False, class_weights=class_weights_t
        )

        # 2. Single-Stream MLP
        m2 = SingleStreamMLP()
        cv_preds["Single-Stream BERT (MLP Head)"][val_idx] = train_model(
            m2, ss_all[tr_idx], y_tr_fold, ss_all[val_idx], is_dual=False, use_focal=False, class_weights=class_weights_t
        )

        # 3. Single-Stream Focal
        m3 = SingleStreamMLP()
        cv_preds["Single-Stream BERT (Focal Loss)"][val_idx] = train_model(
            m3, ss_all[tr_idx], y_tr_fold, ss_all[val_idx], is_dual=False, use_focal=True, class_weights=class_weights_t
        )

        # 4. Dual-Stream No Gate
        m4 = DualStreamNoGate()
        cv_preds["Dual-Stream (Ablation: No Gate)"][val_idx] = train_model(
            m4, (ds_all_nfr[tr_idx], ds_all_fr[tr_idx]), y_tr_fold,
            (ds_all_nfr[val_idx], ds_all_fr[val_idx]), is_dual=True, use_focal=False, class_weights=class_weights_t
        )

        # 5. Proposed Dual-Stream
        m5 = ProposedDualStream()
        cv_preds["Proposed Dual-Stream (Gated Attention + Focal)"][val_idx] = train_model(
            m5, (ds_all_nfr[tr_idx], ds_all_fr[tr_idx]), y_tr_fold,
            (ds_all_nfr[val_idx], ds_all_fr[val_idx]), is_dual=True, use_focal=True, class_weights=class_weights_t
        )

    exp2_results = []
    for k in model_keys:
        m = evaluate_metrics(y_all.numpy(), cv_preds[k])
        m["Model"] = k
        m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
        exp2_results.append(m)

    # Save results
    all_res = pd.concat([pd.DataFrame(exp1_results), pd.DataFrame(exp2_results)], ignore_index=True)
    csv_file  = os.path.join(out_dir, "side_by_side_single_vs_dual_results.csv")
    json_file = os.path.join(out_dir, "side_by_side_single_vs_dual_results.json")
    all_res.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(all_res.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Results saved to: {csv_file}")
    print("=" * 80)

    cols = ["Model", "Accuracy", "Macro_F1", "Weighted_F1", "PE_F1", "SE_F1", "US_F1"]
    print("\n[PROJECT-LEVEL SPLIT: SIDE-BY-SIDE COMPARISON]")
    df1 = pd.DataFrame(exp1_results)[cols].copy()
    for c in cols[1:]:
        df1[c] = (df1[c] * 100).round(2)
    print(df1.to_string(index=False))

    print("\n[10-FOLD STRATIFIED CV: SIDE-BY-SIDE COMPARISON]")
    df2 = pd.DataFrame(exp2_results)[cols].copy()
    for c in cols[1:]:
        df2[c] = (df2[c] * 100).round(2)
    print(df2.to_string(index=False))

if __name__ == "__main__":
    main()
