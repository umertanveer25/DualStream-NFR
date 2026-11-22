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
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
# LOSS FUNCTIONS & NEURAL MODELS
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

class NeuralLinear(nn.Module):
    def __init__(self, in_dim=768, num_classes=4, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.classifier(self.dropout(x))

class NeuralMLP(nn.Module):
    def __init__(self, in_dim=768, num_classes=4, dropout=0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
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
    """Dual-Stream with Hierarchical Attention and Adaptive Context Gating"""
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
    return torch.cat(cls_list, dim=0).numpy()

def extract_dual_stream_features(nfr_texts, fr_texts, tokenizer, bert_model):
    """
    Extracts Dual-Stream representations:
    - 2D representation for classical ML: [h_NFR, h_FR] (1536 dim)
    - 3D representation for proposed neural model: (B, 4, 768) for both streams
    """
    bert_model.eval()
    def get_stream(texts):
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=MAX_LEN_SINGLE, return_tensors="pt")
        ds = TensorDataset(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
        loader = DataLoader(ds, batch_size=32, shuffle=False)
        cls_list, top4_list = [], []
        with torch.no_grad():
            for b_ids, b_mask, b_type in loader:
                outputs = bert_model(input_ids=b_ids, attention_mask=b_mask, token_type_ids=b_type, output_hidden_states=True)
                cls_list.append(outputs.last_hidden_state[:, 0, :].cpu())
                top4 = torch.stack([outputs.hidden_states[k][:, 0, :] for k in range(-4, 0)], dim=1).cpu()
                top4_list.append(top4)
        return torch.cat(cls_list, dim=0), torch.cat(top4_list, dim=0)

    nfr_cls, nfr_top4 = get_stream(nfr_texts)
    fr_cls, fr_top4   = get_stream(fr_texts)

    # 2D concatenated dual representation for classical ML
    dual_2d = torch.cat([nfr_cls, fr_cls], dim=1).numpy()
    return dual_2d, nfr_top4, fr_top4

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

def get_ml_models():
    return {
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
        "RBF SVM": SVC(kernel='rbf', C=1.0, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=16),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=16),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "SGD Classifier": SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42),
        "Gaussian Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    }

def train_neural_model(model_cls, in_dim, tr_x, tr_y, te_x, use_focal=False, class_weights=None):
    torch.manual_seed(42)
    model = model_cls(in_dim=in_dim, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights) if use_focal else nn.CrossEntropyLoss(weight=class_weights)

    ds = TensorDataset(torch.tensor(tr_x, dtype=torch.float32), torch.tensor(tr_y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for b_x, b_y in loader:
            optimizer.zero_grad()
            out = model(b_x)
            loss = criterion(out, b_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(te_x, dtype=torch.float32)), dim=1).numpy()
    return preds

def train_proposed_dual(tr_nfr_top4, tr_fr_top4, tr_y, te_nfr_top4, te_fr_top4, class_weights=None):
    torch.manual_seed(42)
    model = ProposedDualStream(hidden_dim=768, num_classes=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)

    ds = TensorDataset(tr_nfr_top4, tr_fr_top4, torch.tensor(tr_y, dtype=torch.long))
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for _ in range(EPOCHS):
        for b_nfr, b_fr, b_y in loader:
            optimizer.zero_grad()
            logits = model(b_nfr, b_fr)
            loss = criterion(logits, b_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        logits = model(te_nfr_top4, te_fr_top4)
        preds = torch.argmax(logits, dim=1).numpy()
    return preds

def run_project_split_all_techniques(ss_tr, ss_te, ds_tr, ds_te,
                                     tr_nfr_top4, tr_fr_top4, te_nfr_top4, te_fr_top4,
                                     y_tr, y_te, class_weights):
    print("\n" + "=" * 80)
    print("RUNNING ALL TECHNIQUES SIDE-BY-SIDE: PROJECT SPLIT (P1-P10 -> P11-P15)")
    print("=" * 80)

    results = []

    # Scale classical ML inputs
    scaler_ss = StandardScaler()
    ss_tr_scaled = scaler_ss.fit_transform(ss_tr)
    ss_te_scaled = scaler_ss.transform(ss_te)

    scaler_ds = StandardScaler()
    ds_tr_scaled = scaler_ds.fit_transform(ds_tr)
    ds_te_scaled = scaler_ds.transform(ds_te)

    # 1. Classical ML Algorithms
    models_dict = get_ml_models()
    for algo_name, clf in models_dict.items():
        # Single-Stream
        clf.fit(ss_tr_scaled, y_tr)
        preds_ss = clf.predict(ss_te_scaled)
        m_ss = evaluate_metrics(y_te, preds_ss)
        m_ss["Algorithm"] = algo_name
        m_ss["Stream"] = "Single-Stream"
        m_ss["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
        results.append(m_ss)

        # Dual-Stream
        clf.fit(ds_tr_scaled, y_tr)
        preds_ds = clf.predict(ds_te_scaled)
        m_ds = evaluate_metrics(y_te, preds_ds)
        m_ds["Algorithm"] = algo_name
        m_ds["Stream"] = "Dual-Stream"
        m_ds["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
        results.append(m_ds)

    # 2. Neural Linear Head
    preds_ss_lin = train_neural_model(NeuralLinear, 768, ss_tr, y_tr, ss_te, False, class_weights)
    m = evaluate_metrics(y_te, preds_ss_lin)
    m["Algorithm"] = "Neural Linear Head"
    m["Stream"] = "Single-Stream"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    preds_ds_lin = train_neural_model(NeuralLinear, 1536, ds_tr, y_tr, ds_te, False, class_weights)
    m = evaluate_metrics(y_te, preds_ds_lin)
    m["Algorithm"] = "Neural Linear Head"
    m["Stream"] = "Dual-Stream"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    # 3. Neural MLP Head
    preds_ss_mlp = train_neural_model(NeuralMLP, 768, ss_tr, y_tr, ss_te, False, class_weights)
    m = evaluate_metrics(y_te, preds_ss_mlp)
    m["Algorithm"] = "Neural MLP Head"
    m["Stream"] = "Single-Stream"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    preds_ds_mlp = train_neural_model(NeuralMLP, 1536, ds_tr, y_tr, ds_te, False, class_weights)
    m = evaluate_metrics(y_te, preds_ds_mlp)
    m["Algorithm"] = "Neural MLP Head"
    m["Stream"] = "Dual-Stream"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    # 4. Neural Focal Loss Head
    preds_ss_focal = train_neural_model(NeuralMLP, 768, ss_tr, y_tr, ss_te, True, class_weights)
    m = evaluate_metrics(y_te, preds_ss_focal)
    m["Algorithm"] = "Neural Focal Loss Head"
    m["Stream"] = "Single-Stream"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    preds_ds_focal = train_neural_model(NeuralMLP, 1536, ds_tr, y_tr, ds_te, True, class_weights)
    m = evaluate_metrics(y_te, preds_ds_focal)
    m["Algorithm"] = "Neural Focal Loss Head"
    m["Stream"] = "Dual-Stream"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    # 5. Proposed Dual-Stream (Adaptive Gated Cross-Attention)
    preds_proposed = train_proposed_dual(tr_nfr_top4, tr_fr_top4, y_tr, te_nfr_top4, te_fr_top4, class_weights)
    m = evaluate_metrics(y_te, preds_proposed)
    m["Algorithm"] = "Proposed Gated Attention"
    m["Stream"] = "Dual-Stream (Proposed SOTA)"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    results.append(m)

    return pd.DataFrame(results)

def run_10fold_cv_all_techniques(ss_all, ds_all, nfr_top4_all, fr_top4_all, y_all, class_weights):
    print("\n" + "=" * 80)
    print("RUNNING ALL TECHNIQUES SIDE-BY-SIDE: 10-FOLD STRATIFIED CV (208 SAMPLES)")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    algos = list(get_ml_models().keys()) + [
        "Neural Linear Head",
        "Neural MLP Head",
        "Neural Focal Loss Head"
    ]

    preds_ss = {a: np.zeros(len(y_all), dtype=int) for a in algos}
    preds_ds = {a: np.zeros(len(y_all), dtype=int) for a in algos}
    preds_proposed = np.zeros(len(y_all), dtype=int)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(ss_all, y_all)):
        y_tr, y_val = y_all[tr_idx], y_all[val_idx]

        scaler_ss = StandardScaler()
        ss_tr_scaled = scaler_ss.fit_transform(ss_all[tr_idx])
        ss_val_scaled = scaler_ss.transform(ss_all[val_idx])

        scaler_ds = StandardScaler()
        ds_tr_scaled = scaler_ds.fit_transform(ds_all[tr_idx])
        ds_val_scaled = scaler_ds.transform(ds_all[val_idx])

        models = get_ml_models()
        for a_name, clf in models.items():
            # SS
            clf.fit(ss_tr_scaled, y_tr)
            preds_ss[a_name][val_idx] = clf.predict(ss_val_scaled)
            # DS
            clf.fit(ds_tr_scaled, y_tr)
            preds_ds[a_name][val_idx] = clf.predict(ds_val_scaled)

        # Neural Linear
        preds_ss["Neural Linear Head"][val_idx] = train_neural_model(NeuralLinear, 768, ss_all[tr_idx], y_tr, ss_all[val_idx], False, class_weights)
        preds_ds["Neural Linear Head"][val_idx] = train_neural_model(NeuralLinear, 1536, ds_all[tr_idx], y_tr, ds_all[val_idx], False, class_weights)

        # Neural MLP
        preds_ss["Neural MLP Head"][val_idx] = train_neural_model(NeuralMLP, 768, ss_all[tr_idx], y_tr, ss_all[val_idx], False, class_weights)
        preds_ds["Neural MLP Head"][val_idx] = train_neural_model(NeuralMLP, 1536, ds_all[tr_idx], y_tr, ds_all[val_idx], False, class_weights)

        # Neural Focal Loss
        preds_ss["Neural Focal Loss Head"][val_idx] = train_neural_model(NeuralMLP, 768, ss_all[tr_idx], y_tr, ss_all[val_idx], True, class_weights)
        preds_ds["Neural Focal Loss Head"][val_idx] = train_neural_model(NeuralMLP, 1536, ds_all[tr_idx], y_tr, ds_all[val_idx], True, class_weights)

        # Proposed Gated Dual Stream
        preds_proposed[val_idx] = train_proposed_dual(
            nfr_top4_all[tr_idx], fr_top4_all[tr_idx], y_tr,
            nfr_top4_all[val_idx], fr_top4_all[val_idx], class_weights
        )

    results = []
    for a in algos:
        # SS
        m = evaluate_metrics(y_all, preds_ss[a])
        m["Algorithm"] = a
        m["Stream"] = "Single-Stream"
        m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
        results.append(m)

        # DS
        m = evaluate_metrics(y_all, preds_ds[a])
        m["Algorithm"] = a
        m["Stream"] = "Dual-Stream"
        m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
        results.append(m)

    m = evaluate_metrics(y_all, preds_proposed)
    m["Algorithm"] = "Proposed Gated Attention"
    m["Stream"] = "Dual-Stream (Proposed SOTA)"
    m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
    results.append(m)

    return pd.DataFrame(results)

def main():
    print("=" * 80)
    print("FULL BENCHMARK: ALL TECHNIQUES SIDE-BY-SIDE (SINGLE-STREAM vs. DUAL-STREAM)")
    print("Same Backbone: bert-base-uncased | Same Seeds (42) | Same Parameters")
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

    tr_nfr = [clean_text(t) for t in train_df["RequirementTextClean"]]
    tr_fr  = [clean_text(t) for t in train_df["FR_Context"]]
    y_tr   = train_df["Class"].map(LABEL_TO_IDX).values

    te_nfr = [clean_text(t) for t in test_df["RequirementTextClean"]]
    te_fr  = [clean_text(t) for t in test_df["FR_Context"]]
    y_te   = test_df["Class"].map(LABEL_TO_IDX).values

    all_nfr = [clean_text(t) for t in all_df["RequirementTextClean"]]
    all_fr  = [clean_text(t) for t in all_df["FR_Context"]]
    y_all   = all_df["Class"].map(LABEL_TO_IDX).values

    counts = np.bincount(y_all, minlength=4)
    weights = len(y_all) / (4.0 * counts)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)

    print("\n--- Precomputing Representations ---")
    print("Precomputing Single-stream features (P1-P10 train & P11-P15 test)...")
    ss_tr = extract_single_stream_features(tr_nfr, tr_fr, tokenizer, bert_model)
    ss_te = extract_single_stream_features(te_nfr, te_fr, tokenizer, bert_model)

    print("Precomputing Dual-stream features (P1-P10 train & P11-P15 test)...")
    ds_tr, tr_nfr_top4, tr_fr_top4 = extract_dual_stream_features(tr_nfr, tr_fr, tokenizer, bert_model)
    ds_te, te_nfr_top4, te_fr_top4 = extract_dual_stream_features(te_nfr, te_fr, tokenizer, bert_model)

    print("Precomputing Single-stream & Dual-stream features for All 208 samples...")
    ss_all = extract_single_stream_features(all_nfr, all_fr, tokenizer, bert_model)
    ds_all, nfr_top4_all, fr_top4_all = extract_dual_stream_features(all_nfr, all_fr, tokenizer, bert_model)

    # 1. Project Split
    res_split = run_project_split_all_techniques(
        ss_tr, ss_te, ds_tr, ds_te,
        tr_nfr_top4, tr_fr_top4, te_nfr_top4, te_fr_top4,
        y_tr, y_te, class_weights_t
    )

    # 2. 10-Fold CV
    res_10cv = run_10fold_cv_all_techniques(
        ss_all, ds_all, nfr_top4_all, fr_top4_all, y_all, class_weights_t
    )

    combined_df = pd.concat([res_split, res_10cv], ignore_index=True)
    csv_file  = os.path.join(out_dir, "all_techniques_side_by_side_comparison.csv")
    json_file = os.path.join(out_dir, "all_techniques_side_by_side_comparison.json")

    combined_df.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(combined_df.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 80)
    print("ALL TECHNIQUES COMPARISON COMPLETE!")
    print(f"Saved CSV : {csv_file}")
    print(f"Saved JSON: {json_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
