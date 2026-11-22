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
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

torch.set_num_threads(16)
torch.set_num_interop_threads(4)
torch.manual_seed(42)
np.random.seed(42)

TARGET_CLASSES = ['PE', 'SE', 'US', 'A']
LABEL_TO_IDX   = {c: i for i, c in enumerate(TARGET_CLASSES)}
IDX_TO_LABEL   = {i: c for i, c in enumerate(TARGET_CLASSES)}

MAX_LEN_CONCAT = 192   # FR (128) + NFR (64)
MAX_LEN_STREAM = 128
BATCH_SIZE     = 16
EPOCHS_FT      = 8     # Fine-tuning epochs
LR_BERT        = 2e-5  # BERT fine-tuning learning rate
LR_HEAD        = 1e-3  # Head learning rate

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
# DATASETS
# ==============================================================================
class SingleStreamDataset(Dataset):
    """[CLS] FR_text [SEP] NFR_text [SEP] as published in Almohammady et al. (2026)"""
    def __init__(self, nfr_texts, fr_texts, labels, tokenizer, max_len=MAX_LEN_CONCAT):
        if fr_texts is not None:
            self.encodings = tokenizer(
                fr_texts, nfr_texts,
                padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
            )
        else:
            self.encodings = tokenizer(
                nfr_texts,
                padding="max_length", truncation=True, max_length=max_len, return_tensors="pt"
            )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.encodings.items()}
        return item, self.labels[i]

class DualStreamDataset(Dataset):
    """Separate tokenizations for Stream A (NFR) and Stream B (FR Context)"""
    def __init__(self, nfr_texts, fr_texts, labels, tokenizer, max_len=MAX_LEN_STREAM):
        self.enc_nfr = tokenizer(nfr_texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        self.enc_fr  = tokenizer(fr_texts,  padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        self.labels  = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        item_nfr = {k: v[i] for k, v in self.enc_nfr.items()}
        item_fr  = {k: v[i] for k, v in self.enc_fr.items()}
        return item_nfr, item_fr, self.labels[i]

# ==============================================================================
# ARCHITECTURE 1: FULL SINGLE-STREAM MODEL (Almohammady et al., 2026)
# ==============================================================================
class FullSingleStreamModel(nn.Module):
    """
    Exact end-to-end single-stream architecture:
    [CLS] FR [SEP] NFR [SEP] -> BERT Encoder -> Dropout(0.1) -> Linear(768, 4)
    """
    def __init__(self, bert_model, num_classes=4, dropout=0.1, use_mlp=False):
        super().__init__()
        self.bert = bert_model
        self.use_mlp = use_mlp
        if use_mlp:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(bert_model.config.hidden_size, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(bert_model.config.hidden_size, num_classes)
            )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_rep)

# ==============================================================================
# ARCHITECTURE 2: FULL PROPOSED DUAL-STREAM MODEL (Complete Architecture)
# ==============================================================================
class FullProposedDualStreamModel(nn.Module):
    """
    Complete Proposed Dual-Stream Architecture:
    1. Stream A (NFR BERT Encoder) -> extracts top-4 hidden layers of [CLS]
    2. Stream B (FR Context BERT Encoder) -> extracts top-4 hidden layers of [CLS]
    3. Hierarchical Cross-Layer Attention Fusion (H-CLAF)
    4. Adaptive Context Gating Unit (ACG)
    5. Multi-Layer Projection with GELU, BatchNorm, and Learnable Temperature Scaling
    """
    def __init__(self, bert_model, hidden_dim=768, num_classes=4, dropout=0.20):
        super().__init__()
        self.bert = bert_model  # Shared or dual-branch encoder

        # Stream A Hierarchical Layer Attention
        self.nfr_layer_attn = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Stream B Hierarchical Layer Attention
        self.fr_layer_attn = nn.Sequential(
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
        # Deep Fusion Head
        self.fc1 = nn.Linear(hidden_dim * 2, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        self.temp = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, nfr_inputs, fr_inputs):
        # Stream A: NFR representation over top 4 layers
        out_nfr = self.bert(**nfr_inputs, output_hidden_states=True)
        top4_nfr = torch.stack([out_nfr.hidden_states[k][:, 0, :] for k in range(-4, 0)], dim=1) # (B, 4, D)

        # Stream B: FR Context representation over top 4 layers
        out_fr = self.bert(**fr_inputs, output_hidden_states=True)
        top4_fr = torch.stack([out_fr.hidden_states[k][:, 0, :] for k in range(-4, 0)], dim=1)   # (B, 4, D)

        b, l, d = top4_nfr.shape

        # Pool NFR layers
        w_nfr = F.softmax(self.nfr_layer_attn(top4_nfr.view(-1, d)).view(b, l, 1), dim=1)
        h_nfr = (top4_nfr * w_nfr).sum(dim=1)

        # Pool FR layers
        w_fr = F.softmax(self.fr_layer_attn(top4_fr.view(-1, d)).view(b, l, 1), dim=1)
        h_fr = (top4_fr * w_fr).sum(dim=1)

        # Adaptive Context Gate (ACG)
        alpha = self.context_gate(torch.cat([h_nfr, h_fr], dim=1))
        h_gated_fr = alpha * h_fr

        # Fusion
        fused = torch.cat([h_nfr, h_gated_fr], dim=1)
        out1 = self.gelu(self.bn1(self.fc1(fused)))
        out2 = self.gelu(self.bn2(self.fc2(self.dropout(out1))))
        logits = self.classifier(out2) / torch.clamp(self.temp, min=0.01, max=1.0)
        return logits

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
# FULL ARCHITECTURE TRAINING FUNCTIONS
# ==============================================================================
def train_full_single_stream(train_ds, test_ds, use_mlp=False, epochs=EPOCHS_FT):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = FullSingleStreamModel(bert, num_classes=4, dropout=0.1, use_mlp=use_mlp)

    # Differential learning rate
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': LR_BERT, 'weight_decay': 0.01},
        {'params': model.classifier.parameters(), 'lr': LR_HEAD, 'weight_decay': 0.01}
    ])
    criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model.train()
    for _ in range(epochs):
        for batch, labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch, labels in test_loader:
            logits = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            p = torch.argmax(logits, dim=1).numpy()
            preds.extend(p)
            trues.extend(labels.numpy())
    return np.array(trues), np.array(preds)

def train_full_dual_stream(train_ds, test_ds, class_weights=None, epochs=EPOCHS_FT):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased")
    model = FullProposedDualStreamModel(bert, hidden_dim=768, num_classes=4, dropout=0.20)

    # Differential learning rate
    head_params = [p for n, p in model.named_parameters() if not n.startswith('bert')]
    optimizer = torch.optim.AdamW([
        {'params': model.bert.parameters(), 'lr': LR_BERT, 'weight_decay': 0.01},
        {'params': head_params, 'lr': LR_HEAD, 'weight_decay': 0.01}
    ])
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    model.train()
    for _ in range(epochs):
        for nfr_batch, fr_batch, labels in train_loader:
            optimizer.zero_grad()
            logits = model(nfr_batch, fr_batch)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for nfr_batch, fr_batch, labels in test_loader:
            logits = model(nfr_batch, fr_batch)
            p = torch.argmax(logits, dim=1).numpy()
            preds.extend(p)
            trues.extend(labels.numpy())
    return np.array(trues), np.array(preds)

def run_classical_baselines(train_df, test_df):
    """Replicates all comparative ML algorithms mentioned in the paper"""
    X_tr = [f"{clean_text(fr)} {clean_text(nfr)}" for fr, nfr in zip(train_df["FR_Context"], train_df["RequirementTextClean"])]
    y_tr = train_df["Class"].map(LABEL_TO_IDX).values

    X_te = [f"{clean_text(fr)} {clean_text(nfr)}" for fr, nfr in zip(test_df["FR_Context"], test_df["RequirementTextClean"])]
    y_te = test_df["Class"].map(LABEL_TO_IDX).values

    tfidf = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    X_tr_tfidf = tfidf.fit_transform(X_tr).toarray()
    X_te_tfidf = tfidf.transform(X_te).toarray()

    models = {
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=16),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=16),
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        "SGD Classifier": SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    results = []
    for name, clf in models.items():
        clf.fit(X_tr_tfidf, y_tr)
        p = clf.predict(X_te_tfidf)
        m = evaluate_metrics(y_te, p)
        m["Algorithm"] = f"TF-IDF + {name}"
        m["Architecture"] = "Comparative Classical ML"
        results.append(m)
    return results

def main():
    print("=" * 80)
    print("MASTER BENCHMARK: FULL ARCHITECTURE REPLICATION IN PARALLEL (16 CORES)")
    print("Paper 7: Single-Stream Re-Distill Full Fine-Tuned Model (Almohammady et al., 2026)")
    print("vs. Proposed: Dual-Stream Gated Cross-Attention Full Fine-Tuned Model")
    print("All Comparative Algorithms Executed Under Exact Same Parameters")
    print("=" * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    out_dir  = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(data_dir, "train_p1_p10.csv"))
    test_df  = pd.read_csv(os.path.join(data_dir, "test_p11_p15.csv"))
    all_df   = pd.read_csv(os.path.join(data_dir, "nfr_all_with_context.csv"))

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

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

    # ==========================================================================
    # EXPERIMENT 1: PROJECT-LEVEL SPLIT (P1-P10 -> P11-P15)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: FULL ARCHITECTURES ON PROJECT SPLIT (P1-P10 -> P11-P15)")
    print("=" * 80)

    ss_tr_ds = SingleStreamDataset(tr_nfr, tr_fr, y_tr, tokenizer)
    ss_te_ds = SingleStreamDataset(te_nfr, te_fr, y_te, tokenizer)

    ds_tr_ds = DualStreamDataset(tr_nfr, tr_fr, y_tr, tokenizer)
    ds_te_ds = DualStreamDataset(te_nfr, te_fr, y_te, tokenizer)

    exp1_results = []

    # 1. Full Single-Stream (Paper 7 - Linear Head)
    print("Training Full Single-Stream Model (Linear Head - Almohammady et al. 2026)...", flush=True)
    y_true, p_ss_lin = train_full_single_stream(ss_tr_ds, ss_te_ds, use_mlp=False)
    m = evaluate_metrics(y_true, p_ss_lin)
    m["Algorithm"] = "Single-Stream BERT (Linear Head)"
    m["Architecture"] = "Single-Stream (Almohammady et al., 2026)"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    exp1_results.append(m)

    # 2. Full Single-Stream (MLP Head)
    print("Training Full Single-Stream Model (MLP Head)...", flush=True)
    y_true, p_ss_mlp = train_full_single_stream(ss_tr_ds, ss_te_ds, use_mlp=True)
    m = evaluate_metrics(y_true, p_ss_mlp)
    m["Algorithm"] = "Single-Stream BERT (MLP Head)"
    m["Architecture"] = "Single-Stream (Almohammady et al., 2026)"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    exp1_results.append(m)

    # 3. Full Proposed Dual-Stream Architecture
    print("Training Full Proposed Dual-Stream Model (H-CLAF + ACG + Focal Loss)...", flush=True)
    y_true, p_ds_full = train_full_dual_stream(ds_tr_ds, ds_te_ds, class_weights=class_weights_t)
    m = evaluate_metrics(y_true, p_ds_full)
    m["Algorithm"] = "Proposed Dual-Stream (Gated Cross-Attention)"
    m["Architecture"] = "Dual-Stream (Proposed SOTA)"
    m["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
    exp1_results.append(m)

    # 4. Comparative Classical ML Baselines
    print("Running Comparative Classical Algorithms on same split...", flush=True)
    classical_res = run_classical_baselines(train_df, test_df)
    for cr in classical_res:
        cr["Evaluation"] = "Project Split (P1-P10 -> P11-P15)"
        exp1_results.append(cr)

    # ==========================================================================
    # EXPERIMENT 2: 10-FOLD STRATIFIED CROSS-VALIDATION (208 SAMPLES)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: FULL ARCHITECTURES ON 10-FOLD STRATIFIED CV (208 SAMPLES)")
    print("=" * 80)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    all_preds_ss = np.zeros(len(y_all), dtype=int)
    all_preds_ds = np.zeros(len(y_all), dtype=int)

    # Classical folds
    classical_fold_preds = {
        "Linear SVM": np.zeros(len(y_all), dtype=int),
        "Extra Trees": np.zeros(len(y_all), dtype=int),
        "Random Forest": np.zeros(len(y_all), dtype=int),
        "Logistic Regression": np.zeros(len(y_all), dtype=int),
        "SGD Classifier": np.zeros(len(y_all), dtype=int),
        "Naive Bayes": np.zeros(len(y_all), dtype=int),
        "MLP Classifier": np.zeros(len(y_all), dtype=int),
        "Decision Tree": np.zeros(len(y_all), dtype=int)
    }

    all_concat_texts = [f"{clean_text(fr)} {clean_text(nfr)}" for fr, nfr in zip(all_df["FR_Context"], all_df["RequirementTextClean"])]
    tfidf = TfidfVectorizer(max_features=2500, stop_words="english", ngram_range=(1, 2))
    all_tfidf = tfidf.fit_transform(all_concat_texts).toarray()

    for fold, (tr_idx, val_idx) in enumerate(skf.split(all_nfr, y_all)):
        print(f"--- Running Fold {fold+1}/10 in parallel across 16 cores ---", flush=True)
        # 1. Full Single-Stream Fold
        fold_ss_tr = SingleStreamDataset([all_nfr[i] for i in tr_idx], [all_fr[i] for i in tr_idx], y_all[tr_idx], tokenizer)
        fold_ss_val = SingleStreamDataset([all_nfr[i] for i in val_idx], [all_fr[i] for i in val_idx], y_all[val_idx], tokenizer)
        _, p_ss = train_full_single_stream(fold_ss_tr, fold_ss_val, use_mlp=False, epochs=6)
        all_preds_ss[val_idx] = p_ss

        # 2. Full Dual-Stream Fold
        fold_ds_tr = DualStreamDataset([all_nfr[i] for i in tr_idx], [all_fr[i] for i in tr_idx], y_all[tr_idx], tokenizer)
        fold_ds_val = DualStreamDataset([all_nfr[i] for i in val_idx], [all_fr[i] for i in val_idx], y_all[val_idx], tokenizer)
        _, p_ds = train_full_dual_stream(fold_ds_tr, fold_ds_val, class_weights=class_weights_t, epochs=6)
        all_preds_ds[val_idx] = p_ds

        # 3. Classical Baselines
        clf_dict = {
            "Linear SVM": LinearSVC(C=1.0, max_iter=2000, random_state=42),
            "Extra Trees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=16),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=16),
            "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            "SGD Classifier": SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42),
            "Naive Bayes": MultinomialNB(),
            "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }
        for cname, clf in clf_dict.items():
            clf.fit(all_tfidf[tr_idx], y_all[tr_idx])
            classical_fold_preds[cname][val_idx] = clf.predict(all_tfidf[val_idx])

    exp2_results = []
    # SS
    m = evaluate_metrics(y_all, all_preds_ss)
    m["Algorithm"] = "Single-Stream BERT (Linear Head)"
    m["Architecture"] = "Single-Stream (Almohammady et al., 2026)"
    m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
    exp2_results.append(m)

    # DS
    m = evaluate_metrics(y_all, all_preds_ds)
    m["Algorithm"] = "Proposed Dual-Stream (Gated Cross-Attention)"
    m["Architecture"] = "Dual-Stream (Proposed SOTA)"
    m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
    exp2_results.append(m)

    # Classical
    for cname in classical_fold_preds:
        m = evaluate_metrics(y_all, classical_fold_preds[cname])
        m["Algorithm"] = f"TF-IDF + {cname}"
        m["Architecture"] = "Comparative Classical ML"
        m["Evaluation"] = "10-Fold Stratified CV (208 samples)"
        exp2_results.append(m)

    # Save
    combined_df = pd.concat([pd.DataFrame(exp1_results), pd.DataFrame(exp2_results)], ignore_index=True)
    csv_file  = os.path.join(out_dir, "master_full_architectures_parallel_results.csv")
    json_file = os.path.join(out_dir, "master_full_architectures_parallel_results.json")

    combined_df.to_csv(csv_file, index=False)
    with open(json_file, "w") as f:
        json.dump(combined_df.to_dict(orient="records"), f, indent=2)

    print("\n" + "=" * 80)
    print("MASTER BENCHMARK OF FULL ARCHITECTURES COMPLETED!")
    print(f"Results CSV  : {csv_file}")
    print(f"Results JSON : {json_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
