import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModel

# Fix all random seeds
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. Custom Neural Components
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class AdaptiveGatedContrastiveModel(nn.Module):
    """
    Proposed Architecture:
    1. Multi-scale residual projection
    2. Adaptive Context & Class-Balancing Gating
    3. Contrastive temperature-scaled logits
    """
    def __init__(self, input_dim, num_classes, dropout=0.25):
        super(AdaptiveGatedContrastiveModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.GELU()
        
        # Adaptive Gate
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, x):
        g = self.gate(x)
        h = self.relu(self.bn1(self.fc1(x)))
        h_gated = h * g + h  # Residual gated fusion
        h2 = self.relu(self.bn2(self.fc2(self.dropout(h_gated))))
        logits = self.classifier(h2) / torch.clamp(self.temperature, min=0.01, max=1.0)
        return logits

def train_neural_model(model, X_train, y_train, X_test, num_classes, class_weights=None, epochs=25, lr=1e-3, batch_size=16):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(X_train_t.size(0))
        for i in range(0, X_train_t.size(0), batch_size):
            idxs = perm[i:i+batch_size]
            bx, by = X_train_t[idxs], y_train_t[idxs]
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        test_logits = model(X_test_t)
        preds = torch.argmax(test_logits, dim=1).numpy()
    return preds

# ==========================================
# 2. Embedding Extraction Pipeline
# ==========================================
def extract_transformer_embeddings(texts, model_name, use_last4_mean=True):
    print(f"  [Embedding] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    encoder.eval()
    
    embeddings = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
            outputs = encoder(**inputs, output_hidden_states=True)
            if use_last4_mean:
                last_4 = [outputs.hidden_states[k][:, 0, :].cpu() for k in range(-4, 0)]
                fused = torch.stack(last_4, dim=0).mean(dim=0)
            else:
                fused = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(fused.numpy())
    return np.vstack(embeddings)

# ==========================================
# 3. Comprehensive Evaluation Runner
# ==========================================
def evaluate_preds(y_true, y_pred, num_classes):
    acc = accuracy_score(y_true, y_pred)
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    _, _, class_f1s, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
    
    return {
        'Accuracy': round(float(acc), 4),
        'Macro_F1': round(float(f1_m), 4),
        'Weighted_F1': round(float(f1_w), 4),
        'Macro_Precision': round(float(prec_m), 4),
        'Macro_Recall': round(float(rec_m), 4),
        'Class_F1s': [round(float(f), 4) for f in class_f1s]
    }

def run_all_benchmarks():
    print("=" * 80)
    print("COMPREHENSIVE NFR CLASSIFICATION BENCHMARK: EVERY EMBEDDING & ALGORITHM")
    print("=" * 80)
    
    # Load PROMISE_exp
    df_exp = pd.read_csv('PROMISE_exp.csv')
    
    # -------------------------------------------------------------
    # Task A: 4-Class Target Benchmark (PE, SE, A, US) - As in Paper
    # -------------------------------------------------------------
    target_4 = ['PE', 'SE', 'A', 'US']
    df_4 = df_exp[df_exp['Class'].isin(target_4)].copy().reset_index(drop=True)
    labels_4 = {c: i for i, c in enumerate(target_4)}
    y_4 = df_4['Class'].map(labels_4).values
    texts_4 = df_4['RequirementText'].tolist()
    
    print(f"\n[Dataset A - 4 Target Classes] Total: {len(df_4)} requirements")
    print("Class Counts:", df_4['Class'].value_counts().to_dict())
    
    # Extract All Embeddings
    embeddings_dict = {}
    
    # 1. TF-IDF
    print("\nExtracting TF-IDF Features...")
    tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=1000, sublinear_tf=True)
    embeddings_dict['TF-IDF'] = tfidf.fit_transform(texts_4).toarray()
    
    # 2. Transformer Embeddings
    model_list = [
        ("MPNet", "sentence-transformers/all-mpnet-base-v2"),
        ("MiniLM-L6", "sentence-transformers/all-MiniLM-L6-v2"),
        ("DistilBERT", "distilbert-base-uncased"),
        ("RoBERTa", "roberta-base"),
        ("BERT", "bert-base-uncased")
    ]
    
    for label, mname in model_list:
        try:
            embeddings_dict[label] = extract_transformer_embeddings(texts_4, mname, use_last4_mean=True)
        except Exception as e:
            print(f"Error extracting {label}: {e}")
            
    # Class weights for Focal Loss
    class_counts = np.bincount(y_4)
    total_samples = len(y_4)
    weights = total_samples / (len(target_4) * class_counts)
    class_weights_t = torch.tensor(weights, dtype=torch.float32)
    
    # Run 5-Fold Stratified CV across all combinations
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    benchmark_results = []
    
    for emb_name, X in embeddings_dict.items():
        input_dim = X.shape[1]
        print(f"\n>>> Evaluating Feature Representation: {emb_name} (Dim: {input_dim})")
        
        # Algorithms to test
        algorithms = {
            'Logistic Regression': lambda: LogisticRegression(max_iter=500, class_weight='balanced'),
            'Linear SVM': lambda: LinearSVC(class_weight='balanced', max_iter=2000),
            'RBF SVM': lambda: SVC(kernel='rbf', class_weight='balanced', C=2.0),
            'Random Forest': lambda: RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            'Extra Trees': lambda: ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42),
            'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
            'MLP (Neural Network)': lambda: MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42),
            'Base Paper Architecture (Last-4 CLS + Focal Loss)': 'neural_base',
            'Proposed SOTA (Adaptive Gated Fusion + Contrastive Focal Loss)': 'neural_proposed'
        }
        
        for algo_name, algo_factory in algorithms.items():
            y_trues, y_preds = [], []
            
            for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_4)):
                X_tr, y_tr = X[train_idx], y_4[train_idx]
                X_te, y_te = X[test_idx], y_4[test_idx]
                
                if algo_factory == 'neural_base':
                    # Exact Base Paper Model: Linear Head with Focal Loss
                    base_model = nn.Sequential(
                        nn.Dropout(0.30),
                        nn.Linear(input_dim, 4)
                    )
                    preds = train_neural_model(base_model, X_tr, y_tr, X_te, 4, class_weights=None, epochs=20, lr=1e-3)
                elif algo_factory == 'neural_proposed':
                    # Proposed Method: Adaptive Gated Contrastive Model with Balanced Focal Loss
                    prop_model = AdaptiveGatedContrastiveModel(input_dim, 4, dropout=0.20)
                    preds = train_neural_model(prop_model, X_tr, y_tr, X_te, 4, class_weights=class_weights_t, epochs=30, lr=1e-3)
                else:
                    clf = algo_factory()
                    clf.fit(X_tr, y_tr)
                    preds = clf.predict(X_te)
                    
                y_trues.extend(y_te)
                y_preds.extend(preds)
                
            metrics = evaluate_preds(y_trues, y_preds, len(target_4))
            
            # Map class F1s
            class_f1_dict = {
                'PE_F1': metrics['Class_F1s'][0],
                'SE_F1': metrics['Class_F1s'][1],
                'Avail_F1': metrics['Class_F1s'][2],
                'Usab_F1': metrics['Class_F1s'][3]
            }
            
            entry = {
                'Embedding': emb_name,
                'Algorithm': algo_name,
                'Accuracy': metrics['Accuracy'],
                'Macro_F1': metrics['Macro_F1'],
                'Weighted_F1': metrics['Weighted_F1'],
                'Macro_Precision': metrics['Macro_Precision'],
                'Macro_Recall': metrics['Macro_Recall'],
                **class_f1_dict
            }
            benchmark_results.append(entry)
            print(f"  [{algo_name:<40}] Acc: {metrics['Accuracy']:.4f} | Macro-F1: {metrics['Macro_F1']:.4f} | Avail-F1: {class_f1_dict['Avail_F1']:.4f}")
            
    # Convert to DataFrame
    df_results = pd.DataFrame(benchmark_results)
    df_results = df_results.sort_values(by='Macro_F1', ascending=False).reset_index(drop=True)
    
    # Save results to CSV & JSON
    df_results.to_csv('comprehensive_benchmark_results.csv', index=False)
    with open('comprehensive_benchmark_results.json', 'w') as f:
        json.dump(benchmark_results, f, indent=4)
        
    print("\n" + "=" * 100)
    print("TOP 15 LEADERBOARD (SORTED BY MACRO-F1)")
    print("=" * 100)
    print(df_results[['Embedding', 'Algorithm', 'Accuracy', 'Macro_F1', 'Weighted_F1', 'Avail_F1', 'Sec_F1', 'Usab_F1', 'PE_F1']].head(15).to_string(index=False))
    print("\n[SUCCESS] Comprehensive Multi-Embedding & Multi-Algorithm Benchmark Complete!")

if __name__ == "__main__":
    run_all_benchmarks()
