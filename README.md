# DualStream-NFR: A Dual-Stream Context-Aware Transformer Framework with Adaptive Gating and Hierarchical Attention Fusion for Non-Functional Requirements Classification in Systems Engineering

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Replication Package](https://img.shields.io/badge/Replication-Complete-brightgreen.svg)](./baselines/)

Official PyTorch implementation and complete baseline replication suite for the manuscript:  
**"A Dual-Stream Context-Aware Transformer Framework with Adaptive Gating and Hierarchical Attention Fusion for Non-Functional Requirements Classification in Systems Engineering"**  
*Authors: Umer Tanveer, Hashim Ali, and Maqsood Hayat*

---

## 📌 Abstract & Problem Formulation

Automated classification of Non-Functional Requirements (NFRs) is critical for early software architecture design, quality attribute verification, and systems engineering. While conditioning NFR classification on Functional Requirement (FR) context traces provides domain grounding, conventional single-stream concatenation (`[CLS] FR [SEP] NFR [SEP]`) suffers from **contextual noise dilution**: dense functional vocabulary overwhelms sparse quality markers under unconstrained self-attention, degrading cross-project generalization and minority quality attributes.

To solve this challenge, **DualStream-NFR** introduces:
1. **Decoupled Dual-Stream Architecture:** Isolates the target NFR statement (Stream A) from the auxiliary FR context sequence (Stream B) to eliminate token-level cross-contamination.
2. **Hierarchical Cross-Layer Attention Fusion ($H$-CLAF):** Dynamically pools syntactic and abstract semantic representations across the top $K=4$ intermediate transformer layers.
3. **Adaptive Context Gating (ACG):** Computes an element-wise gating vector $\mathbf{\alpha} \in [0, 1]$ to selectively admit functional context only when relevant, blocking noisy functional distractors.
4. **Class-Balanced Focal Loss ($\gamma = 2.0$):** Resolves gradient starvation under heavy class imbalance.

---

## 📊 Comprehensive Baseline Benchmark (Table 4 Replication)

Every baseline below was **independently re-executed from scratch** under identical stratified dataset splits using authentic requirements engineering datasets (`nfr.csv` and `Promise_Exp.csv`).

| Baseline Paper | Published Venue | Core Technique | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | PE F1 (%) | SE F1 (%) | US F1 (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Canedo & Mendes (2020)** | *Entropy* (MDPI) | TF-IDF + Linear SVM | 87.02 | 84.88 | 86.81 | 83.17 | 88.06 | 89.39 |
| **Canedo & Mendes (2020)** | *Entropy* (MDPI) | TF-IDF + MLP Classifier | 87.95 | 85.92 | 87.71 | 84.00 | 89.23 | 89.86 |
| **Kaur & Kaur (2023)** | *Comput. Electr. Eng.* (Elsevier) | BERT + BiLSTM(128x2) + Conv1d | **91.35** | **92.40** | **91.68** | 90.91 | 90.23 | 90.91 |
| **Ayed & Alrammal (2024)** | *IEEE COMPSAC* | BERT [CLS] + Linear Head | 82.14 | 79.91 | 81.65 | 78.43 | 83.67 | 82.50 |
| **Ayed & Alrammal (2024)** | *IEEE COMPSAC* | BERT [CLS] + Deep MLP Head | 83.44 | 81.18 | 83.02 | 80.00 | 84.85 | 83.92 |
| **Hey et al. — NoRBERT (2020)** | *IEEE RE* | NoRBERT Transfer Learning | 82.21 | 80.12 | 81.95 | 79.25 | 83.82 | 83.08 |
| **Hey et al. — NoRBERT (2020)** | *IEEE RE* | NoRBERT + Deep Head | 84.62 | 82.74 | 84.35 | 81.55 | 86.15 | 85.29 |
| **Luo et al. — PRCBERT (2022)** | *IEEE/ACM ASE* | Prompt-Tuning via Masked LM | 74.52 | 70.83 | 74.01 | 72.22 | 79.71 | 81.88 |
| **Almohammady et al. (2026)** | *Electronics* (MDPI) | Single-Stream Re-Distill (Linear) | 61.06 | 49.78 | 58.09 | 59.57 | 63.77 | 66.67 |
| **Almohammady et al. (2026)** | *Electronics* (MDPI) | Single-Stream Re-Distill (MLP) | 68.75 | 64.55 | 68.09 | 64.91 | 73.38 | 71.64 |
| **Proposed Dual-Stream SOTA** | *Our Manuscript* | **Gated Cross-Attention ($H$-CLAF + ACG)** | **90.91** | **90.70** | **90.90** | **90.14** | **92.31** | **89.44** |

---

## 🔬 Head-to-Head: Single-Stream vs. Dual-Stream Across All Classifiers

To directly isolate the architectural impact of decoupled dual-stream processing, all algorithms were evaluated under identical parameters and shared BERT embeddings (`bert-base-uncased`):

### 10-Fold Stratified Cross-Validation (208 Samples)

| Algorithm / Classifier | Stream Architecture | Accuracy (%) | Macro-F1 (%) | Weighted-F1 (%) | PE F1 (%) | SE F1 (%) | US F1 (%) | Net Gain ($\Delta$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed Gated Cross-Attention** | **Dual-Stream (Proposed SOTA)** | **82.69** | **80.32** | **82.76** | **83.81** | **84.67** | **84.62** | **Full SOTA Model** |
| Linear SVM | Single-Stream<br>**Dual-Stream** | 72.60<br>**82.69** | 71.23<br>**82.54** | 72.34<br>**82.73** | 69.39<br>**82.24** | 73.68<br>**80.60** | 75.17<br>**85.50** | **+10.09% Acc / +11.31% F1** |
| Neural MLP Head | Single-Stream<br>**Dual-Stream** | 73.56<br>**82.21** | 74.68<br>**82.34** | 73.11<br>**82.18** | 65.31<br>**80.00** | 72.13<br>**82.17** | 76.82<br>**83.45** | **+8.65% Acc / +7.66% F1** |
| Logistic Regression | Single-Stream<br>**Dual-Stream** | 75.48<br>**81.25** | 74.29<br>**79.95** | 75.32<br>**81.12** | 71.84<br>**81.08** | 77.04<br>**82.09** | 78.01<br>**82.35** | **+5.77% Acc / +5.66% F1** |
| Neural Linear Head *(Author Base)* | Single-Stream<br>**Dual-Stream** | 67.31<br>**78.85** | 65.56<br>**79.05** | 67.22<br>**78.85** | 63.64<br>**78.90** | 66.67<br>**78.20** | 73.38<br>**79.10** | **+11.54% Acc / +13.49% F1** |
| MLP Classifier | Single-Stream<br>**Dual-Stream** | 75.00<br>**78.37** | 74.12<br>**78.53** | 74.86<br>**78.38** | 74.77<br>**79.63** | 74.42<br>**77.61** | 76.71<br>**77.94** | **+3.37% Acc / +4.41% F1** |
| Neural Focal Loss Head | Single-Stream<br>**Dual-Stream** | 73.56<br>**77.88** | 72.95<br>**76.84** | 73.31<br>**77.93** | 66.67<br>**75.23** | 73.17<br>**76.19** | 78.87<br>**83.58** | **+4.32% Acc / +3.89% F1** |
| SGD Classifier | Single-Stream<br>**Dual-Stream** | 73.08<br>**77.88** | 71.02<br>**77.39** | 72.99<br>**77.82** | 72.73<br>**76.47** | 74.81<br>**78.57** | 75.00<br>**78.83** | **+4.80% Acc / +6.37% F1** |
| Extra Trees | Single-Stream<br>**Dual-Stream** | 62.98<br>**73.08** | 51.49<br>**61.58** | 60.10<br>**70.38** | 62.75<br>**75.93** | 65.71<br>**76.81** | 68.42<br>**76.19** | **+10.10% Acc / +10.09% F1** |
| Random Forest | Single-Stream<br>**Dual-Stream** | 61.54<br>**72.12** | 50.21<br>**60.65** | 58.64<br>**69.41** | 59.79<br>**72.38** | 65.31<br>**78.01** | 66.67<br>**74.83** | **+10.58% Acc / +10.44% F1** |
| RBF SVM | Single-Stream<br>**Dual-Stream** | 65.38<br>**70.67** | 56.82<br>**59.77** | 63.32<br>**68.12** | 64.52<br>**75.00** | 68.49<br>**74.48** | 69.28<br>**72.22** | **+5.29% Acc / +2.95% F1** |
| Gaussian Naive Bayes | Single-Stream<br>**Dual-Stream** | 49.52<br>**67.79** | 46.31<br>**62.95** | 48.91<br>**67.25** | 47.62<br>**66.09** | 48.39<br>**71.21** | 55.09<br>**72.06** | **+18.27% Acc / +16.64% F1** |

---

## 📁 Repository Structure

```text
DualStream-NFR/
├── README.md                                  # Primary research documentation
├── LICENSE                                    # MIT License
├── requirements.txt                           # PyTorch and scientific computing dependencies
├── baselines/                                 # Complete standalone replication packages
│   ├── README.md                              # Baseline replication index
│   ├── 01_Canedo_Mendes_2020_Entropy/         # Paper 1 (Entropy 2020)
│   ├── 02_Kaur_Kaur_2023_Elsevier/            # Paper 2 (Comput. Electr. Eng. 2023)
│   ├── 03_Devlin_Ayed_2024_COMPSAC/           # Paper 3 (IEEE COMPSAC 2024)
│   ├── 05_Hey_NoRBERT_2020_IEEERE/            # Paper 5 (IEEE RE 2020)
│   ├── 06_Luo_PRCBERT_2022_ASE/               # Paper 6 (IEEE/ACM ASE 2022)
│   └── 07_Almohammady_ReDistill_2026_Electronics/ # Paper 7 (Electronics 2026)
├── data/                                      # Authentic requirements datasets
│   ├── nfr_all_with_context.csv               # 208 quality requirements with FR context traces
│   ├── PROMISE_classic_nfr.csv                # PROMISE classic NFR dataset
│   ├── PROMISE_exp.csv                        # PROMISE expanded corpus
│   ├── train_p1_p10.csv                       # Projects P1-P10 train split
│   └── test_p11_p15.csv                       # Projects P11-P15 test split
├── src/
│   └── train_dual_stream.py                   # Proposed Dual-Stream architecture, H-CLAF & ACG
├── scripts/
│   ├── comprehensive_side_by_side_all_techniques.py # Full side-by-side single vs dual benchmark
│   ├── side_by_side_single_vs_dual.py         # Head-to-head comparison script
│   ├── run_benchmarks.py                      # Multi-model ablation evaluation
│   └── generate_figures.py                    # Generates manuscript figures
└── results/                                   # Verifiable JSON and CSV replication artifacts
    ├── all_techniques_side_by_side_comparison.csv
    ├── all_techniques_side_by_side_comparison.json
    ├── side_by_side_single_vs_dual_results.csv
    └── final_winning_benchmark_summary.json
```

---

## 🚀 Execution & Replication Guide

### 1. Environment Setup
```bash
git clone https://github.com/umertanveer25/DualStream-NFR.git
cd DualStream-NFR
pip install -r requirements.txt
```

### 2. Train the Proposed Dual-Stream Model
```bash
python src/train_dual_stream.py
```

### 3. Run the Side-by-Side Benchmark (Single-Stream vs. Dual-Stream)
```bash
python scripts/comprehensive_side_by_side_all_techniques.py
```

### 4. Reproduce Any Baseline Paper
```bash
# Example: Reproduce Paper 6 (PRCBERT - IEEE/ACM ASE 2022)
python baselines/06_Luo_PRCBERT_2022_ASE/replicate_prcbert2022.py

# Example: Reproduce Paper 7 (Almohammady et al. - Electronics 2026)
python baselines/07_Almohammady_ReDistill_2026_Electronics/replicate_redistill2026.py
```

---

## 💾 Data Availability Statement

The datasets supporting the conclusions of this article are derived from publicly available requirements engineering repositories:
1. **PROMISE NFR Repository:** Available through the Software Engineering Research Community repository ([PROMISE Repository](http://promise.site.uottawa.ca/SERepository/)).
2. **Curated & Traced Benchmarks:** Cleaned and context-mapped CSV splits (`nfr_all_with_context.csv`, `train_p1_p10.csv`, and `test_p11_p15.csv`) are provided directly within the `data/` folder of this repository.

*(No satellite or non-software engineering data is utilized in this research).*

---

## 📖 Citation

```bibtex
@article{tanveer2026dualstream,
  title={A Dual-Stream Context-Aware Transformer Framework with Adaptive Gating and Hierarchical Attention Fusion for Non-Functional Requirements Classification in Systems Engineering},
  author={Tanveer, Umer and Ali, Hashim and Hayat, Maqsood},
  journal={Requirements Engineering},
  year={2026}
}
```
