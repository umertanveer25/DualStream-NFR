# Replication Package: Authentic Baseline Re-Execution Suite

This directory contains the independent, reproducible replication package for all baseline models evaluated in the paper:
**"A Dual-Stream Context-Aware Transformer Framework with Adaptive Gating and Hierarchical Attention Fusion for Non-Functional Requirements Classification in Systems Engineering"**.

Each baseline was re-executed under identical dataset splits and evaluation protocols across all 16 CPU cores using authentic requirements engineering datasets (`B:\Dataset\nfr.csv` and `Promise_Exp.csv`).

---

## 📁 Baseline Directory Structure

| Directory | Baseline Paper | Venue / Year | Core Method / Architecture | Dataset | Execution Script | Results Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| [`01_Canedo_Mendes_2020_Entropy/`](./01_Canedo_Mendes_2020_Entropy/) | Canedo & Mendes | *Entropy* (MDPI), 2020 | TF-IDF + 9 Classical Classifiers (SVM, MLP, RF, ET, etc.) | `nfr.csv` (625 reqs) | `replicate_canedo2020.py` | `canedo2020_results.csv`, `.json` |
| [`02_Kaur_Kaur_2023_Elsevier/`](./02_Kaur_Kaur_2023_Elsevier/) | Kaur & Kaur | *Comput. Electr. Eng.* (Elsevier), 2023 | BERT + BiLSTM(128x2) + Multi-Kernel Conv1d (2,3,4) | `nfr.csv` (625 reqs) | `replicate_kaur2023.py` | `kaur2023_results.csv`, `.json` |
| [`03_Devlin_Ayed_2024_COMPSAC/`](./03_Devlin_Ayed_2024_COMPSAC/) | Ayed & Alrammal | *IEEE COMPSAC*, 2024 | BERT [CLS] Embedding + Linear / MLP & 8 ML Classifiers | `Promise_Exp.csv` (969 reqs) | `replicate_ayed2024.py` | `ayed2024_results.csv`, `.json` |
| [`05_Hey_NoRBERT_2020_IEEERE/`](./05_Hey_NoRBERT_2020_IEEERE/) | Hey et al. (NoRBERT) | *IEEE RE*, 2020 | Domain-Adapted BERT + Deep Classification Head & ML Suite | `nfr.csv` (625 reqs) | `replicate_norbert2020.py` | `norbert2020_nfr_results.csv`, `.json` |
| [`06_Luo_PRCBERT_2022_ASE/`](./06_Luo_PRCBERT_2022_ASE/) | Luo et al. (PRCBERT) | *IEEE/ACM ASE*, 2022 | Prompt-Tuning via Masked LM (`"[X]. This is a [MASK] requirement."`) | `nfr.csv` (625 reqs) | `replicate_prcbert2022.py` | `prcbert2022_results.csv`, `.json` |
| [`07_Almohammady_ReDistill_2026_Electronics/`](./07_Almohammady_ReDistill_2026_Electronics/) | Almohammady et al. | *Electronics* (MDPI), 2026 | Single-Stream Contextual Concatenation (`[CLS] FR [SEP] NFR [SEP]`) | `nfr_all_with_context.csv` (208 reqs) | `replicate_redistill2026.py` | `redistill2026_results.csv`, `.json` |

*(Note: Paper 4 / Liu et al., 2019 RoBERTa was excluded to focus strictly on peer-reviewed Software Engineering papers).*

---

## ⚡ Quick Replication Commands

To re-run any baseline individually, execute the respective replication script:

```bash
# Paper 1: Canedo & Mendes (Entropy 2020)
python baselines/01_Canedo_Mendes_2020_Entropy/replicate_canedo2020.py

# Paper 2: Kaur & Kaur (Elsevier 2023)
python baselines/02_Kaur_Kaur_2023_Elsevier/replicate_kaur2023.py

# Paper 3: Ayed & Alrammal (IEEE COMPSAC 2024)
python baselines/03_Devlin_Ayed_2024_COMPSAC/replicate_ayed2024.py

# Paper 5: Hey et al. - NoRBERT (IEEE RE 2020)
python baselines/05_Hey_NoRBERT_2020_IEEERE/replicate_norbert2020.py

# Paper 6: Luo et al. - PRCBERT (IEEE/ACM ASE 2022)
python baselines/06_Luo_PRCBERT_2022_ASE/replicate_prcbert2022.py

# Paper 7: Almohammady et al. - Single-Stream Base SOTA (Electronics 2026)
python baselines/07_Almohammady_ReDistill_2026_Electronics/replicate_redistill2026.py
```

---

## 🔬 Head-to-Head Side-by-Side Comparison

To run the complete side-by-side benchmark comparing the Single-Stream baseline against the Proposed Dual-Stream architecture across all algorithms and evaluation protocols:

```bash
# Run side-by-side Single-Stream vs. Dual-Stream across all algorithms
python scripts/comprehensive_side_by_side_all_techniques.py
```
