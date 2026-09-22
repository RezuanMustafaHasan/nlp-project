# Model Report Dossier: BanglaBERT vs. Packed BiLSTM for Bangla Punctuation Restoration

This directory contains the complete curated dossier of source code, empirical metrics, hyperparameter configurations, and scientific manuscript context necessary for an AI agent or researcher to write an exhaustive, publication-grade comparative report on the two leading punctuation restoration models:

1. **Aligned BanglaBERT** (Pretrained Contextual ELECTRA Transformer — **Best Overall Model**)
2. **Packed fastText–BiLSTM** (Recurrent Neural Network — **Best Lightweight/Recurrent Model**)

---

## 1. Directory Structure & File Manifest

```text
report_materials/
├── README.md                                  # This guide & executive data summary
│
├── models/                                    # Standalone PyTorch Source Code
│   ├── banglabert_model.py                    # Complete AlignedBanglaBERTPunctuation architecture
│   ├── banglabert_train.py                    # BanglaBERT training loop, collator, and evaluation
│   ├── banglabert_guide.md                    # Technical documentation for BanglaBERT
│   ├── bilstm_model.py                        # Complete PackedBiLSTMPunctuation architecture
│   ├── bilstm_train.py                        # BiLSTM training loop, collator, and evaluation
│   └── bilstm_guide.md                        # Technical documentation for BiLSTM
│
├── metrics_and_benchmarks/                    # Empirical Logs, Metrics & Benchmark Outputs
│   ├── banglabert_validation_metrics.json     # Detailed validation results (F1, exact match, per-symbol)
│   ├── banglabert_training_config.json        # Hyperparameters (lr 1e-5, epochs 3, batch 32, weight_power 0.5)
│   ├── banglabert_training_history.json       # Epoch-by-epoch validation progression
│   ├── bilstm_validation_metrics.json         # Detailed validation results (F1, exact match, per-symbol)
│   ├── bilstm_training_config.json            # Hyperparameters (lr 1e-4, epochs 4, batch 64, weight_power 0.5)
│   ├── bilstm_training_history.json           # Epoch-by-epoch validation progression
│   ├── test_benchmark_report.md               # 185,370-sentence test set comparative benchmark report
│   └── test_rows.tex                          # LaTeX formatted benchmark table rows
│
└── research_context/                          # Academic Paper & Scientific Background
    ├── paper_manuscript.tex                   # Full ICCIT 2026 conference paper manuscript (LaTeX)
    └── citations.bib                          # Full BibTeX bibliographic database
```

---

## 2. Executive Model Comparison

| Evaluation Metric / Property | **Aligned BanglaBERT** 🏆 | **Packed fastText–BiLSTM** ⚡ | Difference (&Delta;) |
| :--- | :---: | :---: | :---: |
| **Model Family** | Pretrained ELECTRA Transformer | 2-Layer Bidirectional LSTM | Pretrained vs. Recurrent |
| **Backbone** | `csebuetnlp/banglabert` | Frozen 300d fastText | Contextual vs. Static |
| **Trainable Parameters** | ~110.4 Million | ~2.2 Million | ~50&times; smaller |
| **Test Macro-F1 (%)** | **68.93** | **52.05** | **+16.88%** |
| **Test Exact Match (%)** | **83.46** | **77.93** | **+5.53%** |
| **Validation Macro-F1 (%)** | **68.58** | **50.32** | **+18.26%** |
| **Validation Exact Match (%)**| **83.74** | **77.81** | **+5.93%** |
| **Parenthesis Span F1 (%)** | **74.89** | **43.01** | **+31.88%** |
| **False Parentheses (FP / 1k neg)**| **1.69** | **9.09** | **-7.40** (far fewer false marks) |
| **Target Checkpoint (.pt)** | `banglabert_aligned_weighted_s42/best.pt` | `bilstm_packed_weighted_s42/best.pt` | ~422 MB vs ~8.4 MB |

---

## 3. Per-Symbol Detailed Breakdown (Validation Set)

| Punctuation Symbol | Name / Meaning | BanglaBERT Precision (%) | BanglaBERT Recall (%) | **BanglaBERT F1 (%)** | BiLSTM Precision (%) | BiLSTM Recall (%) | **BiLSTM F1 (%)** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `।` | Dari (Full Stop) | 98.77 | 99.13 | **98.95** | 98.64 | 98.60 | **98.62** |
| `,` | Comma | 84.73 | 84.22 | **84.47** | 79.81 | 75.00 | **77.33** |
| `?` | Question Mark | 93.52 | 95.00 | **94.25** | 84.29 | 91.98 | **87.97** |
| `!` | Exclamation Mark | 39.17 | 19.63 | **26.15** | 17.58 | 6.70 | **9.70** |
| `;` | Semicolon | 32.26 | 22.99 | **26.85** | 7.84 | 4.60 | **5.80** |
| `:` | Colon | 65.27 | 69.51 | **67.32** | 34.05 | 45.12 | **38.81** |
| `(` | Open Parenthesis | 84.03 | 69.03 | **75.80** | 38.73 | 43.23 | **40.85** |
| `)` | Close Parenthesis | 82.05 | 68.82 | **74.85** | 41.49 | 45.59 | **43.44** |

---

## 4. Key Architectural Innovations

### A. Aligned BanglaBERT (`models/banglabert_model.py`)
1. **Contextual ELECTRA Discriminator:** Utilizes `csebuetnlp/banglabert`, pretrained on Bangla2B+ (~2.75 billion tokens). The discriminator architecture trains on 100% of tokens rather than the 15% masked in standard BERT.
2. **Subword-to-Word Scatter Pooling:** HuggingFace fast tokenizer tokenizes words into subwords. The model maps subwords back to word indices using `encoding.word_ids` and performs mean scatter pooling across subwords, generating true word-level contextual vectors.
3. **Framed BOS/EOS Embeddings:** Learned vectors represent sequence start and end immediately after the actual sequence length, preventing padding leakage.
4. **Adjacent Context Gap Projection:** Concatenates $[h_i \parallel h_{i+1}]$ ($768 \times 2 = 1536$ dims) for adjacent words, followed by `Linear(1536 -> 384) + GELU + LayerNorm(384)`.
5. **Multi-Slot Classification Head:** Maps $384 \to 4 \times 9 = 36$ logits, predicting up to 4 sequential punctuation marks at any boundary gap across 9 vocabulary tokens (`<STOP>` + 8 punctuation marks).

### B. Packed fastText–BiLSTM (`models/bilstm_model.py`)
1. **Sequential Inductive Bias:** Unlike un-pretrained Transformers trained from scratch, the recurrent inductive bias of BiLSTM excels at local boundary classification on short sentences, beating the scratch 6-layer Compact Transformer by +3.42% Macro-F1 (52.05% vs 48.63%).
2. **FastText Static Embeddings:** Maps words to frozen 300-dimensional fastText subword-aware vectors, projected via `Linear(300 -> 384)`.
3. **Variable-Length Sequence Packing:** Uses PyTorch `pack_padded_sequence` and `pad_packed_sequence` over $[L + 2]$ lengths (including learned BOS/EOS), ensuring zero padding never pollutes internal hidden or cell states.
4. **Adjacent Context Gap Concatenation:** Concatenates $[h_i \parallel h_{i+1}]$ ($384 \times 2 = 768$ dims) and projects to 384 dimensions with GELU and LayerNorm.
5. **Multi-Slot Classification Head:** Identical 4-slot gap head ($384 \to 36$) for direct comparability.

---

## 5. Recommended Outline for the Writing Agent

When prompt-instructing another agent to write the report, recommend structuring it as follows:

1. **Title & Abstract:** Concisely summarize the objective (Bangla punctuation restoration), models compared, test set size (185,370 sentences), and core findings (BanglaBERT 68.93% F1 vs BiLSTM 52.05% F1).
2. **Problem Formulation & Annotation Scheme:** Inter-word gap tagging, 4 slots per gap, 9-class vocabulary, class imbalance challenge.
3. **Model 1: Aligned BanglaBERT Architecture & Training:** ELECTRA pretraining, scatter mean-pooling, loss function, hyperparameters.
4. **Model 2: Packed fastText–BiLSTM Architecture & Training:** Static 300d embeddings, sequence packing, recurrent inductive bias, loss function, hyperparameters.
5. **Comparative Experimental Results:**
   - Overall test and validation Macro-F1 and sentence exact match.
   - Per-symbol performance comparison table and analysis (highlight why rare marks like `;`, `!`, and `:` benefit drastically from pretrained contextual semantics).
   - Structural punctuation (parentheses) analysis (span F1 and false-positive rates).
6. **Efficiency vs. Performance Trade-off:** Parameter count (110M vs 2.2M), inference speed/latency, deployment scenarios (server/cloud GPU vs mobile/edge CPU).
7. **Conclusion & Future Directions:** Summary and practical recommendations.
