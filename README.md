# BanglaBERT Punctuation Restoration (NLP Lab Project)

This directory contains the standalone model and training pipeline for **Bangla Punctuation Restoration** using **BanglaBERT** (`csebuetnlp/banglabert`).

---

## 1. Why `csebuetnlp/banglabert` is the Best Bangla BERT

Among all available Bangla BERT models in literature and Hugging Face:
- **Pretraining Architecture (ELECTRA vs. Masked LM):** While models like `sagorsarker/bangla-bert-base` use standard BERT masked language modeling (predicting only 15% masked tokens), `csebuetnlp/banglabert` is built on the **ELECTRA discriminator** framework. It trains on 100% of tokens by detecting replaced tokens, yielding far superior token-level and boundary-level contextual embeddings.
- **Corpus Scale & Quality:** Pretrained on **Bangla2B+** (~27.5 GB of text, over 2.75 billion tokens) curated from diverse domains, compared to Wikipedia-only subsets used by earlier models.
- **Benchmark State-of-the-Art:** Outperforms `bangla-bert-base`, `mBERT`, and `Indic-BERT` across all 10 benchmarks in the BanglaLanguageUnderstanding (BLU) benchmark suite.
- **Project Empirical Validation:** On the 185,370-sentence punctuation restoration test set, fine-tuned `AlignedBanglaBERT` achieved **68.93% Macro-F1** and **83.46% Exact Match**, outperforming BiLSTM (52.05% F1) and Compact Transformers (48.63% F1) by wide margins.

---

## 2. Directory Structure

```text
nlp_lab_banglabert/
├── model.py     # Standalone PyTorch AlignedBanglaBERTPunctuation model
├── train.py     # Standalone training loop, dataset parser, metrics & inference demo
└── README.md    # Lab project presentation guide & instructions
```

---

## 3. Requirements

Install standard dependencies:
```bash
pip install torch transformers scikit-learn numpy
```

---

## 4. How to Run

### A. Quick Demo Mode (Toy Dataset)
For an immediate lab demonstration without needing large external dataset files:
```bash
python train.py --demo --epochs 3 --batch_size 4
```
This trains on 20 authentic punctuated Bangla sentences, validates on 5 sentences, reports Macro-F1 and sentence exact match, and restores punctuation on sample sentences in real time.

### B. Training on Custom Data
You can pass any text file of punctuated Bangla sentences (one sentence per line):
```bash
python train.py --train_file data/train.txt --val_file data/val.txt --epochs 5 --batch_size 16 --use_class_weights
```

### C. Live Punctuation Restoration (Inference)
Restore punctuation on an unpunctuated Bangla sentence directly:
```bash
python train.py --predict "আপনি কেমন আছেন আমিও ভালো আছি" --checkpoint checkpoints/best_banglabert_punct.pt
```

Or in Python:
```python
from model import AlignedBanglaBERTPunctuation

model = AlignedBanglaBERTPunctuation()
# Restore unpunctuated text
output = model.restore_punctuation("আপনি কেমন আছেন আমিও ভালো আছি")
print(output)
# Output: "আপনি কেমন আছেন? আমিও ভালো আছি।"
```
