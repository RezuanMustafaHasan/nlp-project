# Packed fastText–BiLSTM Punctuation Restoration (NLP Lab Project)

This directory contains the standalone model and training pipeline for **Bangla Punctuation Restoration** using the **Packed fastText–BiLSTM** architecture.

---

## 1. Why the BiLSTM Model is Significant

In our empirical research on the 185,370-sentence benchmark test set (ICCIT 2026):
- **Outperforms Scratch Compact Transformers:** Despite having only ~2.2M parameters compared to the 11.2M-parameter Compact Transformer, the 2-layer BiLSTM achieved **52.05% Macro-F1** and **77.93% Exact Match**, outperforming the Compact Transformer (48.63% Macro-F1) by **+3.42 Macro-F1 points**.
- **Inductive Bias for Sequential Boundaries:** Punctuation decisions are inherently local sequential boundary decisions. Recurrent bidirectional gates naturally enforce sequential dependencies without requiring positional encodings.
- **Sequence Packing Optimization:** By using PyTorch's `pack_padded_sequence`, the model ignores padding tokens and places the learned `[EOS]` boundary token immediately after the final word of each sentence, preventing boundary corruption.
- **Adjacent Context Gap Representation:** For every boundary gap between adjacent words $w_i$ and $w_{i+1}$, the model concatenates the left and right recurrent states $[h_i \parallel h_{i+1}]$ (768 dimensions), projects to 384 dimensions, and classifies up to 4 consecutive punctuation marks across 9 classes.

---

## 2. Directory Structure

```text
nlp_lab_bilstm/
├── model.py         # Standalone PyTorch PackedBiLSTMPunctuation model
├── train.py         # Training pipeline, demo mode, metrics & live inference
├── only_train.py    # Minimalist training script without logging or evaluation
└── README.md        # Lab project presentation guide & instructions
```

---

## 3. Pretrained Model Checkpoint (.pt)

The project's best-performing BiLSTM model checkpoint is located at:
```text
bangla-punctuation/results/post79_improvement/run_24h/bilstm_packed_weighted_s42/best.pt
```

* **Test Macro-F1**: **52.05%** (highest among all recurrent models)
* **Test Exact Match**: **77.93%**
* **Validation Macro-F1**: 50.32%

---

## 4. Requirements

Install standard dependencies:
```bash
pip install torch transformers scikit-learn numpy
```

---

## 5. How to Run

### A. Quick Demo Mode (Toy Dataset)
For an immediate lab demonstration without needing external dataset files or large embeddings:
```bash
python train.py --demo --epochs 3 --batch_size 4
```
This runs training on 20 authentic punctuated Bangla sentences, evaluates validation Macro-F1 and exact match, and restores punctuation on live sample sentences.

### B. Training on Custom Data
Run full training on your own punctuated Bangla text file (one sentence per line):
```bash
python train.py --train_file data/train.txt --val_file data/val.txt --epochs 5 --batch_size 16 --use_class_weights
```

Or using the minimalist script:
```bash
python only_train.py --train_file data/train.txt --epochs 3 --batch_size 16
```

### C. Live Punctuation Restoration (Inference with Pretrained .pt)
Restore punctuation on an unpunctuated Bangla sentence directly:
```bash
python train.py --predict "আপনি কেমন আছেন আমিও ভালো আছি" \
  --checkpoint "../bangla-punctuation/results/post79_improvement/run_24h/bilstm_packed_weighted_s42/best.pt" \
  --word2id "../bangla-punctuation/embeddings/cache/word2id.json" \
  --vocab_vectors "../bangla-punctuation/embeddings/cache/vocab_vectors.npy"
```

### D. Using in Python
```python
import torch
from model import PackedBiLSTMPunctuation

model = PackedBiLSTMPunctuation()

# Load best checkpoint
ckpt_path = "../bangla-punctuation/results/post79_improvement/run_24h/bilstm_packed_weighted_s42/best.pt"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
model.load_state_dict(state)

# Load fastText embedding cache (automatically locates ../bangla-punctuation/embeddings/cache)
model.load_embeddings()

# Restore unpunctuated text
output = model.restore_punctuation("আপনি কেমন আছেন আমিও ভালো আছি")
print("Restored:", output)
# Output: "আপনি কেমন আছেন? আমিও ভালো আছি।"
```
