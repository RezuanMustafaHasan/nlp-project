"""Standalone Training Pipeline for BanglaBERT Punctuation Restoration.

This script trains and evaluates AlignedBanglaBERT for Bangla punctuation
restoration. It is completely self-contained and does not depend on external
internal project files.

Features:
- Automatic parsing of punctuated Bangla text or structured JSONL data.
- Built-in Demo / Synthetic Mode (--demo) for instant demonstration in NLP lab.
- Subword-to-word alignment collator with HuggingFace fast tokenizer.
- Class-imbalanced Cross-Entropy loss with inverse-frequency weighting.
- Full evaluation suite: Macro-F1 (across all 8 punctuation marks), exact match %,
  and per-symbol precision/recall/F1 metrics.
- Learning rate scheduling with linear warmup and cosine decay.
- Interactive test demonstration showing live sentence punctuation restoration.

Usage:
  1. Quick Demo Mode (Toy Dataset):
     python train.py --demo --epochs 3 --batch_size 4

  2. Train on Custom Data:
     python train.py --train_file path/to/train.txt --val_file path/to/val.txt --epochs 5

  3. Restore Punctuation on a Custom Sentence:
     python train.py --predict "আপনি কেমন আছেন আমিও ভালো আছি" --checkpoint best_banglabert_punct.pt
"""

from __future__ import annotations
import argparse
import json
import logging
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import standalone model and vocabulary
from model import (
    ID_TO_PUNCT,
    MAX_SLOTS_PER_GAP,
    NUM_CLASSES,
    PUNCT_TO_ID,
    PUNCTUATION_VOCAB,
    AlignedBanglaBERTPunctuation,
    word_alignment,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger("TrainBanglaBERT")


# ---------------------------------------------------------------------------
# Synthetic / Toy Dataset for Immediate Demonstration
# ---------------------------------------------------------------------------
DEMO_TRAIN_SENTENCES = [
    "বাংলাদেশ একটি সুন্দর দেশ। এখানে অনেক নদী আছে।",
    "আপনি কেমন আছেন? আমি ভালো আছি।",
    "কী চমৎকার দৃশ্য! আমাদের এই দৃশ্যটি খুব ভালো লেগেছে।",
    "তিনি বললেন, আমি আজ যাব না।",
    "ঢাকা, চট্টগ্রাম, এবং সিলেট প্রধান শহর।",
    "তুমি কি বই পড়তে ভালোবাসো? হ্যাঁ, আমি খুব ভালোবাসি।",
    "সাবাশ! তোমরা খেলায় অনেক ভালো করেছ।",
    "কাজী নজরুল ইসলাম (জাতীয় কবি) বিদ্রোহী কবিতা লিখেছিলেন।",
    "তিনি বললেন: আমাদের সততা বজায় রাখা উচিত।",
    "যে পরিশ্রম করে; সে জীবনে সফল হয়।",
    "মেঘ করেছে, বৃষ্টি হতে পারে। তাই ছাতা সঙ্গে নাও।",
    "তুমি কখন আসবে? আমরা তোমার জন্য অপেক্ষা করছি।",
    "আহা! কী সুন্দর গান গাইছে পাখিটি।",
    "রবীন্দ্রনাথ ঠাকুর (নোবেল বিজয়ী) গীতাঞ্জলি রচনা করেছিলেন।",
    "বাংলা ভাষা আমাদের অহংকার; এটি আমাদের মাতৃভাষা।",
    "তিনি বললেন: সত্য সর্বদা সুন্দর।",
    "তুমি কি চা খাবে, নাকি কফি পছন্দ করবে?",
    "কী দারুণ খবর! আমরা সবাই আনন্দিত।",
    "পদ্মা, মেঘনা, এবং যমুনা আমাদের প্রধান নদী।",
    "মন দিয়ে পড়াশোনা করো; তবেই পরীক্ষায় প্রথম হবে।",
]

DEMO_VAL_SENTENCES = [
    "আজকের আবহাওয়া খুব সুন্দর। আমরা কি ঘুরতে যেতে পারি?",
    "শাবাশ! তুমি পরীক্ষায় দারুণ ফল করেছ।",
    "বঙ্গবন্ধু শেখ মুজিবুর রহমান (জাতির পিতা) স্বাধীনতার ডাক দিয়েছিলেন।",
    "তিনি বললেন: সময় খুব মূল্যবান।",
    "বইমেলা শুরু হয়েছে, আমরা সেখানে যাব।",
]


# ---------------------------------------------------------------------------
# Data Preprocessing & Dataset
# ---------------------------------------------------------------------------
# Regex to isolate supported punctuation symbols
PUNCT_CHARS = set(PUNCTUATION_VOCAB[1:])  # {'।', ',', '?', '!', ';', ':', '(', ')'}


def parse_punctuated_text(text: str) -> Optional[Dict[str, any]]:
    """Parse a punctuated Bangla string into words and gap punctuation labels."""
    text = text.strip()
    if not text:
        return None

    # Tokenize words and punctuation while preserving boundaries
    # Standard gap format: for N words, there are N + 1 gaps
    tokens = re.findall(r"[\w\u0980-\u09FF]+|[।,\?!;:()]", text)
    if not tokens:
        return None

    words = []
    # Initialize gaps: gaps[0] is before word 0, gaps[i] is after word i-1
    gaps: List[List[str]] = [[]]

    for tok in tokens:
        if tok in PUNCT_CHARS:
            # Append punctuation to the most recent gap
            if len(gaps[-1]) < MAX_SLOTS_PER_GAP:
                gaps[-1].append(tok)
        else:
            words.append(tok)
            gaps.append([])  # New gap after this word

    if not words:
        return None

    # Map gap symbols to class IDs (pad unused slots with <STOP>)
    gap_label_ids = []
    for gap in gaps:
        slot_ids = [PUNCT_TO_ID[sym] for sym in gap]
        # Pad remaining slots with STOP (ID 0)
        while len(slot_ids) < MAX_SLOTS_PER_GAP:
            slot_ids.append(PUNCT_TO_ID["<STOP>"])
        gap_label_ids.append(slot_ids[:MAX_SLOTS_PER_GAP])

    return {
        "words": words,
        "gap_labels": gap_label_ids,  # Shape: [len(words) + 1, MAX_SLOTS_PER_GAP]
        "length": len(words),
    }


class BanglaPunctuationDataset(Dataset):
    """Dataset for punctuation restoration supporting text files, JSONL, or lists."""

    def __init__(self, data_source: Union[str, Path, List[str]]):
        self.examples = []

        if isinstance(data_source, (list, tuple)):
            for line in data_source:
                parsed = parse_punctuated_text(line)
                if parsed:
                    self.examples.append(parsed)
        else:
            path = Path(data_source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("{") and line.endswith("}"):
                        try:
                            record = json.loads(line)
                            if "words" in record and "gap_labels" in record:
                                self.examples.append(record)
                                continue
                        except json.JSONDecodeError:
                            pass

                    parsed = parse_punctuated_text(line)
                    if parsed:
                        self.examples.append(parsed)

        LOG.info("Loaded %d valid examples from data source.", len(self.examples))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Collate Function for HuggingFace Fast Tokenizer
# ---------------------------------------------------------------------------
class PunctuationCollator:
    """Collates variable-length word sequences and tokenizes using fast subword alignment."""

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        words_list = [ex["words"] for ex in batch]
        lengths = [len(words) for words in words_list]
        max_words = max(lengths)

        # Tokenize with HuggingFace fast tokenizer
        encoding = self.tokenizer(
            words_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Compute subword to word index mapping
        word_ids = word_alignment(encoding, words_list)

        # Pad gap labels to [batch_size, max_words + 1, MAX_SLOTS_PER_GAP]
        # Use -100 for ignored gaps beyond sentence length
        batch_size = len(batch)
        padded_labels = torch.full(
            (batch_size, max_words + 1, MAX_SLOTS_PER_GAP),
            fill_value=-100,
            dtype=torch.long,
        )

        for i, ex in enumerate(batch):
            num_gaps = len(ex["words"]) + 1
            labels_tensor = torch.tensor(ex["gap_labels"][:num_gaps], dtype=torch.long)
            padded_labels[i, :num_gaps] = labels_tensor

        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "word_ids": word_ids,
            "lengths": lengths,
            "gap_labels": padded_labels,
            "raw_words": words_list,
        }


# ---------------------------------------------------------------------------
# Imbalance Weighting & Loss
# ---------------------------------------------------------------------------
def compute_class_weights(dataset: BanglaPunctuationDataset, exponent: float = 0.25) -> torch.Tensor:
    """Compute smoothed inverse-frequency weights to combat extreme STOP imbalance."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for ex in dataset.examples:
        for gap in ex["gap_labels"]:
            for slot in gap:
                if 0 <= slot < NUM_CLASSES:
                    counts[slot] += 1.0

    counts = np.maximum(counts, 1.0)
    # STOP (index 0) gets fixed weight 1.0
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    max_non_stop = np.max(counts[1:]) if len(counts) > 1 else counts[0]

    for c in range(1, NUM_CLASSES):
        # Smoothed inverse-frequency formula
        weights[c] = float(min(5.0, (max_non_stop / counts[c]) ** exponent))

    LOG.info("Class distribution: %s", {ID_TO_PUNCT[i]: int(counts[i]) for i in range(NUM_CLASSES)})
    LOG.info("Calculated class weights: %s", {ID_TO_PUNCT[i]: round(weights[i], 3) for i in range(NUM_CLASSES)})
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training & Validation Loops
# ---------------------------------------------------------------------------
def evaluate(
    model: AlignedBanglaBERTPunctuation,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on validation split, computing Macro-F1 and Exact Match %."""
    model.eval()
    all_preds = []
    all_targets = []
    exact_match_hits = 0
    total_sentences = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids = batch["word_ids"].to(device)
            lengths = batch["lengths"]
            gap_labels = batch["gap_labels"].to(device)

            outputs = model(input_ids, attention_mask, word_ids, lengths)
            logits = outputs["gap_logits"]  # [B, max_words+1, max_slots, num_classes]
            preds = logits.argmax(dim=-1)   # [B, max_words+1, max_slots]

            for b, length in enumerate(lengths):
                num_gaps = length + 1
                sent_preds = preds[b, :num_gaps].cpu().numpy()
                sent_golds = gap_labels[b, :num_gaps].cpu().numpy()

                # Check sentence exact match across all valid gaps & slots
                if np.array_equal(sent_preds, sent_golds):
                    exact_match_hits += 1
                total_sentences += 1

                for g in range(num_gaps):
                    for s in range(MAX_SLOTS_PER_GAP):
                        gold = sent_golds[g, s]
                        if gold != -100:
                            all_targets.append(gold)
                            all_preds.append(sent_preds[g, s])

    # Calculate Macro-F1 across the 8 punctuation marks (excluding STOP index 0)
    # This aligns strictly with standard academic evaluation protocols
    p_classes = list(range(1, NUM_CLASSES))
    per_class_f1 = f1_score(all_targets, all_preds, labels=p_classes, average=None, zero_division=0)
    macro_f1 = float(np.mean(per_class_f1) * 100.0)
    exact_match_pct = float((exact_match_hits / max(1, total_sentences)) * 100.0)

    # Detailed per-symbol report
    precision, recall, _, _ = precision_recall_fscore_support(
        all_targets, all_preds, labels=p_classes, average=None, zero_division=0
    )
    symbol_metrics = {}
    for i, cid in enumerate(p_classes):
        symbol_metrics[ID_TO_PUNCT[cid]] = {
            "Precision": round(float(precision[i]) * 100.0, 2),
            "Recall": round(float(recall[i]) * 100.0, 2),
            "F1": round(float(per_class_f1[i]) * 100.0, 2),
        }

    return {
        "macro_f1": round(macro_f1, 2),
        "exact_match_pct": round(exact_match_pct, 2),
        "symbol_metrics": symbol_metrics,
    }


def train_model(args):
    # Set random seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    LOG.info("Using compute device: %s", device)

    # 1. Prepare Datasets
    if args.demo:
        LOG.info("Initializing in DEMO MODE with built-in Bangla sentences.")
        train_dataset = BanglaPunctuationDataset(DEMO_TRAIN_SENTENCES)
        val_dataset = BanglaPunctuationDataset(DEMO_VAL_SENTENCES)
    else:
        if not args.train_file or not args.val_file:
            raise ValueError("Must specify --train_file and --val_file, or use --demo flag.")
        train_dataset = BanglaPunctuationDataset(args.train_file)
        val_dataset = BanglaPunctuationDataset(args.val_file)

    # 2. Instantiate Model and Tokenizer
    LOG.info("Instantiating AlignedBanglaBERT with pretrained backbone: %s", args.pretrained_model)
    model = AlignedBanglaBERTPunctuation(
        pretrained_model_name=args.pretrained_model,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)

    param_info = model.count_parameters()
    LOG.info(
        "Model parameters: Total=%d, Trainable=%d (Encoder=%d, Head=%d)",
        param_info["total"],
        param_info["trainable"],
        param_info["encoder"],
        param_info["head"],
    )

    collator = PunctuationCollator(model.tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    # 3. Loss function with optional class weights
    if args.use_class_weights:
        weights = compute_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Optimizer & Warmup Scheduler
    # Separate weight decay for non-bias/LayerNorm parameters
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 5. Training Loop
    best_macro_f1 = -1.0
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_banglabert_punct.pt"

    LOG.info("Starting training for %d epochs (%d total updates)...", args.epochs, total_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids = batch["word_ids"].to(device)
            lengths = batch["lengths"]
            gap_labels = batch["gap_labels"].to(device)

            outputs = model(input_ids, attention_mask, word_ids, lengths)
            logits = outputs["gap_logits"]  # [B, max_words+1, max_slots, num_classes]

            # Flatten logits and labels for Cross-Entropy calculation
            loss = criterion(
                logits.view(-1, NUM_CLASSES),
                gap_labels.view(-1),
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            step_count += 1

        avg_loss = epoch_loss / max(1, step_count)

        # Validation at epoch end
        val_metrics = evaluate(model, val_loader, device)
        macro_f1 = val_metrics["macro_f1"]
        exact_match = val_metrics["exact_match_pct"]

        LOG.info(
            "Epoch %02d/%02d | Train Loss: %.4f | Val Macro-F1: %.2f%% | Val Exact Match: %.2f%%",
            epoch,
            args.epochs,
            avg_loss,
            macro_f1,
            exact_match,
        )

        # Save checkpoint if best Macro-F1
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                best_checkpoint_path,
            )
            LOG.info("--> Saved new best checkpoint to %s", best_checkpoint_path)

    LOG.info("Training complete! Best Validation Macro-F1: %.2f%%", best_macro_f1)

    # 6. Interactive Sample Demonstration
    LOG.info("Running live test prediction demonstration...")
    test_samples = [
        "আপনি কেমন আছেন আমিও ভালো আছি",
        "কী চমৎকার খবর আমরা খেলায় জিতে গেছি",
        "ঢাকা চট্টগ্রাম এবং রাজশাহী বাংলাদেশের প্রধান শহর",
    ]
    for sample in test_samples:
        restored = model.restore_punctuation(sample, device=device)
        print(f"\n[Raw Input]   : {sample}")
        print(f"[Restored]    : {restored}")


def main():
    parser = argparse.ArgumentParser(description="Aligned BanglaBERT Punctuation Restoration Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with built-in toy dataset")
    parser.add_argument("--train_file", type=str, default=None, help="Path to training text or JSONL file")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation text or JSONL file")
    parser.add_argument("--pretrained_model", type=str, default="csebuetnlp/banglabert")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--projection_dim", type=int, default=384)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true", help="Apply class-imbalance weights")
    parser.add_argument("--predict", type=str, default=None, help="Restore punctuation on input text and exit")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for prediction")

    args = parser.parse_args()

    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        model = AlignedBanglaBERTPunctuation(pretrained_model_name=args.pretrained_model).to(device)
        if args.checkpoint and Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            LOG.info("Loaded weights from %s", args.checkpoint)
        result = model.restore_punctuation(args.predict, device=device)
        print("\nOriginal :", args.predict)
        print("Restored :", result)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
