"""Training-Only Pipeline for BanglaBERT Punctuation Restoration.

This script contains exclusively the training routine for AlignedBanglaBERT:
- No logging module
- No validation loop
- No F1, precision, recall, or exact match calculation
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import standalone model and vocabulary from model.py
from model import (
    ID_TO_PUNCT,
    MAX_SLOTS_PER_GAP,
    NUM_CLASSES,
    PUNCT_TO_ID,
    PUNCTUATION_VOCAB,
    AlignedBanglaBERTPunctuation,
    word_alignment,
)

# ---------------------------------------------------------------------------
# Synthetic / Toy Dataset for Quick Demo / Fallback
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

PUNCT_CHARS = set(PUNCTUATION_VOCAB[1:])


# ---------------------------------------------------------------------------
# Data Preprocessing & Dataset
# ---------------------------------------------------------------------------
def parse_punctuated_text(text: str) -> Optional[Dict[str, any]]:
    """Parse a punctuated Bangla string into words and gap punctuation labels."""
    text = text.strip()
    if not text:
        return None

    tokens = re.findall(r"[\w\u0980-\u09FF]+|[।,\?!;:()]", text)
    if not tokens:
        return None

    words = []
    gaps: List[List[str]] = [[]]

    for tok in tokens:
        if tok in PUNCT_CHARS:
            if len(gaps[-1]) < MAX_SLOTS_PER_GAP:
                gaps[-1].append(tok)
        else:
            words.append(tok)
            gaps.append([])

    if not words:
        return None

    gap_label_ids = []
    for gap in gaps:
        slot_ids = [PUNCT_TO_ID[sym] for sym in gap]
        while len(slot_ids) < MAX_SLOTS_PER_GAP:
            slot_ids.append(PUNCT_TO_ID["<STOP>"])
        gap_label_ids.append(slot_ids[:MAX_SLOTS_PER_GAP])

    return {
        "words": words,
        "gap_labels": gap_label_ids,
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

        encoding = self.tokenizer(
            words_list,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        word_ids = word_alignment(encoding, words_list)

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
        }


def compute_class_weights(dataset: BanglaPunctuationDataset, exponent: float = 0.25) -> torch.Tensor:
    """Compute smoothed inverse-frequency weights to combat extreme STOP imbalance."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for ex in dataset.examples:
        for gap in ex["gap_labels"]:
            for slot in gap:
                if 0 <= slot < NUM_CLASSES:
                    counts[slot] += 1.0

    counts = np.maximum(counts, 1.0)
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    max_non_stop = np.max(counts[1:]) if len(counts) > 1 else counts[0]

    for c in range(1, NUM_CLASSES):
        weights[c] = float(min(5.0, (max_non_stop / counts[c]) ** exponent))

    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------
def train_model(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    # Prepare training dataset
    if args.train_file:
        train_dataset = BanglaPunctuationDataset(args.train_file)
    else:
        train_dataset = BanglaPunctuationDataset(DEMO_TRAIN_SENTENCES)
    print(f"Loaded {len(train_dataset)} training examples.")

    # Initialize model
    model = AlignedBanglaBERTPunctuation(
        pretrained_model_name=args.pretrained_model,
        projection_dim=args.projection_dim,
        dropout=args.dropout,
    ).to(device)

    collator = PunctuationCollator(model.tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    # Loss function
    if args.use_class_weights:
        weights = compute_class_weights(train_dataset).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Optimizer & Scheduler
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

    # Training Loop
    print(f"Starting training for {args.epochs} epochs ({total_steps} total steps)...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            word_ids = batch["word_ids"].to(device)
            lengths = batch["lengths"]
            gap_labels = batch["gap_labels"].to(device)

            outputs = model(input_ids, attention_mask, word_ids, lengths)
            logits = outputs["gap_logits"]

            loss = criterion(
                logits.view(-1, NUM_CLASSES),
                gap_labels.view(-1),
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {avg_loss:.4f}")

    # Save final model
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = Path(args.output_dir) / "banglabert_punct.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
        },
        checkpoint_path,
    )
    print(f"Training complete. Model checkpoint saved to: {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="BanglaBERT Punctuation Restoration (Training Only)")
    parser.add_argument("--train_file", type=str, default=None, help="Path to training text or JSONL file")
    parser.add_argument("--demo", action="store_true", help="Run with built-in toy dataset")
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

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
