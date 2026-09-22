"""Standalone Training Pipeline for Packed BiLSTM Punctuation Restoration.

This script trains and evaluates PackedBiLSTMPunctuation for Bangla punctuation
restoration. It is completely self-contained and mirrors the nlp_lab_banglabert
pipeline.

Features:
- Automatic parsing of punctuated Bangla text or structured JSONL data.
- Built-in Demo Mode (--demo) for instant demonstration in NLP lab.
- Automatic fastText embedding cache loader (or lightweight on-the-fly fallback).
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
     python train.py --predict "আপনি কেমন আছেন আমিও ভালো আছি" --checkpoint best_bilstm_punct.pt
"""

from __future__ import annotations
import argparse
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

# Import standalone model and vocabulary
from model import (
    ID_TO_PUNCT,
    MAX_SLOTS_PER_GAP,
    NUM_CLASSES,
    PUNCT_TO_ID,
    PUNCTUATION_VOCAB,
    PackedBiLSTMPunctuation,
)

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
LOG = logging.getLogger("TrainBiLSTM")


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


PUNCT_CHARS = set(PUNCTUATION_VOCAB[1:])


def parse_punctuated_text(text: str) -> Optional[Dict[str, any]]:
    """Parse a punctuated Bangla string into words and gap punctuation labels."""
    tokens = re.findall(r"[\w\u0980-\u09FF]+|[।,\?!;:()]", text.strip())
    if not tokens:
        return None

    words: List[str] = []
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

    gap_labels: List[List[int]] = []
    for gap in gaps:
        slot_ids = [PUNCT_TO_ID[sym] for sym in gap]
        while len(slot_ids) < MAX_SLOTS_PER_GAP:
            slot_ids.append(PUNCT_TO_ID["<STOP>"])
        gap_labels.append(slot_ids[:MAX_SLOTS_PER_GAP])

    return {
        "words": words,
        "gap_labels": gap_labels,
        "length": len(words),
    }


class BanglaPunctuationDataset(Dataset):
    """Loads sentences from a list of strings, a text file, or a JSONL file."""

    def __init__(self, data_source: Union[List[str], str, Path]):
        self.examples: List[Dict[str, any]] = []

        if isinstance(data_source, (list, tuple)):
            for sent in data_source:
                parsed = parse_punctuated_text(sent)
                if parsed:
                    self.examples.append(parsed)
        else:
            file_path = Path(data_source)
            if not file_path.exists():
                raise FileNotFoundError(f"Data file not found: {file_path}")

            with file_path.open("r", encoding="utf-8") as f:
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

        if not self.examples:
            raise ValueError(f"No valid examples parsed from {data_source}")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, any]:
        return self.examples[idx]


class EmbeddingLookup:
    """Helper for converting words to 300d embeddings."""

    def __init__(
        self,
        embed_dim: int = 300,
        word2id_path: Optional[str | Path] = None,
        vectors_path: Optional[str | Path] = None,
    ):
        self.embed_dim = embed_dim
        self.word2id: Dict[str, int] = {}
        self.vectors: Optional[np.ndarray] = None
        self._fallback_cache: Dict[str, np.ndarray] = {}

        # Search default locations
        search_dirs = [
            Path(__file__).resolve().parent.parent / "bangla-punctuation" / "embeddings" / "cache",
            Path(__file__).resolve().parent / "embeddings",
        ]

        w_p = Path(word2id_path) if word2id_path else None
        v_p = Path(vectors_path) if vectors_path else None

        if w_p is None or v_p is None:
            for s_dir in search_dirs:
                cand_w = s_dir / "word2id.json"
                cand_v = s_dir / "vocab_vectors.npy"
                if cand_w.is_file() and cand_v.is_file():
                    w_p, v_p = cand_w, cand_v
                    break

        if w_p and v_p and w_p.is_file() and v_p.is_file():
            LOG.info("Loading fastText embedding cache from %s ...", w_p.parent)
            with open(w_p, "r", encoding="utf-8") as f:
                self.word2id = json.load(f)
            self.vectors = np.load(v_p, mmap_mode="r")
            LOG.info("Loaded fastText cache: %d words, shape %s", len(self.word2id), self.vectors.shape)
        else:
            LOG.warning(
                "fastText embedding cache not found. Using deterministic pseudo-embeddings for words."
            )

    def get_vector(self, word: str) -> np.ndarray:
        if self.vectors is not None and self.word2id:
            idx = self.word2id.get(word, -1)
            if 0 <= idx < len(self.vectors):
                return self.vectors[idx]

        if word not in self._fallback_cache:
            # Deterministic pseudo-embedding based on sha256 hash of word
            seed = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(seed)
            self._fallback_cache[word] = rng.normal(0.0, 0.1, size=self.embed_dim).astype(np.float32)

        return self._fallback_cache[word]


class BiLSTMPunctuationCollator:
    """Collates variable-length word sequences into padded vector tensors and gap labels."""

    def __init__(self, embed_lookup: EmbeddingLookup):
        self.lookup = embed_lookup

    def __call__(self, batch: List[Dict[str, any]]) -> Dict[str, torch.Tensor]:
        words_list = [ex["words"] for ex in batch]
        lengths = [len(words) for words in words_list]
        max_words = max(lengths)
        batch_size = len(batch)

        word_vectors = torch.zeros(batch_size, max_words, self.lookup.embed_dim, dtype=torch.float32)
        word_mask = torch.zeros(batch_size, max_words, dtype=torch.bool)
        padded_labels = torch.full(
            (batch_size, max_words + 1, MAX_SLOTS_PER_GAP), fill_value=-100, dtype=torch.long
        )

        for i, ex in enumerate(batch):
            words = ex["words"]
            n_w = len(words)
            for j, w in enumerate(words):
                word_vectors[i, j] = torch.from_numpy(np.array(self.lookup.get_vector(w), copy=True))
            word_mask[i, :n_w] = True
            padded_labels[i, : n_w + 1] = torch.tensor(ex["gap_labels"], dtype=torch.long)

        return {
            "word_vectors": word_vectors,
            "word_mask": word_mask,
            "lengths": lengths,
            "gap_labels": padded_labels,
        }


def compute_class_weights(dataset: BanglaPunctuationDataset, power: float = 0.5, cap: float = 6.0) -> torch.Tensor:
    """Calculate inverse-frequency class weights capped at specified factor."""
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for ex in dataset.examples:
        for gap in ex["gap_labels"]:
            for slot in gap:
                if slot != -100:
                    counts[slot] += 1

    total_valid = np.sum(counts)
    if total_valid == 0:
        return torch.ones(NUM_CLASSES, dtype=torch.float32)

    weights = (total_valid / np.maximum(counts, 1).astype(np.float32)) ** power
    weights = weights / np.mean(weights)
    weights = np.clip(weights, 0.1, cap)
    weights[0] = 1.0  # STOP class normal weight
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(
    model: PackedBiLSTMPunctuation,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, any]:
    """Comprehensive evaluation computing Macro-F1 across 8 marks and Exact Match %."""
    model.eval()
    all_targets: List[int] = []
    all_preds: List[int] = []
    exact_matches = 0
    total_sentences = 0

    with torch.no_grad():
        for batch in dataloader:
            word_vectors = batch["word_vectors"].to(device)
            word_mask = batch["word_mask"].to(device)
            lengths = batch["lengths"]
            gap_labels = batch["gap_labels"].to(device)

            outputs = model(word_vectors=word_vectors, word_mask=word_mask, real_lengths=lengths)
            preds = outputs["gap_logits"].argmax(dim=-1)

            for b, length in enumerate(lengths):
                total_sentences += 1
                n_gaps = length + 1
                sent_preds = preds[b, :n_gaps].cpu().numpy()
                sent_gold = gap_labels[b, :n_gaps].cpu().numpy()

                # Sentence-level exact match across all slots in all gaps
                if np.array_equal(sent_preds, sent_gold):
                    exact_matches += 1

                for g in range(n_gaps):
                    for s in range(MAX_SLOTS_PER_GAP):
                        gold = sent_gold[g, s]
                        pred = sent_preds[g, s]
                        if gold != -100:
                            all_targets.append(gold)
                            all_preds.append(pred)

    # Calculate Macro-F1 strictly across 8 punctuation classes (excluding STOP id 0)
    punct_classes = list(range(1, NUM_CLASSES))
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        all_targets, all_preds, labels=punct_classes, average=None, zero_division=0
    )

    macro_f1 = float(np.mean(f1s) * 100.0)
    exact_match_pct = float((exact_matches / max(1, total_sentences)) * 100.0)

    per_symbol_metrics = {}
    for idx, sym_id in enumerate(punct_classes):
        sym = ID_TO_PUNCT[sym_id]
        per_symbol_metrics[sym] = {
            "precision": round(float(precisions[idx] * 100.0), 2),
            "recall": round(float(recalls[idx] * 100.0), 2),
            "f1": round(float(f1s[idx] * 100.0), 2),
            "support": int(supports[idx]),
        }

    return {
        "macro_f1": round(macro_f1, 2),
        "exact_match_pct": round(exact_match_pct, 2),
        "total_sentences": total_sentences,
        "per_symbol": per_symbol_metrics,
    }


def train_model(args: argparse.Namespace) -> None:
    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
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

    # 2. Instantiate Embedding Lookup and Model
    embed_lookup = EmbeddingLookup(
        embed_dim=args.embed_dim,
        word2id_path=args.word2id,
        vectors_path=args.vocab_vectors,
    )

    model = PackedBiLSTMPunctuation(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Share embedding matrix with model for inference
    model.word2id = embed_lookup.word2id
    model.embedding_matrix = embed_lookup.vectors

    param_info = model.count_parameters()
    LOG.info(
        "Model parameters: Total=%d, Trainable=%d (LSTM=%d, InputProj=%d, GapLayers=%d)",
        param_info["total"],
        param_info["trainable"],
        param_info["lstm"],
        param_info["input_projection"],
        param_info["gap_layers"],
    )

    collator = BiLSTMPunctuationCollator(embed_lookup)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 5. Training Loop
    best_macro_f1 = -1.0
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = checkpoint_dir / "best_bilstm_punct.pt"

    LOG.info("Starting training for %d epochs (%d total updates)...", args.epochs, total_steps)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader, 1):
            optimizer.zero_grad()

            word_vectors = batch["word_vectors"].to(device)
            word_mask = batch["word_mask"].to(device)
            lengths = batch["lengths"]
            gap_labels = batch["gap_labels"].to(device)

            outputs = model(word_vectors=word_vectors, word_mask=word_mask, real_lengths=lengths)
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
        restored = model.restore_punctuation(
            sample, word2vec_fn=embed_lookup.get_vector, device=device
        )
        print(f"\n[Raw Input]   : {sample}")
        print(f"[Restored]    : {restored}")


def main():
    parser = argparse.ArgumentParser(description="Packed BiLSTM Punctuation Restoration Pipeline")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode with built-in toy dataset")
    parser.add_argument("--train_file", type=str, default=None, help="Path to training text or JSONL file")
    parser.add_argument("--val_file", type=str, default=None, help="Path to validation text or JSONL file")
    parser.add_argument("--embed_dim", type=int, default=300, help="Word vector dimension")
    parser.add_argument("--hidden_dim", type=int, default=192, help="LSTM hidden dimension per direction")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--word2id", type=str, default=None, help="Path to fastText word2id.json")
    parser.add_argument("--vocab_vectors", type=str, default=None, help="Path to fastText vocab_vectors.npy")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true", help="Apply class-imbalance weights")
    parser.add_argument("--predict", type=str, default=None, help="Restore punctuation on input text and exit")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for prediction")

    args = parser.parse_args()

    if args.predict:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        embed_lookup = EmbeddingLookup(
            embed_dim=args.embed_dim,
            word2id_path=args.word2id,
            vectors_path=args.vocab_vectors,
        )
        model = PackedBiLSTMPunctuation(
            embed_dim=args.embed_dim,
            hidden_dim=args.hidden_dim,
        ).to(device)
        model.word2id = embed_lookup.word2id
        model.embedding_matrix = embed_lookup.vectors

        if args.checkpoint and Path(args.checkpoint).exists():
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
            state = ckpt.get("model_state", ckpt.get("model_state_dict", ckpt))
            model.load_state_dict(state)
            LOG.info("Loaded weights from %s", args.checkpoint)

        result = model.restore_punctuation(
            args.predict, word2vec_fn=embed_lookup.get_vector, device=device
        )
        print("\nOriginal :", args.predict)
        print("Restored :", result)
    else:
        train_model(args)


if __name__ == "__main__":
    main()
