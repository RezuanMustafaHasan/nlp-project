"""Minimalist Training Script for BiLSTM Bangla Punctuation Restoration."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from model import (
    MAX_SLOTS_PER_GAP,
    NUM_CLASSES,
    PUNCT_TO_ID,
    PUNCTUATION_VOCAB,
    PackedBiLSTMPunctuation,
)

PUNCT_CHARS = set(PUNCTUATION_VOCAB[1:])


def parse_punctuated_text(text: str) -> Optional[Dict[str, any]]:
    """Parse a punctuated Bangla string into words and gap punctuation labels."""
    tokens = re.findall(r"[\w\u0980-\u09FF]+|[।,\?!;:()]", text.strip())
    if not tokens:
        return None

    words, gaps = [], [[]]
    for tok in tokens:
        if tok in PUNCT_CHARS:
            if len(gaps[-1]) < MAX_SLOTS_PER_GAP:
                gaps[-1].append(tok)
        else:
            words.append(tok)
            gaps.append([])

    if not words:
        return None

    gap_labels = []
    for gap in gaps:
        slot_ids = [PUNCT_TO_ID[sym] for sym in gap]
        slot_ids += [PUNCT_TO_ID["<STOP>"]] * (MAX_SLOTS_PER_GAP - len(slot_ids))
        gap_labels.append(slot_ids[:MAX_SLOTS_PER_GAP])

    return {"words": words, "gap_labels": gap_labels}


class BanglaPunctuationDataset(Dataset):
    """Loads sentences from a text or JSONL file."""

    def __init__(self, file_path: str):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
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


class EmbeddingLookup:
    """Helper for converting words to 300d embeddings."""

    def __init__(
        self,
        embed_dim: int = 300,
        word2id_path: Optional[str] = None,
        vectors_path: Optional[str] = None,
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
            with open(w_p, "r", encoding="utf-8") as f:
                self.word2id = json.load(f)
            self.vectors = np.load(v_p, mmap_mode="r")

    def get_vector(self, word: str) -> np.ndarray:
        if self.vectors is not None and self.word2id:
            idx = self.word2id.get(word, -1)
            if 0 <= idx < len(self.vectors):
                return self.vectors[idx]

        if word not in self._fallback_cache:
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


def main():
    parser = argparse.ArgumentParser(description="Minimalist BiLSTM Bangla Punctuation Training")
    parser.add_argument("--train_file", type=str, required=True, help="Path to training text or JSONL file")
    parser.add_argument("--word2id", type=str, default=None, help="Path to fastText word2id.json")
    parser.add_argument("--vocab_vectors", type=str, default=None, help="Path to fastText vocab_vectors.npy")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    train_dataset = BanglaPunctuationDataset(args.train_file)
    embed_lookup = EmbeddingLookup(
        embed_dim=args.embed_dim,
        word2id_path=args.word2id,
        vectors_path=args.vocab_vectors,
    )

    model = PackedBiLSTMPunctuation(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    collator = BiLSTMPunctuationCollator(embed_lookup)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training PackedBiLSTMPunctuation on {len(train_dataset)} examples for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

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

        avg_loss = epoch_loss / max(1, len(train_loader))
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {avg_loss:.4f}")

    save_path = out_dir / "final_bilstm_punct.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Model successfully saved to {save_path}")


if __name__ == "__main__":
    main()
