"""Packed fastText-BiLSTM for Bangla Punctuation Restoration.

This module provides a standalone, production-ready PyTorch model for Bangla
punctuation restoration using a 2-layer Bidirectional LSTM over 300-dimensional
word representations, with sequence packing, learned BOS/EOS boundary framing,
and adjacent-context multi-slot gap classification.

Key architectural features:
1. Input Projection: Projects 300d word vectors into 384d LSTM hidden space.
2. BOS/EOS Boundary Framing: Explicit learned vectors marking sequence boundaries.
3. Variable-Length Sequence Packing: Uses pack_padded_sequence to avoid padding pollution.
4. Bidirectional LSTM: 2-layer BiLSTM (192 units/dir = 384 total hidden dim).
5. Adjacent Context Gap Representation: Concatenates left and right hidden states
   [h_i || h_{i+1}] (768 dim), projected to 384 with GELU and LayerNorm.
6. Multi-Slot Gap Classification: Predicts up to 4 punctuation marks per word gap
   across 9 Bengali punctuation classes.
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# Standard 8 Bangla punctuation classes + 1 STOP/NONE token
PUNCTUATION_VOCAB = [
    "<STOP>",  # 0: No punctuation / slot terminator
    "।",       # 1: Dari (Bengali full stop)
    ",",       # 2: Comma
    "?",       # 3: Question mark
    "!",       # 4: Exclamation mark
    ";",       # 5: Semicolon
    ":",       # 6: Colon
    "(",       # 7: Opening parenthesis
    ")",       # 8: Closing parenthesis
]

PUNCT_TO_ID = {sym: idx for idx, sym in enumerate(PUNCTUATION_VOCAB)}
ID_TO_PUNCT = {idx: sym for idx, sym in enumerate(PUNCTUATION_VOCAB)}
NUM_CLASSES = len(PUNCTUATION_VOCAB)  # 9
MAX_SLOTS_PER_GAP = 4                 # Support up to 4 consecutive marks at a single gap


def framed_words(
    words: torch.Tensor,
    lengths: Sequence[int],
    bos_embedding: nn.Parameter,
    eos_embedding: nn.Parameter,
) -> torch.Tensor:
    """Frame word sequences by placing BOS at index 0 and EOS immediately after each example.

    Args:
        words: [batch_size, max_words, dim]
        lengths: Actual word count for each sentence in the batch
        bos_embedding: [1, 1, dim]
        eos_embedding: [1, 1, dim]

    Returns:
        [batch_size, max_words + 2, dim] tensor with boundary tokens.
    """
    batch_size, max_words, dim = words.shape
    device = words.device
    seq = words.new_zeros(batch_size, max_words + 2, dim)

    # Place BOS at index 0
    seq[:, :1] = bos_embedding
    # Place words in indices 1 .. max_words
    seq[:, 1 : max_words + 1] = words

    # Place EOS immediately after actual length of each sentence
    rows = torch.arange(batch_size, device=device)
    lens = torch.as_tensor(lengths, device=device)
    seq[rows, lens + 1] = eos_embedding[0, 0].to(words.dtype)

    # Zero out padded positions beyond EOS
    mask = torch.arange(max_words + 2, device=device)[None, :] > lens[:, None] + 1
    return seq.masked_fill(mask[..., None], 0.0)


class PackedBiLSTMPunctuation(nn.Module):
    """Packed Bidirectional LSTM with adjacent-gap representation for punctuation restoration.

    Architecture:
      Input (300d fastText Word Vectors)
      -> Linear Projection (300 -> 384) + Dropout
      -> Framed with Learned [BOS] and [EOS] Embeddings
      -> PyTorch Packed Sequence (pack_padded_sequence)
      -> 2-layer Bidirectional LSTM (192 hidden per dir = 384 dim)
      -> Unpacked (pad_packed_sequence)
      -> Adjacent Context Concatenation [h_left || h_right] (768 dim)
      -> Gap Projection (GELU + LayerNorm -> 384 dim)
      -> Multi-Slot Classification Head -> [Batch, Gaps, Slots, Classes]
    """

    def __init__(
        self,
        embed_dim: int = 300,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = NUM_CLASSES,
        max_slots: int = MAX_SLOTS_PER_GAP,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.total_hidden = hidden_dim * 2  # 384
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_slots = max_slots

        # 1. Project input word vectors (e.g. 300d fastText) to LSTM hidden dimension
        self.input_projection = nn.Linear(embed_dim, self.total_hidden)
        self.dropout = nn.Dropout(dropout)

        # 2. Learned BOS and EOS boundary embeddings
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, self.total_hidden) * 0.02)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, self.total_hidden) * 0.02)

        # 3. 2-layer Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.total_hidden,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 4. Gap projection: concat left & right context (384 + 384 = 768) -> 384
        self.gap_projection = nn.Sequential(
            nn.Linear(self.total_hidden * 2, self.total_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(self.total_hidden),
        )

        # 5. Gap slot classification head: 384 -> (4 * 9 = 36)
        self.gap_head = nn.Linear(self.total_hidden, max_slots * num_classes)

        # Optional word embedding table / cache for inference
        self.word2id: Optional[Dict[str, int]] = None
        self.embedding_matrix: Optional[np.ndarray] = None

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts broken down by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lstm_params = sum(p.numel() for p in self.lstm.parameters())
        input_proj_params = sum(p.numel() for p in self.input_projection.parameters())
        head_params = sum(p.numel() for p in self.gap_head.parameters()) + sum(
            p.numel() for p in self.gap_projection.parameters()
        )
        return {
            "total": total,
            "trainable": trainable,
            "lstm": lstm_params,
            "input_projection": input_proj_params,
            "gap_layers": head_params,
        }

    def load_embeddings(
        self,
        word2id_path: Optional[Union[str, Path]] = None,
        vectors_path: Optional[Union[str, Path]] = None,
    ) -> bool:
        """Load external fastText vocabulary and embedding matrix into memory."""
        search_dirs = [
            Path(__file__).resolve().parent.parent / "bangla-punctuation" / "embeddings" / "cache",
            Path(__file__).resolve().parent / "embeddings",
        ]

        w_path = Path(word2id_path) if word2id_path else None
        v_path = Path(vectors_path) if vectors_path else None

        if w_path is None or v_path is None:
            for s_dir in search_dirs:
                cand_w = s_dir / "word2id.json"
                cand_v = s_dir / "vocab_vectors.npy"
                if cand_w.is_file() and cand_v.is_file():
                    w_path, v_path = cand_w, cand_v
                    break

        if w_path and v_path and w_path.is_file() and v_path.is_file():
            with open(w_path, "r", encoding="utf-8") as f:
                self.word2id = json.load(f)
            self.embedding_matrix = np.load(v_path, mmap_mode="r")
            return True
        return False

    def get_word_vector(self, word: str) -> np.ndarray:
        """Retrieve 300-dim vector for a word, or zero vector fallback."""
        if self.word2id is not None and self.embedding_matrix is not None:
            idx = self.word2id.get(word, -1)
            if 0 <= idx < len(self.embedding_matrix):
                return self.embedding_matrix[idx]
        return np.zeros(self.embed_dim, dtype=np.float32)

    def forward(
        self,
        word_vectors: torch.Tensor,
        word_mask: Optional[torch.Tensor] = None,
        real_lengths: Optional[Sequence[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using packed recurrent encoding and adjacent gap projections.

        Args:
            word_vectors: [batch_size, max_words, embed_dim]
            word_mask: Optional [batch_size, max_words] boolean tensor
            real_lengths: Optional sequence of actual sentence lengths (word counts)

        Returns:
            Dictionary containing:
              - 'gap_logits': [batch_size, max_words + 1, max_slots, num_classes]
              - 'gap_representations': [batch_size, max_words + 1, total_hidden]
        """
        batch_size, max_words, _ = word_vectors.shape

        if real_lengths is not None:
            lengths = list(real_lengths)
        elif word_mask is not None:
            lengths = word_mask.sum(1).tolist()
        else:
            lengths = [max_words] * batch_size

        # Step 1: Project word representations to LSTM hidden dimension
        projected = self.dropout(self.input_projection(word_vectors))

        # Step 2: Frame sequence with learned BOS and EOS
        seq = framed_words(projected, lengths, self.bos_embedding, self.eos_embedding)

        # Step 3: Pack sequence to avoid RNN recurrent state corruption on padding
        packed = pack_padded_sequence(
            seq,
            [n + 2 for n in lengths],
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.lstm(packed)
        hidden, _ = pad_packed_sequence(encoded, batch_first=True, total_length=seq.shape[1])

        # Step 4: Concatenate adjacent representations across gaps: [h_i || h_{i+1}]
        gap_inputs = torch.cat((hidden[:, :-1], hidden[:, 1:]), dim=-1)
        reps = self.gap_projection(gap_inputs)

        # Step 5: Multi-slot classification logits
        logits = self.gap_head(reps).reshape(
            len(lengths), seq.shape[1] - 1, self.max_slots, self.num_classes
        )

        return {
            "gap_logits": logits,
            "gap_representations": reps,
        }

    @torch.no_grad()
    def restore_punctuation(
        self,
        text: str,
        word2vec_fn: Optional[Callable[[str], np.ndarray]] = None,
        device: Optional[torch.device] = None,
    ) -> str:
        """Convenience method for end-to-end inference on raw unpunctuated Bangla text.

        Args:
            text: Raw Bangla sentence (e.g. 'আপনি কেমন আছেন আমিও ভালো আছি')
            word2vec_fn: Optional callable mapping word -> 300d vector. If None, uses loaded cache.
            device: Target torch device

        Returns:
            Reconstructed punctuated text (e.g. 'আপনি কেমন আছেন? আমিও ভালো আছি।')
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        words = text.strip().split()
        if not words:
            return ""

        # Vector lookup
        vec_lookup = word2vec_fn if word2vec_fn is not None else self.get_word_vector
        vectors = [vec_lookup(w) for w in words]
        word_vectors = torch.tensor(
            np.array([vectors], dtype=np.float32), device=device
        )
        lengths = [len(words)]

        outputs = self.forward(word_vectors=word_vectors, real_lengths=lengths)
        logits = outputs["gap_logits"][0]  # [len(words) + 1, max_slots, num_classes]
        predictions = logits.argmax(dim=-1).cpu().tolist()  # [N+1, max_slots]

        # Reconstruct sentence by interleaving words and punctuation
        punctuated_parts = []
        for i, word in enumerate(words):
            # Gap i (before word i)
            for slot_id in predictions[i]:
                if slot_id != 0:  # Not STOP
                    punctuated_parts.append(ID_TO_PUNCT[slot_id])

            punctuated_parts.append(word)

        # Final gap (after last word)
        for slot_id in predictions[len(words)]:
            if slot_id != 0:
                punctuated_parts.append(ID_TO_PUNCT[slot_id])

        # Format output string: attach closing punctuation to preceding words
        result = ""
        for token in punctuated_parts:
            if token in PUNCTUATION_VOCAB[1:]:
                if token == "(":
                    result = result + (" " if result else "") + token
                else:
                    result = result + token
            else:
                result = result + (" " if result and not result.endswith("(") else "") + token

        return result
