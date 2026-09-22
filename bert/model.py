"""Aligned BanglaBERT for Punctuation Restoration.

This module provides a standalone, production-ready PyTorch model for Bangla
punctuation restoration using the pre-trained 'csebuetnlp/banglabert' ELECTRA
discriminator as the contextual backbone.

Key architectural features:
1. Contextual Subword Encoding: Uses csebuetnlp/banglabert (ELECTRA architecture).
2. Subword-to-Word Alignment: Mean-pools subword embeddings into word-level vectors.
3. BOS/EOS Boundary Framing: Explicit learned vectors marking start and end of sequence.
4. Adjacent Context Gap Representation: Concatenates left and right word contexts
   around each boundary gap, followed by a non-linear projection.
5. Multi-Slot Gap Classification: Predicts punctuation marks across gap slots.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


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


def word_alignment(encoding, words_list: List[List[str]]) -> torch.Tensor:
    """Extract word indices for each token from HuggingFace tokenizer encoding.

    Returns a tensor of shape [batch_size, max_subwords] where each entry is:
      - The 0-based word index if the subword belongs to that word.
      - -1 if the token is a special token ([CLS], [SEP], [PAD]).
    """
    result = []
    for i, words in enumerate(words_list):
        ids = encoding.word_ids(i)
        seen = {x for x in ids if x is not None}
        if seen != set(range(len(words))):
            raise ValueError(
                f"Tokenizer dropped words in example {i}. Expected {len(words)} words, got {len(seen)}."
            )
        result.append([-1 if x is None else x for x in ids])
    return torch.tensor(result, dtype=torch.long)


def frame_word_representations(
    words: torch.Tensor,
    lengths: Sequence[int],
    bos_proj: nn.Parameter,
    eos_proj: nn.Parameter,
) -> torch.Tensor:
    """Framed representation inserting BOS at index 0 and EOS immediately after each sequence.

    Args:
        words: [batch_size, max_words, hidden_dim]
        lengths: Actual word count for each sentence in the batch
        bos_proj: [1, 1, hidden_dim]
        eos_proj: [1, 1, hidden_dim]

    Returns:
        [batch_size, max_words + 2, hidden_dim] tensor with BOS and EOS tokens.
    """
    batch_size, max_words, dim = words.shape
    device = words.device
    seq = words.new_zeros(batch_size, max_words + 2, dim)

    # Place BOS at index 0
    seq[:, 0] = bos_proj.squeeze(1).expand(batch_size, -1)
    # Place words in indices 1 .. max_words
    seq[:, 1 : max_words + 1] = words

    # Place EOS immediately after the actual word length of each example
    for b, length in enumerate(lengths):
        seq[b, length + 1] = eos_proj.squeeze(1)

    # Zero out padded areas beyond length + 1
    lens_tensor = torch.as_tensor(lengths, device=device)
    mask = torch.arange(max_words + 2, device=device)[None, :] > lens_tensor[:, None] + 1
    return seq.masked_fill(mask[..., None], 0.0)


class AlignedBanglaBERTPunctuation(nn.Module):
    """Fine-tuned BanglaBERT (ELECTRA) with subword-to-word pooling and gap tagging.

    Architecture:
      Input (Words) -> Tokenizer -> Subword IDs -> Pretrained BanglaBERT
      -> Subword-to-Word Scatter Pooling -> Word Contexts
      -> Framed with Learned [BOS] and [EOS]
      -> Adjacent Context Concatenation [h_left || h_right] (1536 dim)
      -> Gap Projection (GELU + LayerNorm -> 384 dim)
      -> Multi-Slot Classification Head -> [Batch, Gaps, Slots, Classes]
    """

    def __init__(
        self,
        pretrained_model_name: str = "csebuetnlp/banglabert",
        projection_dim: int = 384,
        num_classes: int = NUM_CLASSES,
        max_slots: int = MAX_SLOTS_PER_GAP,
        dropout: float = 0.1,
        load_pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name
        self.num_classes = num_classes
        self.max_slots = max_slots
        self.projection_dim = projection_dim

        # 1. Contextual Encoder (BanglaBERT ELECTRA discriminator)
        self.config = AutoConfig.from_pretrained(pretrained_model_name)
        # Inference checkpoints already contain the full encoder state. Building
        # from config avoids downloading a duplicate 400+ MB backbone before the
        # local .pt weights are loaded.
        self.encoder = (
            AutoModel.from_pretrained(pretrained_model_name, config=self.config)
            if load_pretrained_backbone
            else AutoModel.from_config(self.config)
        )
        self.hidden_dim = self.config.hidden_size  # 768 for banglabert-base

        # 2. Learned BOS and EOS boundary embeddings
        self.bos_proj = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)
        self.eos_proj = nn.Parameter(torch.randn(1, 1, self.hidden_dim) * 0.02)

        # 3. Gap projection: [h_left || h_right] (768 * 2 = 1536) -> projection_dim (384)
        self.gap_projection = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_dim),
        )

        # 4. Multi-slot classification head: projection_dim -> (max_slots * num_classes)
        self.gap_head = nn.Linear(projection_dim, max_slots * num_classes)

        # Tokenizer initialized lazily for inference
        self._tokenizer = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.pretrained_model_name, use_fast=True
            )
        return self._tokenizer

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts broken down by component."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        head_params = total - encoder_params
        return {
            "total": total,
            "trainable": trainable,
            "encoder": encoder_params,
            "head": head_params,
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: torch.Tensor,
        lengths: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass using subword-to-word alignment.

        Args:
            input_ids: [batch_size, num_subwords]
            attention_mask: [batch_size, num_subwords]
            word_ids: [batch_size, num_subwords] mapping to 0-based word index (-1 for special)
            lengths: List/Sequence of real word counts per sentence in batch

        Returns:
            Dictionary containing:
              - "gap_logits": [batch_size, max_words + 1, max_slots, num_classes]
              - "gap_representations": [batch_size, max_words + 1, projection_dim]
        """
        # Step 1: Contextual encoding via BanglaBERT
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        subword_hidden = outputs.last_hidden_state  # [B, S, 768]

        batch_size, _, dim = subword_hidden.shape
        max_words = max(lengths)

        # Step 2: Mean-pool subwords into corresponding words using scatter_add
        valid = (word_ids >= 0) & attention_mask.bool()
        index = word_ids.clamp_min(0)

        # Accumulate subword vectors for each word
        sums = subword_hidden.new_zeros(batch_size, max_words, dim)
        sums.scatter_add_(1, index[..., None].expand(-1, -1, dim), subword_hidden * valid[..., None])

        # Accumulate counts per word
        counts = subword_hidden.new_zeros(batch_size, max_words)
        counts.scatter_add_(1, index, valid.to(subword_hidden.dtype))

        # Mean pooling (prevent division by zero for padding)
        words = sums / counts.clamp_min(1.0)[..., None]  # [B, max_words, 768]

        # Step 3: Frame word sequence with learned BOS and EOS
        framed_seq = frame_word_representations(words, lengths, self.bos_proj, self.eos_proj)
        # framed_seq shape: [B, max_words + 2, 768]

        # Step 4: Represent gap i by concatenating context i (left) and context i+1 (right)
        # For N words, there are N + 1 gaps (including initial gap 0 and final gap N)
        left_ctx = framed_seq[:, : max_words + 1, :]
        right_ctx = framed_seq[:, 1 : max_words + 2, :]
        gap_inputs = torch.cat([left_ctx, right_ctx], dim=-1)  # [B, max_words + 1, 1536]

        gap_representations = self.gap_projection(gap_inputs)  # [B, max_words + 1, 384]

        # Step 5: Multi-slot gap logits
        logits = self.gap_head(gap_representations)
        logits = logits.view(batch_size, max_words + 1, self.max_slots, self.num_classes)

        return {
            "gap_logits": logits,
            "gap_representations": gap_representations,
        }

    @torch.no_grad()
    def restore_punctuation(self, text: str, device: Optional[torch.device] = None) -> str:
        """Convenience method for end-to-end inference on raw unpunctuated Bangla text.

        Args:
            text: Raw Bangla sentence (e.g. 'আপনি কেমন আছেন আমিও ভালো আছি')
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

        encoding = self.tokenizer([words], is_split_into_words=True, return_tensors="pt")
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        word_ids = word_alignment(encoding, [words]).to(device)
        lengths = [len(words)]

        outputs = self.forward(input_ids, attention_mask, word_ids, lengths)
        logits = outputs["gap_logits"][0]  # [len(words) + 1, max_slots, num_classes]
        predictions = logits.argmax(dim=-1).cpu().tolist()  # [N+1, max_slots]

        # Reconstruct sentence by interleaving words and punctuation
        punctuated_parts = []
        for i, word in enumerate(words):
            # Check gap i (before word i)
            gap_preds = predictions[i]
            for slot_id in gap_preds:
                if slot_id != 0:  # Not STOP
                    punctuated_parts.append(ID_TO_PUNCT[slot_id])

            punctuated_parts.append(word)

        # Check final gap (after the last word)
        final_gap_preds = predictions[len(words)]
        for slot_id in final_gap_preds:
            if slot_id != 0:
                punctuated_parts.append(ID_TO_PUNCT[slot_id])

        # Format output string: attach closing punctuation to preceding words
        result = ""
        for token in punctuated_parts:
            if token in PUNCTUATION_VOCAB[1:]:
                # If opening bracket, space before it
                if token == "(":
                    result = result + (" " if result else "") + token
                else:
                    # Closing marks attach directly to the previous word
                    result = result + token
            else:
                result = result + (" " if result and not result.endswith("(") else "") + token

        return result
