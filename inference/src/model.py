"""Bidirectional Transformer encoder and gap classification network for Bangla punctuation restoration.

Implements Step 18:
- Word vector projection: 300 -> 384
- Learned BOS and EOS token embeddings
- Learned position embeddings (up to 130 positions)
- 6-layer bidirectional TransformerEncoder (6 heads, d_model=384, d_ff=1536, dropout=0.1)
- Contextual gap representations: concat(h_{i}^left, h_{i}^right) in R^768 -> Linear(768, 384) + GELU
- Gap classification head: Linear(384, 4 * 9 = 36) -> reshape to [B, N+1, 4, 9]
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional

from labels import MAX_SLOTS_PER_GAP, NUM_CLASSES


class BanglaPunctuationTransformer(nn.Module):
    """Transformer for predicting punctuation at word gaps."""

    def __init__(
        self,
        embed_dim: int = 300,
        hidden_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 6,
        ffn_dim: int = 1536,
        dropout: float = 0.1,
        max_positions: int = 130,  # 128 words + BOS + EOS
        num_classes: int = NUM_CLASSES,  # 9 classes (0: STOP, 1..8: symbols)
        max_slots: int = MAX_SLOTS_PER_GAP,  # 4 slots per gap
        has_span_head: bool = True,
        span_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_positions = max_positions
        self.num_classes = num_classes
        self.max_slots = max_slots
        self.has_span_head = has_span_head
        self.span_dim = span_dim

        # 1. Project fixed input word vectors to model hidden dimension
        self.input_projection = nn.Linear(embed_dim, hidden_dim)

        # 2. Learned BOS and EOS tokens
        self.bos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.eos_embedding = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # 3. Learned positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_positions, hidden_dim) * 0.02)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 4. Bidirectional Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. Gap representation: concat left & right context (384 + 384 = 768) -> 384
        self.gap_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )

        # 6. Gap slot classification head: 384 -> (max_slots * num_classes = 4 * 9 = 36)
        self.gap_head = nn.Linear(hidden_dim, max_slots * num_classes)

        # 7. Parenthesis span boundary head (Step 22)
        # start_vector[i] = learned projection of gap i: 384 -> 128
        # end_vector[j]   = learned projection of gap j: 384 -> 128
        if self.has_span_head:
            self.span_start_proj = nn.Linear(hidden_dim, span_dim)
            self.span_end_proj = nn.Linear(hidden_dim, span_dim)
        else:
            self.span_start_proj = None
            self.span_end_proj = None

        self._init_weights()

    def _init_weights(self):
        """Initialize linear layers with standard normal."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        word_vectors: torch.Tensor,
        word_mask: torch.Tensor,
        encoder_mask: torch.Tensor,
        real_lengths: Optional[Sequence[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the punctuation network.

        Args:
            word_vectors: [B, N, 300] float32 tensor of word vectors
            word_mask: [B, N] bool tensor indicating real words
            encoder_mask: [B, N+2] bool tensor (True for padding positions to ignore)
            real_lengths: optional list/tensor of actual word counts per example

        Returns:
            Dict containing:
                "gap_logits": [B, N+1, 4, 9] raw prediction scores
                "gap_reprs": [B, N+1, 384] gap contextual representations
        """
        B, N, _ = word_vectors.shape

        # Step 1: Project word vectors
        projected_words = self.input_projection(word_vectors)  # [B, N, hidden_dim]

        # Step 2: Construct token sequence [B, N+2, hidden_dim]
        # Token sequence is: [BOS, word_1, ..., word_{n_w}, EOS, PAD, ...]
        tokens = torch.zeros((B, N + 2, self.hidden_dim), device=word_vectors.device, dtype=word_vectors.dtype)
        # Position 0 is BOS
        tokens[:, 0:1, :] = self.bos_embedding.expand(B, 1, self.hidden_dim)

        # If real_lengths is not provided, infer from word_mask
        if real_lengths is None:
            real_lengths = word_mask.sum(dim=1).tolist()

        # Place words and dynamic EOS immediately after the last real word of each example
        for b in range(B):
            n_w = int(real_lengths[b])
            if n_w > 0:
                tokens[b, 1:n_w + 1, :] = projected_words[b, :n_w, :]
            # Place EOS at position n_w + 1
            tokens[b, n_w + 1:n_w + 2, :] = self.eos_embedding

        # Step 3: Add learned position embeddings
        tokens = tokens + self.pos_embedding[:, :N + 2, :]
        tokens = self.input_norm(tokens)
        tokens = self.dropout(tokens)

        # Step 4: Run bidirectional Transformer encoder
        # src_key_padding_mask: True indicates positions to be ignored
        contextual_tokens = self.encoder(tokens, src_key_padding_mask=encoder_mask)  # [B, N+2, hidden_dim]

        # Step 5: Gather gap representations for each of the N+1 gaps
        # Gap i (0 <= i <= N) has left context at pos i, right context at pos i+1
        # For examples shorter than N, we rearrange EOS so right context of gap n_w is EOS
        # Vectorized assembly:
        # Create gap left and right tensors of shape [B, N+1, hidden_dim]
        gap_left = torch.zeros((B, N + 1, self.hidden_dim), device=word_vectors.device, dtype=word_vectors.dtype)
        gap_right = torch.zeros((B, N + 1, self.hidden_dim), device=word_vectors.device, dtype=word_vectors.dtype)

        for b in range(B):
            n_w = int(real_lengths[b])
            # Valid gaps are 0..n_w
            # Gap i (0 <= i <= n_w):
            # Left token is at pos i
            # Right token is at pos i+1 (for gap n_w, pos n_w+1 is EOS)
            if n_w + 1 > 0:
                gap_left[b, :n_w + 1, :] = contextual_tokens[b, :n_w + 1, :]
                gap_right[b, :n_w + 1, :] = contextual_tokens[b, 1:n_w + 2, :]

        # Concatenate left and right contextual vectors: [B, N+1, 768]
        gap_cat = torch.cat([gap_left, gap_right], dim=-1)
        gap_reprs = self.gap_projection(gap_cat)  # [B, N+1, 384]

        # Step 6: Classification logits
        gap_logits_raw = self.gap_head(gap_reprs)  # [B, N+1, 36]
        gap_logits = gap_logits_raw.view(B, N + 1, self.max_slots, self.num_classes)  # [B, N+1, 4, 9]

        # Step 7: Parenthesis span scores [B, N+1, N+1] (Step 22)
        # span_score[i, j] = dot(start_vector[i], end_vector[j]) / sqrt(span_dim)
        # Computed via batched matrix multiplication [B, G, 128] @ [B, 128, G] -> [B, G, G]
        span_scores = None
        if self.has_span_head and self.span_start_proj is not None and self.span_end_proj is not None:
            start_reprs = self.span_start_proj(gap_reprs)  # [B, N+1, 128]
            end_reprs = self.span_end_proj(gap_reprs)      # [B, N+1, 128]
            span_scores = torch.bmm(start_reprs, end_reprs.transpose(1, 2)) / math.sqrt(self.span_dim)  # [B, N+1, N+1]

        return {
            "gap_logits": gap_logits,
            "gap_reprs": gap_reprs,
            "span_scores": span_scores,
        }
