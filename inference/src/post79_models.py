"""Opt-in corrected baselines; historical model implementations stay unchanged."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from baselines import FastTextBiLSTMPunctuation, BanglaBERTPunctuation


def framed_words(words, lengths, bos, eos):
    """Put EOS immediately after each example, never after batch padding."""
    batch, width, dim = words.shape
    seq = words.new_zeros(batch, width + 2, dim)
    seq[:, :1] = bos
    seq[:, 1:width + 1] = words
    rows = torch.arange(batch, device=words.device)
    lens = torch.as_tensor(lengths, device=words.device)
    seq[rows, lens + 1] = eos[0, 0].to(words.dtype)
    mask = torch.arange(width + 2, device=words.device)[None, :] > lens[:, None] + 1
    return seq.masked_fill(mask[..., None], 0)


class PackedBiLSTM(FastTextBiLSTMPunctuation):
    """Same parameter names as the old checkpoint, corrected recurrent padding."""
    def forward(self, word_vectors, word_mask, real_lengths=None):
        lengths = real_lengths if real_lengths is not None else word_mask.sum(1).tolist()
        projected = self.dropout(self.input_projection(word_vectors))
        seq = framed_words(projected, lengths, self.bos_embedding, self.eos_embedding)
        packed = pack_padded_sequence(seq, [n + 2 for n in lengths], batch_first=True,
                                      enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        hidden, _ = pad_packed_sequence(encoded, batch_first=True, total_length=seq.shape[1])
        reps = self.gap_projection(torch.cat((hidden[:, :-1], hidden[:, 1:]), -1))
        logits = self.gap_head(reps).reshape(len(lengths), seq.shape[1] - 1,
                                           self.max_slots, self.num_classes)
        return {"gap_logits": logits, "gap_representations": reps}


def word_alignment(encoding, words_list):
    """Return tokenizer word indices; special tokens/padding always map to -1."""
    result = []
    for i, words in enumerate(words_list):
        ids = encoding.word_ids(i)
        seen = {x for x in ids if x is not None}
        if seen != set(range(len(words))):
            raise ValueError("Tokenizer dropped a word; truncation/empty word is forbidden")
        result.append([-1 if x is None else x for x in ids])
    return torch.tensor(result, dtype=torch.long)


class AlignedBanglaBERT(BanglaBERTPunctuation):
    """Explicit encoder injection prevents the historical silent random fallback."""
    def __init__(self, encoder, **kwargs):
        if encoder is None:
            raise ValueError("An explicitly loaded pretrained/checkpoint encoder is required")
        super().__init__(encoder=encoder, hidden_dim=encoder.config.hidden_size, **kwargs)

    def forward_aligned(self, input_ids, attention_mask, word_ids, lengths):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        batch, _, dim = hidden.shape
        width = max(lengths)
        valid = (word_ids >= 0) & attention_mask.bool()
        index = word_ids.clamp_min(0)
        sums = hidden.new_zeros(batch, width, dim)
        sums.scatter_add_(1, index[..., None].expand(-1, -1, dim), hidden * valid[..., None])
        counts = hidden.new_zeros(batch, width)
        counts.scatter_add_(1, index, valid.to(hidden.dtype))
        expected = torch.arange(width, device=hidden.device)[None, :] < torch.tensor(
            lengths, device=hidden.device)[:, None]
        if torch.any((counts == 0) & expected):
            raise ValueError("Missing subword representation for a real word")
        words = sums / counts.clamp_min(1)[..., None]
        seq = framed_words(words, lengths, self.bos_proj, self.eos_proj)
        reps = self.gap_projection(torch.cat((seq[:, :-1], seq[:, 1:]), -1))
        return {"gap_logits": self.gap_head(reps).reshape(batch, width + 1,
                                                          self.max_slots, self.num_classes)}
