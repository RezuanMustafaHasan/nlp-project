"""Decoding algorithms for Bengali punctuation restoration and parenthesis balancing.

Implements Step 23:
1. Independent decoding: standard slot argmax until STOP (ordinary learned baseline).
2. Pairing repair: stack-based post-processing that eliminates orphan closing and
   unfinished opening parentheses without altering word order or other marks.
3. Constrained beam search decoder:
   - Evaluates gap sequence candidates ranked by slot log probabilities.
   - Enforces strict grammatical invariants: no orphan ')', no empty pairs '()',
     maximum nesting depth <= 2, and no duplicate completed spans.
   - Requires an empty stack at sentence end.
   - Supports auxiliary learned span score bonuses: score += beta * span_score[s, e].
   - Deterministic fallback if search degenerates.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Sequence, Optional, Dict, Set, Any
import torch
import torch.nn.functional as F

from labels import ID_TO_LABEL, LABEL_TO_ID, TARGET_SYMBOLS

FALLBACK_STATS = {
    "beam_repair_fallbacks": 0,
    "empty_fallbacks": 0
}

def get_fallback_stats() -> Dict[str, int]:
    """Returns current counts of decoder fallback invocations."""
    return dict(FALLBACK_STATS)

def reset_fallback_stats() -> None:
    """Resets decoder fallback invocation counters to zero."""
    FALLBACK_STATS["beam_repair_fallbacks"] = 0
    FALLBACK_STATS["empty_fallbacks"] = 0

def decode_independent_example(
    gap_logits: torch.Tensor,
    num_gaps: int
) -> List[List[str]]:
    """Decodes a single example's gap logits using greedy slot argmax.

    Args:
        gap_logits: [G, 4, 9] prediction scores
        num_gaps: number of valid gaps (n_words + 1)

    Returns:
        List of gap mark lists for the example.
    """
    pred_classes = torch.argmax(gap_logits[:num_gaps], dim=-1).cpu().numpy()
    example_gaps: List[List[str]] = []

    for g in range(num_gaps):
        marks: List[str] = []
        for s in range(4):
            class_id = int(pred_classes[g, s])
            if class_id == LABEL_TO_ID['STOP']:
                break
            if class_id in ID_TO_LABEL:
                marks.append(ID_TO_LABEL[class_id])
        example_gaps.append(marks)

    return example_gaps


def decode_gap_logits(
    gap_logits: torch.Tensor,
    real_lengths: Sequence[int]
) -> List[List[List[str]]]:
    """Batch wrapper for independent greedy decoding (backwards compatible)."""
    B = gap_logits.shape[0]
    return [
        decode_independent_example(gap_logits[b], real_lengths[b] + 1)
        for b in range(B)
    ]


def decode_independent(
    gap_logits: torch.Tensor,
    real_lengths: Sequence[int]
) -> List[List[List[str]]]:
    """Decodes batch using independent slot argmax."""
    return decode_gap_logits(gap_logits, real_lengths)


# ==============================================================================
# 2. Stack-Based Pairing Repair
# ==============================================================================

def repair_parentheses_stack(gaps: List[List[str]]) -> List[List[str]]:
    """Repairs unmatched parentheses using a linear stack traversal.

    Rules:
    - Traverses all predicted marks in gap and slot order.
    - An opening '(' is pushed onto the stack with its (gap_idx, mark_idx).
    - A closing ')' pops the matching '(' if stack is non-empty.
    - If stack is empty, the closing ')' has no opening and is dropped.
    - After scanning all gaps, any unclosed '(' remaining in the stack are dropped.
    - All other punctuation marks are preserved in their original order.

    Args:
        gaps: List of mark lists for each gap.

    Returns:
        Cleaned gaps guaranteed to have 100% structurally balanced parentheses.
    """
    stack: List[Tuple[int, int]] = []
    drop_indices: Set[Tuple[int, int]] = set()

    # Pass 1: find orphan closing parens and identify open paren positions
    for g_idx, gap in enumerate(gaps):
        for m_idx, mark in enumerate(gap):
            if mark == "(":
                stack.append((g_idx, m_idx))
            elif mark == ")":
                if stack:
                    open_pos = stack.pop()
                    if open_pos[0] == g_idx:
                        # Empty parenthesis pair enclosing zero words -> drop both
                        drop_indices.add(open_pos)
                        drop_indices.add((g_idx, m_idx))
                else:
                    drop_indices.add((g_idx, m_idx))

    # Pass 2: find unfinished opening parens left in the stack
    for g_idx, m_idx in stack:
        drop_indices.add((g_idx, m_idx))

    # Pass 3: reconstruct gap lists omitting dropped parentheses
    repaired_gaps: List[List[str]] = []
    for g_idx, gap in enumerate(gaps):
        cleaned = [
            mark for m_idx, mark in enumerate(gap)
            if (g_idx, m_idx) not in drop_indices
        ]
        repaired_gaps.append(cleaned)

    return repaired_gaps


def decode_pairing_repair(
    gap_logits: torch.Tensor,
    real_lengths: Sequence[int]
) -> List[List[List[str]]]:
    """Decodes with independent slot argmax followed by stack-based pairing repair."""
    raw_predictions = decode_independent(gap_logits, real_lengths)
    return [repair_parentheses_stack(gaps) for gaps in raw_predictions]


# ==============================================================================
# 3. Constrained Beam Search Decoder
# ==============================================================================

def get_gap_candidates(
    slot_logits: torch.Tensor,
    max_candidates: int = 16
) -> List[Tuple[List[str], float]]:
    """Enumerates top candidate mark sequences for a single gap.

    Args:
        slot_logits: [4, 9] float tensor of logits for the 4 slots of one gap.
        max_candidates: maximum number of candidates to return (up to 16).

    Returns:
        List of (marks_list, log_prob_sum) sorted descending by score.
        Always includes the empty sequence [].
    """
    log_probs = F.log_softmax(slot_logits.float(), dim=-1).detach().cpu().numpy()  # [4, 9]

    # Priority queue/beam over slot extensions
    # Each state: (score, marks_ids, is_stopped)
    beam: List[Tuple[float, List[int], bool]] = [(0.0, [], False)]
    completed: List[Tuple[float, List[int]]] = []

    stop_id = LABEL_TO_ID["STOP"]

    for slot_idx in range(4):
        next_beam: List[Tuple[float, List[int], bool]] = []
        for score, seq, stopped in beam:
            if stopped:
                completed.append((score, seq))
                continue

            # Branch 1: STOP at this slot
            stop_score = score + float(log_probs[slot_idx, stop_id])
            completed.append((stop_score, seq))

            # Branch 2: Emit a punctuation mark (classes 1..8)
            for c_id in range(1, 9):
                mark_score = score + float(log_probs[slot_idx, c_id])
                is_last_slot = (slot_idx == 3)
                next_beam.append((mark_score, seq + [c_id], is_last_slot))

        # Prune intermediate beam
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:max_candidates]

    # Collect all completed paths
    for score, seq, _ in beam:
        completed.append((score, seq))

    # Sort completed by score
    completed.sort(key=lambda x: x[0], reverse=True)

    # Convert class IDs to strings and eliminate duplicates
    seen_seqs: Set[Tuple[str, ...]] = set()
    unique_candidates: List[Tuple[List[str], float]] = []

    has_empty = False
    empty_score = float(log_probs[0, stop_id])

    for score, seq in completed:
        marks = [ID_TO_LABEL[cid] for cid in seq if cid in ID_TO_LABEL]
        marks_tuple = tuple(marks)
        if marks_tuple in seen_seqs:
            continue
        seen_seqs.add(marks_tuple)
        if not marks:
            has_empty = True
        unique_candidates.append((marks, score))
        if len(unique_candidates) >= max_candidates:
            break

    # Ensure empty sequence is explicitly present
    if not any(len(c[0]) == 0 for c in unique_candidates):
        if len(unique_candidates) >= max_candidates:
            unique_candidates[-1] = ([], empty_score)
        else:
            unique_candidates.append(([], empty_score))

    return unique_candidates


class BeamHypothesis:
    """Represents a partial decoded sentence hypothesis in beam search."""

    def __init__(
        self,
        score: float,
        gaps: List[List[str]],
        stack: List[int],
        completed_spans: Set[Tuple[int, int]]
    ):
        self.score = score
        self.gaps = gaps
        self.stack = stack
        self.completed_spans = completed_spans


def decode_passage_beam(
    gap_logits: torch.Tensor,
    span_scores: Optional[torch.Tensor] = None,
    num_gaps: int = 1,
    beam_width: int = 8,
    max_candidates: int = 16,
    beta: float = 0.0
) -> List[List[str]]:
    """Decodes a single passage using constrained beam search.

    Enforces invariants:
    - No closing ')' without matching opening '('.
    - No empty paired spans '()' enclosing 0 words.
    - Maximum parenthesis nesting depth <= 2.
    - No repeated completed intervals (s, e).
    - Requires empty stack at sentence end.
    - Adds beta * span_score[s, e] for each completed span if beta > 0.
    """
    # Pre-calculate candidates for each gap
    gap_cand_list = [
        get_gap_candidates(gap_logits[g], max_candidates=max_candidates)
        for g in range(num_gaps)
    ]

    # Convert span_scores to CPU numpy for fast scalar indexing
    span_scores_np = None
    if span_scores is not None and beta != 0.0:
        span_scores_np = span_scores.float().detach().cpu().numpy()

    # Initial beam
    beam = [BeamHypothesis(score=0.0, gaps=[], stack=[], completed_spans=set())]

    # Fallback path tracker in case beam degenerates
    best_empty_stack_path: Optional[BeamHypothesis] = None

    for g in range(num_gaps):
        candidates = gap_cand_list[g]
        next_beam: List[BeamHypothesis] = []

        for hyp in beam:
            for cand_marks, cand_log_prob in candidates:
                # Check grammatical validity of cand_marks at gap g
                new_stack = list(hyp.stack)
                new_completed = set(hyp.completed_spans)
                valid = True
                span_bonus = 0.0

                for mark in cand_marks:
                    if mark == "(":
                        if len(new_stack) >= 2:
                            # Invariant 1: Nesting depth must not exceed 2
                            valid = False
                            break
                        new_stack.append(g)
                    elif mark == ")":
                        if not new_stack:
                            # Invariant 2: No closing paren without prior open paren
                            valid = False
                            break
                        s = new_stack.pop()
                        if s == g:
                            # Invariant 3: No empty parenthesis span '()'
                            valid = False
                            break
                        if (s, g) in new_completed:
                            # Invariant 4: No duplicate completed intervals
                            valid = False
                            break
                        new_completed.add((s, g))

                        if span_scores_np is not None:
                            span_bonus += beta * float(span_scores_np[s, g])

                if not valid:
                    continue

                new_score = hyp.score + cand_log_prob + span_bonus
                new_hyp = BeamHypothesis(
                    score=new_score,
                    gaps=hyp.gaps + [cand_marks],
                    stack=new_stack,
                    completed_spans=new_completed
                )
                next_beam.append(new_hyp)

        if not next_beam:
            # Fallback: keep previous hypotheses and pad current gap with empty []
            next_beam = [
                BeamHypothesis(
                    score=hyp.score - 5.0,
                    gaps=hyp.gaps + [[]],
                    stack=hyp.stack,
                    completed_spans=hyp.completed_spans
                )
                for hyp in beam
            ]

        # Prune beam to beam_width
        next_beam.sort(key=lambda h: h.score, reverse=True)
        beam = next_beam[:beam_width]

    # Select best hypothesis with empty stack
    valid_final = [h for h in beam if len(h.stack) == 0]
    if valid_final:
        return valid_final[0].gaps

    # Fallback 1: repair best available hypothesis
    if beam:
        FALLBACK_STATS["beam_repair_fallbacks"] += 1
        return repair_parentheses_stack(beam[0].gaps)

    # Fallback 2: all empty gaps
    FALLBACK_STATS["empty_fallbacks"] += 1
    return [[] for _ in range(num_gaps)]


def decode_constrained(
    gap_logits: torch.Tensor,
    span_scores: Optional[torch.Tensor],
    real_lengths: Sequence[int],
    beam_width: int = 8,
    max_candidates: int = 16,
    beta: float = 0.0
) -> List[List[List[str]]]:
    """Decodes batch using constrained beam search."""
    B = gap_logits.shape[0]
    results: List[List[List[str]]] = []

    for b in range(B):
        n_w = real_lengths[b]
        num_gaps = n_w + 1
        b_span_scores = span_scores[b] if span_scores is not None else None
        gaps = decode_passage_beam(
            gap_logits=gap_logits[b],
            span_scores=b_span_scores,
            num_gaps=num_gaps,
            beam_width=beam_width,
            max_candidates=max_candidates,
            beta=beta
        )
        results.append(gaps)

    return results


# ==============================================================================
# 4. Universal Decoding Dispatcher
# ==============================================================================

def decode_batch(
    gap_logits: torch.Tensor,
    span_scores: Optional[torch.Tensor],
    real_lengths: Sequence[int],
    method: str = "independent",
    beam_width: int = 8,
    max_candidates: int = 16,
    beta: float = 0.0
) -> List[List[List[str]]]:
    """Universal batch decoder supporting all 3 strategies.

    Args:
        gap_logits: [B, G, 4, 9]
        span_scores: [B, G, G] or None
        real_lengths: list of word lengths per example
        method: "independent" (A/D), "pairing_repair" (B), or "constrained" (C/E/F)
        beam_width: beam width for constrained decoder (default 8)
        max_candidates: max gap candidate sequences (default 16)
        beta: weight on learned span score bonus in constrained search (default 0.0)

    Returns:
        List of B items, each containing num_gaps lists of predicted marks.
    """
    if method == "independent":
        return decode_independent(gap_logits, real_lengths)
    elif method in ("pairing_repair", "repair"):
        return decode_pairing_repair(gap_logits, real_lengths)
    elif method == "constrained":
        return decode_constrained(
            gap_logits=gap_logits,
            span_scores=span_scores,
            real_lengths=real_lengths,
            beam_width=beam_width,
            max_candidates=max_candidates,
            beta=beta
        )
    else:
        raise ValueError(f"Unknown decoding method: {method}")
