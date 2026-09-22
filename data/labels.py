"""Label extraction, gap parsing, and parenthesis span processing.

Defines:
- 8-symbol vocabulary mapping and slot constants
- Gap parsing for words and punctuation
- Stack-based parenthesis span extraction with structural validation
- Structured example creation
"""

from __future__ import annotations
import re
from typing import List, Tuple, Dict, Any, Optional

# The 8 target punctuation symbols:
TARGET_SYMBOLS: List[str] = ['।', ',', '?', '!', ';', ':', '(', ')']
TARGET_SYMBOL_SET = set(TARGET_SYMBOLS)

# Gap slot label map: 0 is STOP, 1-8 are the symbols, -100 is IGNORE
LABEL_TO_ID: Dict[str, int] = {
    'STOP': 0,
    '।': 1,
    ',': 2,
    '?': 3,
    '!': 4,
    ';': 5,
    ':': 6,
    '(': 7,
    ')': 8,
}
ID_TO_LABEL: Dict[int, str] = {v: k for k, v in LABEL_TO_ID.items()}
IGNORE_LABEL_ID = -100
MAX_SLOTS_PER_GAP = 4
NUM_CLASSES = 9

# Regex to scan target punctuation marks and non-punctuation words
PIECES_PATTERN = re.compile(r'[।,?!;:()]|[^\s।,?!;:()]+')


def parse_gaps(text: str) -> Tuple[List[str], List[List[str]]]:
    """Parse text into words and gaps.

    For N words, returns (words, gaps) where len(gaps) == N + 1.
    gap[0] precedes word[0], gap[i] is between word[i-1] and word[i],
    and gap[N] follows word[N-1].
    """
    words: List[str] = []
    gaps: List[List[str]] = [[]]

    for piece in PIECES_PATTERN.findall(text):
        if piece in TARGET_SYMBOL_SET:
            gaps[-1].append(piece)
        else:
            words.append(piece)
            gaps.append([])

    return words, gaps


def extract_parenthesis_spans(
    gaps: List[List[str]],
    max_nesting_depth: int = 2
) -> Tuple[List[Tuple[int, int]], Optional[str]]:
    """Extract (start_gap, end_gap) parenthesis spans using a stack.

    Enforces:
    - start_gap < end_gap
    - non-empty enclosed word interval (at least 1 word inside)
    - nesting depth <= max_nesting_depth (e.g. 2)
    - no duplicate intervals with identical (start_gap, end_gap)
    - all parentheses must be properly matched

    Returns:
        (spans, error_reason)
    """
    stack: List[int] = []  # stores start_gap indices
    spans: List[Tuple[int, int]] = []
    seen_spans = set()

    for gap_idx, marks in enumerate(gaps):
        for mark in marks:
            if mark == '(':
                stack.append(gap_idx)
                if len(stack) > max_nesting_depth:
                    return [], f"nesting_exceeds_max_{len(stack)}"
            elif mark == ')':
                if not stack:
                    return [], "unmatched_closing_paren"
                start_gap = stack.pop()
                if start_gap == gap_idx:
                    return [], "empty_parenthesis_pair"
                span = (start_gap, gap_idx)
                if span in seen_spans:
                    return [], "duplicate_identical_span"
                seen_spans.add(span)
                spans.append(span)

    if stack:
        return [], f"unmatched_opening_paren_count_{len(stack)}"

    # Sort spans by start_gap then end_gap
    spans.sort(key=lambda s: (s[0], s[1]))
    return spans, None


def gaps_to_slot_ids(gaps: List[List[str]]) -> List[List[int]]:
    """Converts gaps into 4-slot label ID arrays.

    Punctuation sequence is followed by a STOP token.
    Unused trailing slots are filled with IGNORE (-100).
    """
    slot_ids: List[List[int]] = []
    for marks in gaps:
        row = [IGNORE_LABEL_ID] * MAX_SLOTS_PER_GAP
        num_marks = min(len(marks), MAX_SLOTS_PER_GAP)
        for s in range(num_marks):
            row[s] = LABEL_TO_ID.get(marks[s], IGNORE_LABEL_ID)
        # If there are fewer than 4 marks, place STOP immediately after
        if num_marks < MAX_SLOTS_PER_GAP:
            row[num_marks] = LABEL_TO_ID['STOP']
        slot_ids.append(row)
    return slot_ids


def slot_ids_to_gaps(slot_ids: List[List[int]]) -> List[List[str]]:
    """Converts 4-slot label ID arrays back to gap punctuation mark lists."""
    gaps: List[List[str]] = []
    for row in slot_ids:
        marks: List[str] = []
        for class_id in row:
            if class_id == LABEL_TO_ID['STOP'] or class_id == IGNORE_LABEL_ID:
                break
            if class_id in ID_TO_LABEL:
                marks.append(ID_TO_LABEL[class_id])
        gaps.append(marks)
    return gaps
