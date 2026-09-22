"""Canonical renderer assembling words and gap punctuation into formatted Bangla text.

Adheres strictly to the standard punctuation spacing conventions:
- Trailing punctuation (।, ?, !, ,, ;, :, )) attaches to the immediately preceding token.
- Opening parenthesis ('(') attaches to the succeeding word/token, preceded by a space
  unless following another opening parenthesis.
- Preserves exact symbol ordering at gaps.
"""

from __future__ import annotations
from typing import List, Sequence

TRAILING_MARKS = {'।', ',', '?', '!', ';', ':', ')'}


def render(words: Sequence[str], gaps: Sequence[Sequence[str]]) -> str:
    """Renders words and punctuation at gaps into canonical punctuated text.

    Args:
        words: List of N word tokens.
        gaps: List of N+1 gap symbol lists.

    Returns:
        Canonical punctuated string.
    """
    if len(gaps) != len(words) + 1:
        raise ValueError(
            f"Expected {len(words) + 1} gaps for {len(words)} words, got {len(gaps)}"
        )

    # Flatten pieces into a linear sequence of tokens and symbols
    pieces: List[str] = list(gaps[0])
    for i, word in enumerate(words):
        pieces.append(word)
        pieces.extend(gaps[i + 1])

    output_parts: List[str] = []
    previous: str | None = None

    for piece in pieces:
        if not piece:
            continue

        if piece in TRAILING_MARKS:
            # Trailing mark attaches directly to the preceding content
            output_parts.append(piece)
        elif piece == '(':
            # Opening parenthesis preceded by a space unless it is at the start or follows another '('
            if output_parts and previous != '(':
                output_parts.append(' ')
            output_parts.append(piece)
        else:
            # A regular word
            if output_parts and previous != '(':
                output_parts.append(' ')
            output_parts.append(piece)

        previous = piece

    return ''.join(output_parts)


# Public alias
render_punctuated_text = render
