"""Normalization and exclusion filtering for Bangla punctuation restoration.

Defines text normalization (Unicode NFC, whitespace) and validates that texts
conform to the 8-symbol benchmark scope, recording explicit exclusion reasons.
"""

from __future__ import annotations
import re
import unicodedata
from typing import Tuple, Optional, List

# Target 8 punctuation symbols: 7 ASCII symbols + Bangla dari (U+0964)
TARGET_MARKS = {'।', ',', '?', '!', ';', ':', '(', ')'}
TARGET_MARKS_STR = '।,?!;:()'

# Non-target punctuation characters that trigger exclusion
UNSUPPORTED_BRACKETS = {'[', ']', '{', '}'}
QUOTATION_MARKS = {'"', "'", '“', '”', '‘', '’', '`'}
STANDALONE_DASHES = {'—', '–'}  # Em-dash, en-dash

# Regex to detect URLs
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

# Regex to detect times with internal colon (e.g. 12:30 or ১২:৩০)
TIME_PATTERN = re.compile(r'\b(?:\d{1,2}|[০-৯]{1,2}):(?:\d{2}|[০-৯]{2})\b')


def normalize_text(text: str) -> str:
    """Normalize text using Unicode NFC and standardized single whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize('NFC', text)
    return ' '.join(text.split())


def check_scope_and_exclusions(
    normalized_text: str,
    max_words: int = 128
) -> Tuple[bool, Optional[str]]:
    """Checks whether a normalized text is within benchmark scope.

    Returns:
        (is_valid, exclusion_reason)
    """
    if not normalized_text or normalized_text.isspace():
        return False, "empty_or_whitespace"

    # Check for URLs
    if URL_PATTERN.search(normalized_text):
        return False, "contains_url"

    # Check for time format with internal colon
    if TIME_PATTERN.search(normalized_text):
        return False, "contains_time_colon"

    # Check for unsupported brackets
    for ch in UNSUPPORTED_BRACKETS:
        if ch in normalized_text:
            return False, f"unsupported_bracket_{ch}"

    # Check for quotation structures
    for ch in QUOTATION_MARKS:
        if ch in normalized_text:
            return False, "contains_quotation_mark"

    # Check for standalone em/en dashes
    for ch in STANDALONE_DASHES:
        if ch in normalized_text:
            return False, f"unsupported_dash_{ch}"

    return True, None
