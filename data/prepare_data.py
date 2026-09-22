"""Data processing, validation, deduplication, and split leakage protection.

Implements Steps 7, 8, 10, and 11:
1. Normalizes text using NFC and whitespace conventions.
2. Filters out-of-scope examples (URLs, colons in times, non-target brackets, quotes, dashes, length > 128).
3. Parses gaps and extracts parenthesis spans (max nesting <= 2, matched pairs).
4. Collapses identical canonical word-and-label rows within each split, preserving row IDs and nopr values.
5. Identifies ambiguous input sequences across splits and excludes them from the single-reference benchmark.
6. Resolves cross-split duplicates using strict priority: test > validation > train.
7. Saves processed records in JSONL and outputs frozen manifests with zero cross-split overlap assertions.
"""

from __future__ import annotations
import json
import hashlib
import logging
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple, Set

from normalize import normalize_text, check_scope_and_exclusions
from labels import parse_gaps, extract_parenthesis_spans, MAX_SLOTS_PER_GAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_sha256(text: str) -> str:
    """Compute SHA-256 hash of a UTF-8 string."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def process_raw_row(
    raw_id: str,
    target_raw: str,
    original_nopr: Any,
    original_split: str,
    max_words: int = 128
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validates and processes a single raw corpus row.

    Returns:
        (record_dict, exclusion_reason)
    """
    normalized = normalize_text(target_raw)
    is_valid, reason = check_scope_and_exclusions(normalized, max_words=max_words)
    if not is_valid:
        return None, reason

    words, gaps = parse_gaps(normalized)

    # Word count check
    if len(words) == 0:
        return None, "zero_words"
    if len(words) > max_words:
        return None, f"words_exceed_max_{len(words)}"

    # Max slots per gap check
    for gap in gaps:
        if len(gap) > MAX_SLOTS_PER_GAP:
            return None, f"gap_marks_exceed_max_{len(gap)}"

    # Parenthesis span extraction & validation
    spans, span_err = extract_parenthesis_spans(gaps)
    if span_err:
        return None, span_err

    input_text = " ".join(words)
    input_key = compute_sha256(input_text)
    canonical_repr = json.dumps({"words": words, "gaps": gaps}, ensure_ascii=False)
    target_key = compute_sha256(canonical_repr)

    gold_mark_count = sum(len(g) for g in gaps)

    record = {
        "example_id": "",  # Assigned after deduplication
        "raw_row_ids": [raw_id],
        "original_split": original_split,
        "input_key": input_key,
        "target_key": target_key,
        "words": words,
        "input_text": input_text,
        "gap_punctuation": gaps,
        "parenthesis_spans": spans,
        "gold_mark_count": gold_mark_count,
        "original_nopr_values": [original_nopr] if original_nopr is not None else [],
    }
    return record, None


def deduplicate_and_partition(
    split_records: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    manifests_dir: Path
) -> Dict[str, Any]:
    """Applies cross-split grouping, ambiguity filtering, and priority assignment.

    Priority order: test > validation > train
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Internal split dedup (identical canonical input_key AND target_key)
    logger.info("Collapsing internal split duplicates...")
    collapsed_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {
        "train": {}, "validation": {}, "test": {}
    }

    internal_dedup_counts = Counter()
    for split_name in ["train", "validation", "test"]:
        for rec in split_records.get(split_name, []):
            unique_k = f"{rec['input_key']}:{rec['target_key']}"
            if unique_k in collapsed_by_split[split_name]:
                existing = collapsed_by_split[split_name][unique_k]
                existing["raw_row_ids"].extend(rec["raw_row_ids"])
                existing["original_nopr_values"].extend(rec["original_nopr_values"])
                internal_dedup_counts[split_name] += 1
            else:
                collapsed_by_split[split_name][unique_k] = rec

    logger.info(f"Internal duplicate collapses: {dict(internal_dedup_counts)}")

    # Step 2 & 3: Group by input_key across all splits to check ambiguity
    logger.info("Grouping by input_key to detect ambiguous references...")
    all_by_input_key: Dict[str, Dict[str, List[Tuple[str, Dict[str, Any]]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for split_name in ["train", "validation", "test"]:
        for rec in collapsed_by_split[split_name].values():
            inp_k = rec["input_key"]
            tgt_k = rec["target_key"]
            all_by_input_key[inp_k][tgt_k].append((split_name, rec))

    ambiguous_input_keys = set()
    ambiguous_records_report = []

    for inp_k, target_map in all_by_input_key.items():
        if len(target_map) > 1:
            ambiguous_input_keys.add(inp_k)
            sample_rec = next(iter(target_map.values()))[0][1]
            ambiguous_records_report.append({
                "input_key": inp_k,
                "input_text": sample_rec["input_text"],
                "num_variants": len(target_map),
                "variants": [
                    {
                        "target_key": tgt_k,
                        "splits": [s for s, _ in items],
                        "gaps": items[0][1]["gap_punctuation"],
                    }
                    for tgt_k, items in target_map.items()
                ]
            })

    logger.info(f"Identified {len(ambiguous_input_keys):,} ambiguous input groups. Saving exclusion manifest...")
    (manifests_dir / "ambiguous_exclusions.json").write_text(
        json.dumps(ambiguous_records_report, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )

    # Step 4: For unambiguous inputs, enforce split priority: test > validation > train
    logger.info("Applying split priority (test > validation > train) on unambiguous inputs...")
    priority_order = ["test", "validation", "train"]
    final_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }
    cross_split_leakage_prevented = Counter()

    for inp_k, target_map in all_by_input_key.items():
        if inp_k in ambiguous_input_keys:
            continue

        tgt_k, split_items = next(iter(target_map.items()))
        splits_present = [s for s, _ in split_items]

        # Determine winning split by priority
        assigned_split = None
        for p_split in priority_order:
            if p_split in splits_present:
                assigned_split = p_split
                break

        # Merge metadata from all appearances into the winning split record
        winning_rec = None
        for s, rec in split_items:
            if s == assigned_split:
                winning_rec = rec
                break

        for s, rec in split_items:
            if s != assigned_split:
                winning_rec["raw_row_ids"].extend(rec["raw_row_ids"])
                winning_rec["original_nopr_values"].extend(rec["original_nopr_values"])
                cross_split_leakage_prevented[f"{assigned_split}_held_over_{s}"] += 1

        final_records[assigned_split].append(winning_rec)

    # Assign final sequential example IDs and sort deterministically
    assigned_counts = {}
    for split_name in priority_order:
        final_records[split_name].sort(key=lambda r: r["input_key"])
        for idx, rec in enumerate(final_records[split_name]):
            rec["example_id"] = f"{split_name}:{idx:08d}"
        assigned_counts[split_name] = len(final_records[split_name])

    logger.info(f"Final partitioned counts: {assigned_counts}")
    logger.info(f"Cross-split leakage prevented: {dict(cross_split_leakage_prevented)}")

    # Step 5: Assert zero overlap
    train_keys = {r["input_key"] for r in final_records["train"]}
    val_keys = {r["input_key"] for r in final_records["validation"]}
    test_keys = {r["input_key"] for r in final_records["test"]}

    assert len(train_keys & val_keys) == 0, f"Train and Validation overlap: {len(train_keys & val_keys)}"
    assert len(train_keys & test_keys) == 0, f"Train and Test overlap: {len(train_keys & test_keys)}"
    assert len(val_keys & test_keys) == 0, f"Validation and Test overlap: {len(val_keys & test_keys)}"
    logger.info("Zero cross-split overlap assertion PASSED successfully.")

    # Save to jsonl
    for split_name in ["train", "validation", "test"]:
        out_file = output_dir / f"{split_name}.jsonl"
        with out_file.open("w", encoding="utf-8") as f:
            for rec in final_records[split_name]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(final_records[split_name]):,} records to {out_file}")

    # Write split manifest
    summary = {
        "final_counts": assigned_counts,
        "total_retained": sum(assigned_counts.values()),
        "internal_dedup_collapsed": dict(internal_dedup_counts),
        "ambiguous_groups_excluded": len(ambiguous_input_keys),
        "cross_split_preventions": dict(cross_split_leakage_prevented),
    }
    (manifests_dir / "partition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    return summary


def run_prepare_pipeline(
    raw_dir: Path = Path("data/raw_hf"),
    output_dir: Path = Path("data/processed"),
    manifests_dir: Path = Path("data/manifests"),
    max_words: int = 128
):
    """Executes end-to-end data preparation, filtering, and partitioning."""
    import csv

    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    manifests_dir = Path(manifests_dir)

    split_files = {
        "train": raw_dir / "train.csv",
        "validation": raw_dir / "validation.csv",
        "test": raw_dir / "test.csv",
    }

    split_records: Dict[str, List[Dict[str, Any]]] = {
        "train": [], "validation": [], "test": []
    }
    exclusion_reasons = Counter()

    for split_name, csv_path in split_files.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing raw CSV file: {csv_path}")

        logger.info(f"Processing raw rows from {csv_path}...")
        with csv_path.open("r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                raw_id = f"{split_name}:{row_idx:08d}"
                target_text = row.get("target", "")
                nopr_val = row.get("nopr", "")

                rec, reason = process_raw_row(
                    raw_id=raw_id,
                    target_raw=target_text,
                    original_nopr=nopr_val,
                    original_split=split_name,
                    max_words=max_words
                )

                if rec is not None:
                    split_records[split_name].append(rec)
                else:
                    exclusion_reasons[f"{split_name}:{reason}"] += 1

                if (row_idx + 1) % 250000 == 0:
                    logger.info(f"[{split_name}] Processed {row_idx + 1:,} rows. Valid: {len(split_records[split_name]):,}")

        logger.info(f"Finished {split_name}: {len(split_records[split_name]):,} valid examples.")

    (manifests_dir / "exclusions_breakdown.json").write_text(
        json.dumps(dict(exclusion_reasons), ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    logger.info(f"Saved exclusions breakdown ({sum(exclusion_reasons.values()):,} total exclusions) to manifests.")

    summary = deduplicate_and_partition(split_records, output_dir, manifests_dir)
    logger.info("Data preparation pipeline complete!")
    return summary


if __name__ == "__main__":
    run_prepare_pipeline()

