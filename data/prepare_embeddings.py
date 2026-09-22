"""Embedding downloading, vector extraction, and memory-efficient caching.

Implements Steps 14 & 15:
- Downloads official fastText Bengali crawl model (cc.bn.300.bin)
- Builds unique vocabulary from processed corpus splits
- Pre-extracts 300-dimensional float32 vector matrix once
- Saves numpy matrix and word2id mapping to disk
- Supports OOV fallback via fastText subword synthesis
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.bin.gz"


def collect_vocabulary(jsonl_paths: List[Path]) -> List[str]:
    """Collects sorted distinct word vocabulary across given jsonl splits."""
    vocab: Set[str] = set()
    for path in jsonl_paths:
        if not path.exists():
            continue
        logger.info(f"Scanning vocabulary from {path}...")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    for w in rec.get("words", []):
                        vocab.add(w)
    sorted_vocab = sorted(vocab)
    logger.info(f"Total unique vocabulary size: {len(sorted_vocab):,} words.")
    return sorted_vocab


def build_embedding_cache(
    vocab: List[str],
    fasttext_model_path: Path,
    output_dir: Path
) -> Tuple[Path, Path]:
    """Extracts vectors using fastText and saves float32 numpy array and word2id map."""
    output_dir.mkdir(parents=True, exist_ok=True)
    import fasttext

    logger.info(f"Loading fastText model from {fasttext_model_path}...")
    ft = fasttext.load_model(str(fasttext_model_path))

    dim = ft.get_dimension()
    assert dim == 300, f"Expected 300 dimensions, got {dim}"

    n_words = len(vocab)
    logger.info(f"Extracting vectors for {n_words:,} unique words...")

    matrix = np.zeros((n_words, dim), dtype=np.float32)
    word2id: Dict[str, int] = {}

    for idx, word in enumerate(vocab):
        word2id[word] = idx
        matrix[idx] = ft.get_word_vector(word)

    matrix_path = output_dir / "vocab_vectors.npy"
    map_path = output_dir / "word2id.json"

    np.save(matrix_path, matrix)
    map_path.write_text(json.dumps(word2id, ensure_ascii=False), encoding='utf-8')

    logger.info(f"Saved {n_words:,} vectors to {matrix_path} ({matrix.nbytes / (1024*1024):.1f} MB)")
    logger.info(f"Saved word2id mapping to {map_path}")
    return matrix_path, map_path


if __name__ == "__main__":
    import sys
    # Usage: python prepare_embeddings.py <fasttext_bin_path>
    raw_dir = Path("embeddings/raw")
    cache_dir = Path("embeddings/cache")
    processed_dir = Path("data/processed")

    model_path = raw_dir / "cc.bn.300.bin"
    if model_path.exists():
        v = collect_vocabulary([
            processed_dir / "train.jsonl",
            processed_dir / "validation.jsonl",
            processed_dir / "test.jsonl"
        ])
        build_embedding_cache(v, model_path, cache_dir)
    else:
        logger.warning(f"Model {model_path} not found. Please download fastText model first.")
