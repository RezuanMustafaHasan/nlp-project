"""Interactive and programmatic prediction tool for Bangla Punctuation Restoration.

Supports multi-model inference across recent high-accuracy and baseline models:
1. banglabert: Fine-tuned BanglaBERT with aligned pooling (68.58% Macro-F1, 83.74% Exact Match - Highest Accuracy)
2. bilstm: Packed fastText–BiLSTM with per-sentence EOS (50.32% Macro-F1, 77.81% Exact Match)
3. compact_joint: Compact Joint Transformer Post-79 continuation (47.25% Macro-F1, 78.15% Exact Match)
4. legacy: Historical 5-epoch Compact Joint baseline (45.99% Macro-F1, 77.11% Exact Match)
"""

from __future__ import annotations
import argparse
import json
import logging
import math
from pathlib import Path
import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Sequence

# Ensure local 'src' modules can be imported
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

from model import BanglaPunctuationTransformer
from post79_models import PackedBiLSTM, AlignedBanglaBERT, word_alignment
from normalize import normalize_text
from labels import extract_parenthesis_spans, TARGET_SYMBOLS
from decode import decode_batch
from render import render

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = SRC_DIR.parent
MAX_WORDS_LIMIT = 128


def resolve_cache_dir() -> Path:
    candidates = [
        REPO_ROOT / "embeddings" / "cache",
        REPO_ROOT / "models" / "cache",
        REPO_ROOT.parent / "embeddings" / "cache",
    ]
    for c in candidates:
        if (c / "word2id.json").is_file():
            return c
    return candidates[0]


DEFAULT_CACHE_DIR = resolve_cache_dir()


def resolve_default_checkpoint(model_key: str) -> Path:
    candidates = [
        REPO_ROOT / "models" / model_key / "best.pt",
        REPO_ROOT / "checkpoints" / model_key / "best.pt",
        REPO_ROOT / "models" / f"{model_key}.pt",
    ]
    if model_key == "banglabert":
        candidates.extend([
            REPO_ROOT / "results" / "post79_improvement" / "run_24h" / "banglabert_aligned_weighted_s42" / "best.pt",
            REPO_ROOT.parent / "results" / "post79_improvement" / "run_24h" / "banglabert_aligned_weighted_s42" / "best.pt",
            REPO_ROOT.parent / "bert" / "model" / "bert" / "best.pt",
        ])
    elif model_key == "bilstm":
        candidates.extend([
            REPO_ROOT / "results" / "post79_improvement" / "run_24h" / "bilstm_packed_weighted_s42" / "best.pt",
            REPO_ROOT.parent / "results" / "post79_improvement" / "run_24h" / "bilstm_packed_weighted_s42" / "best.pt",
            REPO_ROOT.parent / "bilstm" / "checkpoints" / "best_bilstm_punct.pt",
        ])
    elif model_key == "compact_distilled":
        candidates.extend([
            REPO_ROOT / "results" / "post79_improvement" / "distillation_s42" / "best.pt",
            REPO_ROOT.parent / "results" / "post79_improvement" / "distillation_s42" / "best.pt",
        ])
    elif model_key == "compact_joint":
        candidates.extend([
            REPO_ROOT / "results" / "post79_improvement" / "run_24h" / "compact_ce_s42" / "best.pt",
            REPO_ROOT.parent / "results" / "post79_improvement" / "run_24h" / "compact_ce_s42" / "best.pt",
        ])
    elif model_key == "legacy":
        candidates.extend([
            REPO_ROOT / "checkpoints" / "revision_l4_50h" / "compact_joint" / "seed_42" / "best.pt",
            REPO_ROOT.parent / "checkpoints" / "revision_l4_50h" / "compact_joint" / "seed_42" / "best.pt",
        ])

    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


MODEL_REGISTRY = {
    "banglabert": {
        "name": "BanglaBERT (Fine-Tuned & Aligned)",
        "family": "banglabert",
        "default_checkpoint": resolve_default_checkpoint("banglabert"),
        "macro_f1": 68.58,
        "exact_match": 83.74,
        "has_span_head": False,
        "description": "Pretrained foundation model with aligned pooling (Highest Accuracy).",
    },
    "bilstm": {
        "name": "Packed fastText–BiLSTM",
        "family": "bilstm",
        "default_checkpoint": resolve_default_checkpoint("bilstm"),
        "macro_f1": 50.32,
        "exact_match": 77.81,
        "has_span_head": False,
        "description": "Lightweight 2-layer recurrent baseline with packed sequences.",
    },
    "compact_distilled": {
        "name": "Compact Distilled Transformer (Step 88b)",
        "family": "compact_joint",
        "default_checkpoint": resolve_default_checkpoint("compact_distilled"),
        "macro_f1": 47.70,
        "exact_match": 78.53,
        "has_span_head": True,
        "description": "Distilled from fine-tuned BanglaBERT teacher into Compact Joint with Bilinear Span Head (78.53% Exact Match).",
    },
    "compact_joint": {
        "name": "Compact Joint Transformer (Post-79)",
        "family": "compact_joint",
        "default_checkpoint": resolve_default_checkpoint("compact_joint"),
        "macro_f1": 47.25,
        "exact_match": 78.15,
        "has_span_head": True,
        "description": "Our proposed lightweight Transformer with bilinear parenthesis span head.",
    },
    "legacy": {
        "name": "Legacy Compact Joint (Epoch 5)",
        "family": "compact_joint",
        "default_checkpoint": resolve_default_checkpoint("legacy"),
        "macro_f1": 45.99,
        "exact_match": 77.11,
        "has_span_head": True,
        "description": "Historical 5-epoch baseline checkpoint.",
    },
}


class PunctuationRestorer:
    """Production inference engine supporting multiple Bangla punctuation models."""

    def __init__(
        self,
        default_model: str = "banglabert",
        cache_dir: str | Path = DEFAULT_CACHE_DIR,
        device: Optional[str] = None,
        checkpoint_path: Optional[str | Path] = None,
    ):
        if default_model not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {default_model}")
        self.default_model = default_model
        self.default_checkpoint = checkpoint_path
        self.cache_dir = Path(cache_dir)

        # Select device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using compute device: {self.device}")

        # Load embedding cache (for fastText models)
        self.word2id: Dict[str, int] = {}
        self.embedding_matrix: Optional[np.ndarray] = None
        self._load_embeddings()

        # Model instances cache
        self.loaded_models: Dict[str, Dict[str, Any]] = {}

        # Preload the default model
        self._get_or_load_model(self.default_model, checkpoint_path)

    def _load_embeddings(self) -> None:
        """Loads word2id mapping and pre-extracted vector matrix into memory."""
        word2id_path = self.cache_dir / "word2id.json"
        vocab_vectors_path = self.cache_dir / "vocab_vectors.npy"

        if word2id_path.exists() and vocab_vectors_path.exists():
            logger.info(f"Loading vocabulary mapping from {word2id_path}...")
            with word2id_path.open("r", encoding="utf-8") as f:
                self.word2id = json.load(f)

            logger.info(f"Loading embedding matrix from {vocab_vectors_path}...")
            self.embedding_matrix = np.load(vocab_vectors_path, mmap_mode="r")
            if (self.embedding_matrix.ndim != 2 or self.embedding_matrix.shape[1] != 300
                    or not self.word2id or min(self.word2id.values()) < 0
                    or max(self.word2id.values()) >= len(self.embedding_matrix)):
                raise ValueError("Invalid fastText embedding cache")
            logger.info(
                f"Embedding cache loaded: {len(self.word2id):,} words, "
                f"shape {self.embedding_matrix.shape}."
            )
        else:
            logger.warning(
                f"Embedding cache not found at {self.cache_dir}. "
                "fastText models will be unavailable until the cache is restored."
            )

    def _resolve_checkpoint(self, model_key: str, custom_path: Optional[str | Path] = None) -> Tuple[Path, str, bool]:
        """Resolves checkpoint path and family configuration."""
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {model_key}")
        info = MODEL_REGISTRY[model_key]
        checkpoint = Path(custom_path) if custom_path is not None else info["default_checkpoint"]
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint unavailable for {model_key}: {checkpoint}")
        return checkpoint, info["family"], info["has_span_head"]

    def _get_or_load_model(self, model_key: str, custom_path: Optional[str | Path] = None) -> Dict[str, Any]:
        """Retrieves an already loaded model or instantiates and caches it."""
        cache_key = f"{model_key}:{str(custom_path) if custom_path else 'default'}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]

        ckpt_path, family, has_span_head = self._resolve_checkpoint(model_key, custom_path)
        if family != "banglabert" and (self.embedding_matrix is None or not self.word2id):
            raise FileNotFoundError(f"fastText embedding cache is required for {model_key}: {self.cache_dir}")
        logger.info(f"Loading model '{model_key}' (family: {family}) from {ckpt_path}...")

        tokenizer = None
        if family == "banglabert":
            from transformers import AutoConfig, AutoModel, AutoTokenizer
            model_name = "csebuetnlp/banglabert"
            rev = "9ce791f330578f50da6bc52b54205166fb5d1c8c"
            try:
                hf_config = AutoConfig.from_pretrained(model_name, revision=rev, local_files_only=True)
                tokenizer = AutoTokenizer.from_pretrained(model_name, revision=rev, use_fast=True, local_files_only=True)
            except Exception:
                hf_config = AutoConfig.from_pretrained(model_name, revision=rev)
                tokenizer = AutoTokenizer.from_pretrained(model_name, revision=rev, use_fast=True)
            if not tokenizer.is_fast:
                raise ValueError("BanglaBERT requires a fast tokenizer with word alignment")
            model = AlignedBanglaBERT(AutoModel.from_config(hf_config))
        elif family == "bilstm":
            model = PackedBiLSTM()
        else:
            model = BanglaPunctuationTransformer(
                embed_dim=300,
                hidden_dim=384,
                num_layers=6,
                num_heads=6,
                ffn_dim=1536,
                dropout=0.0,
                max_positions=130,
                has_span_head=has_span_head,
                span_dim=128,
            )

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = state.get("student_state", state.get("model_state", state.get("model_state_dict", state)))
        model.load_state_dict(state_dict, strict=True)
        logger.info(f"Checkpoint {ckpt_path} successfully loaded.")

        model.to(self.device)
        model.eval()

        bundle = {
            "model": model,
            "family": family,
            "tokenizer": tokenizer,
            "has_span_head": has_span_head,
            "checkpoint_path": ckpt_path,
            "info": {**MODEL_REGISTRY[model_key],
                     "macro_f1": None if custom_path else MODEL_REGISTRY[model_key]["macro_f1"],
                     "exact_match": None if custom_path else MODEL_REGISTRY[model_key]["exact_match"]},
        }
        self.loaded_models[cache_key] = bundle
        return bundle

    def get_word_vector(self, word: str) -> np.ndarray:
        """Retrieves 300-dim vector for word or zero vector fallback."""
        if self.embedding_matrix is not None:
            w_id = self.word2id.get(word, -1)
            if 0 <= w_id < len(self.embedding_matrix):
                return self.embedding_matrix[w_id]
        return np.zeros(300, dtype=np.float32)

    def restore_punctuation(
        self,
        text: str,
        model_name: Optional[str] = None,
        method: str = "constrained",
        beam_width: int = 8,
        max_candidates: int = 16,
        beta: float = 0.3,
        custom_checkpoint: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Restores punctuation for an unpunctuated Bangla sentence."""
        start_time = time.perf_counter()
        chosen_model_key = model_name or self.default_model
        if chosen_model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {chosen_model_key}")
        if method not in ("constrained", "pairing_repair", "independent"):
            raise ValueError(f"Unknown decoder: {method}")
        if not isinstance(beam_width, int) or isinstance(beam_width, bool) or not 1 <= beam_width <= 32:
            raise ValueError("Beam width must be an integer from 1 to 32")
        if not isinstance(max_candidates, int) or not 1 <= max_candidates <= 64:
            raise ValueError("Candidate count must be an integer from 1 to 64")
        if not isinstance(beta, (int, float)) or not math.isfinite(beta) or not 0 <= beta <= 10:
            raise ValueError("Span bonus must be finite and between 0 and 10")
        if not isinstance(text, str):
            raise ValueError("Text must be a string")
        if custom_checkpoint is None and chosen_model_key == self.default_model:
            custom_checkpoint = self.default_checkpoint

        # Step 1: Input normalization
        norm_text = normalize_text(text)
        if not norm_text or norm_text.isspace():
            return {
                "original_text": text,
                "restored_text": "",
                "words": [],
                "predicted_gaps": [],
                "parenthesis_spans": [],
                "is_valid_syntax": True,
                "method": method,
                "model_name": chosen_model_key,
                "num_words": 0,
                "num_marks": 0,
                "latency_ms": 0.0,
                "error": "Empty or whitespace-only input",
            }

        # Step 2: Tokenize words
        raw_words = norm_text.split()
        if len(raw_words) > MAX_WORDS_LIMIT:
            return {
                "original_text": text,
                "restored_text": norm_text,
                "words": raw_words,
                "predicted_gaps": [],
                "parenthesis_spans": [],
                "method": method,
                "model_name": chosen_model_key,
                "num_words": len(raw_words),
                "num_marks": 0,
                "latency_ms": 0.0,
                "error": f"Input exceeds maximum word limit ({len(raw_words)} > {MAX_WORDS_LIMIT})",
            }

        n_w = len(raw_words)

        # Retrieve model bundle
        bundle = self._get_or_load_model(chosen_model_key, custom_checkpoint)
        model = bundle["model"]
        family = bundle["family"]
        tokenizer = bundle["tokenizer"]
        has_span_head = bundle["has_span_head"]

        # Step 3: Neural forward pass
        with torch.no_grad():
            if family == "banglabert":
                encoded = tokenizer([raw_words], is_split_into_words=True, padding=True, truncation=False, return_tensors="pt")
                if encoded["input_ids"].shape[1] > 512:
                    raise ValueError("Input exceeds the 512-subword limit; no words were truncated")
                word_ids = word_alignment(encoded, [raw_words])
                outputs = model.forward_aligned(
                    encoded["input_ids"].to(self.device),
                    encoded["attention_mask"].to(self.device),
                    word_ids.to(self.device),
                    [n_w]
                )
                gap_logits = outputs["gap_logits"]
                span_scores = None
            elif family == "bilstm":
                vectors = [self.get_word_vector(w) for w in raw_words]
                word_vectors_tensor = torch.tensor(np.array([vectors], dtype=np.float32), device=self.device)
                word_mask_tensor = torch.ones((1, n_w), dtype=torch.bool, device=self.device)
                outputs = model(
                    word_vectors=word_vectors_tensor,
                    word_mask=word_mask_tensor,
                    real_lengths=[n_w],
                )
                gap_logits = outputs["gap_logits"]
                span_scores = None
            else:  # compact_joint or legacy
                vectors = [self.get_word_vector(w) for w in raw_words]
                word_vectors_tensor = torch.tensor(np.array([vectors], dtype=np.float32), device=self.device)
                word_mask_tensor = torch.ones((1, n_w), dtype=torch.bool, device=self.device)
                encoder_mask_tensor = torch.zeros((1, n_w + 2), dtype=torch.bool, device=self.device)
                outputs = model(
                    word_vectors=word_vectors_tensor,
                    word_mask=word_mask_tensor,
                    encoder_mask=encoder_mask_tensor,
                    real_lengths=[n_w],
                )
                gap_logits = outputs["gap_logits"]
                span_scores = outputs.get("span_scores") if has_span_head else None

            # Step 4: Decode gap marks
            effective_beta = beta if (has_span_head and span_scores is not None) else 0.0
            decoded_batch = decode_batch(
                gap_logits=gap_logits,
                span_scores=span_scores,
                real_lengths=[n_w],
                method=method,
                beam_width=beam_width,
                max_candidates=max_candidates,
                beta=effective_beta,
            )

        predicted_gaps = decoded_batch[0]

        # Step 5: Render text and extract spans
        restored_text = render(raw_words, predicted_gaps)
        spans, syntax_error = extract_parenthesis_spans(predicted_gaps, max_nesting_depth=2)

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        num_marks = sum(len(g) for g in predicted_gaps)

        return {
            "original_text": text,
            "restored_text": restored_text,
            "words": raw_words,
            "predicted_gaps": predicted_gaps,
            "parenthesis_spans": spans,
            "is_valid_syntax": syntax_error is None,
            "syntax_error": syntax_error,
            "method": method,
            "model_name": chosen_model_key,
            "model_display": bundle["info"].get("name", chosen_model_key),
            "macro_f1": bundle["info"].get("macro_f1"),
            "exact_match": bundle["info"].get("exact_match"),
            "score_context": "Natural validation; independent decoding; not a confidence score for this input",
            "num_words": n_w,
            "num_marks": num_marks,
            "latency_ms": round(elapsed_ms, 2),
            "error": None,
        }


# Global singleton
_DEFAULT_RESTORER: Optional[PunctuationRestorer] = None


def get_restorer(
    default_model: str = "banglabert",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: Optional[str] = None,
) -> PunctuationRestorer:
    """Returns singleton instance of PunctuationRestorer."""
    global _DEFAULT_RESTORER
    if _DEFAULT_RESTORER is None:
        _DEFAULT_RESTORER = PunctuationRestorer(
            default_model=default_model,
            cache_dir=cache_dir,
            device=device,
        )
    return _DEFAULT_RESTORER


def restore_punctuation(
    text: str,
    model: str = "banglabert",
    method: str = "constrained",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    device: Optional[str] = None,
    beam_width: int = 8,
    beta: float = 0.3,
) -> str:
    """Convenience functional API returning restored text string."""
    restorer = get_restorer(default_model=model, cache_dir=cache_dir, device=device)
    result = restorer.restore_punctuation(
        text=text, model_name=model, method=method, beam_width=beam_width, beta=beta
    )
    return result["restored_text"]


def main():
    parser = argparse.ArgumentParser(description="Bangla Punctuation Restoration Prediction Tool")
    parser.add_argument("--text", type=str, default=None, help="Input unpunctuated text")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI prompt")
    parser.add_argument("--input-file", type=Path, default=None, help="Input text file (1 sentence/line)")
    parser.add_argument("--output-file", type=Path, default=None, help="Output file for restored text")
    parser.add_argument(
        "--model",
        type=str,
        default="banglabert",
        choices=["banglabert", "bilstm", "compact_distilled", "compact_joint", "legacy"],
        help="Model architecture (default: banglabert)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="constrained",
        choices=["constrained", "pairing_repair", "independent"],
        help="Decoding method (default: constrained)",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional custom checkpoint path")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Embeddings cache dir")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (auto, cuda, mps, cpu)")
    parser.add_argument("--beta", type=float, default=0.3, help="Learned span score bonus in beam search")
    parser.add_argument("--beam-width", type=int, default=8, help="Beam width for constrained search")
    parser.add_argument("--daemon", action="store_true", help="Run persistent IPC daemon over stdin/stdout")
    parser.add_argument("--json", action="store_true", help="Output full JSON result")

    args = parser.parse_args()

    restorer = PunctuationRestorer(
        default_model=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
        checkpoint_path=args.checkpoint,
    )

    if args.daemon:
        logger.info("PunctuationRestorer daemon active. Listening on stdin for JSON requests...")
        sys.stdout.write(json.dumps({"status": "ready", "device": str(restorer.device), "default_model": args.model}) + "\n")
        sys.stdout.flush()
        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                req = {}
                try:
                    req = json.loads(line)
                    req_id = req.get("id")
                    text = req.get("text", "")
                    req_model = req.get("model", args.model)
                    method = req.get("method", "constrained")
                    beam_width = int(req.get("beam_width", 8))
                    beta = float(req.get("beta", 0.3))
                    res = restorer.restore_punctuation(
                        text=text,
                        model_name=req_model,
                        method=method,
                        beam_width=beam_width,
                        beta=beta,
                        custom_checkpoint=args.checkpoint,
                    )
                    res["id"] = req_id
                    sys.stdout.write(json.dumps(res, ensure_ascii=False) + "\n")
                    sys.stdout.flush()
                except Exception as ex:
                    sys.stdout.write(
                        json.dumps({"id": req.get("id"), "error": str(ex)}, ensure_ascii=False) + "\n"
                    )
                    sys.stdout.flush()
        except (KeyboardInterrupt, BrokenPipeError):
            pass
        return

    if args.interactive:
        print("\n=======================================================")
        print(" Bangla Punctuation Restoration - Interactive Console")
        print(f" Model: {args.model} | Method: {args.method} | Device: {restorer.device}")
        print(" Type a Bangla sentence and press Enter (or 'quit' to exit)")
        print("=======================================================\n")
        try:
            while True:
                user_input = input("Input > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                result = restorer.restore_punctuation(
                    text=user_input,
                    model_name=args.model,
                    method=args.method,
                    beam_width=args.beam_width,
                    beta=args.beta,
                    custom_checkpoint=args.checkpoint,
                )
                if args.json:
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                else:
                    print(f"Output: {result['restored_text']}")
                    print(
                        f"[{result.get('model_display', args.model)} | Tokens: {result['num_words']} | "
                        f"Marks: {result['num_marks']} | Spans: {len(result['parenthesis_spans'])} | {result['latency_ms']} ms]\n"
                    )
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
        return

    if args.text:
        result = restorer.restore_punctuation(
            text=args.text,
            model_name=args.model,
            method=args.method,
            beam_width=args.beam_width,
            beta=args.beta,
            custom_checkpoint=args.checkpoint,
        )
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(result["restored_text"])
        return

    if args.input_file:
        if not args.input_file.exists():
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)

        lines = [l.strip() for l in args.input_file.read_text(encoding="utf-8").splitlines() if l.strip()]
        logger.info(f"Processing {len(lines)} lines from {args.input_file}...")

        results = []
        for line in lines:
            res = restorer.restore_punctuation(
                text=line,
                model_name=args.model,
                method=args.method,
                beam_width=args.beam_width,
                beta=args.beta,
                custom_checkpoint=args.checkpoint,
            )
            results.append(res)

        if args.output_file:
            args.output_file.parent.mkdir(parents=True, exist_ok=True)
            with args.output_file.open("w", encoding="utf-8") as f:
                if args.json:
                    for r in results:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                else:
                    for r in results:
                        f.write(r["restored_text"] + "\n")
            logger.info(f"Saved {len(results)} restored sentences to {args.output_file}.")
        else:
            for r in results:
                print(r["restored_text"])
        return

    parser.print_help()


if __name__ == "__main__":
    main()
