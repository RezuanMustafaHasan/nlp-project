"""Persistent JSON-lines inference worker for the Node/Express API."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS: dict[str, Any] = {}


def import_module(name: str, source: Path):
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def checkpoint_state(checkpoint: Any) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("model_state", "model_state_dict"):
        if key in checkpoint:
            return checkpoint[key]
    return checkpoint


def load_model(model_name: str):
    if model_name in MODELS:
        return MODELS[model_name]

    if model_name == "bert":
        module = import_module("punctuation_bert_model", ROOT / "bert" / "model.py")
        model = module.AlignedBanglaBERTPunctuation(
            load_pretrained_backbone=False,
        ).to(DEVICE)
        checkpoint_path = ROOT / "bert" / "model" / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint_state(checkpoint))
        del checkpoint
        model.eval()
        MODELS[model_name] = (model, None)
    elif model_name == "bilstm":
        module = import_module("punctuation_bilstm_model", ROOT / "bilstm" / "model.py")
        model = module.PackedBiLSTMPunctuation().to(DEVICE)
        checkpoint_path = ROOT / "bilstm" / "model" / "best_bilstm_punct.pt"
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint_state(checkpoint))
        del checkpoint
        model.eval()
        embeddings_loaded = model.load_embeddings()

        embedding_cache: dict[str, np.ndarray] = {}

        def word_vector(word: str) -> np.ndarray:
            if embeddings_loaded:
                return model.get_word_vector(word)
            if word not in embedding_cache:
                seed = int(hashlib.sha256(word.encode("utf-8")).hexdigest()[:8], 16)
                generator = np.random.RandomState(seed)
                embedding_cache[word] = generator.normal(0.0, 0.1, 300).astype(np.float32)
            return embedding_cache[word]

        MODELS[model_name] = (model, word_vector)
    else:
        raise ValueError("Unknown model selection.")

    return MODELS[model_name]


def restore(text: str, model_name: str) -> str:
    model, word_vector = load_model(model_name)
    if model_name == "bert":
        return model.restore_punctuation(text, device=DEVICE)
    return model.restore_punctuation(text, word2vec_fn=word_vector, device=DEVICE)


def respond(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    for raw_line in sys.stdin:
        request_id = None
        try:
            request = json.loads(raw_line)
            request_id = request.get("id")
            if request.get("action") != "restore":
                raise ValueError("Unsupported inference action.")

            text = str(request.get("text", "")).strip()
            model_name = request.get("model")
            if not text:
                raise ValueError("Input text is empty.")

            output = restore(text, model_name)
            respond({"id": request_id, "output": output})
        except Exception as error:  # Keep the worker alive for future requests.
            respond({"id": request_id, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
