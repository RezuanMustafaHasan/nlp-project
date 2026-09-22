"""Persistent JSON-lines bridge to the bundled canonical inference engine.

The React application uses ``bert`` as its public model id. The canonical
inference implementation calls the same model ``banglabert``; this worker is
the only compatibility layer between those two interfaces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


INFERENCE_ROOT = Path(__file__).resolve().parent
REFERENCE_SOURCE = INFERENCE_ROOT / "src"
REFERENCE_CACHE = INFERENCE_ROOT / "embeddings" / "cache"

if not REFERENCE_SOURCE.is_dir():
    raise FileNotFoundError(f"Canonical inference source is missing: {REFERENCE_SOURCE}")

# predict.py deliberately imports its sibling modules by their top-level names.
sys.path.insert(0, str(REFERENCE_SOURCE))

from predict import PunctuationRestorer  # noqa: E402


PUBLIC_TO_REFERENCE_MODEL = {
    "bert": "banglabert",
    "banglabert": "banglabert",
    "bilstm": "bilstm",
}


def respond(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main() -> None:
    # Eagerly load the default BanglaBERT checkpoint and lazily cache the
    # BiLSTM on first use.
    restorer = PunctuationRestorer(
        default_model="banglabert",
        cache_dir=REFERENCE_CACHE,
        device="auto",
    )
    respond({
        "status": "ready",
        "device": str(restorer.device),
        "default_model": "banglabert",
    })

    for raw_line in sys.stdin:
        request_id = None
        try:
            request = json.loads(raw_line)
            request_id = request.get("id")
            if request.get("action") != "restore":
                raise ValueError("Unsupported inference action.")

            text = request.get("text")
            public_model = request.get("model")
            reference_model = PUBLIC_TO_REFERENCE_MODEL.get(public_model)
            if reference_model is None:
                raise ValueError("Model must be either bert or bilstm.")

            result = restorer.restore_punctuation(
                text=text,
                model_name=reference_model,
                method=request.get("method", "constrained"),
                beam_width=request.get("beam_width", 8),
                beta=request.get("beta", 0.3),
            )
            result["id"] = request_id
            respond(result)
        except Exception as error:  # Keep the worker alive for future requests.
            respond({"id": request_id, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
