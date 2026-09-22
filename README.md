# Bangla Punctuation Restoration

A minimal MERN interface for running the canonical inference pipeline bundled
under `inference/`:

- fine-tuned BanglaBERT with aligned subword pooling;
- packed fastText–BiLSTM with the shared embedding cache; and
- constrained gap decoding and canonical punctuation rendering.

The React client keeps its public `bert` / `bilstm` model choices. The backend
maps `bert` to the reference pipeline's `banglabert` identifier and returns the
restored text through the existing API response shape.

## Run locally

1. Install the JavaScript dependencies:

   ```bash
   npm install
   ```

2. Ensure the Python environment has the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the React client, Express API, and PyTorch inference worker:

   ```bash
   npm run dev
   ```

4. Open `http://127.0.0.1:5173`.

The API automatically uses `venv/Scripts/python.exe` on Windows when available. Add `MONGO_URI` to a local `.env` file if request history should be saved; model inference does not require MongoDB.

The backend expects these reference assets to remain available:

```text
inference/models/banglabert/best.pt
inference/models/bilstm/best.pt
inference/embeddings/cache/word2id.json
inference/embeddings/cache/vocab_vectors.npy
```

## Production-style local run

```bash
npm run build
npm start
```

Then open `http://127.0.0.1:5000`.
