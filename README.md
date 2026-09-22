# Bangla Punctuation Restoration

A minimal MERN interface for running the included BanglaBERT and BiLSTM PyTorch checkpoints.

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

## Production-style local run

```bash
npm run build
npm start
```

Then open `http://127.0.0.1:5000`.
