import "dotenv/config";
import express from "express";
import cors from "cors";
import mongoose from "mongoose";
import path from "node:path";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { modelWorker } from "./modelWorker.js";

const serverDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(serverDirectory, "../..");
const app = express();
const port = Number(process.env.PORT) || 5000;

app.use(cors());
app.use(express.json({ limit: "16kb" }));

const restorationSchema = new mongoose.Schema({
  input: { type: String, required: true },
  output: { type: String, required: true },
  model: { type: String, enum: ["bert", "bilstm"], required: true },
  durationMs: { type: Number, required: true },
}, { timestamps: true });

const Restoration = mongoose.model("Restoration", restorationSchema);
let databaseState = "not configured";

if (process.env.MONGO_URI) {
  mongoose.connect(process.env.MONGO_URI)
    .then(() => { databaseState = "connected"; })
    .catch((error) => {
      databaseState = "unavailable";
      console.error("MongoDB connection failed:", error.message);
    });
}

app.get("/api/health", (_request, response) => {
  const assets = {
    bertCheckpoint: existsSync(path.join(projectRoot, "inference", "models", "banglabert", "best.pt")),
    bilstmCheckpoint: existsSync(path.join(projectRoot, "inference", "models", "bilstm", "best.pt")),
    vocabulary: existsSync(path.join(projectRoot, "inference", "embeddings", "cache", "word2id.json")),
    vectors: existsSync(path.join(projectRoot, "inference", "embeddings", "cache", "vocab_vectors.npy")),
  };

  const healthy = modelWorker.active && Object.values(assets).every(Boolean);
  response.status(healthy ? 200 : 503).json({
    status: healthy ? (modelWorker.ready ? "ready" : "initializing") : "unavailable",
    inferenceReady: modelWorker.ready,
    device: modelWorker.device,
    assets,
    database: databaseState,
  });
});

app.post("/api/restore", async (request, response) => {
  const text = typeof request.body.text === "string" ? request.body.text.trim() : "";
  const model = request.body.model;

  if (!text) {
    return response.status(400).json({ message: "Bangla input text is required." });
  }

  if (text.length > 1500 || text.split(/\s+/u).length > 128) {
    return response.status(400).json({ message: "Input must be no more than 1,500 characters and 128 words." });
  }

  if (!["bert", "bilstm"].includes(model)) {
    return response.status(400).json({ message: "Model must be either bert or bilstm." });
  }

  const startedAt = Date.now();

  try {
    const result = await modelWorker.request({
      action: "restore",
      text,
      model,
      method: "constrained",
      beam_width: 8,
      beta: 0.3,
    });
    const durationMs = Date.now() - startedAt;

    if (result.error) {
      return response.status(422).json({ message: result.error });
    }

    if (mongoose.connection.readyState === 1) {
      Restoration.create({ input: text, output: result.restored_text, model, durationMs }).catch(() => {});
    }

    return response.json({
      output: result.restored_text,
      model,
      durationMs,
      inference: {
        modelName: result.model_name,
        modelDisplay: result.model_display,
        method: result.method,
        words: result.num_words,
        marks: result.num_marks,
        parenthesisSpans: result.parenthesis_spans,
        validSyntax: result.is_valid_syntax,
        latencyMs: result.latency_ms,
      },
    });
  } catch (error) {
    console.error(`${model} inference failed:`, error.message);
    return response.status(500).json({ message: error.message || "Model inference failed." });
  }
});

const clientBuild = path.join(projectRoot, "client", "dist");
if (existsSync(clientBuild)) {
  app.use(express.static(clientBuild));
  app.get("/{*route}", (_request, response) => response.sendFile(path.join(clientBuild, "index.html")));
}

const server = app.listen(port, "127.0.0.1", () => {
  console.log(`API listening at http://127.0.0.1:${port}`);
});

let shuttingDown = false;
function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  server.close();
  void Promise.allSettled([modelWorker.stop(), mongoose.disconnect()]);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
