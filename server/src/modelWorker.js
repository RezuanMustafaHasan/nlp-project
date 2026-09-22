import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";

const serverDirectory = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(serverDirectory, "../..");

function resolvePython() {
  if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;

  const localCandidates = process.platform === "win32"
    ? [path.join(projectRoot, "venv", "Scripts", "python.exe")]
    : [path.join(projectRoot, "venv", "bin", "python")];

  return localCandidates.find(existsSync) || (process.platform === "win32" ? "python" : "python3");
}

class ModelWorker {
  constructor() {
    this.pending = new Map();
    this.process = null;
    this.start();
  }

  start() {
    const script = path.join(projectRoot, "inference", "worker.py");
    this.process = spawn(resolvePython(), [script], {
      cwd: projectRoot,
      env: { ...process.env, PYTHONIOENCODING: "utf-8" },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });

    const output = readline.createInterface({ input: this.process.stdout });
    output.on("line", (line) => {
      let message;
      try {
        message = JSON.parse(line);
      } catch {
        console.error("Unexpected inference output:", line);
        return;
      }

      const request = this.pending.get(message.id);
      if (!request) return;

      this.pending.delete(message.id);
      clearTimeout(request.timeout);
      if (message.error) request.reject(new Error(message.error));
      else request.resolve(message);
    });

    this.process.stderr.on("data", (data) => {
      process.stderr.write(`[models] ${data}`);
    });

    this.process.on("exit", (code) => {
      const error = new Error(`Inference worker stopped with code ${code}.`);
      for (const request of this.pending.values()) {
        clearTimeout(request.timeout);
        request.reject(error);
      }
      this.pending.clear();
      this.process = null;
    });
  }

  get active() {
    return Boolean(this.process && !this.process.killed);
  }

  request(payload, timeoutMs = 180_000) {
    if (!this.active) this.start();

    return new Promise((resolve, reject) => {
      const id = randomUUID();
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error("Model inference timed out."));
      }, timeoutMs);

      this.pending.set(id, { resolve, reject, timeout });
      this.process.stdin.write(`${JSON.stringify({ id, ...payload })}\n`);
    });
  }

  stop() {
    this.process?.kill();
  }
}

export const modelWorker = new ModelWorker();
