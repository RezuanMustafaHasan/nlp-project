import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync } from "node:fs";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";

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
    this.state = "stopped";
    this.device = "unknown";
    this.startupWaiters = [];
    this.start();
  }

  start() {
    if (this.process) return;

    const script = path.join(projectRoot, "inference", "worker.py");
    this.state = "initializing";
    this.process = spawn(resolvePython(), ["-u", script], {
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

      if (message.status === "ready") {
        this.state = "ready";
        this.device = message.device || "unknown";
        for (const waiter of this.startupWaiters.splice(0)) waiter.resolve();
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

    this.process.on("error", (error) => this.fail(error));
    this.process.on("exit", (code) => {
      this.fail(new Error(`Inference worker stopped with code ${code}.`));
      this.process = null;
      this.state = "stopped";
    });
  }

  fail(error) {
    for (const waiter of this.startupWaiters.splice(0)) waiter.reject(error);
    for (const request of this.pending.values()) {
      clearTimeout(request.timeout);
      request.reject(error);
    }
    this.pending.clear();
  }

  get active() {
    return Boolean(this.process && !this.process.killed);
  }

  get ready() {
    return this.state === "ready";
  }

  waitUntilReady(timeoutMs) {
    if (this.ready) return Promise.resolve();

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        const index = this.startupWaiters.findIndex((waiter) => waiter.resolve === wrappedResolve);
        if (index >= 0) this.startupWaiters.splice(index, 1);
        reject(new Error("Inference worker is still initializing."));
      }, timeoutMs);
      const wrappedResolve = () => {
        clearTimeout(timeout);
        resolve();
      };
      const wrappedReject = (error) => {
        clearTimeout(timeout);
        reject(error);
      };
      this.startupWaiters.push({ resolve: wrappedResolve, reject: wrappedReject });
    });
  }

  async request(payload, timeoutMs = 180_000) {
    if (!this.active) this.start();
    await this.waitUntilReady(timeoutMs);

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
    const child = this.process;
    if (!child || child.exitCode !== null) return Promise.resolve();

    this.state = "stopping";
    child.stdin.end();
    return new Promise((resolve) => {
      const forceStop = setTimeout(() => child.kill(), 3_000);
      child.once("exit", () => {
        clearTimeout(forceStop);
        resolve();
      });
    });
  }
}

export const modelWorker = new ModelWorker();
