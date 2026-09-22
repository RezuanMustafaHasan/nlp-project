import { useEffect, useMemo, useState } from "react";

const EXAMPLE = "আজ আকাশ খুব সুন্দর তুমি কি বাইরে যাবে আমি বিকেলে মাঠে যাব";
const MAX_CHARACTERS = 1500;
const MAX_WORDS = 180;

function App() {
  const [input, setInput] = useState("");
  const [output, setOutput] = useState("");
  const [model, setModel] = useState("bert");
  const [status, setStatus] = useState("checking");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);

  const wordCount = useMemo(
    () => (input.trim() ? input.trim().split(/\s+/u).length : 0),
    [input],
  );

  useEffect(() => {
    fetch("/api/health")
      .then((response) => {
        if (!response.ok) throw new Error();
        return response.json();
      })
      .then(() => setStatus("ready"))
      .catch(() => setStatus("offline"));
  }, []);

  async function restore(event) {
    event.preventDefault();
    const cleanInput = input.trim();

    if (!cleanInput) {
      setError("Enter Bangla text before restoring punctuation.");
      return;
    }

    if (wordCount > MAX_WORDS) {
      setError(`Please keep the input within ${MAX_WORDS} words.`);
      return;
    }

    setError("");
    setCopied(false);
    setIsLoading(true);

    try {
      const response = await fetch("/api/restore", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: cleanInput, model }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || "Punctuation restoration failed.");
      }

      setOutput(data.output);
    } catch (requestError) {
      setError(requestError.message || "The inference service is unavailable.");
    } finally {
      setIsLoading(false);
    }
  }

  async function copyOutput() {
    if (!output) return;
    await navigator.clipboard.writeText(output);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1600);
  }

  function clearAll() {
    setInput("");
    setOutput("");
    setError("");
    setCopied(false);
  }

  return (
    <main className="app-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true">ব</div>
        <div className="title-block">
          <p className="eyebrow">Natural Language Processing</p>
          <h1>Bangla Punctuation Restoration</h1>
          <p className="subtitle">Add accurate punctuation to unpunctuated Bangla text.</p>
        </div>
        <div className={`service-status ${status}`} role="status">
          <span aria-hidden="true" />
          {status === "ready" ? "Models ready" : status === "offline" ? "Service offline" : "Checking service"}
        </div>
      </header>

      <form className="workspace" onSubmit={restore}>
        <section className="model-row" aria-labelledby="model-label">
          <div>
            <p className="section-label" id="model-label">Model</p>
            <p className="section-help">Choose the inference architecture.</p>
          </div>
          <div className="model-switch" role="radiogroup" aria-label="Select model">
            <label className={model === "bert" ? "selected" : ""}>
              <input
                type="radio"
                name="model"
                value="bert"
                checked={model === "bert"}
                onChange={(event) => setModel(event.target.value)}
              />
              <span>BERT</span>
              <small>Context-aware</small>
            </label>
            <label className={model === "bilstm" ? "selected" : ""}>
              <input
                type="radio"
                name="model"
                value="bilstm"
                checked={model === "bilstm"}
                onChange={(event) => setModel(event.target.value)}
              />
              <span>BiLSTM</span>
              <small>Lightweight</small>
            </label>
          </div>
        </section>

        <div className="editor-grid">
          <section className="editor-panel">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Input</p>
                <h2>Unpunctuated text</h2>
              </div>
              <button className="text-action" type="button" onClick={() => setInput(EXAMPLE)}>
                Use example
              </button>
            </div>
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder="এখানে বাংলা লেখা লিখুন"
              maxLength={MAX_CHARACTERS}
              aria-label="Unpunctuated Bangla input"
              lang="bn"
            />
            <div className="panel-footer">
              <span>{wordCount} / {MAX_WORDS} words</span>
              {input && <button className="text-action" type="button" onClick={clearAll}>Clear</button>}
            </div>
          </section>

          <section className="editor-panel output-panel" aria-live="polite">
            <div className="panel-heading">
              <div>
                <p className="panel-kicker">Output</p>
                <h2>Punctuation restored</h2>
              </div>
              <button className="text-action" type="button" onClick={copyOutput} disabled={!output}>
                {copied ? "Copied" : "Copy text"}
              </button>
            </div>
            <div className={`output-area ${!output ? "empty" : ""}`} lang="bn">
              {isLoading ? (
                <div className="loading-state"><span />Running {model === "bert" ? "BERT" : "BiLSTM"} inference…</div>
              ) : output || "Restored text will appear here."}
            </div>
            <div className="panel-footer">
              <span>{output ? `${output.length} characters` : "Awaiting input"}</span>
              <span className="model-note">{model === "bert" ? "BanglaBERT" : "Packed BiLSTM"}</span>
            </div>
          </section>
        </div>

        <div className="action-row">
          <div className="feedback" role="alert">{error}</div>
          <button className="primary-button" type="submit" disabled={isLoading || status === "offline"}>
            {isLoading ? "Restoring…" : "Restore punctuation"}
          </button>
        </div>
      </form>

      <footer>
        <p>Bangla NLP research interface</p>
        <p>PyTorch inference · BERT &amp; BiLSTM</p>
      </footer>
    </main>
  );
}

export default App;
