# Running

[简体中文](running.zh-CN.md) · [README](../README.md)

Run commands from the repository root. Python 3.11 is the reference environment. Commands on one line work in PowerShell and Bash.

## Install and check offline

```bash
uv sync --frozen --python 3.11 --extra dev
uv run --frozen --extra dev python -m pytest -q
uv run --frozen python -m experiments.reliability.reliability_demo
uv run --frozen python -m experiments.evaluation.verify_release_results
```

Without uv, activate a Python 3.11 virtual environment and run `python -m pip install -e ".[dev]"`; this resolves compatible dependencies rather than reproducing the lockfile. The repository's `experiments` tools run from the source checkout; they are not console commands installed by the wheel.

| Optional extra | Purpose |
| --- | --- |
| `dev` | pytest, distribution build tools, Ruff |
| `benchmark` | PDF/HTML extraction and BGE embeddings; semantic retrieval downloads model weights on first use |

## Live report

Copy `.env.example` to `.env` (`Copy-Item .env.example .env` in PowerShell, `cp .env.example .env` in Bash). Set `OPENAI_API_KEY` and `TAVILY_API_KEY`. The runner loads `.env` automatically.

```bash
uv run --frozen python -m experiments.runners.run_research --question "Compare task-level and graph-level recovery in agent workflows. Cite primary sources."
```

This makes paid requests. Default research/compression/report models are `openai:gpt-4.1`; summarization uses `openai:gpt-4.1-mini`. To use one model across all roles, pass `--model openai:gpt-4.1`. A run can fail because of insufficient evidence, provider errors or exhausted budgets. Check the acceptance summary before treating `report.md` as a deliverable. `incomplete_evidence` means the report passed its reliability checks while still retaining answer gaps; `degraded` means it needs review and returns a nonzero exit.

The entry point runs Reliable Deep Research; defaults use the same bounded task and per-call token settings as the documented mainline comparison. The default web search and model differ from that fixed-corpus historical experiment. The released profile enables question decomposition, paragraph audit, at most two feedback repair rounds/two gap tasks and one budgeted JSON/schema retry. Global defaults are 40 model calls, 4 searches, 6 task attempts, 1,200 seconds, 250,000 estimated input tokens and 120,000 observed output tokens. Input tokens are estimated and output usage is observed after requests, so token controls are not exact billing caps. `--help` lists supported options.

### Custom API endpoint

Set your key in `.env` as `EXPERIMENT_API_KEY=...`, then use your provider's model name and API base URL:

```bash
uv run --frozen python -m experiments.runners.run_research --question "How do durable agent tasks recover?" --model openai:YOUR_MODEL --base-url https://YOUR_HOST/v1 --api-key-env EXPERIMENT_API_KEY --plain-json-output --streaming --reasoning-effort low
```

`YOUR_MODEL` and `YOUR_HOST` are placeholders. Streaming and reasoning options require a compatible provider; omit unsupported options. The key itself must not appear in command arguments or published files. Tracing is off by default in `.env.example`.

### Multiple cases and local retrieval

A JSONL input contains one object per line:

```json
{"id":"my_question","category":"reliability","prompt":"What recovery guarantees do durable agent tasks provide? Cite primary sources."}
```

```bash
uv run --frozen python -m experiments.runners.run_research --questions path/to/questions.jsonl --case my_question
```

Omit `--case` to run every row, each with an independent database. Reference annotations in benchmark files are not passed into generation.

For a prepared local corpus, install the benchmark extra and choose BM25 or BM25+dense retrieval:

```bash
uv sync --frozen --extra dev --extra benchmark
uv run --frozen python -m experiments.runners.run_research --question "What does the corpus establish about task recovery?" --search-api local --local-corpus-path benchmark_data/expanded_20261021 --local-retrieval-strategy bm25_semantic_rrf --local-embedding-cache-path benchmark_data/expanded_20261021/embeddings/bge-small-en-v1.5.npz
```

[Benchmark inputs and corpus preparation](../benchmarks/README.md). Local retrieval still requires a model API for research and writing.

## Inspect output

Each invocation creates a unique run directory under `artifacts/research/`:

| File | Meaning |
| --- | --- |
| `manifest.json` at run root | Per-case summaries, including failures |
| `<case>/input.json` | Question text used |
| `<case>/summary.json` | Status, configuration, acceptance, timing and usage |
| `<case>/report.md` | Generated report or failure text |
| `<case>/events.jsonl` | Node/tool/task events and budget observations |
| `<case>/runtime.db` | Tasks, results, Documents, Evidence and Claims |

```bash
uv run --frozen python -m experiments.reliability.inspect_run artifacts/research/ACTUAL_RUN/ACTUAL_CASE
```

Replace the uppercase placeholders with a real output path. The research runner returns a nonzero exit code for failed acceptance. The runner creates a new case each time; it does not automatically resume an arbitrary interrupted LangGraph session. Never share one runtime database across unrelated questions.

The public runner delegates to `experiments.runners.run_baseline`, a shared internal engine whose historical filename does not select the profile. The supported entry point runs Reliable Deep Research; other development profiles and direct Studio graph runs are outside this workflow.
