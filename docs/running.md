# Installation and usage

[简体中文](running.zh-CN.md) · [README](../README.md)

## 1. Environment and installation

Use Python 3.11+; Python 3.11 is the reference environment. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and run from the repository root. The following single-line commands work in PowerShell and Bash.

```bash
uv sync --frozen --python 3.11 --extra dev
uv run --frozen python -m experiments.reliability.reliability_demo
uv run --frozen --extra dev python -m pytest -q
```

The demo and tests need no model keys. The demo injects a task failure and checks retry, completed-task reuse and dependency replay. Outputs are saved under `artifacts/reliability_demo/`.

Without uv, run `python -m pip install -e ".[dev]"` in a Python 3.11 virtual environment; this does not reproduce locked dependency versions exactly. The repository's `experiments` commands must run from the source checkout.

## 2. Model and retrieval configuration

Copy the environment template:

```bash
cp .env.example .env
```

Use `Copy-Item .env.example .env` in PowerShell. Set these values in `.env`:

```dotenv
OPENAI_API_KEY=YOUR_MODEL_KEY
TAVILY_API_KEY=YOUR_SEARCH_KEY
```

The runner loads `.env` automatically. Retrieval defaults to Tavily. Research, compression and report models use `openai:gpt-4.1`; summarization uses `openai:gpt-4.1-mini`. Keep keys in local configuration and out of the repository.

For a compatible API service, also set `EXPERIMENT_API_KEY=...` in `.env` and supply the model name, API prefix and key variable name together:

```bash
uv run --frozen python -m experiments.runners.run_research --question "How do durable agent tasks recover?" --model openai:YOUR_MODEL --base-url https://YOUR_HOST/v1 --api-key-env EXPERIMENT_API_KEY
```

Replace `YOUR_MODEL` and `YOUR_HOST`; do not append `/chat/completions` to the address. Depending on provider support, add `--plain-json-output`, `--streaming` or `--reasoning-effort low`. Use `--plain-json-output` when native structured output is unsupported.

## 3. Run research

```bash
uv run --frozen python -m experiments.runners.run_research --question "Compare task-level and graph-level recovery in agent workflows. Cite primary sources."
```

Live research incurs model and search costs. The entry point fixes `coverage_v2`, with default caps of 40 model calls, 4 searches, 6 task attempts and 1,200 seconds. Retrieval, writing, auditing and repair share the budget. Token usage combines input estimates and output observations after requests; these are not exact billing caps.

Pass `--model openai:gpt-4.1` to use one model across roles, or `--output path/to/output` to choose an output location. List all options with:

```bash
uv run --frozen python -m experiments.runners.run_research --help
```

## 4. Inspect results

Each invocation creates a separate directory under `artifacts/research/`. Start with `manifest.json` at the run root to locate the question, then inspect its subdirectory:

| File | Content |
| --- | --- |
| `report.md` | Research report or failure text |
| `summary.json` | Status, configuration, acceptance, timing, usage and errors |
| `input.json` | Actual question input |
| `events.jsonl`, `runtime.db` | Event trace, task results and evidence |

```bash
uv run --frozen python -m experiments.reliability.inspect_run artifacts/research/ACTUAL_RUN/ACTUAL_CASE
```

Replace uppercase placeholders with the actual directory. A `status` of `succeeded` means run acceptance passed; `incomplete_evidence` means reliability checks passed with answer gaps, and `degraded` needs review. Unaccepted runs return a nonzero exit code; inspect the summary and events for the reason.

Each invocation creates a new run rather than resuming an arbitrary interrupted session. Databases are isolated by question and must not be shared across questions. See [System architecture](project_overview.md) for task-result reuse.

## 5. Optional usage

**Batch questions:** a JSONL file contains one object per line:

```json
{"id":"my_question","category":"reliability","prompt":"What recovery guarantees do durable agent tasks provide? Cite primary sources."}
```

```bash
uv run --frozen python -m experiments.runners.run_research --questions path/to/questions.jsonl --case my_question
```

Omit `--case` to run all rows, each with an independent database.

**Local corpus:** follow the [benchmark guide](../benchmarks/README.md) to prepare inputs. Use BM25 on a prepared corpus:

```bash
uv run --frozen python -m experiments.runners.run_research --question "What does the corpus establish about task recovery?" --search-api local --local-corpus-path benchmark_data/expanded_20261021
```

Local retrieval needs no Tavily key; research and writing still need a model API. For BM25 + vector fusion, install the extra dependencies and append `--local-retrieval-strategy bm25_semantic_rrf` to the command above. Embedding weights download on first use.

```bash
uv sync --frozen --extra dev --extra benchmark
```

**Experiment verification:** see [Evaluation](evaluation.md) for report metrics, numeric verification and process recovery experiments.

## 6. Common issues

| Symptom | Action |
| --- | --- |
| Module import fails | Run from the repository root, install dependencies first, and use the project environment through `uv run --frozen` |
| Model authentication or connection fails | Check `.env`, model access, API address, quota and provider rate limits; custom APIs require all three related arguments together |
| Provider rejects structured output | Try `--plain-json-output`; remove streaming or reasoning options when unsupported |
| Local corpus cannot load | Check that `--local-corpus-path` points to a prepared corpus directory and follow the benchmark preparation steps |
| Report fails or exhausts budget | Check the error in `summary.json` and `events.jsonl`; narrow the question or adjust `--max-llm-calls`, `--max-search-calls` and `--max-total-seconds` |
| Report has gaps or needs review | Read the acceptance summary and gap notes, then check cited passages before using the report |
