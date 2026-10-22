# Benchmark evidence

[简体中文](README.zh-CN.md) · [Evaluation protocol](../docs/evaluation.md)

| File | Purpose |
| --- | --- |
| [30 questions](expanded_20261021/questions.jsonl) | Frozen prompts, assistant-authored reference points and source annotations |
| [24 sources](expanded_20261021/sources.jsonl) | Source metadata and content hashes; raw documents are excluded |
| [180 comparison slots](results/release_comparison.json) | 60 evaluation slots per approach, with per-record source mapping |
| [Scoring records](results/coverage_formal_20261021/model_audit_numeric.json), [Feedback-repair evaluation records](results/coverage_retry_20261021/model_audit_numeric.json) | Scoring data underlying the aggregate metrics |
| [Repair manifest](coverage_retry_20261021/frozen_manifest.json) and [environment](coverage_retry_20261021/environment.json) | Feedback-repair configuration, evaluation cases and global budgets |
| [Recovery measurements](results/recovery.json) | Synthetic fault matrix and actual process-exit experiments |
| [Generation manifest](expanded_20261021/frozen_manifest.json) and [environment](expanded_20261021/environment.json) | Historical generation provenance |
| [Reevaluation manifest](coverage_formal_20261021/frozen_manifest.json) and [environment](coverage_formal_20261021/environment.json) | Provenance of the common scoring pass |
| [ID-format addendum](coverage_formal_20261021/evaluation_id_format_addendum.json), [execution addendum](coverage_formal_20261021/evaluation_execution_addendum.json) | Disclosed extra format recovery for one judgment |

Public data retains run statuses, individual scores and source mappings for offline metric verification. See the [evaluation protocol](../docs/evaluation.md) for methodology and limitations. Original reports, passages and judge explanations are excluded; arithmetic checks cannot establish semantic correctness.

```bash
uv run --frozen python -m experiments.evaluation.verify_release_results
```

## Rebuild local inputs

Install `--extra benchmark`. First acquire the seven-paper seed corpus:

```bash
uv run --frozen --extra benchmark python -m experiments.benchmarks.acquire_benchmark_sources --help
uv run --frozen --extra benchmark python -m experiments.benchmarks.build_local_corpus --help
```

The [seed corpus instructions](researchbench_en/README.md) give the concrete acquisition/build commands. Then add the remaining sources:

```bash
uv run --frozen --extra benchmark python -m experiments.benchmarks.prepare_expanded_corpus --existing-corpus benchmark_data/researchbench_en/derived
uv run --frozen --extra benchmark python -m experiments.benchmarks.evaluate_semantic_retrieval --corpus-dir benchmark_data/expanded_20261021 --questions benchmarks/expanded_20261021/questions.jsonl --embedding-cache benchmark_data/expanded_20261021/embeddings/bge-small-en-v1.5.npz --threads 4 --output artifacts/retrieval.json
```

Use a fresh working copy for corpus acquisition: preparation writes source metadata, and upstream documentation can change. Compare new content hashes with the published metadata rather than silently treating newly downloaded text as the historical snapshot. Do not overwrite historical manifests for a fresh experiment.

For new end-to-end reports use `experiments.runners.run_research` with this question file, prepared local corpus and provider settings from the [running guide](../docs/running.md). Its default mainline limits match the documented comparison settings. Run the question file twice into separate output directories for two independent attempts. Record new model, environment, source and code hashes; a later run reproduces the procedure, not historical answers or timings. The public cleanup is later than the frozen experiment code.
