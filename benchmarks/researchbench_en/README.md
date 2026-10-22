# Seed corpus

[Benchmark index](../README.md) · [简体中文](README.zh-CN.md)

These seven papers seed the 24-document mainline corpus. `sources.jsonl` records source versions, hashes and admission status. Raw and derived files stay under ignored `benchmark_data/`; `approved_local_only` does not grant redistribution rights. The builder rejects sources without an approved admission status. Question references are assistant-authored, not human-validated gold.

Install `--extra benchmark`, then acquire and build:

```bash
uv run --frozen --extra benchmark python -m experiments.benchmarks.acquire_benchmark_sources --manifest benchmarks/researchbench_en/sources.jsonl --raw-dir benchmark_data/researchbench_en/raw
uv run --frozen --extra benchmark python -m experiments.benchmarks.build_local_corpus --manifest benchmarks/researchbench_en/sources.jsonl --raw-dir benchmark_data/researchbench_en/raw --output-dir benchmark_data/researchbench_en/derived
```

The builder reads one raw PDF/HTML/Markdown file per source ID, writes deterministic documents/chunks and reports missing sources. It does not invoke a model. Keep one file extension per source ID. To reproduce the mainline setup, continue with the [expanded corpus preparation](../README.md).

The nine seed questions remain as inputs for code-level checks and corpus tooling. The release headline results use the separate 30-question file; results from different studies are not pooled.
