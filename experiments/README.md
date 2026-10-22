# Runners and checks

[简体中文](README.zh-CN.md) · [Running guide](../docs/running.md)

The supported workflow has one entry point:

```bash
uv run --frozen python -m experiments.runners.run_research --question "How do durable research tasks recover?"
```

| Directory | Published purpose |
| --- | --- |
| `runners/` | Public Reliable Deep Research entry point and shared internal run/acceptance engine |
| `reliability/` | Key-free demo, fault injection, real process restart and output inspection |
| `benchmarks/` | Fixed-corpus acquisition, chunking and semantic retrieval checks |
| `evaluation/` | Public numeric verification and report-grounding evaluation helpers |
| `configs/` | Seed questions used by the internal engine |

Generated reports, model outputs, task databases and downloaded corpora belong in ignored `artifacts/` and `benchmark_data/`. The release evidence is indexed in [benchmarks](../benchmarks/README.md). Tools run from the repository root; they are not installed CLI commands in the wheel.

The internal engine retains historical profile switches for source compatibility. Only Reliable Deep Research is selected by the supported entry point. Development-only runners and historical study writeups are excluded by the release file list.
