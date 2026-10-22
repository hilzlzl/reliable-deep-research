# Task recovery

[简体中文](recovery.zh-CN.md) · [Architecture](project_overview.md) · [Numeric evidence](../benchmarks/results/recovery.json)

The runtime stores task states and commits a successful task's result with its completion state in SQLite. Restarting the same task DAG against the same database reuses committed results. Explicit replay invalidates selected tasks and their descendants. Evidence and Claims from an unfinished or replayed attempt are invalidated before retry so stale records cannot enter the new report.

## Existing measurements

| Synthetic fault | Trials | Successful runs | Expected outcomes matched |
| --- | ---: | ---: | ---: |
| Transient exception | 30 | 30 | 30 |
| Transient timeout | 30 | 30 | 30 |
| Persistent exception | 30 | 0 | 30 |
| Persistent timeout | 30 | 0 | 30 |

Persistent faults should fail after the retry limit; a zero success count is the intended behavior in those scenarios.

A separate experiment uses `os._exit` and a fresh process, with 20 trials per scenario per mode:

| Scenario | Repository control | Reliable runtime |
| --- | --- | --- |
| Exit after root task commits | Restart succeeds 20/20; completed root reexecutes 20/20 | Restart succeeds 20/20; root executes once across both processes |
| Exit during evidence write before task commit | Restart succeeds 20/20; no evidence store | Restart succeeds 20/20; zero stale Evidence/Claims after cleanup |

These synthetic tasks establish tested runtime semantics, not production failure rates or guarantees about paid calls. A request sent before a crash may be billed even if no result commits. Budget counters are in memory, so this is not persistent cross-process billing enforcement. A database belongs to one research case.

## Reproduce without keys

```bash
uv run --frozen python -m experiments.reliability.reliability_demo
uv run --frozen python -m experiments.reliability.process_recovery_experiment --trials 2 --output artifacts/recovery_smoke
```

Use a new or empty output directory for process recovery. Change `--trials 2` to `--trials 20` for the published scale. The lightweight demo injects an exception and then replays verification/report tasks; the process experiment deliberately terminates workers.

