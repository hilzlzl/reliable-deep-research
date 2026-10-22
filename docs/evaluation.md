# Deep research and task recovery evaluation

[简体中文](evaluation.zh-CN.md) · [README](../README.md)

## Experimental setup

Compare Repository Baseline, the earlier Claim-first approach and Reliable Deep Research on questions about agent systems, AI applications, software engineering and CAD generation.

| Item | Repository Baseline | Claim-first | Reliable Deep Research |
| --- | --- | --- | --- |
| Scale | 30 questions × 2 runs, 60 attempts | 30 questions × 2 runs, 60 attempts | 60 selected slots: 47 original accepted runs + 13 reruns[^replacement] |
| Workflow | Upstream research flow with added reliability, evidence review and repair disabled | Evidence → admitted Claims → report | Question decomposition, paragraph audit, bounded gap search and feedback repair |
| Profile ID | `baseline` | `reliable_v2` | Released profile `coverage_v2`; retained records from `coverage_v1` |

Repository Baseline retains this repository's experimental adaptations for retrieval, model access and budgets. Its results do not represent an unmodified official Open Deep Research release.

All three groups use 24 fixed sources, split into 1,200-character chunks with 200-character overlap, and BM25 + BGE-small-en-v1.5 fused retrieval returning at most 5 passages per search. The model gateway alias is `gpt-6.1-sol`, with low reasoning effort and streaming; the hosted model implementation was not independently verified. Generation receives only the question text, without reference answers.

Each attempt is capped at 40 model calls, 4 searches, 6 task attempts, 1,200 seconds, 250,000 estimated input tokens and 120,000 observed output tokens, with research concurrency 2. Historical batches used 18 report processes and rerun batches used 6. Shared API load and configuration changes affect timing.

[^replacement]: The original coverage experiment produced 47 accepted, 11 degraded and 2 failed attempts. All 13 unsuccessful slots were rerun once, producing 12 accepted and 1 failed attempt; the failed rerun is retained without selecting the best result. The table is a development summary across two configurations, consuming 73 generation attempts. A fresh independent set of 60 runs using the released profile has not been completed.

## Results

| Metric | Repository Baseline | Claim-first | Reliable Deep Research[^replacement] |
| --- | ---: | ---: | ---: |
| Accepted | 52/60 | 51/60 | **59/60** |
| Degraded / needs review | 0 | 0 | **0** |
| Failed or budget-exhausted | 8 | 9 | **1** |
| Reference-point coverage, all slots | 135/168 (80.4%) | 109/168 (64.9%) | **149/168 (88.7%)** |
| Coverage within accepted reports | 135/145 (93.1%) | 109/144 (75.7%) | **149/165 (90.3%)** |
| Supported sampled citations | 123/202 (60.9%) | 244/254 (96.1%) | **219/234 (93.6%)** |
| Mean seconds, including failures | 404.9 | 667.6 | **319.3** |
| Mean model calls, including failures | 9.52 | 19.42 | **15.25** |
| Mean searches, including failures | 1.95 | 3.72 | 2.50 |
| Observed output tokens, total | 517,433 | 1,016,755 | 514,544 |

Time and usage cover only the 60 retained slots, excluding the 13 displaced original attempts and external judging costs. Calls and tokens are not monetary costs. Full numeric records and per-slot mappings are in the [experiment records](../benchmarks/results/README.md).

## Scoring rules

| Metric | Calculation |
| --- | --- |
| Accepted | Meets the workflow's acceptance checks; contracts differ across systems, so acceptance is not equivalent to semantic quality |
| Reference-point coverage, all slots | Reference points explicitly covered by accepted reports ÷ reference points across all 60 slots; failed and degraded runs contribute zero |
| Coverage within accepted reports | Reference points explicitly covered by accepted reports ÷ all reference points for those reports |
| Supported sampled citations | Select at most five cited statements per accepted report by stable hash, provide source passages for judging, and report the fraction judged supported |

Three accepted Repository Baseline reports had no extractable citations and do not enter the citation denominator. `incomplete_evidence` can pass reliability acceptance while retaining answer gaps; `degraded` needs review. Repeats are nested within 30 questions, not 60 independent questions.

Reference points are assistant-authored, judging and generation use the same gateway model alias, and independent human review is pending. Some questions were used during development, so this is not an unseen test set. Only five rerun reports used repair followed by verification, and only eight of the twelve accepted reports were judged externally to cover every reference point. The acceptance improvement cannot be attributed entirely to content repair or establish general superiority over upstream. The remaining failure is `agent_02`, repeat 2, which timed out and exhausted research search allocation on retry.

## Task recovery experiments

Independent synthetic tasks test recovery separately from report quality above.

| Scenario | Scale | Result |
| --- | --- | --- |
| Transient exception, transient timeout | 30 trials each | Recovery succeeds 30/30 each |
| Persistent exception, persistent timeout | 30 trials each | All fail after the retry limit, matching expected outcomes 30/30 |
| Process exits after root task commits | 20 trials per implementation | Both restart successfully 20/20; the repository control reexecutes the root, while the reliable runtime reuses its result |
| Process exits during evidence write before task commit | 20 trials per implementation | Both restart successfully 20/20; the control has no evidence store, while the reliable runtime leaves zero stale Evidence/Claims after cleanup |

These tests establish task-result reuse and invalidation semantics, not production failure rates. The full LangGraph cursor and budget counters are not persisted across processes; external requests sent before a crash may still be billed. See the [recovery records](../benchmarks/results/recovery.json).

## Offline verification

Run from the repository root without model keys. Use a new or empty output directory for the process recovery experiment.

```bash
uv run --frozen python -m experiments.evaluation.verify_release_results
uv run --frozen python -m experiments.reliability.process_recovery_experiment --trials 2 --output artifacts/recovery_smoke
```

The first command recalculates results and checks source hashes and all 13 replacement mappings; it cannot validate semantic judgments. Change `--trials 2` to `--trials 20` in the second command for the published scale. Questions, source metadata, environments and experiment addenda are linked from the [benchmark index](../benchmarks/README.md).
