# Design and code map

[简体中文](project_overview.zh-CN.md) · [Run it](running.md)

## Goal and scope

Deliver a completed research report with inspectable evidence, while preserving completed subtask work across failures. The upstream project supplies supervisor/researcher orchestration, prompts and provider tools. This fork concentrates on durable task execution, evidence lineage and controlled evaluation.

Evaluate answer coverage and sampled citation support separately. Completion and recovery measure execution behavior; latency and usage measure resource cost.

## Runtime path

1. The LangGraph supervisor emits research work. The reliable path in `supervisor_tools` converts it into typed `ResearchTask` objects.
2. `DagTaskScheduler` checks dependencies/cycles, executes ready tasks in bounded batches, and applies per-task retries and timeouts. A shared in-memory `RunBudget` prevents concurrent workers from independently spending the same remaining call allowance.
3. `RuntimeStore` commits task success and its result together. Evidence and Claims can be written before success; failed/replayed/unfinished attempts invalidate their derived records before retry. This cleanup matters because replaying with stale evidence would mix two attempts.
4. The same database restores successful task results. Explicit replay also invalidates dependent tasks. A new database is created per research case by the runner.

This is task-level recovery. It does not persist the whole LangGraph execution cursor or in-memory budget counters, and it cannot roll back an external model/search request already sent before a process died. The process-exit experiment uses synthetic tasks and validates these specific semantics.

## Released evidence, coverage and repair path

The entry point selects **Reliable Deep Research**. It reuses the durable research runtime above; its report path organizes answers around question requirements without first going through the earlier claim-extraction pipeline.

1. Derive stable requirements and constraints from the user's question before examining retrieved evidence.
2. Persist retrieved Documents/Evidence, then plan answers and gaps for each requirement with Evidence IDs.
3. Write `[EVIDENCE:<id>]` markers. Audit each paragraph for semantic support and assess answer completeness separately.
4. Repair from current audit feedback and current evidence, preserving verified prose rather than reintroducing rejected stale planned answers. Re-audit citation-binding edits.
5. Allow at most two repair rounds and two total targeted gap tasks. Gap search may spend the remaining fourth search while preserving model/time reserve. One JSON/schema format retry counts against the shared budget.
6. Resolve citations from stored evidence IDs and audit source provenance. Reliability acceptance remains separate from completeness: `incomplete_evidence` can pass reliability checks with answer gaps; `degraded` needs review and returns a nonzero exit.

Semantic audits are fallible model judgments. The older Claim-first claim-admission path remains as control code, not the default release workflow.

## Read the implementation

| Order | Files | What to look for |
| --- | --- | --- |
| 1 | [Research graph](../src/open_deep_research/deep_researcher.py), [runner](../experiments/runners/run_research.py) | Graph flow, supported entry point, shared budget creation and final acceptance |
| 2 | [Task runtime](../src/open_deep_research/task_runtime.py), [store](../src/open_deep_research/runtime_store.py) | State transitions, atomic result commit, restart and replay invalidation |
| 3 | [Evidence records](../src/open_deep_research/evidence.py), [coverage reporting](../src/open_deep_research/coverage_reporting.py) | Source identity, question requirements, paragraph audits and feedback repair |
| 4 | [Report planning](../src/open_deep_research/report_planning.py) | Shared Evidence ID citation rendering and source audit |
| 5 | [Budget](../src/open_deep_research/run_budget.py), [events](../src/open_deep_research/observability.py) | Atomic reservations, token estimation and observable failures |
| 6 | [Recovery experiment](../experiments/reliability/process_recovery_experiment.py), [numeric verifier](../experiments/evaluation/verify_release_results.py) | Actual process exits, complete attempt slots, metrics and denominators |

## Design decisions and trade-offs

- **SQLite instead of a service stack:** inspectable, transactional persistence for a local prototype. This is not a distributed scheduler or a multi-tenant deployment.
- **Task results instead of entire-session checkpoints:** a smaller recovery boundary that can be tested with process exits. External side effects still need idempotency if stronger guarantees are required.
- **Evidence IDs instead of writer-supplied URLs:** deterministic provenance enforcement. It cannot prevent every unsupported natural-language inference or guarantee that the selected evidence answers the question fully.
- **Bounded validation instead of an unbounded critique loop:** limits additional work, but finite-budget runs can still fail or remain incomplete; gap search and verification share resources.
- **Separate metrics instead of one quality score:** answer coverage, citation support and resource use describe different aspects of performance. A single score would conceal those trade-offs.

## Iteration and remaining work

Process-exit tests exposed duplicate completed work and stale derived evidence. Transactional task-result commit and invalidation address those cases. URL provenance alone then proved insufficient, so evaluation moved from broad report judging to sampled citation support plus explicit answer-point coverage.

See [evaluation](evaluation.md) for the controls, metric definitions and known limitations. Independent human review and a full-set evaluation of the released configuration remain follow-up work.
