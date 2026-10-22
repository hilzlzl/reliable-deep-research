# Evaluation and release choice

[简体中文](evaluation.zh-CN.md) · [README](../README.md)

## Released workflow

The project is published as **Reliable Deep Research**, with bounded repair driven by audit feedback. Claim-first is an earlier evaluation control. The public entry point enables question-oriented reporting, paragraph support checks, bounded gap search and verified local repair; it does not select the old Claim-first path.

<a id="experiment-identifiers"></a>

## Approaches

| Approach | Description |
| --- | --- |
| Repository Baseline | Upstream-derived workflow with repository experimental adaptations; added reliability, evidence review and repair disabled |
| Claim-first (earlier approach) | Evidence → admitted Claims → report; earlier evaluation control |
| Reliable Deep Research | Question-oriented reporting, semantic audit and bounded repair |

Repository Baseline is the upstream-derived research workflow as adapted in this repository for retrieval, model access and budget-controlled experiments. It disables the added reliability runtime, evidence review and repair features. It is not an unmodified official Open Deep Research release, and the comparison does not establish performance against such a release.

## Results

| Metric | Repository Baseline | Claim-first (earlier approach) | Reliable Deep Research |
| --- | ---: | ---: | ---: |
| Accepted / slots | 52/60 | 51/60 | **59/60** |
| Degraded / needs review | 0 | 0 | **0** |
| Failed or budget-exhausted | 8 | 9 | **1** |
| Reference-point coverage, all slots | 135/168 = 80.4% | 109/168 = 64.9% | **149/168 = 88.7%** |
| Coverage within accepted reports | 135/145 = 93.1% | 109/144 = 75.7% | **149/165 = 90.3%** |
| Supported sampled citations | 123/202 = 60.9% | 244/254 = 96.1% | **219/234 = 93.6%** |
| Mean seconds, including failures | 404.9 | 667.6 | **319.3** |
| Mean model calls, including failures | 9.52 | 19.42 | **15.25** |
| Mean searches, including failures | 1.95 | 3.72 | 2.50 |
| Observed output tokens, total | 517,433 | 1,016,755 | 514,544 |

All 60 displayed slots remain in each group's coverage denominator, with failed runs contributing zero covered points. Quality numerators count accepted reports only. Citation support samples at most five statements per report; three accepted Repository Baseline reports had no extractable citations. Acceptance contracts differ between systems, so acceptance alone is not a semantic quality score. Repeats are nested within 30 questions, not 60 independent questions.

## Statistical scope and data verification

Results summarize saved development-stage evaluation records across configurations, including cases used for failure analysis and improvement. They do not constitute an independent full-set evaluation of the released configuration. Mean time and calls cover only the displayed records, excluding other development attempts and external judging costs.

The [release data](../benchmarks/results/release_comparison.json) retains all 180 comparison slots. Each record identifies its provenance through `source_run_id`, `source_profile` and `source_experiment`. The corresponding [scoring records](../benchmarks/results/coverage_formal_20261021/model_audit_numeric.json) and [feedback-repair evaluation records](../benchmarks/results/coverage_retry_20261021/model_audit_numeric.json) support individual checks.

```bash
uv run --frozen python -m experiments.evaluation.verify_release_results
```

The offline verifier checks slot completeness, source hashes, record mappings, retained failures and aggregate metrics. It cannot establish whether a model's semantic verdict is correct.

## Why the repair revision exists

The original coverage path could remove unsupported material yet return a report needing review. The repair revision uses the current paragraph audit and current evidence instead of reinserting a stale planned answer, preserves already verified prose, repairs citation bindings with another audit and permits at most two repair rounds and two total gap tasks. Gap tasks may use the remaining fourth search while reserving model/time budget. Research unit allocation increases from three to four, with the same six global task-attempt cap. JSON/schema errors get one budgeted format retry. Acceptance thresholds were not relaxed.

During feedback-repair evaluation, five reports used repair followed by verification and seven passed directly. Of the twelve reports passing reliability acceptance, the external judge found eight covered all reference points; internal checks marked eleven complete and one `incomplete_evidence`. Reliability acceptance, internal completeness and external answer coverage are distinct, so acceptance changes cannot be attributed solely to content repair.

## Data, models and budgets

- 30 curated questions: 8 agent systems, 8 AI applications, 8 software engineering, 6 CAD generation; 8 require multiple documents and 5 are deliberately unanswerable.
- 24 documents; 1,200-character chunks with 200-character overlap; BM25 + BGE-small-en-v1.5 reciprocal rank fusion; up to 5 chunks per search.
- Assistant-authored reference points were mechanically checked against documents. Generation receives question text only. Six CAD questions reuse earlier work and eight questions were used for development diagnostics. This is not an unseen test set.
- Gateway alias `gpt-6.1-sol`, low reasoning effort, streaming. The hosted model implementation was not independently verified. The live quickstart uses different default OpenAI models and web search.
- Per-case global caps remain 40 model calls, 4 searches, 6 task attempts, 1,200 seconds, 250,000 estimated input tokens and 120,000 observed output tokens. Per-call output maxima: research 2,048, compression 3,072, report 4,096, summary 2,048. Research concurrency 2, task timeout 480 seconds.
- Different development stages used 18 and 6 report processes. Shared API load and configuration changes affect latency. Calls and tokens are cost proxies, not currency amounts.

## Scoring and provenance

Repository Baseline/Claim-first reports were saved historical controls rejudged under the same corrected protocol as the original coverage reports; controls were not regenerated. All stages use the same strict anonymous criteria and source extraction. At most five cited statements are selected by stable hash; all resolved cited sources are supplied, using the exact chunk or two lexically closest chunks per source at 1,400 characters per chunk. Bibliography parsing handles bullet-prefixed entries. Missing, extra and duplicate scoring IDs are rejected. Feedback-repair judge prompts map long claim IDs bijectively to short IDs, then map back.

One Repository Baseline judgment needed two extra missing-only format recovery passes beyond the planned two rounds. Existing statements, evidence, references and valid judgments were preserved. The [ID-format addendum](../benchmarks/coverage_formal_20261021/evaluation_id_format_addendum.json) and [execution addendum](../benchmarks/coverage_formal_20261021/evaluation_execution_addendum.json) disclose this deviation.

The failed record (`agent_02`, repeat 2) timed out, invalidated its unfinished task's 14 evidence records and exhausted research search allocation on retry. Its no-evidence branch exits before the gap-search route. This remains unresolved. A subsequent summary-recording error was recovered offline; after generation and judging, a one-line null guard was added without another paid request. See the [failure-record addendum](../benchmarks/results/coverage_retry_20261021/failure_record_recovery_addendum.json) and [post-run fix record](../benchmarks/results/coverage_retry_20261021/postrun_summary_guard_fix.json).

Historical [generation manifest](../benchmarks/expanded_20261021/frozen_manifest.json), [reevaluation manifest](../benchmarks/coverage_formal_20261021/frozen_manifest.json) and [repair manifest](../benchmarks/coverage_retry_20261021/frozen_manifest.json) identify frozen experiment code and inputs. Public packaging is later than those snapshots. Original reports, source passages, databases and judge explanations remain local; numeric records permit arithmetic checks rather than independent factual review.

Same-model judging, assistant references, development sample selection, generation randomness, changed parallelism and lack of independent human review limit interpretation. No claim is made of general superiority over upstream or an isolated causal effect of repair. The next validation is independent review and a fresh full-set run of the released profile, with failures and resource totals retained.
