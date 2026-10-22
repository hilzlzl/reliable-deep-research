# Numeric results

[简体中文](README.zh-CN.md) · [Protocol and limitations](../../docs/evaluation.md) · [Benchmark index](../README.md)

- [release_comparison.json](release_comparison.json): 60 evaluation slots per approach, including run statuses, scores, aggregate metrics and source mappings.
- [Scoring records](coverage_formal_20261021/model_audit_numeric.json): individual scores for the three approaches.
- [Feedback-repair evaluation records](coverage_retry_20261021/model_audit_numeric.json): individual scores from the feedback-repair phase.
- [Failure-record recovery](coverage_retry_20261021/failure_record_recovery_addendum.json), [post-run null guard](coverage_retry_20261021/postrun_summary_guard_fix.json): offline record recovery and summary-handling fixes.
- [recovery.json](recovery.json): synthetic fault and process-exit checks.

Run `python -m experiments.evaluation.verify_release_results` from the repository root to verify source hashes, record mappings and aggregate metrics. See the [evaluation protocol](../../docs/evaluation.md) for statistical definitions. These files omit report text, source excerpts and judge explanations; arithmetic verification does not validate semantic judgments.
