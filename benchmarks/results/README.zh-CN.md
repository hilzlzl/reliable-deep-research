# 数值结果

[English](README.md) · [协议与局限](../../docs/evaluation.zh-CN.md) · [基准索引](../README.zh-CN.md)

- [release_comparison.json](release_comparison.json)：三组方案各 60 个评估槽位，包含运行状态、评分、汇总指标与来源映射。
- [评分记录](coverage_formal_20261021/model_audit_numeric.json)：三组方案的逐项评分数据。
- [反馈修复评测记录](coverage_retry_20261021/model_audit_numeric.json)：反馈修复阶段的逐项评分数据。
- [失败记录恢复](coverage_retry_20261021/failure_record_recovery_addendum.json)、[运行后空值保护](coverage_retry_20261021/postrun_summary_guard_fix.json)：离线记录恢复及摘要处理修复说明。
- [recovery.json](recovery.json)：合成故障与进程退出检查。

在仓库根目录执行 `python -m experiments.evaluation.verify_release_results`，核查来源哈希、逐条记录映射和汇总指标。统计口径见[评测协议](../../docs/evaluation.zh-CN.md)。这些文件不含报告正文、来源片段与评审解释；算术核查不验证语义判断。
