# 设计与代码阅读地图

[English](project_overview.md) · [运行说明](running.zh-CN.md)

## 目标与边界

交付完成且来源可检查的研究报告，并在故障后复用已完成的研究子任务。上游提供 Supervisor/Researcher 编排、提示词和服务商工具；本仓库聚焦任务持久化、证据血缘及受控评测。

分别衡量答案覆盖与抽样引用支持度；报告完成和任务恢复衡量执行表现，耗时与调用量衡量资源开销。

## 任务运行链路

1. LangGraph 的 Supervisor 生成研究工作；`supervisor_tools` 的可靠路径将其转为类型化的 `ResearchTask`。
2. `DagTaskScheduler` 校验依赖和环，以有界批次执行就绪任务，处理任务重试与超时。共享的内存 `RunBudget` 原子预留调用额度，避免并发任务各自花掉同一份剩余额度。
3. `RuntimeStore` 将任务成功状态和结果一起提交。Evidence/Claim 可能在成功前写入，因此失败、重放或中断的尝试会在重试前清理派生产物，避免新旧证据混用。
4. 读取同一个数据库可恢复成功任务的结果；显式重放还会使下游任务失效。运行入口为每道题创建独立数据库。

这是任务级恢复：没有持久化整个 LangGraph 执行位置或内存预算计数，也无法撤销进程退出前已发送的外部模型/搜索请求。真实进程退出实验使用合成任务，只验证这些具体语义。

## 发布主线：证据、问题覆盖与修复

发布入口固定选择 **Reliable Deep Research**。它复用上面的持久化研究运行时，报告阶段按问题要点组织证据，不要求先经过早期的 Claim 提取路径。

1. 在检索证据之前，从用户问题拆出稳定要点和约束，避免把已经搜到的内容误当成全部需求。
2. 检索结果持久化为 Document/Evidence；报告计划围绕每个问题记录可回答内容、证据 ID 和缺口。
3. 写作使用 `[EVIDENCE:<id>]` 标记；按证据审核每个段落的语义支持，同时单独检查问题是否答全。
4. 依据当前审核意见与当前证据修复，保留已核实段落，不把旧计划里被否定的答案重新塞回。引用绑定修正后也要再次审核。
5. 最多 2 轮修复、总共 2 个定向补搜任务；补搜可以使用全局剩余第 4 次检索，模型与时间仍预留审核资源。一次 JSON/schema 格式重试也计入预算。
6. 渲染器从存储证据 ID 解析引用并审计来源。可靠性验收与答案完整性分开；`incomplete_evidence` 可以通过可靠性检查但仍有答案缺口，`degraded` 需要复核并返回非零退出码。

上述语义审核是模型判断，不能代替独立事实核查。旧 Claim-first 的 Claim 准入实现保留为对照代码，不是发布默认流程。

## 推荐阅读顺序

| 顺序 | 文件 | 阅读重点 |
| --- | --- | --- |
| 1 | [研究图](../src/open_deep_research/deep_researcher.py)、[运行入口](../experiments/runners/run_research.py) | 发布入口、共享预算创建和结果验收 |
| 2 | [任务运行时](../src/open_deep_research/task_runtime.py)、[存储](../src/open_deep_research/runtime_store.py) | 状态迁移、任务与结果事务、重启和重放清理 |
| 3 | [证据模型](../src/open_deep_research/evidence.py)、[覆盖报告](../src/open_deep_research/coverage_reporting.py) | 来源身份、问题拆解、段落复核与当前反馈修复 |
| 4 | [报告规划](../src/open_deep_research/report_planning.py) | 共享证据 ID 引用渲染和来源审计 |
| 5 | [预算](../src/open_deep_research/run_budget.py)、[事件](../src/open_deep_research/observability.py) | 原子预留、token 估算与故障观测 |
| 6 | [恢复实验](../experiments/reliability/process_recovery_experiment.py)、[数值核查](../experiments/evaluation/verify_release_results.py) | 真实进程退出、全部尝试、指标与分母 |

## 关键取舍

- **选择 SQLite：** 为本地原型提供可检查、事务化的存储；当前不是分布式调度或多租户服务。
- **恢复任务结果：** 把恢复边界缩小到可用进程退出实验验证的子任务；更强的外部副作用保证仍需要幂等设计。
- **使用 Evidence ID：** 确定性约束引用来源，但不能阻止所有无证据推断，也不能保证选取的证据足够完整。
- **限制验证与补搜：** 避免无界批评循环；有限预算仍可能导致失败或不完整回答；缺口补搜与审核共享额度。
- **分别汇报指标：** 报告覆盖、引用支持与资源开销反映不同方面的表现，不能用单一总分掩盖这些取舍。

## 迭代依据与下一步

进程退出实验暴露了已完成工作重复执行、旧证据残留的问题，随后通过任务结果事务和失效清理解决。仅检查来源 URL 又无法确认语义支持，因此评测从宽泛质量打分转向抽样引用检查，并单独检查明确回答的要点覆盖。

[实验评测](evaluation.zh-CN.md)说明对照方案、指标口径与已知局限。后续需要独立人工复核，并用发布配置完成全量验证。
