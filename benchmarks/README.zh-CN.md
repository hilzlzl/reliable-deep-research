# 基准证据

[English](README.md) · [评测协议](../docs/evaluation.zh-CN.md)

| 文件 | 用途 |
| --- | --- |
| [30 道问题](expanded_20261021/questions.jsonl) | 冻结的问题、助手编写的参考要点及来源标注 |
| [24 份来源](expanded_20261021/sources.jsonl) | 来源元数据与内容哈希；不含原始文档 |
| [180 个比较槽位](results/release_comparison.json) | 三组方案各 60 个评估槽位，含逐条来源映射 |
| [评分记录](results/coverage_formal_20261021/model_audit_numeric.json)、[反馈修复评测记录](results/coverage_retry_20261021/model_audit_numeric.json) | 汇总指标对应的评分数据 |
| [修复清单](coverage_retry_20261021/frozen_manifest.json)与[环境](coverage_retry_20261021/environment.json) | 反馈修复配置、评测样本与全局预算 |
| [恢复测量](results/recovery.json) | 合成故障矩阵和真实进程退出实验 |
| [生成清单](expanded_20261021/frozen_manifest.json)与[环境](expanded_20261021/environment.json) | 历史生成来源 |
| [重评清单](coverage_formal_20261021/frozen_manifest.json)与[环境](coverage_formal_20261021/environment.json) | 统一评分轮次来源 |
| [ID 格式补充记录](coverage_formal_20261021/evaluation_id_format_addendum.json)、[执行补充记录](coverage_formal_20261021/evaluation_execution_addendum.json) | 一次评审额外格式恢复的披露 |

公开数据保留评估状态、逐项评分和来源映射，可离线复算汇总指标。实验口径与局限见[评测协议](../docs/evaluation.zh-CN.md)。原始报告、来源片段和评审解释不随仓库发布，数值核查不能代替独立语义验证。

```bash
uv run --frozen python -m experiments.evaluation.verify_release_results
```

## 重建本地输入

安装 `--extra benchmark`，先获取 7 篇论文的种子语料：

```bash
uv run --frozen --extra benchmark python -m experiments.benchmarks.acquire_benchmark_sources --help
uv run --frozen --extra benchmark python -m experiments.benchmarks.build_local_corpus --help
```

[种子语料说明](researchbench_en/README.zh-CN.md)提供具体下载和构建命令，再加入其他来源：

```bash
uv run --frozen --extra benchmark python -m experiments.benchmarks.prepare_expanded_corpus --existing-corpus benchmark_data/researchbench_en/derived
uv run --frozen --extra benchmark python -m experiments.benchmarks.evaluate_semantic_retrieval --corpus-dir benchmark_data/expanded_20261021 --questions benchmarks/expanded_20261021/questions.jsonl --embedding-cache benchmark_data/expanded_20261021/embeddings/bge-small-en-v1.5.npz --threads 4 --output artifacts/retrieval.json
```

使用新的工作副本下载语料：准备过程会写入来源元数据，上游文档也可能变化。将新内容哈希与公开元数据比较，不要把新下载文本默认为历史快照。新实验不得覆盖历史清单。

新端到端报告使用 `experiments.runners.run_research`，传入该问题文件、准备好的本地语料和[运行指南](../docs/running.zh-CN.md)中的服务商设置。正式入口默认主线上限与比较设置一致。将问题文件运行两次，输出到不同目录，以得到两次独立尝试。记录新模型、环境、来源与代码哈希；后续运行复现的是程序步骤，不是历史回答或耗时。公开整理晚于冻结的实验代码。
