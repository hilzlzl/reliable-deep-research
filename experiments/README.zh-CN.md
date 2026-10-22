# 运行与验证

[English](README.md) · [运行指南](../docs/running.zh-CN.md)

发布主线只有一个入口：

```bash
uv run --frozen python -m experiments.runners.run_research --question "How do durable research tasks recover?"
```

| 目录 | 发布用途 |
| --- | --- |
| `runners/` | Reliable Deep Research 公开入口与共享内部执行/验收引擎 |
| `reliability/` | 无密钥演示、故障注入、真实进程重启及产物检查 |
| `benchmarks/` | 固定语料下载、分块及语义检索检查 |
| `evaluation/` | 公开数值核查与报告证据支持评估工具 |
| `configs/` | 内部引擎使用的种子问题 |

生成报告、模型输出和任务数据库进入被忽略的 `artifacts/`，下载语料进入 `benchmark_data/`。公开证据见[基准索引](../benchmarks/README.zh-CN.md)。工具从仓库根目录运行，不是 wheel 安装的命令行程序。

共享引擎保留历史配置开关以兼容已有源码，正式入口只选择 Reliable Deep Research。开发期专用运行器和历史实验报告不在发布清单内。
