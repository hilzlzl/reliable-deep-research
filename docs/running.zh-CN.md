# 运行指南

[English](running.md) · [中文首页](../README.zh-CN.md)

从仓库根目录执行命令，参考环境为 Python 3.11。单行命令适用于 PowerShell 和 Bash。

## 安装与离线检查

```bash
uv sync --frozen --python 3.11 --extra dev
uv run --frozen --extra dev python -m pytest -q
uv run --frozen python -m experiments.reliability.reliability_demo
uv run --frozen python -m experiments.evaluation.verify_release_results
```

不使用 uv 时，先激活 Python 3.11 虚拟环境，再执行 `python -m pip install -e ".[dev]"`；此方式解析兼容依赖，不复现锁文件。`experiments` 工具从源码目录运行，不是 wheel 安装的命令行程序。

| 可选依赖 | 用途 |
| --- | --- |
| `dev` | pytest、发行包构建工具、Ruff |
| `benchmark` | PDF/HTML 提取与 BGE 嵌入；语义检索首次使用时下载模型权重 |

## 实际研究报告

复制 `.env.example` 为 `.env`（PowerShell 使用 `Copy-Item .env.example .env`，Bash 使用 `cp .env.example .env`），填写 `OPENAI_API_KEY` 和 `TAVILY_API_KEY`。入口自动加载 `.env`。

```bash
uv run --frozen python -m experiments.runners.run_research --question "Compare task-level and graph-level recovery in agent workflows. Cite primary sources."
```

此命令产生付费请求。研究、压缩和报告默认使用 `openai:gpt-4.1`，摘要使用 `openai:gpt-4.1-mini`；传入 `--model openai:gpt-4.1` 可统一所有角色。证据不足、服务商错误或预算耗尽均可能导致失败。交付 `report.md` 前应检查验收摘要。`incomplete_evidence` 表示通过可靠性检查但仍有答案缺口；`degraded` 表示需要复核，返回非零退出码。

正式入口使用 Reliable Deep Research，任务与单次输出 token 设置和主线比较的有界设置一致；默认网络检索与模型不同于历史固定语料实验。发布配置启用问题拆解、段落审核、最多 2 轮反馈修复、总共 2 个补搜任务和 1 次计入预算的 JSON/schema 格式重试。全局默认上限为 40 次模型调用、4 次检索、6 次任务尝试、1,200 秒、250,000 个估算输入 token 和 120,000 个观测输出 token。输入 token 采用估算，输出用量在请求后记录，因此 token 限制不是精确的账单上限。`--help` 列出支持的选项。

### 自定义 API 地址

在 `.env` 中设置 `EXPERIMENT_API_KEY=...`，使用服务商的模型名称和 API 基础地址：

```bash
uv run --frozen python -m experiments.runners.run_research --question "How do durable agent tasks recover?" --model openai:YOUR_MODEL --base-url https://YOUR_HOST/v1 --api-key-env EXPERIMENT_API_KEY --plain-json-output --streaming --reasoning-effort low
```

`YOUR_MODEL` 和 `YOUR_HOST` 是占位符。流式和推理选项需要服务商兼容，不支持时省略。密钥值不应出现在命令参数或发布文件中。`.env.example` 默认关闭远程追踪。

### 多题运行与本地检索

JSONL 输入每行一个对象：

```json
{"id":"my_question","category":"reliability","prompt":"What recovery guarantees do durable agent tasks provide? Cite primary sources."}
```

```bash
uv run --frozen python -m experiments.runners.run_research --questions path/to/questions.jsonl --case my_question
```

省略 `--case` 即运行所有行，每题使用独立数据库。基准文件中的参考标注不会传入生成流程。

准备好本地语料后，安装 benchmark extra，选择 BM25 或 BM25+dense 检索：

```bash
uv sync --frozen --extra dev --extra benchmark
uv run --frozen python -m experiments.runners.run_research --question "What does the corpus establish about task recovery?" --search-api local --local-corpus-path benchmark_data/expanded_20261021 --local-retrieval-strategy bm25_semantic_rrf --local-embedding-cache-path benchmark_data/expanded_20261021/embeddings/bge-small-en-v1.5.npz
```

参见[基准输入与语料准备](../benchmarks/README.zh-CN.md)。本地检索仍需要模型 API 完成研究和写作。

## 查看产物

每次调用在 `artifacts/research/` 下创建唯一运行目录：

| 文件 | 含义 |
| --- | --- |
| 运行根目录的 `manifest.json` | 各题摘要，含失败记录 |
| `<case>/input.json` | 实际使用的问题文本 |
| `<case>/summary.json` | 状态、配置、验收、耗时与用量 |
| `<case>/report.md` | 生成报告或失败文本 |
| `<case>/events.jsonl` | 节点、工具、任务事件及预算观测 |
| `<case>/runtime.db` | 任务、结果、Document、Evidence 和 Claim |

```bash
uv run --frozen python -m experiments.reliability.inspect_run artifacts/research/ACTUAL_RUN/ACTUAL_CASE
```

将大写占位符替换为实际产物路径。研究入口在验收失败时返回非零退出码。入口每次创建新题目运行，不会自动恢复任意中断的 LangGraph 会话。不同问题不得共享一个运行时数据库。

公开入口委托给共享内部引擎 `experiments.runners.run_baseline`；其历史文件名不决定运行配置。正式入口运行 Reliable Deep Research，其他开发配置和直接在 Studio 中运行研究图不属于此流程。
