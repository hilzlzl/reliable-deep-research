# 安装与使用

[English](running.md) · [中文 README](../README.zh-CN.md)

## 1. 环境与安装

使用 Python 3.11+，参考环境为 Python 3.11。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，再在仓库根目录执行；以下单行命令适用于 PowerShell 和 Bash。

```bash
uv sync --frozen --python 3.11 --extra dev
uv run --frozen python -m experiments.reliability.reliability_demo
uv run --frozen --extra dev python -m pytest -q
```

演示和测试不需要模型密钥。演示会注入一次任务失败，验证重试、已完成任务复用与依赖重放，结果保存在 `artifacts/reliability_demo/`。

没有 uv 时，可在 Python 3.11 虚拟环境中运行 `python -m pip install -e ".[dev]"`，但依赖版本不会严格按锁文件复现。仓库的 `experiments` 命令需要在源码目录运行。

## 2. 模型与检索配置

复制环境配置：

```bash
cp .env.example .env
```

PowerShell 使用 `Copy-Item .env.example .env`。在 `.env` 中填写：

```dotenv
OPENAI_API_KEY=你的模型密钥
TAVILY_API_KEY=你的检索密钥
```

入口自动加载 `.env`。默认通过 Tavily 搜索，研究、压缩和报告模型为 `openai:gpt-4.1`，摘要模型为 `openai:gpt-4.1-mini`。密钥留在本地配置中，不提交到仓库。

使用兼容 API 服务时，在 `.env` 中另设 `EXPERIMENT_API_KEY=...`，并在运行命令中同时提供模型名、API 前缀和密钥变量名：

```bash
uv run --frozen python -m experiments.runners.run_research --question "How do durable agent tasks recover?" --model openai:YOUR_MODEL --base-url https://YOUR_HOST/v1 --api-key-env EXPERIMENT_API_KEY
```

替换 `YOUR_MODEL` 和 `YOUR_HOST`，地址不要追加 `/chat/completions`。服务商支持时，可追加 `--plain-json-output`、`--streaming` 或 `--reasoning-effort low`；其中 `--plain-json-output` 用于不支持原生结构化输出的服务。

## 3. 运行研究

```bash
uv run --frozen python -m experiments.runners.run_research --question "Compare task-level and graph-level recovery in agent workflows. Cite primary sources."
```

真实研究会产生模型与检索费用。入口固定使用 `coverage_v2`，默认上限为 40 次模型调用、4 次检索、6 次任务尝试和 1,200 秒；检索、写作、审核与修复共享预算。token 用量包含输入估算和请求后的输出观测，不是精确计费上限。

可用 `--model openai:gpt-4.1` 统一所有角色的模型，用 `--output path/to/output` 指定输出位置。全部参数见：

```bash
uv run --frozen python -m experiments.runners.run_research --help
```

## 4. 查看结果

每次运行在 `artifacts/research/` 下创建独立目录。先看运行根目录的 `manifest.json` 定位对应问题，再检查其子目录：

| 文件 | 内容 |
| --- | --- |
| `report.md` | 研究报告或失败文本 |
| `summary.json` | 状态、配置、验收、耗时、用量与错误 |
| `input.json` | 实际问题输入 |
| `events.jsonl`、`runtime.db` | 事件轨迹、任务结果与证据 |

```bash
uv run --frozen python -m experiments.reliability.inspect_run artifacts/research/ACTUAL_RUN/ACTUAL_CASE
```

将大写占位符替换为真实目录。`status` 为 `succeeded` 表示通过运行验收；`incomplete_evidence` 表示可靠性检查通过但回答仍有缺口；`degraded` 表示需要复核。运行未通过时退出码非零，应结合摘要和事件查看原因。

入口每次创建新运行，不会自动恢复任意中断会话。数据库按问题隔离，不能在不同问题间混用；任务结果复用机制见[系统架构](project_overview.zh-CN.md)。

## 5. 可选用法

**批量问题：** JSONL 文件每行一个对象：

```json
{"id":"my_question","category":"reliability","prompt":"What recovery guarantees do durable agent tasks provide? Cite primary sources."}
```

```bash
uv run --frozen python -m experiments.runners.run_research --questions path/to/questions.jsonl --case my_question
```

省略 `--case` 运行所有行，每道题使用独立数据库。

**本地语料：** 准备步骤见[基准说明（英文）](../benchmarks/README.md)。已准备语料可使用 BM25：

```bash
uv run --frozen python -m experiments.runners.run_research --question "What does the corpus establish about task recovery?" --search-api local --local-corpus-path benchmark_data/expanded_20261021
```

本地检索无需 Tavily 密钥，研究与写作仍需要模型 API。若要启用 BM25 + 向量融合，先安装依赖，再在上述命令追加 `--local-retrieval-strategy bm25_semantic_rrf`；首次使用会下载嵌入模型权重。

```bash
uv sync --frozen --extra dev --extra benchmark
```

**实验复核：** 报告指标、数值复算与进程恢复实验见[实验评测](evaluation.zh-CN.md)。

## 6. 常见运行问题

| 现象 | 处理方式 |
| --- | --- |
| 模块无法导入 | 在仓库根目录运行，先执行安装命令，并通过 `uv run --frozen` 使用项目环境 |
| 模型认证或连接失败 | 检查 `.env`、模型权限、API 地址、额度与服务商限流；自定义 API 的三个参数需同时提供 |
| 服务商不支持结构化输出 | 尝试 `--plain-json-output`；不支持流式或推理选项时去掉对应参数 |
| 本地语料无法加载 | 检查 `--local-corpus-path` 是否指向已准备的语料目录，按基准说明完成准备 |
| 报告失败或预算耗尽 | 查看 `summary.json` 的错误和 `events.jsonl`；可缩小问题范围，或用 `--max-llm-calls`、`--max-search-calls`、`--max-total-seconds` 调整预算 |
| 报告存在缺口或需要复核 | 查看验收摘要与报告中的缺口说明，结合引用原文检查内容后再使用 |
