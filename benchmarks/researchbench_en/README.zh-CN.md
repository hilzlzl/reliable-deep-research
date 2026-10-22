# 种子语料

[English](README.md) · [基准索引](../README.zh-CN.md)

这 7 篇论文是主线 24 来源语料的种子。`sources.jsonl` 记录版本、哈希和准入状态；原始文件与派生产物保存在忽略的 `benchmark_data/` 中。`approved_local_only` 不代表允许再分发；构建器拒绝没有批准准入状态的来源。问题参考要点由助手编写，尚无独立人工验证。

安装 `--extra benchmark` 后，下载并构建：

```bash
uv run --frozen --extra benchmark python -m experiments.benchmarks.acquire_benchmark_sources --manifest benchmarks/researchbench_en/sources.jsonl --raw-dir benchmark_data/researchbench_en/raw
uv run --frozen --extra benchmark python -m experiments.benchmarks.build_local_corpus --manifest benchmarks/researchbench_en/sources.jsonl --raw-dir benchmark_data/researchbench_en/raw --output-dir benchmark_data/researchbench_en/derived
```

构建器按来源 ID 读取一个原始 PDF/HTML/Markdown 文件，生成确定性的文档与分块，并报告缺失来源，不调用模型。每个来源 ID 只保留一种文件扩展名。复现主线设置时，继续[扩充语料准备](../README.zh-CN.md)。

这里的 9 道种子问题供代码检查和语料工具使用；发布主表使用独立的 30 题文件，不混合不同实验结果。
