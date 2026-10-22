# 发布说明

[English](PUBLISHING.md)

本次发布固定使用 Reliable Deep Research。保留上游 [MIT 许可证](LICENSE) 和归属说明。发布内容是源码与精简数值证据；完整实验历史、原始语料、生成产物、凭据和本地 Git 历史留在研究工作区。

## 生成发布目录

`release-files.txt` 是明确的文件白名单。从研究工作区或干净快照的仓库根目录运行：

```bash
python scripts/prepare_release.py --output ../github-release
```

目标目录必须不存在。脚本只复制清单文件，同时生成旁边的 ZIP 与 SHA-256 文件；不会覆盖目录、提交代码或上传。实际审查和提交的对象是生成后的目录。

## 在发布目录验收

从生成的发布目录根目录运行：

```bash
uv sync --frozen --python 3.11 --extra dev
uv run --frozen --extra dev python -m pytest -q
uv run --frozen python -m experiments.reliability.reliability_demo
uv run --frozen python -m experiments.evaluation.verify_release_results
uv build
```

CI 同样运行离线测试、演示、数值复算和发行包构建，不需要模型密钥。此次整理没有重新进行付费端到端实验，也不产生新版本的性能结论。

## 上传 GitHub

在审查后的**发布目录内**新建 Git 仓库，检查文件与暂存差异，再提交并添加自己的 GitHub 仓库地址。继承的上游远端不作为发布目标。如果选择保留原有 Git 历史，需要另行检查历史内容，快照文件检查不能代替历史审查。

当前步骤只准备可发布内容，不创建 GitHub 仓库、不推送，也不发布 PyPI 包。
