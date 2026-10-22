# Publishing

[简体中文](PUBLISHING.zh-CN.md)

This release uses the name Reliable Deep Research. Preserve the upstream [MIT license](LICENSE) and attribution. The release contains source and compact numerical evidence, not raw corpora, generated reports, credentials or local Git history.

## Prepare a reviewable snapshot

`release-files.txt` is the explicit publication allowlist. From the repository root of the research workspace or a clean snapshot:

```bash
python scripts/prepare_release.py --output ../github-release
```

The destination must not exist. The script copies only listed files and creates a sibling ZIP and SHA-256 record. It never commits, uploads or overwrites an existing directory. Full development records stay in the original workspace. Review the generated directory as the GitHub candidate.

## Validate the actual release directory

Run from the generated directory:

```bash
uv sync --frozen --python 3.11 --extra dev
uv run --frozen --extra dev python -m pytest -q
uv run --frozen python -m experiments.reliability.reliability_demo
uv run --frozen python -m experiments.evaluation.verify_release_results
uv build
```

CI repeats offline tests, demo, numerical verification and distribution builds without model keys. Changes to packaging do not constitute a new end-to-end benchmark. The live entry point was not rebenchmarked by the public cleanup.

## GitHub

Initialize a new Git repository **inside the reviewed release directory**. Inspect the files and staged diff before committing. Add your own GitHub repository as the remote and push only when ready. Do not use the inherited upstream remote as a publishing destination. If preserving old Git history instead, audit that history separately: snapshot checks do not inspect prior commits.

This preparation does not create a GitHub repository, push a commit or publish a package to PyPI.
