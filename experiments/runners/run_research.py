"""Run Reliable Deep Research with bounded evidence repair."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def positive_number(value: str) -> float:
    """Reject budgets that cannot permit useful work."""
    import math

    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def positive_int(value: str) -> int:
    """Parse a positive integral budget."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Expose the supported workflow rather than development profiles."""
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--question", help="Research question, with no reference answers")
    inputs.add_argument("--questions", type=Path, help="JSONL file with id and prompt fields")
    parser.add_argument("--case", action="append", default=[], help="Select an ID from --questions")
    parser.add_argument("--output", type=Path, default=Path("artifacts/research"))
    parser.add_argument("--search-api", choices=("tavily", "local"), default="tavily")
    parser.add_argument("--local-corpus-path", type=Path)
    parser.add_argument("--local-retrieval-strategy", choices=("bm25", "bm25_semantic_rrf"), default="bm25")
    parser.add_argument("--local-embedding-cache-path", type=Path)
    parser.add_argument("--model", help="Use one model for research, compression, report and summary")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL")
    parser.add_argument("--api-key-env", help="Environment variable name containing the API key")
    parser.add_argument("--plain-json-output", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"))
    parser.add_argument("--max-llm-calls", type=positive_int, default=40)
    parser.add_argument("--max-search-calls", type=positive_int, default=4)
    parser.add_argument("--max-total-seconds", type=positive_number, default=1200)
    args = parser.parse_args(argv)
    if args.question is not None and (not args.question.strip() or args.case):
        parser.error("--question must be nonempty and cannot be combined with --case")
    if args.search_api == "local" and args.local_corpus_path is None:
        parser.error("--search-api local requires --local-corpus-path")
    if (args.streaming or args.reasoning_effort) and not (args.model and args.base_url and args.api_key_env):
        parser.error("--streaming/--reasoning-effort require --model, --base-url and --api-key-env")
    if bool(args.base_url) != bool(args.api_key_env) or (args.base_url and not args.model):
        parser.error("Custom endpoints require --model, --base-url and --api-key-env together")
    return args


def runner_arguments(args: argparse.Namespace) -> list[str]:
    """Use the evaluated coverage repair profile and existing acceptance engine."""
    result = [
        "--profile", "coverage_v2", "--output", str(args.output),
        "--search-api", args.search_api, "--concurrency", "2",
        "--max-researcher-iterations", "2", "--max-react-tool-calls", "4",
        "--max-search-queries", "1", "--max-search-results", "5",
        "--research-task-timeout", "480", "--claim-extraction-timeout", "90",
        "--evidence-context-max-characters", "9000",
        "--max-llm-calls", str(args.max_llm_calls),
        "--max-search-calls", str(args.max_search_calls),
        "--max-total-seconds", str(args.max_total_seconds),
        "--max-task-attempts", "6", "--max-prompt-tokens", "250000",
        "--max-completion-tokens", "120000", "--research-max-tokens", "2048",
        "--compression-max-tokens", "3072", "--final-report-max-tokens", "4096",
        "--summarization-max-tokens", "2048",
        "--local-retrieval-strategy", args.local_retrieval_strategy,
        "--local-embedding-threads", "2",
    ]
    for name in ("question", "questions", "local_corpus_path", "local_embedding_cache_path", "base_url", "api_key_env", "reasoning_effort"):
        value = getattr(args, name)
        if value is not None:
            result.extend(["--" + name.replace("_", "-"), str(value)])
    for case_id in args.case:
        result.extend(["--case", case_id])
    if args.model:
        for role in ("research", "compression", "final-report", "summarization"):
            result.extend([f"--{role}-model", args.model])
    for name in ("streaming", "plain_json_output"):
        if getattr(args, name):
            result.append("--" + name.replace("_", "-"))
    return result


def main(argv: list[str] | None = None) -> None:
    """Run a report, returning a failing exit status when acceptance fails."""
    args = parse_args(argv)
    from experiments.runners.run_baseline import main as run

    asyncio.run(run(runner_arguments(args)))


if __name__ == "__main__":
    main()
