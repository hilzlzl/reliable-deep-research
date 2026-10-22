"""Download approved, pinned arXiv sources and record content hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from open_deep_research.local_benchmark import SourceRecord, read_jsonl

ARXIV_PATTERN = re.compile(r"^https://arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?$")


def download(url: str, destination: Path, *, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "ResearchBench-EN/0.1"})
    digest = hashlib.sha256()
    size = 0
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as stream:
        while block := response.read(1024 * 1024):
            stream.write(block)
            digest.update(block)
            size += len(block)
    with temporary.open("rb") as stream:
        signature = stream.read(4)
    if signature != b"%PDF":
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Downloaded content is not a PDF: {url}")
    temporary.replace(destination)
    return {"bytes": size, "sha256": digest.hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    selected = set(args.source)
    sources = read_jsonl(args.manifest, SourceRecord)
    if selected:
        known = {source.source_id for source in sources}
        unknown = selected - known
        if unknown:
            raise ValueError(f"Unknown source IDs: {', '.join(sorted(unknown))}")
        sources = [source for source in sources if source.source_id in selected]

    results = []
    for source in sources:
        if source.acquisition_status != "approved_local_only":
            continue
        match = ARXIV_PATTERN.fullmatch(source.locator)
        if not match:
            results.append({"source_id": source.source_id, "status": "skipped_not_pinned_arxiv"})
            continue
        destination = args.raw_dir / f"{source.source_id}.pdf"
        if destination.exists() and not args.force:
            content = destination.read_bytes()
            results.append({
                "source_id": source.source_id,
                "status": "already_present",
                "bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            })
            continue
        metadata = download(
            f"https://arxiv.org/pdf/{match.group(1)}",
            destination,
            timeout=args.timeout,
        )
        results.append({"source_id": source.source_id, "status": "downloaded", **metadata})

    report = {"sources": results}
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    (args.raw_dir / "acquisition_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
