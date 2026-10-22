"""Copy an explicit source allowlist into a new reviewable GitHub directory."""

from __future__ import annotations

import argparse
import hashlib
import re
import zipfile
from pathlib import Path, PurePosixPath

KEY_PATTERN = re.compile(rb"sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")


def prepare(source: Path, output: Path) -> dict:
    """Copy only allowlisted files, without overwriting existing user content."""
    source, output = source.resolve(), output.resolve()
    archive = output.with_name(output.name + ".zip")
    checksum = archive.with_name(archive.name + ".sha256")
    if output == source or source in output.parents:
        raise ValueError("Place the release outside the source directory")
    for path in (output, archive, checksum):
        if path.exists():
            raise FileExistsError(f"Destination already exists: {path}")
    entries = [line.strip() for line in (source / "release-files.txt").read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    if len(entries) != len(set(entries)):
        raise ValueError("Duplicate paths in release-files.txt")
    files = []
    for entry in entries:
        rel = PurePosixPath(entry)
        if rel.is_absolute() or ".." in rel.parts or "\\" in entry:
            raise ValueError(f"Invalid release path: {entry}")
        path = source / entry
        if not path.is_file() or path.is_symlink() or source not in path.resolve().parents:
            raise ValueError(f"Missing or unsafe release file: {entry}")
        if any(part in {".git", ".venv", "artifacts", "benchmark_data", "__pycache__"} for part in rel.parts):
            raise ValueError(f"Local artifact in allowlist: {entry}")
        if (path.name.startswith(".env") and path.name != ".env.example") or path.suffix in {".db", ".sqlite", ".sqlite3", ".pem", ".key"}:
            raise ValueError(f"Private or generated file in allowlist: {entry}")
        raw = path.read_bytes()
        if KEY_PATTERN.search(raw):
            raise ValueError(f"Possible credential in {entry}; content withheld")
        files.append((entry, raw))
    output.mkdir(parents=True)
    for entry, raw in files:
        target = output / entry
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for entry, raw in sorted(files):
            info = zipfile.ZipInfo("reliable-deep-research/" + entry, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            bundle.writestr(info, raw)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return {"directory": str(output), "archive": str(archive), "files": len(files), "sha256": digest}


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(Path(__file__).resolve().parents[1], args.output), indent=2))


if __name__ == "__main__":
    main()
