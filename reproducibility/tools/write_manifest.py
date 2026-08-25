#!/usr/bin/env python3
"""Write or verify SHA-256 hashes for the public package's source inputs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.sha256"
EXCLUDED_DIRECTORIES = {
    ".git",
    ".mplconfig",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "figures",
}
EXCLUDED_FILES = {"MANIFEST.sha256"}


def included_files() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(
            part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if relative.as_posix() in EXCLUDED_FILES:
            continue
        paths.append(path)
    return sorted(paths, key=lambda item: item.relative_to(ROOT).as_posix())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_text() -> str:
    return "".join(
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}\n"
        for path in included_files()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    expected = manifest_text()
    if args.write:
        MANIFEST.write_text(expected, encoding="utf-8")
        print(f"wrote {MANIFEST} with {len(included_files())} entries")
        return
    if not MANIFEST.is_file():
        raise RuntimeError("MANIFEST.sha256 is missing; run make manifest")
    actual = MANIFEST.read_text(encoding="utf-8")
    if actual != expected:
        raise RuntimeError("MANIFEST.sha256 is stale; review changes and run make manifest")
    print(f"verified {MANIFEST} ({len(included_files())} entries)")


if __name__ == "__main__":
    main()
