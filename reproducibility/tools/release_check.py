#!/usr/bin/env python3
"""Fail-closed scan that the public package contains no private material.

Verifies that the required package files are present and that nothing
private has leaked into the export: no referee or review filenames, no
absolute local or private-repository paths in text files or NPZ string
arrays, no private working directories, and no data archive whose public
provenance has not been cleared.  With ``--require-license`` it also
enforces the software license and the third-party detector-data
redistribution clearance.  Any failure raises an error rather than
printing a warning, so an incomplete export cannot pass silently.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "README.md",
    "PUBLIC_RELEASE_CHECKLIST.md",
    "CITATION.cff",
    "pyproject.toml",
    "Makefile",
    "verify_ceilings.py",
    "results/step3-numbers.json",
    "results/step3-spectra.npz",
    "results/fmerge/summary.json",
    "results/fmerge/posterior_draws.npz",
    "archive/author-figure3/gwb-ceilings-final-prepatch.pdf",
)
TEXT_SUFFIXES = {".cff", ".json", ".md", ".py", ".toml", ".txt"}
FORBIDDEN_PARTS = (
    "/Users/",
    "gwb_ceilings-private",
)
FORBIDDEN_PATH_COMPONENTS = {"codes-copies"}
FORBIDDEN_NAMES = (
    "referee",
    "response-letter",
    "review-report",
)
PRIVATE_PTA_ARCHIVE = "step3-figure-context.npz"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-license",
        action="store_true",
        help="enforce license and third-party-data clearance for release",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            failures.append(f"missing required file: {relative}")

    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative_path = path.relative_to(ROOT)
        if any(
            part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info")
            for part in relative_path.parts
        ):
            continue
        relative = relative_path.as_posix()
        lowered = relative.lower()
        if any(part in FORBIDDEN_PATH_COMPONENTS for part in relative_path.parts):
            failures.append(f"private working directory in package: {relative}")
        if any(name in lowered for name in FORBIDDEN_NAMES):
            failures.append(f"private/review filename in package: {relative}")
        if path.name == PRIVATE_PTA_ARCHIVE:
            failures.append(
                f"PTA context lacks cleared public provenance: {relative}"
            )
        if (
            path.resolve() != Path(__file__).resolve()
            and path.suffix.lower() in TEXT_SUFFIXES
        ):
            text = path.read_text(encoding="utf-8", errors="replace")
            for forbidden in FORBIDDEN_PARTS:
                if forbidden in text:
                    failures.append(f"forbidden private path in {relative}: {forbidden}")
        if path.suffix.lower() == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    values = archive[key]
                    if values.dtype.kind not in {"S", "U"}:
                        continue
                    text = "\n".join(str(value) for value in values.ravel())
                    for forbidden in FORBIDDEN_PARTS:
                        if forbidden in text:
                            failures.append(
                                f"forbidden private path in {relative}:{key}: "
                                f"{forbidden}"
                            )

    license_path = ROOT / "LICENSE"
    detector_clearance = (
        ROOT / "data" / "detectors" / "REDISTRIBUTION_CLEARANCE.md"
    )
    if args.require_license and not license_path.is_file():
        failures.append("LICENSE is required before public release")
    elif not license_path.is_file():
        print("warning: no LICENSE selected; public release is not yet ready")
    if args.require_license and not detector_clearance.is_file():
        failures.append(
            "third-party detector-file redistribution terms must be "
            "documented or the files replaced by fetch-and-hash inputs"
        )
    elif not detector_clearance.is_file():
        print(
            "warning: detector-file redistribution clearance is not yet "
            "documented"
        )

    if failures:
        raise RuntimeError("release check failed:\n- " + "\n- ".join(failures))
    print("release-safety checks passed")


if __name__ == "__main__":
    main()
