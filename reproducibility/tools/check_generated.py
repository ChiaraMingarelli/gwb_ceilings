#!/usr/bin/env python3
"""Regenerate the frozen numerical results and compare scientific content.

Reruns ``verify_ceilings.py`` and the f_merge propagation in a temporary
directory, then compares the regenerated JSON/NPZ products with the frozen
copies under ``results/`` at strict floating-point tolerances, ignoring
only environment metadata (Python and NumPy versions).  The frozen files
keep their historical names (``step3-numbers.json``, ``step3-spectra.npz``)
so that pinned hashes remain valid.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_NUMBERS = ROOT / "results" / "step3-numbers.json"
REFERENCE_SPECTRA = ROOT / "results" / "step3-spectra.npz"
REFERENCE_FMERGE_SUMMARY = ROOT / "results" / "fmerge" / "summary.json"
REFERENCE_FMERGE_DRAWS = (
    ROOT / "results" / "fmerge" / "posterior_draws.npz"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compare_json(reference: Any, candidate: Any, path: str = "root") -> None:
    """Compare nested deterministic JSON values with strict float tolerances."""

    if isinstance(reference, dict):
        require(isinstance(candidate, dict), f"{path}: expected mapping")
        require(
            set(reference) == set(candidate),
            f"{path}: keys differ: {set(reference) ^ set(candidate)}",
        )
        for key in sorted(reference):
            compare_json(reference[key], candidate[key], f"{path}.{key}")
        return
    if isinstance(reference, list):
        require(isinstance(candidate, list), f"{path}: expected list")
        require(len(reference) == len(candidate), f"{path}: length differs")
        for index, (left, right) in enumerate(zip(reference, candidate)):
            compare_json(left, right, f"{path}[{index}]")
        return
    if isinstance(reference, float):
        require(
            np.isclose(reference, candidate, rtol=2.0e-13, atol=0.0),
            f"{path}: {candidate!r} != {reference!r}",
        )
        return
    require(reference == candidate, f"{path}: {candidate!r} != {reference!r}")


def compare_npz(reference_path: Path, candidate_path: Path) -> None:
    with np.load(reference_path, allow_pickle=False) as reference, np.load(
        candidate_path, allow_pickle=False
    ) as candidate:
        require(
            set(reference.files) == set(candidate.files),
            f"{reference_path.name}: array names differ",
        )
        for key in reference.files:
            left = reference[key]
            right = candidate[key]
            require(left.shape == right.shape, f"{key}: shape differs")
            require(left.dtype == right.dtype, f"{key}: dtype differs")
            if key == "metadata_json":
                left_metadata = json.loads(str(left))
                right_metadata = json.loads(str(right))
                for metadata in (left_metadata, right_metadata):
                    implementation = metadata.get("implementation", {})
                    implementation.pop("python_version", None)
                    implementation.pop("numpy_version", None)
                compare_json(
                    left_metadata,
                    right_metadata,
                    "fmerge.metadata_json",
                )
                continue
            if np.issubdtype(left.dtype, np.number):
                np.testing.assert_allclose(
                    right,
                    left,
                    rtol=2.0e-13,
                    atol=0.0,
                    equal_nan=True,
                    err_msg=f"generated array changed: {key}",
                )
            else:
                np.testing.assert_array_equal(
                    right,
                    left,
                    err_msg=f"generated array changed: {key}",
                )


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="gwb-ceilings-check-") as raw_tmp:
        temporary = Path(raw_tmp)
        numbers = temporary / "step3-numbers.json"
        spectra = temporary / "step3-spectra.npz"
        fmerge_summary = temporary / "summary.json"
        fmerge_draws = temporary / "posterior_draws.npz"

        run(
            [
                sys.executable,
                "verify_ceilings.py",
                "--json",
                str(numbers),
                "--npz",
                str(spectra),
            ]
        )
        compare_json(
            json.loads(REFERENCE_NUMBERS.read_text(encoding="utf-8")),
            json.loads(numbers.read_text(encoding="utf-8")),
        )
        compare_npz(REFERENCE_SPECTRA, spectra)

        run(
            [
                sys.executable,
                "fmerge/lm24_cnm_posterior.py",
                "--samples",
                "262144",
                "--seed",
                "20240724",
                "--output",
                str(fmerge_summary),
                "--sample-output",
                str(fmerge_draws),
            ]
        )
        reference_summary = json.loads(
            REFERENCE_FMERGE_SUMMARY.read_text(encoding="utf-8")
        )
        candidate_summary = json.loads(fmerge_summary.read_text(encoding="utf-8"))
        reference_summary.pop("sample_archive", None)
        candidate_summary.pop("sample_archive", None)
        for summary in (reference_summary, candidate_summary):
            implementation = summary.get("implementation", {})
            implementation.pop("python_version", None)
            implementation.pop("numpy_version", None)
        compare_json(reference_summary, candidate_summary, "fmerge")
        compare_npz(REFERENCE_FMERGE_DRAWS, fmerge_draws)

    print("generated JSON and NPZ scientific contents match the frozen results")


if __name__ == "__main__":
    main()
