#!/usr/bin/env python3
"""Correct only the AGN--IMRI normalization in the author-finished Figure 3.

The publication PDF was finished in Keynote after the code-generated plot was
exported.  Its labels and page layout therefore cannot be reproduced by simply
rerunning ``make_fig2_corrected.py``.  The only stale plotted quantity is the
AGN--IMRI benchmark, which used R_BBH=30 instead of the adopted 30.4
Gpc^-3 yr^-1.  Because Omega_gw is linear in this rate, the correction is a
uniform translation in log(Omega_gw).

This script edits the single vector content stream, changing only:

* the upper boundary of the translucent orange AGN--IMRI fill; and
* the coordinate transforms of the dashed orange continuation, the white
  halo beneath the solid curve, and the solid orange curve.

Every identifying hash, marker, and geometry count is checked before a
temporary candidate is written.  The candidate is then checked for unchanged
metadata, text, page boxes, and font-resource names before it is atomically
written to a separate output path.  Any unexpected input fails without
changing the source.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import tempfile

from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "archive"
    / "author-figure3"
    / "gwb-ceilings-final-prepatch.pdf"
)
DEFAULT_OUTPUT = ROOT / "figures" / "gwb-ceilings-author-layout.pdf"

EXPECTED_SOURCE_SHA256 = (
    "cc6fb9cb9ad0a368dbc3167c2f26bd754c2ebbe57efe9a1e42f3830ed2c4ab77"
)
EXPECTED_SOURCE_CONTENT_SHA256 = (
    "27b8a7b2753ad7437e26a606d2cf1cab50e26ffc996574084e3195e5adfcdcbf"
)
EXPECTED_PATCHED_CONTENT_SHA256 = (
    "03e0821df5366d98701bce3dbd308c88a179e5130a8cc3d7705394c567a67837"
)

OLD_AGN_RATE_GPC3_YR = 30.0
NEW_AGN_RATE_GPC3_YR = 30.4

# Coordinates of consecutive two-decade y ticks in the original Matplotlib
# axes, before the Keynote page transform.  The displayed range is
# 1e-18 <= Omega_gw <= 1e-6.
AXIS_Y_TICK_LOW = 93.154503
AXIS_Y_TICK_HIGH = 133.624438
AXIS_TICK_DECADES = 2.0
PAGE_SCALE = 3.198748
FILL_BASELINE_Y = 280.043

FILL_MARKER = b"/Cs1 cs 0.8352941 0.3686275 0 sc /Gs2 gs "
FILL_END_MARKER = b" h f Q q "
ORANGE_DASH_MARKER = b"0.8352941 0.3686275 0 SC /Gs5 gs"
ORANGE_SOLID_MARKER = b"0.8352941 0.3686275 0 SC /Gs7 gs"
WHITE_HALO_MARKER = b"/Cs2 CS 1 SC /Gs6 gs"

EXPECTED_FILL_COORDINATES = 4539
EXPECTED_FILL_MOVES = 1
EXPECTED_FILL_LINES = 4538
EXPECTED_FILL_BASELINE_COORDINATES = 2279
EXPECTED_FILL_SHIFTED_COORDINATES = 2260

COORDINATE_PATTERN = re.compile(
    rb"(-?\d+(?:\.\d+)?) (-?\d+(?:\.\d+)?) ([ml])"
)
OLD_CTM_PATTERN = re.compile(
    rb"(3\.198748\s+0\s+0\s+3\.198748\s+-4\.729817\s+)"
    rb"-17\.93481(\s+cm)"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def unique_index(data: bytes, marker: bytes, label: str) -> int:
    count = data.count(marker)
    require(count == 1, f"expected one {label} marker, found {count}")
    return data.index(marker)


def patch_fill(data: bytes, delta_page: float) -> tuple[bytes, int]:
    marker_position = unique_index(data, FILL_MARKER, "orange fill")
    start = marker_position + len(FILL_MARKER)
    end = data.index(FILL_END_MARKER, start)
    region = data[start:end]

    coordinates = list(COORDINATE_PATTERN.finditer(region))
    move_count = sum(match.group(3) == b"m" for match in coordinates)
    line_count = sum(match.group(3) == b"l" for match in coordinates)
    baseline_count = sum(
        abs(float(match.group(2)) - FILL_BASELINE_Y) < 1.0e-5
        for match in coordinates
    )
    below_count = sum(
        float(match.group(2)) < FILL_BASELINE_Y - 1.0e-5
        for match in coordinates
    )
    require(
        len(coordinates) == EXPECTED_FILL_COORDINATES,
        f"orange fill coordinate count changed to {len(coordinates)}",
    )
    require(move_count == EXPECTED_FILL_MOVES, "orange fill move count changed")
    require(line_count == EXPECTED_FILL_LINES, "orange fill line count changed")
    require(
        baseline_count == EXPECTED_FILL_BASELINE_COORDINATES,
        f"orange fill baseline count changed to {baseline_count}",
    )
    require(below_count == 0, "orange fill extends below its expected baseline")

    shifted_count = 0

    def shift_upper_boundary(match: re.Match[bytes]) -> bytes:
        nonlocal shifted_count
        y_value = float(match.group(2))
        if y_value <= FILL_BASELINE_Y + 1.0e-5:
            return match.group(0)
        shifted_count += 1
        shifted_y = f"{y_value + delta_page:.4f}".rstrip("0").rstrip(".")
        return (
            match.group(1)
            + b" "
            + shifted_y.encode("ascii")
            + b" "
            + match.group(3)
        )

    patched_region = COORDINATE_PATTERN.sub(shift_upper_boundary, region)
    require(
        shifted_count == EXPECTED_FILL_SHIFTED_COORDINATES,
        f"shifted {shifted_count} orange fill coordinates, expected "
        f"{EXPECTED_FILL_SHIFTED_COORDINATES}",
    )
    return data[:start] + patched_region + data[end:], shifted_count


def shifted_ctm_value(delta_page: float) -> bytes:
    shifted_translation = -17.93481 + delta_page
    return f"{shifted_translation:.6f}".encode("ascii")


def replace_one_ctm(
    data: bytes,
    start: int,
    end: int,
    new_translation: bytes,
    label: str,
) -> bytes:
    region = data[start:end]
    patched_region, count = OLD_CTM_PATTERN.subn(
        lambda match: match.group(1) + new_translation + match.group(2),
        region,
        count=1,
    )
    require(count == 1, f"expected one stale CTM in {label}, found {count}")
    require(
        OLD_CTM_PATTERN.search(patched_region) is None,
        f"more than one stale CTM remains in {label}",
    )
    return data[:start] + patched_region + data[end:]


def patch_curve_paths(data: bytes, delta_page: float) -> tuple[bytes, int]:
    unique_index(data, ORANGE_DASH_MARKER, "orange dashed curve")
    unique_index(data, ORANGE_SOLID_MARKER, "orange solid curve")
    new_translation = shifted_ctm_value(delta_page)
    patched_count = 0

    # Dashed continuation: patch the first CTM between its color marker and
    # the solid orange marker.
    dashed_start = data.index(ORANGE_DASH_MARKER)
    solid_start = data.index(ORANGE_SOLID_MARKER)
    halo_start = data.rindex(WHITE_HALO_MARKER, dashed_start, solid_start)
    data = replace_one_ctm(
        data,
        dashed_start,
        halo_start,
        new_translation,
        "orange dashed continuation",
    )
    patched_count += 1

    # The white path-effect halo immediately preceding the solid orange line
    # has the same geometry and must move with it.
    solid_start = data.index(ORANGE_SOLID_MARKER)
    halo_start = data.rindex(WHITE_HALO_MARKER, dashed_start, solid_start)
    data = replace_one_ctm(
        data,
        halo_start,
        solid_start,
        new_translation,
        "white halo beneath orange solid curve",
    )
    patched_count += 1

    # Solid orange line.  Its complete path is shorter than 2,000 bytes.
    solid_start = data.index(ORANGE_SOLID_MARKER)
    solid_end = min(len(data), solid_start + 2_000)
    data = replace_one_ctm(
        data,
        solid_start,
        solid_end,
        new_translation,
        "orange solid curve",
    )
    patched_count += 1

    require(patched_count == 3, "did not patch exactly three curve CTMs")
    return data, patched_count


def normalized_metadata(reader: PdfReader) -> dict[str, str]:
    if reader.metadata is None:
        return {}
    return {str(key): str(value) for key, value in reader.metadata.items()}


def page_box_tuple(page: object, attribute: str) -> tuple[float, ...]:
    box = getattr(page, attribute)
    return tuple(float(value) for value in box)


def font_resource_names(reader: PdfReader) -> tuple[str, ...]:
    fonts = reader.pages[0]["/Resources"].get("/Font", {})
    return tuple(sorted(str(key) for key in fonts.keys()))


def validate_source(reader: PdfReader, content: bytes) -> None:
    require(len(reader.pages) == 1, "author figure must contain exactly one page")
    require(
        page_box_tuple(reader.pages[0], "mediabox") == (0.0, 0.0, 1920.0, 1080.0),
        "author figure media box changed",
    )
    require(
        sha256_bytes(content) == EXPECTED_SOURCE_CONTENT_SHA256,
        "author figure content-stream hash changed",
    )


def validate_candidate(
    source_reader: PdfReader,
    candidate_path: Path,
) -> None:
    candidate_reader = PdfReader(candidate_path, strict=False)
    require(len(candidate_reader.pages) == 1, "candidate page count changed")
    for box_name in ("mediabox", "cropbox", "trimbox", "bleedbox", "artbox"):
        require(
            page_box_tuple(candidate_reader.pages[0], box_name)
            == page_box_tuple(source_reader.pages[0], box_name),
            f"candidate {box_name} changed",
        )
    require(
        normalized_metadata(candidate_reader) == normalized_metadata(source_reader),
        "candidate PDF metadata changed",
    )
    require(
        candidate_reader.pages[0].extract_text()
        == source_reader.pages[0].extract_text(),
        "candidate extracted text changed",
    )
    require(
        font_resource_names(candidate_reader) == font_resource_names(source_reader),
        "candidate font-resource names changed",
    )
    candidate_content = candidate_reader.pages[0].get_contents().get_data()
    require(
        sha256_bytes(candidate_content) == EXPECTED_PATCHED_CONTENT_SHA256,
        "candidate patched content-stream hash changed",
    )


def patch_pdf(source: Path, output: Path) -> None:
    source = source.resolve()
    output = output.resolve()
    require(source.is_file(), f"source does not exist: {source}")
    require(source != output, "source and output must be different files")
    require(
        sha256_file(source) == EXPECTED_SOURCE_SHA256,
        "refusing to patch: source is not the audited author-finished PDF",
    )

    source_reader = PdfReader(source, strict=False)
    source_content = source_reader.pages[0].get_contents().get_data()
    validate_source(source_reader, source_content)

    rate_scale = NEW_AGN_RATE_GPC3_YR / OLD_AGN_RATE_GPC3_YR
    y_per_decade = (AXIS_Y_TICK_HIGH - AXIS_Y_TICK_LOW) / AXIS_TICK_DECADES
    delta_base = y_per_decade * math.log10(rate_scale)
    delta_page = PAGE_SCALE * delta_base

    patched_content, fill_count = patch_fill(source_content, delta_page)
    patched_content, curve_count = patch_curve_paths(patched_content, delta_page)
    require(
        sha256_bytes(patched_content) == EXPECTED_PATCHED_CONTENT_SHA256,
        "patched content does not match the audited result",
    )

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            writer = PdfWriter(clone_from=str(source))
            stream = DecodedStreamObject()
            stream.set_data(patched_content)
            stream = stream.flate_encode()
            writer.pages[0][NameObject("/Contents")] = writer._add_object(stream)
            writer.write(temporary_file)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        validate_candidate(source_reader, temporary_path)
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    print(f"wrote {output}")
    print(
        "AGN--IMRI normalization: "
        f"R={OLD_AGN_RATE_GPC3_YR:g} -> {NEW_AGN_RATE_GPC3_YR:g}, "
        f"Omega scale={rate_scale:.12f}"
    )
    print(
        f"vector shift: base={delta_base:.9f}, page={delta_page:.9f} pt; "
        f"fill coordinates={fill_count}, curve CTMs={curve_count}"
    )
    print(f"patched content SHA256={EXPECTED_PATCHED_CONTENT_SHA256}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="audited author-finished source PDF",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="path for the corrected author-layout PDF",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    patch_pdf(args.source, args.output)


if __name__ == "__main__":
    main()
