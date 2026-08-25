#!/usr/bin/env python3
"""Check or explicitly copy the three approved PDFs into the manuscript."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT_DIR = ROOT.parent / "article" / "LISA-IMBH-GWB"
ASSETS = {
    ROOT / "figures" / "pta-amplitude-forest.pdf": "pta-amplitude-forest.pdf",
    ROOT / "figures" / "fmerge-posterior.pdf": "fmerge-posterior.pdf",
    ROOT / "figures" / "gwb-ceilings-reproducible.pdf": (
        "gwb-ceilings-final.pdf"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def page_box(page: object, name: str) -> tuple[float, ...]:
    return tuple(float(value) for value in getattr(page, name))


def pdf_content_signature(path: Path) -> tuple[object, ...]:
    """Return a signature that ignores volatile PDF creation metadata."""

    reader = PdfReader(path, strict=False)
    pages: list[object] = []
    for page in reader.pages:
        content = page.get_contents()
        content_data = b"" if content is None else content.get_data()
        fonts = page.get("/Resources", {}).get("/Font", {})
        pages.append(
            (
                page_box(page, "mediabox"),
                page_box(page, "cropbox"),
                tuple(sorted(str(key) for key in fonts.keys())),
                page.extract_text(),
                hashlib.sha256(content_data).hexdigest(),
            )
        )
    return tuple(pages)


def assets_match(source: Path, destination: Path, destination_name: str) -> bool:
    if not destination.is_file():
        return False
    if destination_name == "gwb-ceilings-final.pdf":
        return sha256(source) == sha256(destination)
    return pdf_content_signature(source) == pdf_content_signature(destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manuscript-dir",
        type=Path,
        default=DEFAULT_MANUSCRIPT_DIR,
        help="directory containing the manuscript figure assets",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="copy the approved PDFs; without this flag, only compare",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manuscript_dir = args.manuscript_dir.resolve()
    if not manuscript_dir.is_dir():
        raise RuntimeError(f"manuscript directory not found: {manuscript_dir}")

    mismatches: list[str] = []
    for source, destination_name in ASSETS.items():
        if not source.is_file():
            raise RuntimeError(f"generated asset missing; run make figures: {source}")
        destination = manuscript_dir / destination_name
        source_hash = sha256(source)
        if not assets_match(source, destination, destination_name):
            if args.write:
                shutil.copy2(source, destination)
                if not assets_match(source, destination, destination_name):
                    raise RuntimeError(f"copy verification failed: {destination}")
                print(f"updated {destination}")
            else:
                mismatches.append(destination_name)
        else:
            print(f"matches {destination_name}: {source_hash}")

    if mismatches:
        joined = ", ".join(mismatches)
        raise RuntimeError(
            f"manuscript assets differ or are absent: {joined}; "
            "review them, then run make sync-paper-assets"
        )
    print("all three approved manuscript assets match")


if __name__ == "__main__":
    main()
