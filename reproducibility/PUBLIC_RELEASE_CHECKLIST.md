# Public release procedure

The public repository <https://github.com/ChiaraMingarelli/gwb_ceilings> was
created from this package under the MIT license (`LICENSE`), with the
redistribution terms for the third-party detector files documented in
`data/detectors/REDISTRIBUTION_CLEARANCE.md`.  This document describes how
that public export is reproduced or updated.

## Safety rules (always in force)

- The public repository is produced by an allowlisted export containing this
  directory and the final manuscript source/assets only.  Never publish the
  private working tree or its history.
- Never copy referee reports, response letters, literature PDFs, private
  collaboration files, or old notebooks into the export.
- Keep the derived PTA free-spectrum context archive out of the public
  export unless its machine-readable source, license, and permission have
  been verified.  The public figure must build without it (see
  `data/pta/README.md`).
- Any new third-party data file needs its redistribution terms recorded in
  `data/detectors/REDISTRIBUTION_CLEARANCE.md`, or must be replaced by an
  official fetch-and-hash step.

## Updating the public export

1. Make and verify the changes in this package: run `make check-generated`
   and `make figures`, and confirm the unit tests pass.
2. Confirm the three manuscript figure assets with `make check-paper-assets`.
3. After all source, metadata, and license edits, review and refresh
   `MANIFEST.sha256` with `make manifest`.
4. Rebuild the public export from a fresh clone using the allowlist, and run
   `make release-check` from that fresh checkout before pushing.
5. For a versioned release, archive a tagged release and add its DOI to
   `CITATION.cff`.

The software license, the public GitHub URL in `CITATION.cff` and the
package metadata, and the detector-data clearance were settled at the
initial release; revisit them only when the corresponding inputs change.
