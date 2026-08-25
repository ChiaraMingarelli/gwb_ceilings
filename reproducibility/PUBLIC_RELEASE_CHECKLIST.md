# Public release checklist

- [ ] Select and add a software `LICENSE`.
- [ ] Confirm the public GitHub URL in `CITATION.cff` and package metadata.
- [ ] Create the public repository from an allowlist containing this directory
      and the final manuscript source/assets only.
- [ ] Do not copy referee reports, response letters, literature PDFs, private
      collaboration files, old notebooks, `codes-copies/`, or repository
      history from the private working tree.
- [ ] Keep the derived PTA free-spectrum context archive out of the public
      export unless its machine-readable source, license, and permission have
      been verified.  The public figure must build without it.
- [ ] Verify the redistribution terms for the three detector context files
      and record them in `data/detectors/REDISTRIBUTION_CLEARANCE.md`, or
      replace the copies with an official fetch-and-hash step.
- [ ] Run `make release-check` from a fresh checkout.
- [ ] Run `make check-generated` and `make figures`.
- [ ] Confirm the three manuscript assets with `make check-paper-assets`.
- [ ] After all source, metadata, and license edits, review and refresh
      `MANIFEST.sha256` with `make manifest`.
- [ ] Archive a tagged release and add its DOI to `CITATION.cff`.
