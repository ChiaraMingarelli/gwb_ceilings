# Reproducibility package for *Energetic Ceilings and Benchmarks of Astrophysical Gravitational-Wave Backgrounds*

This directory is the public-facing numerical package for the paper.  It
contains the authoritative six-channel energy ledger, the published-summary
SMBHB merger-factor inference, regression tests, detector inputs, and figure
builders.  Drafts, referee material, literature PDFs, old notebooks, and the
companion AGN--IMRI paper are deliberately outside this directory.

## Scientific scope

The package reproduces:

- the six benchmark GWB spectra and integrated energies;
- the physical SMBHB fixed-frequency ceiling and the five theoretical upper
  limits over their stated inputs;
- the propagated LVK merger-rate intervals used for the AGN--IMRI, BNS, and
  sBBH benchmarks;
- the one-at-a-time LVK rate/evolution checks and the O1--O4a stochastic-limit
  inversion;
- the approximate posterior for the SMBHB processed-mass factor
  `f_merge`, constructed from published summary information; and
- the numerical content of the three manuscript figures.

The LVK intervals propagated here are rate-only unless the output explicitly
says otherwise.  Representative masses, efficiencies, spectra, and declared
histories remain fixed.  The five theoretical upper limits are deterministic
endpoint evaluations, not credible intervals.

Some frozen array and JSON keys retain older schema tokens such as
`conditional_*` and `*_ceiling` for compatibility with earlier analysis
scripts.  For the five non-SMBHB populations, those keys denote the paper's
stated-input theoretical upper-limit cases, not physical ceilings.

## Layout

```text
reproducibility/
  verify_ceilings.py                 authoritative deterministic ledger
  make_figure1_plane_forest.py       PTA comparison figure
  make_fig2_corrected.py             fully reproducible multiband figure
  fmerge/                            published-summary posterior calculation
  tests/                             numerical and figure regression tests
  data/detectors/                    detector inputs with provenance and hashes
  data/pta/                          provenance note for optional PTA context
  results/                           frozen JSON/NPZ reference results
  figures/                           generated figures
  tools/                             release checks and manuscript sync
  archive/author-figure3/            audited input for the author-layout figure
```

## Quick start

Python 3.10 or newer is required.

```bash
cd reproducibility
python -m pip install -e '.[figures,legacy-pdf]'
make verify
make figures
```

The dependency ranges are declared in `pyproject.toml`.  For an exact copy of
the final tested environment, install `requirements-tested.txt` first.

This is a source-checkout workflow.  The editable install supplies the Python
dependencies and command-line entry points while the scripts read the
versioned `data/`, `results/`, and `archive/` files in this directory.  A
standalone wheel is not currently a supported distribution format.

The individual commands are:

```bash
# Deterministic ledger and spectra
python verify_ceilings.py \
  --json results/step3-numbers.json \
  --npz results/step3-spectra.npz

# Published-summary f_merge propagation
python fmerge/lm24_cnm_posterior.py \
  --samples 262144 \
  --seed 20240724 \
  --output results/fmerge/summary.json \
  --sample-output results/fmerge/posterior_draws.npz

# Tests
python -m unittest -v \
  tests/test_verify_ceilings.py \
  tests/test_phasev_figure_context.py
python -m unittest -v fmerge/test_lm24_cnm_posterior.py
```

`make check-generated` recalculates the frozen JSON/NPZ results in a temporary
directory and compares their scientific contents with `results/`.

## Figures and manuscript assets

`make figures` writes all generated PDFs and PNGs to `figures/`.  The PTA
forest, `f_merge` plot, and `gwb-ceilings-reproducible.pdf` multiband plot are
the direct manuscript assets.  The multiband figure contains the six benchmark
spectra and intentionally omits the cumulative energy panel.  The full-census
SMBHB curve is solid from $10^{-9}$ to $10^{-6}$ Hz, with short dashed guide
extensions on both sides; its total modeled energy is calculated separately
from the global mass budget.

Run `make check-paper-assets` to compare generated assets with the sibling
manuscript.  `make sync-paper-assets` is the explicit write operation; it
copies only the three approved PDFs.

## Optional PTA plotting context

When the repository-level `step3-figure-context.npz` archive is present, the
multiband build includes the five light-gray PTA free-spectrum marginals
automatically.  A public build without that optional archive omits them.  The
compact archive used for those contextual violins was derived
from a source table for which a public machine-readable release and reuse
license have not yet been identified.  It is therefore deliberately excluded
from this package.  This omission does not affect any benchmark, LVK
calculation, integrated energy, or the fixed-slope `f_merge` inference, which
uses only published amplitude quantiles.

An archive at another location can be supplied explicitly with
`make_fig2_corrected.py --pta-context PATH`.  See `data/pta/README.md`.

## Public release notes

- This directory is published through an allowlisted export at
  <https://github.com/ChiaraMingarelli/gwb_ceilings>; the full working
  repository is not published.
- The software is released under the MIT license (see `LICENSE`).
- The detector inputs have pinned official sources and hashes.  They are
  third-party data products outside the software license; their verified
  redistribution terms are documented in
  `data/detectors/REDISTRIBUTION_CLEARANCE.md`.
- The package uses only published summary values for the NANOGrav fixed-slope
  amplitude inference.  No collaboration-internal posterior product is
  included or required.

## Citation

Please cite the associated paper and the software release.  The provisional
software metadata are in `CITATION.cff`; add the public repository URL and
release DOI when available.
