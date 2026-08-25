# Gravitational Wave Background Ceilings

Code accompanying *Energetic Ceilings and Benchmarks of Astrophysical
Gravitational-Wave Backgrounds* (Mingarelli 2026): an interactive
visualization app and the paper's full reproducibility package.

## Repository contents

- **`app.py`** — a Streamlit app for exploring how GWB amplitudes scale with
  the available mass reservoirs.
- **[`reproducibility/`](reproducibility/)** — the paper's numerical package:
  the six-channel energy ledger, the SMBHB merger-factor inference, detector
  inputs with provenance and hashes, regression tests, and the builders for
  the manuscript figures. See its [README](reproducibility/README.md) for the
  scientific scope and quick start.

## The six modeled GWB populations

- **SMBHBs**: Supermassive black hole binaries
- **EMRIs**: Extreme mass-ratio inspirals
- **AGN–IMRIs**: Intermediate-mass black holes inspiralling into SMBHs
- **sBBHs**: Stellar-mass binary black holes
- **BNSs**: Binary neutron stars
- **Pop III**: Population III black hole binaries

## Interactive app

Run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproducing the paper's numbers and figures

Python 3.10+:

```bash
cd reproducibility
python -m pip install -e '.[figures,legacy-pdf]'
make verify
make figures
```

`make verify` runs the regression tests, regenerates the frozen ledger
outputs, and checks the release manifest; `make figures` rebuilds the
manuscript figures from scratch.

## Citation

If you use this code or figures from it, please cite:

> Mingarelli, C. M. F. (2026), "Energetic Ceilings and Benchmarks of
> Astrophysical Gravitational-Wave Backgrounds",
> [arXiv:2601.18859](https://arxiv.org/abs/2601.18859)

Machine-readable citation metadata is in
[`reproducibility/CITATION.cff`](reproducibility/CITATION.cff).

## License

MIT (see [LICENSE](LICENSE)). The third-party detector data files in
`reproducibility/data/detectors/` are excluded; their terms and provenance
are documented in
[`REDISTRIBUTION_CLEARANCE.md`](reproducibility/data/detectors/REDISTRIBUTION_CLEARANCE.md).
