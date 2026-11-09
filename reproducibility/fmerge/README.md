# LM24–CNM merger-factor propagation

This directory contains the reproducible calculation that combines the
population uncertainty reported by Liepold & Ma (2024; LM24) with the
NANOGrav 15-year customized-noise-model (CNM) amplitude. Its primary output
is the paper's distribution for the SMBHB processed-mass factor f_merge —
Eq. (15) of the paper, plotted in Figure 2 — using the LM24 Appendix-B
approximation directly.

The calculation is deliberately separate from verify_ceilings.py. That file
is the deterministic central-value calculation used by the manuscript
figures. This directory reconstructs probability distributions and uses
random draws, so its assumptions, prior, seed, and outputs are versioned
together.

## Scientific status

The default result is constructed entirely from published summary
information.

Version 0.4.0 uses the full LM24 reconstruction as the inference denominator.

- The LM24 input is the galaxy stellar mass-function approximation in its
  Appendix B and author-maintained helper repository.
- The CNM input is the published median and 90% bounds for its power-law amplitude.
  The default code treats those bounds as the 5th and 95th percentiles and
  uses a piecewise-normal surrogate in ln(A_yr) constrained by those three
  numbers. Three quantiles do not determine a unique posterior, so this is an
  explicit approximation.
- Optional externally supplied one-dimensional distributions can replace
  either approximation. The manuscript result does not use that option.
  The free-spectrum-bin data elsewhere in this repository are not valid
  inputs to this calculation and are explicitly rejected.

For the fixed-slope CNM analysis, NANOGrav used a prior uniform in
log10(A_yr). With the declared prior uniform in log10(f_merge) on [-3, 1],
transforming independent amplitude draws according to

    f_merge = (A_CNM / A_model[f_merge=1])^2

is equivalent to marginalizing the CNM likelihood over the LM24 population
distribution, away from the stated prior boundaries. The primary output is
therefore an approximate posterior under a log-uniform f_merge prior. The
code also reports the same samples reweighted to a prior uniform in f_merge;
that is a prior-sensitivity check, not a second measurement.

This inference is conditional on the LM24 mass-ratio and redshift kernels.
Those kernels were chosen to approximate an earlier NANOGrav population fit,
so this is not a fully independent joint population analysis.

## What the calculation does

For each realization, the code:

1. Draws with equal weight from LM24's dynamical-stellar-mass and
   stellar-population-synthesis approximations. The sampled variables are the
   galaxy mass-function amplitudes at 10^10 and 10^12 solar masses, its two
   slopes, and its characteristic mass (LM24 Table B1).
2. Solves for the ordinary double-Schechter amplitudes. As in the author
   helper, negative amplitudes are clipped to 10^-10 per cubic megaparsec;
   the fraction of clipped draws is reported.
3. Independently draws the McConnell–Ma black-hole–bulge intercept and slope,
   and convolves with the 0.34-dex intrinsic scatter published by LM24. The
   later author helper uses 0.35 dex; that alternative can be checked
   with `--intrinsic-scatter-dex 0.35`. The selected value is recorded in every
   output. Covariance between the fitted intercept and slope, and uncertainty
   in the intrinsic scatter, are not available and are not propagated. The
   public helper also reuses a realization index for both the galaxy and
   scaling-relation draws. Because no cross-covariance is supplied, this
   implementation uses independent random streams and records that choice.
4. Computes the full-census black-hole mass moments over host stellar masses
   from 10^8.5 to 10^12 solar masses. It uses the LM24 mass-ratio
   distribution, proportional to q^2 between q=0.1 and q=1.
5. Uses the LM24 redshift distribution and the Phinney population relation to
   calculate A_model for f_merge=1.
6. Pairs independent model draws with draws from the quantile-matched CNM
   surrogate and evaluates the amplitude ratio above.

The calculation depends only on NumPy. Gauss–Legendre quadrature is used for
the stellar-mass, mass-ratio, and redshift integrals.

Here f_merge retains the manuscript definition: the mass involved in mergers,
summed over all events, relative to the available SMBH mass reservoir. For each
LM24 BHMF realization, the reservoir, binary-mass distribution, mass-ratio
and redshift distributions, radiative efficiency, and history-spectrum factor
H(f) are fixed as f_merge varies. Thus h_c^2 is proportional to f_merge, and
the reference LM24 model prediction has f_merge=1. The calculation propagates
uncertainty between BHMF realizations while applying this scaling within
each realization.

Adding generations to the mass-conserving SZQ24 hierarchy is a different
variation: each generation involves the same total mass, but earlier
generations contain less massive binaries and change H(f). Consequently
f_merge grows linearly with the number of generations while the
fixed-frequency strain converges. The amplitude-ratio inference above is
conditional on the fixed population inputs, not a reconstruction of that
hierarchy. This clarification leaves the saved draws and prior unchanged.

## Reproduce the calculation

Python 3.10 or newer is recommended. From the `reproducibility/` package
root:

~~~bash
python -m pip install -e '.[figures]'
python fmerge/lm24_cnm_posterior.py \
  --samples 262144 \
  --seed 20240724 \
  --output results/fmerge/summary.json \
  --sample-output results/fmerge/posterior_draws.npz
python -m unittest -v fmerge/test_lm24_cnm_posterior.py
~~~

The JSON file contains all inputs, citations, numerical settings,
diagnostics, limitations, input hashes, and 5th/50th/95th percentiles. The
optional NPZ contains the paired arrays used to make the summary. Output
directories are created automatically.

## Replace either public approximation

To replace the quantile-matched CNM approximation:

~~~bash
lm24-cnm-posterior \
  --cnm-samples path/to/cnm_amplitude_samples.npz \
  --confirm-external-cnm-semantics \
  --cnm-source-id DOI-or-release-ID
~~~

The input must represent the fixed-gamma=13/3, Hellings–Downs CNM amplitude
posterior with the published log-uniform amplitude prior. If the source chain
was obtained by importance reweighting, retain its weights in an NPZ array
named weight or weights. A CURN chain or an unweighted reweighting chain is
not interchangeable with this posterior.
The enforced amplitude-prior bounds are `-18 <= log10(A_yr) <= -14`.

To replace the reconstructed LM24 distribution, the supplied samples must
already represent A_model for f_merge=1 at one inverse year and must already
represent the full census with the declared LM24 mass-ratio and redshift
kernels:

~~~bash
lm24-cnm-posterior \
  --lm24-amplitude-samples path/to/full_census_model_amplitudes.npz \
  --confirm-external-lm24-full-census-semantics \
  --lm24-source-id DOI-or-release-ID
~~~

Text, CSV, NPY, and NPZ files are accepted. Every value must be a positive
amplitude, not its logarithm. An NPZ may contain amplitude or amplitude_yr
plus an optional weight or weights array; weights are linear, not logarithmic.
The source file's SHA-256 digest
and array keys are written to the JSON summary.
The confirmation flags are deliberate: the program can validate arrays and
hashes, but it cannot infer their scientific meaning from numbers alone.

## Tests and validation targets

The tests cover:

- recovery of the two tabulated galaxy mass-function reference amplitudes;
- quadrature convergence and batch-size invariance;
- exact log-space mapping between the two amplitudes and f_merge;
- invariance under a common population-density rescaling;
- weighted-chain loading and flat-prior importance reweighting;
- external-chain confirmation and invalid-input rejection;
- linked JSON/NPZ hashes and embedded metadata;
- deterministic seeded regression values;
- rejection of a two-dimensional free-spectrum table.

The reconstructed uncut population should recover the published LM24 checks,
within the accuracy expected from their Appendix-B approximation:

    rho_BH = 1.8 (+0.8, -0.5) x 10^6 solar masses per cubic megaparsec
    A_yr   = 2.0 (+0.7, -0.5) x 10^-15

Both are reported 90% intervals. LM24 state that the approximate population
distribution is slightly broader than their unavailable full posterior at
the high-mass end, so exact agreement is not expected.

## Frozen reference result

The canonical files in `results/fmerge/` use 262,144 draws and seed 20240724.
They give the following median and equal-tailed 90% intervals:

```text
LM24 full-census model amplitude       1.983 [1.465, 2.744] x 10^-15
Manuscript f_merge                     1.110 [0.478, 2.463]
f_merge, prior-uniform sensitivity     1.413 [0.625, 3.099]
```

The LM24 model prediction, `f_merge=1`, lies within the reconstructed interval,
so the data do not require `f_merge>1` under the stated population model.
Doubling the draw count from 131,072 to 262,144 moves every reported 5th,
50th, and 95th percentile above by at most 0.34%. The tests separately verify
quadrature convergence at much tighter tolerance.

These results depend on the stated approximations to the published LM24 and
CNM summaries. The JSON status and limitations fields record those inputs;
the numbers should be presented as a quantile-based reconstruction under the
declared prior.

## Relation to the deterministic benchmark

The manuscript uses f_merge = 1.11 (+1.35, -0.63) at 90% credibility
(Eq. (15) of the paper), obtained directly from
the full-census LM24 Appendix-B approximation. Its median amplitude,
1.983e-15, is close to but not identical to the deterministic full-census
benchmark, A_bench = 2.108e-15. No recentering is applied.

## Quantified reconstruction sensitivities

None of these change the conclusion that f_merge = 1 lies inside every
computed 90% interval; together they mean the quoted percentiles should not
be read beyond two significant figures.

- Schechter-amplitude flooring: 18.09% of archived draws floor at least one
  coefficient, following the LM24 helper procedure at the pinned commit.
  Alternative
  handling of those draws is a reconstruction sensitivity.
- Host-mass domain: the calculation follows the helper's declared 10^8.5 to
  10^12 solar-mass range and does not extrapolate beyond it.
- CNM surrogate family: replacing the split-lognormal surrogate with a
  single lognormal fitted to the same three quantiles moves the f_merge
  percentiles by roughly 1-2%.
- Monte Carlo noise: the third decimal of the quoted 90% endpoints is at the
  sampling noise floor for 262,144 draws.

**Thinning warning:** the archived draws alternate LM24 branches
deterministically (index % 2), so any even-stride subsample of
posterior_draws.npz (for example arr[::2]) contains only the dynamical
branch and is biased by 2-3%. Thin with random indices or an odd stride.

The reported flat_f_effective_sample_size is only the effective size of the
importance reweighting. It is not the MCMC effective sample size of an input
chain, and resampling cannot create additional independent information.

## Sources and version choices

- [Liepold & Ma (2024)](https://doi.org/10.3847/2041-8213/ad66b8),
  especially Eq. (2), Eqs. (4)–(6), and Appendix B.
- [LM24 author helper repository at the pinned commit](https://github.com/EmilyLiepold/LM24_helpers/commit/2092837e91197df16d5231d5845628638afd62ea).
- [Liepold & Ma (2026) erratum](https://doi.org/10.3847/2041-8213/ae5f96).
  It corrects only the printed single-Schechter shortcut and explicitly leaves
  the galaxy-derived black-hole mass function and all published results
  unchanged. This calculation therefore does not use that shortcut.
- [McConnell & Ma (2013)](https://doi.org/10.1088/0004-637X/764/2/184).
- [The MASSIVE Survey](https://doi.org/10.1088/0004-637X/795/2/158).
- [NANOGrav customized-noise analysis](https://arxiv.org/abs/2606.28554).

The implementation is independently derived from these published equations
and parameters; the external helper package is not imported or copied.
