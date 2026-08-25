# Optional external distribution inputs

The manuscript calculation uses no external sample file. It reconstructs both
inputs from the published summaries described in the parent README.

For other applications, an optional one-dimensional distribution may be
supplied as a compressed NPZ file with this schema:

```text
amplitude     one-dimensional positive A_yr samples
weight        optional nonnegative importance weights of the same length
```

The key `amplitude_yr` may replace `amplitude`, and `weights` may replace
`weight`. Amplitudes are dimensionless values at one inverse Julian year, not
their logarithms.

CNM samples must represent the fixed-gamma=13/3, Hellings--Downs posterior
with its published log-uniform amplitude prior on
`-18 <= log10(A_yr) <= -14`. LM24 samples must represent the one-pass model
amplitude for the full census with the declared LM24 mass-ratio and redshift
kernels. Any supplied weights must be linear.
The program records the source-file SHA-256 digest and array keys in its JSON
output, but it cannot verify these scientific semantics from an array alone.
For that reason, the corresponding command must include
`--confirm-external-cnm-semantics` or
`--confirm-external-lm24-full-census-semantics`, together with a source DOI,
release ID, or URL through the corresponding `--*-source-id` option.

Do not use `log10_omega_gw_cnm.txt`: it contains independent free-spectrum-bin
marginals, not samples of a single power-law amplitude.
