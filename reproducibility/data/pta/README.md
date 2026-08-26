# Optional PTA figure context

The manuscript's multiband figure includes five light-gray PTA free-spectrum
marginals as visual context.  They are not independent estimates of a smooth
spectrum and are not used in the fixed-slope `f_merge` inference.

The derived plotting archive is intentionally absent from this public
package.  Its source table does not yet have an identified public
machine-readable release and reuse license.  A file hash alone is not enough
to establish public provenance or permission.

The public figure therefore renders without these marginals.  A future,
provenance-cleared archive with arrays named `pta_frequency_hz`,
`pta_log10_omega_grid`, and `pta_half_width_hz` can be supplied explicitly:

```bash
python make_figure3_multiband.py --pta-context /path/to/context.npz
```

This optional file has no effect on the six benchmark spectra or their
integrated energies.
