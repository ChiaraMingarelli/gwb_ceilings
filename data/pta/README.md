# PTA sensitivity data

## NANOGrav 15-year official sensitivity curve

`sensitivity_curves_NG15yr_fullPTA.txt` is the official full-PTA stochastic
gravitational-wave-background sensitivity curve for the NANOGrav 15-year
data set, built from the real per-pulsar noise (white noise, the common
uncorrelated process, and significant individual red noise):

- Data product: "Noise Spectra and Stochastic Background Sensitivity Curve
  for the NG15-year Dataset", v1.0.0, Zenodo,
  <https://doi.org/10.5281/zenodo.8092346> (CC-BY-4.0)
- Associated paper: G. Agazie et al., "The NANOGrav 15 yr Data Set:
  Detector Characterization and Noise Budget", ApJL 951, L10 (2023),
  <https://doi.org/10.3847/2041-8213/acda88>
- Source-archive SHA-256 (NANOGrav15yr_Sensitivity-Curves_v1.0.0.tar.gz):
  `78264e02e24970afe2b9df72ee65e66742ea6ccd6039e47ea73eeecff169be24`
- Retrieved: 2026-07-28

Columns: frequency [Hz], characteristic strain h_c, strain PSD S_eff
[strain^2/Hz], and Omega_GW computed with H0 = 67.4 km/s/Mpc (verified
identical to this app's Omega convention at machine precision).
