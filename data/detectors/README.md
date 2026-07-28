# Detector context data

## Advanced-LIGO/Virgo design stochastic sensitivity

`Figures_3_and_4_PICurve_Design.dat` is the one-sigma power-law-integrated
design-sensitivity curve distributed with the LIGO/Virgo O2 isotropic
stochastic-background analysis:

- Data product: LIGO-T1900058-v3,
  <https://dcc.ligo.org/LIGO-T1900058/public>
- Direct source:
  <https://dcc.ligo.org/public/0158/T1900058/003/Figures_3_and_4_PICurve_Design.dat>
- Associated paper: B. P. Abbott et al., *Phys. Rev. D* **100**, 061101
  (2019), <https://doi.org/10.1103/PhysRevD.100.061101>
- SHA-256:
  `7d32bbf49db02a653da043266c74efce7761490c978b9f1e1ae92d06711f5ef4`
- Retrieved: 2026-07-28

The columns are frequency in Hz and the one-sigma
\(\Omega_{\rm gw}\) power-law-integrated curve. Figure 2 follows the official
plotting notebook and displays twice the second column, i.e. the two-sigma
design curve. The projection assumes two years of coincident operation by the
Advanced-LIGO/Virgo network at design sensitivity.

## Big Bang Observer power-law-integrated sensitivity

`BBO_PLIS_Schmitz2021.dat` is the published one-year, signal-to-noise-ratio-one
power-law-integrated sensitivity for a BBO cross-correlation search:

- Data product: K. Schmitz, *New Sensitivity Curves for Gravitational-Wave
  Experiments*, Zenodo,
  <https://doi.org/10.5281/zenodo.3689582>
- Associated paper: K. Schmitz, *JHEP* **01**, 097 (2021),
  <https://doi.org/10.1007/JHEP01(2021)097>
- Method and BBO response: E. Thrane and J. D. Romano, *Phys. Rev. D* **88**,
  124032 (2013), <https://doi.org/10.1103/PhysRevD.88.124032>
- Source archive: `power-law-integrated_sensitivities.tar.gz`
- Source-archive SHA-256:
  `a5c0d07648c9e18d522d210e8f92418fb1d32ebdb1f88fa8bfb5f768f1ade338`
- Extracted-file SHA-256:
  `e2a005bc0d57090c7dea355f4fd3e869d1da31cab011488cf8d29991a91b61ae`
- Retrieved: 2026-07-28

The four columns are \(\log_{10}(f/{\rm Hz})\),
\(\log_{10}(h^2\Omega_{\rm PLIS})\), the corresponding strain-amplitude
spectrum, and characteristic strain. Figure 2 converts the second column to
\(\Omega_{\rm gw}\) with the manuscript value \(h=0.674\), then rescales the
one-year curve by \(1/\sqrt{5}\) for five years.

## Cosmic Explorer 40-km strain sensitivity

`cosmic_explorer_strain_T2000017-v9.txt` is the official baseline 40-km Cosmic
Explorer strain amplitude spectral density:

- Data product: K. Kuns, P. Fulda, L. Barsotti, and M. Evans,
  *Cosmic Explorer Strain and Displacement Sensitivity*,
  CE-T2000017-v9,
  <https://dcc.cosmicexplorer.org/CE-T2000017/public>
- Version-pinned archive:
  <https://dcc.cosmicexplorer.org/public/0163/T2000017/009/ce_sensitivity.zip>
- Source-archive SHA-256:
  `ae7941e9072994aca948724f3049785dda132d66590b9f686a108c5e55fcbd6b`
- Extracted-file SHA-256:
  `ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c`
- Retrieved: 2026-07-28

The columns are frequency in Hz and strain ASD in
\({\rm Hz}^{-1/2}\), spanning 5--5000 Hz. The DCC calculates the strain curve
for a source \(15^\circ\) from normal incidence. Figure 2 converts the squared
ASD to a one-year, one-natural-log-frequency-bin, signal-to-noise-ratio-one
orientation curve,
\[
\Omega_{\rm orient}(f)=
\frac{2\pi^2 f^3 S_h(f)}{3H_0^2\sqrt{Tf}},
\]
without adding an unstated detector-pair or overlap-reduction factor.

## Analytic space-mission orientation curves

The LISA and \(\mu\)Ares context curves use the same one-natural-log-bin
orientation convention as the CE conversion above. LISA uses the
Robson--Cornish--Liu four-year noise prescription
<https://doi.org/10.1088/1361-6382/ab1101>. The \(\mu\)Ares curve uses the
detailed Mars-orbit and total-readout specifications in A. Sesana et al.,
*Experimental Astronomy* **51**, 1333 (2021),
<https://doi.org/10.1007/s10686-021-09709-9>: 395 million km arms, total
readout noise \(50\,{\rm pm}/\sqrt{\rm Hz}\), and flat
\(10^{-15}\,{\rm m\,s^{-2}}/\sqrt{\rm Hz}\) acceleration noise down to
\(10^{-7}\) Hz. The paper's \(35\,{\rm pm}/\sqrt{\rm Hz}\) value is the
shot-noise allocation, not the total readout budget. The response uses the
same sky-averaged Robson form as LISA, with the \(\mu\)Ares arm length and
noise levels substituted.
