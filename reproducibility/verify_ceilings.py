#!/usr/bin/env python3
"""Authoritative calculations for the Paper-I channel ledger.

The module is deliberately import-safe: figure scripts import ``build_results``
and ``build_spectra`` so that manuscript numbers and plotted spectra cannot
drift apart.  Running it prints the calculation ledger and can also emit JSON
scalars or NPZ spectra.

The scalar construction is applied channel by channel:

    rho_src = f_merge * rho_res,       f_merge >= 0.

Values above unity are allowed when the same mass contributes to successive
events.  Efficiency, cosmological dilution, and spectral support are then
applied for that channel.  Normalized spectral shapes are used only as an internal
numerical device to keep a channel's integrated radiated energy tied to its
own waveform support; they are not the paper's organizing formalism.  The
SMBHB calculation remains a separate full Liepold--Ma population moment and
never passes through a lumped efficiency/cutoff proxy.

Primary numerical sources
-------------------------
* Liepold & Ma (2024) plus their 2026 erratum:
  arXiv:2407.14595; doi:10.3847/2041-8213/ae5f96.
* Casey-Clyde et al. (2022), arXiv:2107.11390, and Mingarelli et al.
  (2026), arXiv:2603.13643v2, for the high-mass dominance of the PTA-band
  SMBHB strain.  The adopted LM24 moments have no additional primary-mass
  selection.
* Ajith et al. aligned-spin IMR phenomenology:
  arXiv:0909.2867, Eq. (1), Eq. (2), and Table I.
* Zhu et al. (2011) conversion of the Ajith waveform to dE/df:
  arXiv:1104.3565, Eq. (5).
* Husa et al. (2016) aligned-spin radiated-energy fit used by IMRPhenomD,
  with the Jimenez-Forteza et al. (2016) nonspinning polynomial as an
  independent equal-mass cross-check.
* GWTC-5.0 population rates: the named FullPop BNS median, Default-BBH
  median, and union-of-model 90-percent upper endpoints used below.
* Liu & Bromm (2020), arXiv:2003.00065, for the Pop-III star-formation
  history and its approximately 7e4 Msun/Mpc^3 integrated inventory.
* Inayoshi et al. (2021), arXiv:2103.12755, for the Planck-calibrated
  Pop-III mass range used for the ceiling.
* Liu et al. (2024), arXiv:2406.17397, as an independent Pop-III inventory
  and spectral-location cross-check.
* Phinney (2001), arXiv:astro-ph/0108028, for the redshift-diluted integrated
  energy theorem and additivity of distinct source classes.
* Giblin & Thrane (2014), arXiv:1410.4779, for contextual early-Universe
  peak-energy rule-of-thumb levels (not astrophysical bounds).
* Wright, Giblin & Hazboun (2026), arXiv:2409.15572 and
  doi:10.1103/clwq-n2vp, for the pre-recombination Delta-N_eff integral.

The LVK rate-based entries use a declared constant comoving source-frame rate
over a 10-Gyr lookback interval and integrate redshift dilution explicitly.
They are benchmarks and stated-domain maxima, not population forecasts.  A
GWTC-5 Default-BBH ``R(z) proportional to (1+z)^2.5`` variant is emitted
only as a sensitivity calculation.  The adopted Pop-III benchmark sends one
percent of the Liu--Bromm formation history through prompt, nonspinning
50+50-Msun mergers; the prompt mapping is a declared model assumption.

Reproduce the ledger from this directory
----------------------------------------
    python verify_ceilings.py --json results/step3-numbers.json \
        --npz results/step3-spectra.npz
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Callable

import numpy as np

# NumPy 2 renamed ``trapz`` to ``trapezoid``.  The calculation runtime uses
# the new spelling, while the local plotting environment still uses NumPy 1.x.
if not hasattr(np, "trapezoid"):
    setattr(np, "trapezoid", np.trapz)


# ---------------------------------------------------------------------------
# Constants and fixed cosmology
# ---------------------------------------------------------------------------

G_SI = 6.67430e-11
C_SI = 299_792_458.0
MSUN_SI = 1.98847e30
MPC_SI = 3.0856775814913673e22
YEAR_SI = 365.25 * 86_400.0
H0_KM_S_MPC = 67.4
H0_SI = H0_KM_S_MPC * 1_000.0 / MPC_SI
F_YEAR_HZ = 1.0 / YEAR_SI
OMEGA_M = 0.3
OMEGA_LAMBDA = 0.7

RHO_CRIT_SI = 3.0 * H0_SI**2 / (8.0 * math.pi * G_SI)
RHO_CRIT = RHO_CRIT_SI * MPC_SI**3 / MSUN_SI
RHO_STAR = 5.0e-3 * RHO_CRIT
RHO_BH_LM24 = 1.8e6
SMBHB_PTA_DISPLAY_MIN_HZ = 1.0e-9
SMBHB_PTA_DISPLAY_MAX_HZ = 1.0e-7
SMBHB_Q_MIN = 0.1
SMBHB_Q_MAX = 1.0
SMBHB_REPEATED_CEILING_Q = 1.0
SZQ_HIERARCHY_HC2_BOOST = 1.0 / (1.0 - 2.0 ** (-2.0 / 3.0))
SZQ_HIERARCHY_STRAIN_BOOST = math.sqrt(SZQ_HIERARCHY_HC2_BOOST)
ETA_NSC_FIDUCIAL = 1.0e-3
EMRI_CLOSED_POOL_F_MERGE = 0.03
POPIII_CEILING_RHO_MSUN_MPC3 = 2.0e5
WRIGHT_DELTA_NEFF_95 = 0.284
WRIGHT_H = 0.674
GIBLIN_THRANE_PEAK_LEVELS = {
    "pessimistic": 9.93e-15,
    "realistic": 1.49e-12,
    "optimistic": 4.97e-10,
}

# Model-specific GWTC-5 90-percent merger-rate intervals used for the
# benchmark uncertainty propagation.  The separately quoted unions of the
# FullPop and PixelPop intervals remain inputs to the theoretical upper-limit
# cases; they are not benchmark posterior intervals.
GWTC5_BNS_FULLPOP_RATE_90 = (15.4, 59.3, 154.7)
GWTC5_DEFAULT_BBH_RATE_90 = (25.0, 30.4, 36.9)
GWTC5_DEFAULT_BBH_KAPPA_90 = (1.8, 2.5, 3.2)
LVK_O4A_OMEGA_25_LOG_UNIFORM_95 = 2.0e-9
LVK_O4A_OMEGA_25_UNIFORM_95 = 6.3e-9


def trapz_log(y: np.ndarray, f: np.ndarray) -> float:
    """Return integral y dln(f), requiring a positive increasing grid."""

    return float(np.trapezoid(y, x=np.log(f)))


def cumulative_trapz_log(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Return the lower cumulative integral of ``y dln(f)``."""

    values = np.asarray(y, dtype=float)
    frequency = np.asarray(f, dtype=float)
    if values.shape != frequency.shape:
        raise ValueError("y and f must have the same shape.")
    if np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise ValueError("Frequency grid must be positive and increasing.")
    increments = (
        0.5
        * (values[1:] + values[:-1])
        * np.diff(np.log(frequency))
    )
    return np.concatenate((np.asarray([0.0]), np.cumsum(increments)))


def expansion_function(redshift: float | np.ndarray) -> np.ndarray:
    """Dimensionless flat-Lambda-CDM expansion function E(z)."""

    z = np.asarray(redshift, dtype=float)
    return np.sqrt(OMEGA_M * (1.0 + z) ** 3 + OMEGA_LAMBDA)


def source_time_per_redshift_yr(redshift: float | np.ndarray) -> np.ndarray:
    """Positive |dt_source/dz| in years."""

    z = np.asarray(redshift, dtype=float)
    return 1.0 / ((1.0 + z) * H0_SI * expansion_function(z)) / YEAR_SI


def lookback_time_gyr(redshift: float, intervals: int = 20_000) -> float:
    """Lookback time to ``redshift`` in Gyr."""

    if redshift < 0.0:
        raise ValueError("redshift must be non-negative.")
    if redshift == 0.0:
        return 0.0
    z = np.linspace(0.0, redshift, intervals + 1)
    return float(np.trapezoid(source_time_per_redshift_yr(z), x=z) / 1.0e9)


@lru_cache(maxsize=32)
def redshift_at_lookback_gyr(duration_gyr: float) -> float:
    """Invert the lookback time in the adopted flat-Lambda-CDM cosmology."""

    if not 0.0 <= duration_gyr < 13.9:
        raise ValueError("duration_gyr must lie in [0, 13.9).")
    if duration_gyr == 0.0:
        return 0.0
    lower, upper = 0.0, 1.0
    while lookback_time_gyr(upper, intervals=4_000) < duration_gyr:
        upper *= 2.0
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        if lookback_time_gyr(midpoint, intervals=4_000) < duration_gyr:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def trapezoid_quadrature_weights(grid: np.ndarray) -> np.ndarray:
    """Weights whose dot product implements trapezoidal integration."""

    x = np.asarray(grid, dtype=float)
    if x.ndim != 1 or x.size < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("grid must be one-dimensional and strictly increasing.")
    weights = np.empty_like(x)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def hc_from_omega(f_hz: float | np.ndarray, omega: float | np.ndarray) -> np.ndarray:
    """Characteristic strain corresponding to Omega_gw."""

    f = np.asarray(f_hz, dtype=float)
    om = np.asarray(omega, dtype=float)
    return np.sqrt(3.0 * H0_SI**2 * om / (2.0 * math.pi**2 * f**2))


def omega_from_hc(f_hz: float | np.ndarray, hc: float | np.ndarray) -> np.ndarray:
    """Omega_gw corresponding to characteristic strain."""

    f = np.asarray(f_hz, dtype=float)
    h = np.asarray(hc, dtype=float)
    return 2.0 * math.pi**2 * f**2 * h**2 / (3.0 * H0_SI**2)


def kerr_isco_radius(spin: float) -> float:
    """Prograde equatorial Kerr ISCO radius in units of ``GM/c^2``.

    Negative ``spin`` selects the corresponding retrograde branch.  The
    expression is the standard Bardeen--Press--Teukolsky result.
    """

    a = float(spin)
    if not -1.0 <= a <= 1.0:
        raise ValueError("spin must lie in [-1, 1].")
    z1 = 1.0 + (1.0 - a * a) ** (1.0 / 3.0) * (
        (1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0)
    )
    z2 = math.sqrt(3.0 * a * a + z1 * z1)
    branch = 1.0 if a >= 0.0 else -1.0
    return 3.0 + z2 - branch * math.sqrt(
        (3.0 - z1) * (3.0 + z1 + 2.0 * z2)
    )


def kerr_isco_frequency(
    total_mass_msun: float,
    spin: float,
    redshift: float = 0.0,
) -> float:
    """Observed quadrupolar GW frequency at the Kerr ISCO."""

    if total_mass_msun <= 0.0:
        raise ValueError("total_mass_msun must be positive.")
    if redshift < 0.0:
        raise ValueError("redshift must be non-negative.")
    radius = kerr_isco_radius(spin)
    source_frequency = C_SI**3 / (
        math.pi
        * G_SI
        * total_mass_msun
        * MSUN_SI
        * (radius ** (3.0 / 2.0) + float(spin))
    )
    return source_frequency / (1.0 + redshift)


def mass_for_kerr_isco_frequency(
    observed_frequency_hz: float,
    spin: float,
    redshift: float = 0.0,
) -> float:
    """Total source-frame mass whose Kerr ISCO is at an observed frequency."""

    if observed_frequency_hz <= 0.0:
        raise ValueError("observed_frequency_hz must be positive.")
    if redshift < 0.0:
        raise ValueError("redshift must be non-negative.")
    radius = kerr_isco_radius(spin)
    return C_SI**3 / (
        math.pi
        * G_SI
        * MSUN_SI
        * observed_frequency_hz
        * (1.0 + redshift)
        * (radius ** (3.0 / 2.0) + float(spin))
    )


def peters_energy_spectrum(
    source_frequency_hz: float | np.ndarray,
    chirp_mass_msun: float,
) -> np.ndarray:
    """Leading-order circular-inspiral ``dE/df`` in SI units.

    This is the Peters--Mathews spectrum

        dE/df = (pi G)^(2/3) M_chirp^(5/3) f^(-1/3) / 3.
    """

    frequency = np.asarray(source_frequency_hz, dtype=float)
    if np.any(frequency <= 0.0):
        raise ValueError("source frequencies must be positive.")
    if chirp_mass_msun <= 0.0:
        raise ValueError("chirp_mass_msun must be positive.")
    coefficient = (
        (math.pi * G_SI) ** (2.0 / 3.0)
        * (chirp_mass_msun * MSUN_SI) ** (5.0 / 3.0)
        / 3.0
    )
    return coefficient * frequency ** (-1.0 / 3.0)


def peters_inspiral_efficiency(
    total_mass_msun: float,
    eta: float,
    source_cutoff_hz: float,
) -> float:
    """Rest-mass efficiency obtained by integrating the Peters spectrum."""

    if total_mass_msun <= 0.0:
        raise ValueError("total_mass_msun must be positive.")
    if not 0.0 < eta <= 0.25:
        raise ValueError("eta must lie in (0, 0.25].")
    if source_cutoff_hz <= 0.0:
        raise ValueError("source_cutoff_hz must be positive.")
    chirp_mass_msun = total_mass_msun * eta ** (3.0 / 5.0)
    dedf_at_cutoff = float(
        peters_energy_spectrum(source_cutoff_hz, chirp_mass_msun)
    )
    emitted_energy = 1.5 * dedf_at_cutoff * source_cutoff_hz
    return emitted_energy / (total_mass_msun * MSUN_SI * C_SI**2)


def inspiral_amplitude(
    epsilon: float,
    rho_src: float,
    f_max_hz: float,
    f_ref_hz: float,
    redshift: float = 0.0,
) -> float:
    """Amplitude of the normalized f^(2/3) energy shape for one redshift.

    This helper is used for channels whose model is explicitly truncated in
    the inspiral.  It is not used for the SMBHB population-moment result or
    the full-IMR BBH entries.  ``f_max_hz`` is the source-frame cutoff.
    """

    present_budget = epsilon * rho_src / ((1.0 + redshift) * RHO_CRIT)
    observed_f_max = f_max_hz / (1.0 + redshift)
    return (
        H0_SI
        / math.pi
        * f_ref_hz ** (-2.0 / 3.0)
        * observed_f_max ** (-1.0 / 3.0)
        * math.sqrt(present_budget)
    )


def inspiral_shape(f_hz: np.ndarray, f_max_hz: float) -> np.ndarray:
    """Normalized p(f) for dE/df proportional to f^(-1/3)."""

    f = np.asarray(f_hz, dtype=float)
    p = np.zeros_like(f)
    mask = (f > 0.0) & (f <= f_max_hz)
    p[mask] = (2.0 / 3.0) * (f[mask] / f_max_hz) ** (2.0 / 3.0)
    return p


def lisa_transfer_parenthesization_ratio(
    frequency_hz: float | np.ndarray,
) -> np.ndarray:
    """Correct/legacy LISA noise-power ratio for the V5 audit.

    The legacy expression multiplied only the acceleration term by the
    Robson--Cornish--Liu transfer factor.  The correct expression multiplies
    the full optical-metrology-plus-acceleration bracket.
    """

    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(frequency <= 0.0):
        raise ValueError("frequency_hz must be positive.")
    f_star = 19.09e-3
    p_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / frequency) ** 4)
    p_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / frequency) ** 2) * (
        1.0 + (frequency / 8.0e-3) ** 4
    )
    acceleration = 4.0 * p_acc / (2.0 * math.pi * frequency) ** 4
    transfer = 1.0 + 0.6 * (frequency / f_star) ** 2
    legacy = p_oms + acceleration * transfer
    corrected = (p_oms + acceleration) * transfer
    return corrected / legacy


def detector_context_audit() -> dict[str, object]:
    """Fixed detector anchors; Figure 2 uses the LISA/BBO/CE subset."""

    lisa_ratios = lisa_transfer_parenthesization_ratio(
        np.asarray([20.0e-3, 40.0e-3])
    )
    return {
        "lisa_correct_to_legacy_noise_power_20_mhz": float(lisa_ratios[0]),
        "lisa_correct_to_legacy_noise_power_40_mhz": float(lisa_ratios[1]),
        "one_sided_strain_psd_to_omega_prefactor": (
            2.0 * math.pi**2 / (3.0 * H0_SI**2)
        ),
        "muares": {
            "arm_length_m": 3.95e11,
            "position_noise_asd_m_per_sqrt_hz": 50.0e-12,
            "acceleration_noise_asd_m_s2_per_sqrt_hz": 1.0e-15,
            "duration_yr": 10.0,
        },
        "lisa_duration_yr": 4.0,
        "bbo_duration_yr": 5.0,
        "bbo_log_bin_scaling": "1/sqrt(T*f)",
        "ce_duration_yr": 1.0,
        "detector_scope": (
            "illustrative energy-density sensitivity context, not "
            "population-ceiling inputs"
        ),
    }


# ---------------------------------------------------------------------------
# LM24 SMBHB population moment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LM24Parameters:
    """Erratum-corrected single-Schechter fit, for h=0.7."""

    alpha: float = -1.27
    beta: float = 0.45
    log10_phi_mpc3: float = -2.46
    log10_mscale_msun: float = 8.40

    @property
    def phi_mpc3(self) -> float:
        return 10.0**self.log10_phi_mpc3

    @property
    def mscale_msun(self) -> float:
        return 10.0**self.log10_mscale_msun


LM24 = LM24Parameters()


def lm24_mass_moment(power: float, params: LM24Parameters = LM24) -> float:
    """Integral M^power dn for the corrected analytic BHMF."""

    gamma_arg = (params.alpha + power + 1.0) / params.beta
    if gamma_arg <= 0.0:
        raise ValueError("Requested LM24 mass moment diverges.")
    return (
        params.phi_mpc3
        * params.mscale_msun**power
        / params.beta
        * math.gamma(gamma_arg)
    )


def simpson_scalar(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    intervals: int,
) -> float:
    """Dependency-free composite Simpson integration."""

    if intervals % 2:
        intervals += 1
    step = (upper - lower) / intervals
    total = function(lower) + function(upper)
    for index in range(1, intervals):
        total += (4.0 if index % 2 else 2.0) * function(lower + index * step)
    return total * step / 3.0


def lm24_eta_moment() -> float:
    """<eta> for p(q) proportional to q^2 on 0.1 <= q <= 1."""

    numerator = simpson_scalar(
        lambda q: q**2 * q / (1.0 + q) ** 2,
        SMBHB_Q_MIN,
        SMBHB_Q_MAX,
        20_000,
    )
    denominator = simpson_scalar(
        lambda q: q**2,
        SMBHB_Q_MIN,
        SMBHB_Q_MAX,
        20_000,
    )
    return numerator / denominator


def lm24_redshift_moment() -> float:
    """<(1+z)^(-1/3)> for p(z) proportional to z exp[-(z/0.5)^2]."""

    weight = lambda z: z * math.exp(-((z / 0.5) ** 2))
    numerator = simpson_scalar(
        lambda z: weight(z) * (1.0 + z) ** (-1.0 / 3.0),
        0.0,
        6.0,
        40_000,
    )
    denominator = simpson_scalar(weight, 0.0, 6.0, 40_000)
    return numerator / denominator


def lm24_energy_dilution_moment() -> float:
    """<(1+z)^(-1)> for the LM24 benchmark redshift distribution."""

    weight = lambda z: z * math.exp(-((z / 0.5) ** 2))
    numerator = simpson_scalar(
        lambda z: weight(z) / (1.0 + z),
        0.0,
        6.0,
        40_000,
    )
    denominator = simpson_scalar(weight, 0.0, 6.0, 40_000)
    return numerator / denominator


def smbhb_strain_from_i53(
    i53_msun53_mpc3: float,
    eta_moment: float,
    redshift_moment: float,
    frequency_hz: float = F_YEAR_HZ,
) -> float:
    """Phinney population-moment strain for circular GW-driven binaries."""

    i53_si = i53_msun53_mpc3 * MSUN_SI ** (5.0 / 3.0) / MPC_SI**3
    hc2 = (
        4.0
        / (3.0 * math.pi ** (1.0 / 3.0))
        * G_SI ** (5.0 / 3.0)
        / C_SI**2
        * frequency_hz ** (-4.0 / 3.0)
        * eta_moment
        * redshift_moment
        * i53_si
    )
    return math.sqrt(hc2)


@dataclass(frozen=True)
class SZQ24Parameters:
    """Fiducial VDF and M-sigma inputs of Sato-Polito et al. (2024)."""

    phi_mpc3: float = 2.61e-2
    sigma_star_km_s: float = 159.6
    alpha: float = 0.41
    beta: float = 2.59
    log10_mass_normalization: float = 8.32
    mass_slope: float = 5.64
    scatter_dex: float = 0.38
    q_min: float = 0.1
    q_max: float = 1.0
    redshift_power: float = 0.5
    redshift_scale: float = 0.3

    @property
    def mstar_msun(self) -> float:
        return (
            10.0**self.log10_mass_normalization
            * (self.sigma_star_km_s / 200.0) ** self.mass_slope
        )


SZQ24 = SZQ24Parameters()


def szq24_vdf_mass_moment(
    power: float,
    params: SZQ24Parameters = SZQ24,
) -> float:
    """Return integral phi(M) M^power dM for the fiducial SZQ24 census."""

    gamma_arg = (params.alpha + power * params.mass_slope) / params.beta
    if gamma_arg <= 0.0:
        raise ValueError("Requested SZQ24 mass moment diverges.")
    scatter_factor = math.exp(
        0.5 * (power * params.scatter_dex * math.log(10.0)) ** 2
    )
    return (
        params.phi_mpc3
        * params.mstar_msun**power
        * math.gamma(gamma_arg)
        / math.gamma(params.alpha / params.beta)
        * scatter_factor
    )


def szq24_eta_moment(params: SZQ24Parameters = SZQ24) -> float:
    """Return <eta> for normalized p(q) proportional to q^-1."""

    numerator = (
        1.0 / (1.0 + params.q_min)
        - 1.0 / (1.0 + params.q_max)
    )
    return numerator / math.log(params.q_max / params.q_min)


def szq24_redshift_moment(params: SZQ24Parameters = SZQ24) -> float:
    """Return <(1+z)^(-1/3)> for the fiducial SZQ24 redshift law."""

    normalization = (
        params.redshift_scale ** (params.redshift_power + 1.0)
        * math.gamma(params.redshift_power + 1.0)
    )
    # Substitute z=x^2 to remove the square-root endpoint in the fiducial
    # z^(1/2) exp(-z/z_*) distribution.
    numerator = simpson_scalar(
        lambda x: (
            2.0
            * x
            * (x * x) ** params.redshift_power
            * math.exp(-(x * x) / params.redshift_scale)
            * (1.0 + x * x) ** (-1.0 / 3.0)
        ),
        0.0,
        math.sqrt(10.0),
        4_000,
    )
    return numerator / normalization


def szq24_population_moment_audit() -> dict[str, object]:
    """Independent evaluation of the SZQ24 one-pass population moment."""

    rho_bh = szq24_vdf_mass_moment(1.0)
    i53 = szq24_vdf_mass_moment(5.0 / 3.0)
    eta = szq24_eta_moment()
    redshift = szq24_redshift_moment()
    fiducial = smbhb_strain_from_i53(i53, eta, redshift)
    equal_mass_same_redshift = smbhb_strain_from_i53(i53, 0.25, redshift)
    equal_mass_z0 = smbhb_strain_from_i53(i53, 0.25, 1.0)
    fiducial_plus_hierarchy = fiducial * SZQ_HIERARCHY_STRAIN_BOOST
    equal_mass_same_redshift_plus_hierarchy = (
        equal_mass_same_redshift * SZQ_HIERARCHY_STRAIN_BOOST
    )
    equal_mass_z0_plus_hierarchy = (
        equal_mass_z0 * SZQ_HIERARCHY_STRAIN_BOOST
    )

    # Literal evaluation of the rounded normalization printed in SZQ24
    # Eq. (21).  It is retained as a comparison, not used by this work.
    literal_eq21_hc2 = (
        1.2e-30
        * (SZQ24.mstar_msun / 5.8e7) ** (2.0 / 3.0)
        * (rho_bh / 4.5e5)
        * 2.0
        / math.exp(
            (8.0 / 9.0)
            * SZQ24.scatter_dex**2
            * math.log(10.0) ** 2
        )
    )
    literal_eq21 = math.sqrt(literal_eq21_hc2)

    return {
        "mass_scale_msun": SZQ24.mstar_msun,
        "rho_bh_msun_mpc3": rho_bh,
        "i53_msun53_mpc3": i53,
        "eta_log_uniform": eta,
        "redshift_moment": redshift,
        "fiducial_log_uniform_one_pass_A_1yr": fiducial,
        "equal_mass_over_log_uniform_hc2_ratio": 0.25 / eta,
        "equal_mass_over_log_uniform_strain_ratio": math.sqrt(0.25 / eta),
        "equal_mass_fiducial_redshift_one_pass_A_1yr": (
            equal_mass_same_redshift
        ),
        "equal_mass_z0_one_pass_A_1yr": equal_mass_z0,
        "fiducial_plus_hierarchy_A_1yr": fiducial_plus_hierarchy,
        "equal_mass_fiducial_redshift_infinite_A_1yr": (
            equal_mass_same_redshift_plus_hierarchy
        ),
        "equal_mass_z0_infinite_A_1yr": equal_mass_z0_plus_hierarchy,
        "hierarchy_hc2_boost": SZQ_HIERARCHY_HC2_BOOST,
        "hierarchy_strain_boost": SZQ_HIERARCHY_STRAIN_BOOST,
        "literal_eq21_hc2": literal_eq21_hc2,
        "literal_eq21_A_1yr": literal_eq21,
        "literal_eq21_over_direct_strain_ratio": literal_eq21 / fiducial,
        "literal_eq21_over_direct_hc2_ratio": literal_eq21_hc2 / fiducial**2,
        "normalization_status": (
            "unresolved: literal SZQ24 Eq. (21) exceeds the direct "
            "population moment"
        ),
        "scope": (
            "full SZQ24 VDF census with no primary-mass participation cut"
        ),
    }


def smbhb_population_results() -> dict[str, float | str | None]:
    """LM24 SMBHB results from the full analytic population moment.

    The adopted result imposes no additional primary-mass selection.
    """

    i1_fit = lm24_mass_moment(1.0)
    i53_fit = lm24_mass_moment(5.0 / 3.0)
    density_scale = RHO_BH_LM24 / i1_fit
    i53 = density_scale * i53_fit
    unrestricted_eta = lm24_eta_moment()
    zfac = lm24_redshift_moment()
    benchmark_i53_factor = smbhb_population_moment_factor(
        "benchmark",
        5.0 / 3.0,
        True,
    )
    benchmark_i53_unweighted = smbhb_population_moment_factor(
        "benchmark",
        5.0 / 3.0,
        False,
    )
    ceiling_i53_factor = smbhb_population_moment_factor(
        "conditional_ceiling",
        5.0 / 3.0,
        True,
    )
    benchmark_mass_fraction = smbhb_population_moment_factor(
        "benchmark",
        1.0,
        False,
    )
    benchmark_mass_eta_factor = smbhb_population_moment_factor(
        "benchmark",
        1.0,
        True,
    )
    ceiling_mass_fraction = smbhb_population_moment_factor(
        "conditional_ceiling",
        1.0,
        False,
    )
    benchmark = smbhb_strain_from_i53(
        i53,
        benchmark_i53_factor,
        zfac,
    )
    equal_mass_latest_generation = smbhb_strain_from_i53(
        i53,
        ceiling_i53_factor,
        1.0,
    )
    repeated_merger_ceiling = (
        equal_mass_latest_generation * SZQ_HIERARCHY_STRAIN_BOOST
    )
    cnm_central = 2.1e-15
    return {
        "rho_fit_msun_mpc3": i1_fit,
        "i53_fit_msun53_mpc3": i53_fit,
        "primary_minimum_mass_msun": None,
        "primary_mass_selection": "none; full analytic LM24 moment",
        "minimum_total_mass_benchmark_msun": None,
        "conditional_ceiling_mass_ratio": (
            SMBHB_REPEATED_CEILING_Q
        ),
        "minimum_total_mass_conditional_ceiling_msun": None,
        "eta_moment_unrestricted": unrestricted_eta,
        "eta_moment_participating_i53": (
            benchmark_i53_factor / benchmark_i53_unweighted
        ),
        "eta_moment_participating_mass": (
            benchmark_mass_eta_factor / benchmark_mass_fraction
        ),
        "retained_i53_fraction_benchmark": benchmark_i53_unweighted,
        "retained_i53_eta_factor_benchmark": benchmark_i53_factor,
        "retained_i53_eta_factor_conditional_ceiling": ceiling_i53_factor,
        "retained_mass_density_fraction_benchmark": benchmark_mass_fraction,
        "retained_mass_density_fraction_conditional_ceiling": (
            ceiling_mass_fraction
        ),
        "participant_mass_density_benchmark_msun_mpc3": (
            RHO_BH_LM24 * benchmark_mass_fraction
        ),
        "participant_mass_density_conditional_ceiling_msun_mpc3": (
            RHO_BH_LM24 * ceiling_mass_fraction
        ),
        "redshift_moment": zfac,
        "benchmark_A_1yr": benchmark,
        # Base normalization for the equal-mass, z=0 spectrum before the
        # earlier generations in the SZQ24 repeated-merger history are added.
        "conditional_ceiling_A_1yr": equal_mass_latest_generation,
        "equal_mass_latest_generation_A_1yr": equal_mass_latest_generation,
        "szq_hierarchy_hc2_boost": SZQ_HIERARCHY_HC2_BOOST,
        "szq_hierarchy_strain_boost": SZQ_HIERARCHY_STRAIN_BOOST,
        "repeated_merger_ceiling_A_1yr": repeated_merger_ceiling,
        "cnm_central_A_1yr": cnm_central,
        "cnm_low_A_1yr": 1.6e-15,
        "cnm_high_A_1yr": 2.7e-15,
        "fmerge_cnm_central_fixed_shape": (cnm_central / benchmark) ** 2,
        "fmerge_cnm_low_fixed_shape": (1.6e-15 / benchmark) ** 2,
        "fmerge_cnm_high_fixed_shape": (2.7e-15 / benchmark) ** 2,
        "shape_conditional_product_msun_mpc3": (
            (cnm_central / benchmark) ** 2 * RHO_BH_LM24
        ),
    }


@lru_cache(maxsize=4)
def smbhb_isco_support_audit(
    spin: float = 0.0,
) -> dict[str, float | str | None]:
    """Effect of a mass/redshift-dependent ISCO mask at ``1/yr``.

    The quoted low-frequency population moment uses the full analytic LM24
    fit.  This diagnostic measures the small additional effect of terminating
    every source at its own Kerr ISCO.
    """

    mass_limit_z0 = mass_for_kerr_isco_frequency(F_YEAR_HZ, spin, 0.0)
    quoted_amplitude = smbhb_population_results()["benchmark_A_1yr"]
    supported_omega = float(
        smbhb_population_omega(
            np.asarray([F_YEAR_HZ]),
            kind="benchmark",
            spin=spin,
        )[0]
    )
    supported_amplitude = float(hc_from_omega(F_YEAR_HZ, supported_omega))
    retained_hc2_fraction = (supported_amplitude / quoted_amplitude) ** 2
    return {
        "spin": float(spin),
        "isco_radius_gm_c2": kerr_isco_radius(spin),
        "mass_limit_at_1yr_z0_msun": mass_limit_z0,
        "primary_minimum_mass_msun": None,
        "primary_mass_selection": "none; full analytic LM24 moment",
        "supported_hc_at_1yr": supported_amplitude,
        "retained_hc_squared_fraction": retained_hc2_fraction,
        "fractional_hc_change": math.sqrt(retained_hc2_fraction) - 1.0,
    }


@lru_cache(maxsize=4)
def _lm24_cumulative_mass_moment(
    power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerical lower cumulative moment of the analytic LM24 fit.

    The broad quadrature bounds approximate the analytic zero-to-infinity
    moment; they do not assert empirical BHMF support over that full range.
    """

    log_mass = np.linspace(math.log(1.0e-6), math.log(1.0e16), 80_001)
    mass = np.exp(log_mass)
    x = mass / LM24.mscale_msun
    dndlnm = LM24.phi_mpc3 * x ** (LM24.alpha + 1.0) * np.exp(
        -(x**LM24.beta)
    )
    integrand = dndlnm * mass**power
    cumulative = np.concatenate(
        (
            np.asarray([0.0]),
            np.cumsum(
                0.5
                * (integrand[1:] + integrand[:-1])
                * np.diff(log_mass)
            ),
        )
    )
    analytic_total = lm24_mass_moment(power)
    cumulative /= analytic_total
    if not math.isclose(float(cumulative[-1]), 1.0, rel_tol=3.0e-7):
        raise RuntimeError("LM24 cumulative moment failed to close.")
    cumulative[-1] = 1.0
    return log_mass, cumulative


def lm24_moment_fraction_below(
    mass_limit_msun: float | np.ndarray,
    power: float = 5.0 / 3.0,
) -> np.ndarray:
    """Fraction of an LM24 mass moment below a source-frame mass limit."""

    limits = np.asarray(mass_limit_msun, dtype=float)
    if np.any(limits <= 0.0):
        raise ValueError("mass limits must be positive.")
    log_mass, cumulative = _lm24_cumulative_mass_moment(float(power))
    return np.interp(
        np.log(limits),
        log_mass,
        cumulative,
        left=0.0,
        right=1.0,
    )


def smbhb_population_moment_factor(
    kind: str,
    power: float,
    eta_weighted: bool = True,
) -> float:
    """LM24 moment factor for the adopted full analytic census."""

    if kind == "benchmark":
        return lm24_eta_moment() if eta_weighted else 1.0
    if kind == "conditional_ceiling":
        q = SMBHB_REPEATED_CEILING_Q
        return q / (1.0 + q) ** 2 if eta_weighted else 1.0
    raise ValueError("kind must be 'benchmark' or 'conditional_ceiling'.")


def smbhb_supported_moment_factor(
    mass_limit_msun: float | np.ndarray,
    kind: str,
    *,
    power: float = 5.0 / 3.0,
) -> np.ndarray:
    """Eta-weighted moment below a frequency-dependent total-mass limit."""

    limits = np.asarray(mass_limit_msun, dtype=float)
    upper_fraction = lm24_moment_fraction_below(limits, power=power)
    if kind == "benchmark":
        return lm24_eta_moment() * upper_fraction
    if kind == "conditional_ceiling":
        q = SMBHB_REPEATED_CEILING_Q
        return q / (1.0 + q) ** 2 * upper_fraction
    raise ValueError("kind must be 'benchmark' or 'conditional_ceiling'.")


def peters_isco_efficiency_per_eta(spin: float = 0.0) -> float:
    """Newtonian inspiral-to-Kerr-ISCO efficiency divided by eta."""

    radius = kerr_isco_radius(spin)
    frequency_factor = 1.0 / (radius ** (3.0 / 2.0) + float(spin))
    return 0.5 * frequency_factor ** (2.0 / 3.0)


def smbhb_inspiral_present_budget(
    kind: str = "benchmark",
    spin: float = 0.0,
) -> float:
    """Integrated present-day LM24 inspiral energy in units of rho_crit.

    The benchmark uses the LM24 q and z distributions.  The conditional
    ceiling uses z=0 and q=1, as required by the adopted SZQ24 equal-mass
    repeated-merger history.  Both integrate the modeled Newtonian inspiral
    through the mass-dependent Kerr ISCO.  The adopted calculation uses the
    full analytic LM24 mass moment.
    """

    if kind == "benchmark":
        dilution = lm24_energy_dilution_moment()
    elif kind == "conditional_ceiling":
        dilution = 1.0
    else:
        raise ValueError("kind must be 'benchmark' or 'conditional_ceiling'.")
    eta_mass_factor = smbhb_population_moment_factor(
        kind,
        1.0,
        True,
    )
    return (
        RHO_BH_LM24
        / RHO_CRIT
        * eta_mass_factor
        * peters_isco_efficiency_per_eta(spin)
        * dilution
    )


def smbhb_population_omega(
    f_hz: np.ndarray,
    kind: str = "benchmark",
    spin: float = 0.0,
) -> np.ndarray:
    """Formal LM24 inspiral spectrum with source-by-source ISCO support.

    This helper supports PTA checks.  The emitted archive masks the adopted
    full-census result to its declared PTA display segment; scalar energies
    come from analytic mass moments, not from integrating an emitted spectrum.
    """

    frequency = np.asarray(f_hz, dtype=float)
    if np.any(frequency <= 0.0):
        raise ValueError("frequencies must be positive.")

    smbhb = smbhb_population_results()
    mass_at_one_hz = mass_for_kerr_isco_frequency(1.0, spin, 0.0)
    retained_i53_eta_factor = smbhb_population_moment_factor(
        kind,
        5.0 / 3.0,
        True,
    )
    if kind == "benchmark":
        amplitude = smbhb["benchmark_A_1yr"]
        redshift = np.linspace(0.0, 6.0, 2_001)
        weight = (
            redshift
            * np.exp(-((redshift / 0.5) ** 2))
            * (1.0 + redshift) ** (-1.0 / 3.0)
        )
        trapezoid_weight = weight.copy()
        trapezoid_weight[[0, -1]] *= 0.5
        trapezoid_weight *= (
            (redshift[1] - redshift[0])
            / np.trapezoid(weight, x=redshift)
        )
        support = np.zeros_like(frequency)
        for z_value, z_weight in zip(redshift, trapezoid_weight):
            mass_limit = mass_at_one_hz / (
                frequency * (1.0 + z_value)
            )
            retained_moment = smbhb_supported_moment_factor(
                mass_limit,
                kind,
            )
            support += z_weight * retained_moment
    elif kind == "conditional_ceiling":
        amplitude = smbhb["conditional_ceiling_A_1yr"]
        support = smbhb_supported_moment_factor(
            mass_at_one_hz / frequency,
            kind,
        )
    else:
        raise ValueError("kind must be 'benchmark' or 'conditional_ceiling'.")

    support /= retained_i53_eta_factor
    untruncated_hc = amplitude * (frequency / F_YEAR_HZ) ** (-2.0 / 3.0)
    return omega_from_hc(frequency, untruncated_hc) * support


def smbhb_fmerge_for_amplitude(amplitude: np.ndarray, rho_bh: np.ndarray) -> np.ndarray:
    """Uniform-processing fraction for the fixed central LM24 shape."""

    anchor = smbhb_population_results()["benchmark_A_1yr"]
    return (np.asarray(amplitude) / anchor) ** 2 * RHO_BH_LM24 / np.asarray(rho_bh)


# ---------------------------------------------------------------------------
# Ajith IMRPhenomB normalized energy shapes
# ---------------------------------------------------------------------------


def phenomb_frequencies(
    total_mass_msun: float,
    eta: float,
    chi: float,
    redshift: float = 0.0,
) -> tuple[float, float, float, float]:
    """Observed (f_merg, f_ring, sigma, f_cut) from Ajith et al. (2009)."""

    if not 0.0 < eta <= 0.25:
        raise ValueError("eta must lie in (0, 0.25].")
    if not -1.0 < chi < 1.0:
        raise ValueError("chi must lie in (-1, 1).")
    e2 = eta * eta
    e3 = e2 * eta
    c2 = chi * chi
    echi = eta * chi
    ec2 = eta * c2
    e2chi = e2 * chi
    one_minus = 1.0 - chi

    f_merg = (
        1.0
        - 4.4547 * one_minus**0.217
        + 3.5210 * one_minus**0.260
        + 0.64365 * eta
        + 0.82696 * echi
        - 0.27063 * ec2
        - 0.058218 * e2
        - 3.9346 * e2chi
        - 7.0916 * e3
    )
    f_ring = (
        (1.0 - 0.63 * one_minus**0.3) / 2.0
        + 0.14690 * eta
        - 0.12281 * echi
        - 0.026091 * ec2
        - 0.024900 * e2
        + 0.17013 * e2chi
        + 2.3252 * e3
    )
    sigma = (
        (1.0 - 0.63 * one_minus**0.3) * one_minus**0.45 / 4.0
        - 0.40979 * eta
        - 0.035226 * echi
        + 0.10082 * ec2
        + 1.8286 * e2
        - 0.020169 * e2chi
        - 2.8698 * e3
    )
    f_cut = (
        0.32361
        + 0.048935 * chi
        + 0.013463 * c2
        - 0.13313 * eta
        - 0.081719 * echi
        + 0.14512 * ec2
        - 0.27140 * e2
        + 0.12788 * e2chi
        + 4.9220 * e3
    )
    pi_m = math.pi * total_mass_msun * G_SI * MSUN_SI / C_SI**3
    scale = pi_m * (1.0 + redshift)
    return f_merg / scale, f_ring / scale, sigma / scale, f_cut / scale


def _phenomb_raw_p(
    f_hz: np.ndarray,
    total_mass_msun: float,
    eta: float,
    chi: float,
    redshift: float,
) -> np.ndarray:
    """Unnormalized f dE/df, proportional to f^3 |h_tilde(f)|^2."""

    f = np.asarray(f_hz, dtype=float)
    f_merg, f_ring, sigma, f_cut = phenomb_frequencies(
        total_mass_msun, eta, chi, redshift
    )
    source_mass_seconds = total_mass_msun * G_SI * MSUN_SI / C_SI**3
    # v is source-frame invariant: pi M f_source.
    v = np.cbrt(math.pi * source_mass_seconds * (1.0 + redshift) * f)
    alpha2 = -323.0 / 224.0 + 451.0 * eta / 168.0
    alpha3 = (27.0 / 8.0 - 11.0 * eta / 6.0) * chi
    eps1 = 1.4547 * chi - 1.8897
    eps2 = -1.8153 * chi + 1.6557

    v_merg = np.cbrt(
        math.pi * source_mass_seconds * (1.0 + redshift) * f_merg
    )
    v_ring = np.cbrt(
        math.pi * source_mass_seconds * (1.0 + redshift) * f_ring
    )
    w1 = (1.0 + alpha2 * v_merg**2 + alpha3 * v_merg**3) / (
        1.0 + eps1 * v_merg + eps2 * v_merg**2
    )
    w2 = (
        w1
        * math.pi
        * sigma
        / 2.0
        * (f_ring / f_merg) ** (-2.0 / 3.0)
        * (1.0 + eps1 * v_ring + eps2 * v_ring**2)
    )

    amp = np.zeros_like(f)
    inspiral = (f > 0.0) & (f <= f_merg)
    merger = (f > f_merg) & (f <= f_ring)
    ringdown = (f > f_ring) & (f <= f_cut)
    amp[inspiral] = (
        (f[inspiral] / f_merg) ** (-7.0 / 6.0)
        * (
            1.0
            + alpha2 * v[inspiral] ** 2
            + alpha3 * v[inspiral] ** 3
        )
    )
    amp[merger] = (
        w1
        * (f[merger] / f_merg) ** (-2.0 / 3.0)
        * (1.0 + eps1 * v[merger] + eps2 * v[merger] ** 2)
    )
    lorentz = sigma / (
        2.0
        * math.pi
        * ((f[ringdown] - f_ring) ** 2 + sigma**2 / 4.0)
    )
    amp[ringdown] = w2 * lorentz
    return f**3 * amp**2


@lru_cache(maxsize=512)
def _phenomb_reference_normalization(eta: float, chi: float) -> float:
    """Normalize the scale-free IMR shape for a 1-Msun, z=0 reference."""

    _, _, _, f_cut = phenomb_frequencies(1.0, eta, chi, 0.0)
    norm_grid = np.logspace(math.log10(f_cut) - 12.0, math.log10(f_cut), 80_000)
    raw_norm = _phenomb_raw_p(norm_grid, 1.0, eta, chi, 0.0)
    return trapz_log(raw_norm, norm_grid)


def phenomb_shape(
    f_hz: np.ndarray,
    total_mass_msun: float,
    eta: float = 0.25,
    chi: float = 0.0,
    redshift: float = 0.0,
) -> np.ndarray:
    """Normalized observer-frame IMR energy shape for one domain point.

    IMRPhenomB is scale-free in ``M(1+z)f``.  Evaluating a cached 1-Msun
    reference shape makes redshift-history integration fast while preserving
    the exact normalization in logarithmic frequency.
    """

    normalization = _phenomb_reference_normalization(float(eta), float(chi))
    if not normalization > 0.0:
        raise RuntimeError("IMR shape normalization failed.")
    scaled_frequency = (
        np.asarray(f_hz, dtype=float)
        * float(total_mass_msun)
        * (1.0 + float(redshift))
    )
    p = _phenomb_raw_p(
        scaled_frequency,
        1.0,
        float(eta),
        float(chi),
        0.0,
    ) / normalization
    if np.any(p < 0.0):
        raise RuntimeError("IMR energy shape became negative.")
    return p


# ---------------------------------------------------------------------------
# Checkable GWTC-5-inspired sBBH population weighting
# ---------------------------------------------------------------------------


def husa_radiated_fraction(
    m1_msun: np.ndarray,
    m2_msun: np.ndarray,
    chi1z: np.ndarray,
    chi2z: np.ndarray,
) -> np.ndarray:
    """Aligned-spin radiated fraction used by IMRPhenomD.

    This is the public LALSuite implementation of Husa et al. (2016), Eqs.
    (3.6) and (3.8), copied algebraically so that this audit has no LALSuite
    dependency.  It gives 0.04839 for equal-mass nonspinning binaries and
    0.08051 for q=1, chi1z=chi2z=+0.7.
    """

    m1 = np.asarray(m1_msun, dtype=float)
    m2 = np.asarray(m2_msun, dtype=float)
    c1 = np.asarray(chi1z, dtype=float)
    c2 = np.asarray(chi2z, dtype=float)
    total = m1 + m2
    eta = m1 * m2 / total**2
    eta2 = eta**2
    eta3 = eta2 * eta
    eta4 = eta3 * eta
    spin = (c1 * m1**2 + c2 * m2**2) / total**2
    shat = spin / (1.0 - 2.0 * eta)
    nonspin = (
        0.055974469826360077 * eta
        + 0.5809510763115132 * eta2
        - 0.9606726679372312 * eta3
        + 3.352411249771192 * eta4
    )
    numerator = 1.0 + (
        -0.0030302335878845507
        - 2.0066110851351073 * eta
        + 7.7050567802399215 * eta2
    ) * shat
    denominator = 1.0 + (
        -0.6714403054720589
        - 1.4756929437702908 * eta
        + 7.304676214885011 * eta2
    ) * shat
    return nonspin * numerator / denominator


def jimenez_forteza_nonspinning_radiated_fraction(
    eta: float | np.ndarray,
) -> np.ndarray:
    """UIB2016/Jimenez-Forteza nonspinning radiated-energy polynomial."""

    symmetric_mass_ratio = np.asarray(eta, dtype=float)
    if np.any((symmetric_mass_ratio <= 0.0) | (symmetric_mass_ratio > 0.25)):
        raise ValueError("eta must lie in (0, 0.25].")
    return (
        0.0559745 * symmetric_mass_ratio
        + 0.580951 * symmetric_mass_ratio**2
        - 0.960599 * symmetric_mass_ratio**3
        + 3.35818 * symmetric_mass_ratio**4
    )


@lru_cache(maxsize=1)
def gwtc5_bbh_population_weighting(samples: int = 300_000) -> dict[str, float]:
    """Reproduce the rate-weighted mass and mass-weighted IMR efficiency.

    GWTC-5.0's compact public summaries do not provide a small posterior file
    for direct propagation.  This transparent approximation uses its quoted
    broken-power-law slopes, break mass, q slope, low-mass concentration, and
    low-spin/isotropic-dominated constraints.  The first Gaussian mixture
    weight is fixed by the published statement that 72% of primary masses lie
    in 8--15 Msun.  The output is used only to justify the rounded 5% benchmark
    and the 21.7-Msun rate-weighted mass; it is not presented as a posterior.
    """

    rng = np.random.default_rng(20_260_724)
    mass_grid = np.logspace(math.log10(5.0), math.log10(150.0), 6_000)
    power_law = np.where(
        mass_grid < 37.5,
        (mass_grid / 37.5) ** (-1.5),
        (mass_grid / 37.5) ** (-5.4),
    )
    power_law /= np.trapezoid(power_law, x=mass_grid)
    peak1 = np.exp(-0.5 * ((mass_grid - 9.8) / 1.5) ** 2)
    peak1 /= np.trapezoid(peak1, x=mass_grid)
    peak2 = np.exp(-0.5 * ((mass_grid - 33.0) / 3.1) ** 2)
    peak2 /= np.trapezoid(peak2, x=mass_grid)
    in_low_peak = (mass_grid >= 8.0) & (mass_grid <= 15.0)
    frac_pl = np.trapezoid(power_law[in_low_peak], x=mass_grid[in_low_peak])
    frac_peak1 = np.trapezoid(peak1[in_low_peak], x=mass_grid[in_low_peak])
    frac_peak2 = np.trapezoid(peak2[in_low_peak], x=mass_grid[in_low_peak])
    lambda2 = 0.06
    lambda1 = (
        0.72 - (1.0 - lambda2) * frac_pl - lambda2 * frac_peak2
    ) / (frac_peak1 - frac_pl)
    primary_pdf = (
        (1.0 - lambda1 - lambda2) * power_law
        + lambda1 * peak1
        + lambda2 * peak2
    )
    cdf = np.concatenate(
        [
            [0.0],
            np.cumsum(
                0.5
                * (primary_pdf[1:] + primary_pdf[:-1])
                * np.diff(mass_grid)
            ),
        ]
    )
    cdf /= cdf[-1]
    m1 = np.interp(rng.random(samples), cdf, mass_grid)

    # p(q|m1) proportional to q^1, with both components above 5 Msun.
    q_min = np.clip(5.0 / m1, 1.0e-3, 0.999)
    q = np.sqrt(q_min**2 + rng.random(samples) * (1.0 - q_min**2))
    m2 = m1 * q

    # A compact realization of the GWTC-5 low-spin, mostly isotropic result.
    magnitude1 = np.clip(rng.normal(0.20, 0.22, samples), 0.0, 1.0)
    magnitude2 = np.clip(rng.normal(0.20, 0.22, samples), 0.0, 1.0)

    def tilt_cosines() -> np.ndarray:
        isotropic = rng.uniform(-1.0, 1.0, samples)
        preferentially_aligned = np.clip(
            rng.normal(1.0, 0.6, samples), -1.0, 1.0
        )
        return np.where(
            rng.random(samples) < 0.30,
            preferentially_aligned,
            isotropic,
        )

    chi1z = magnitude1 * tilt_cosines()
    chi2z = magnitude2 * tilt_cosines()
    total = m1 + m2
    erad = husa_radiated_fraction(m1, m2, chi1z, chi2z)
    return {
        "samples": float(samples),
        "lambda_low_mass_peak": float(lambda1),
        "mean_total_mass_msun": float(np.mean(total)),
        "rate_weighted_mean_efficiency": float(np.mean(erad)),
        "mass_weighted_mean_efficiency": float(
            np.sum(total * erad) / np.sum(total)
        ),
        "equal_mass_nonspinning_efficiency": float(
            husa_radiated_fraction(
                np.asarray([1.0]),
                np.asarray([1.0]),
                np.asarray([0.0]),
                np.asarray([0.0]),
            )[0]
        ),
        "equal_mass_chi0p7_efficiency": float(
            husa_radiated_fraction(
                np.asarray([1.0]),
                np.asarray([1.0]),
                np.asarray([0.7]),
                np.asarray([0.7]),
            )[0]
        ),
    }


# ---------------------------------------------------------------------------
# Channel inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedRedshiftHistory:
    """A declared delta-function merger history."""

    rho_src_msun_mpc3: float
    redshift: float
    description: str

    def mass_elements(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([self.redshift], dtype=float),
            np.asarray([self.rho_src_msun_mpc3], dtype=float),
        )


@dataclass(frozen=True)
class RateHistory:
    """A comoving source-frame rate integrated over a declared history.

    The rate is normalized at ``rate_anchor_redshift``.  ``rate_evolution_kappa
    = 0`` is the adopted fixed-rate construction.  Nonzero kappa values are
    sensitivity models, not posterior propagation.
    """

    rate_gpc3_yr: float
    mean_mass_msun: float
    duration_gyr: float | None = 10.0
    rate_anchor_redshift: float = 0.0
    rate_evolution_kappa: float = 0.0
    processing_fraction: float = 1.0
    max_redshift: float | None = None
    samples: int = 801
    description: str = ""

    def __post_init__(self) -> None:
        if self.rate_gpc3_yr < 0.0 or self.mean_mass_msun <= 0.0:
            raise ValueError("Rate and mean mass must be physical.")
        if not 0.0 <= self.processing_fraction <= 1.0:
            raise ValueError("processing_fraction must lie in [0, 1].")
        if (self.duration_gyr is None) == (self.max_redshift is None):
            raise ValueError("Specify exactly one of duration_gyr or max_redshift.")
        if self.duration_gyr is not None and self.duration_gyr <= 0.0:
            raise ValueError("duration_gyr must be positive.")
        if self.max_redshift is not None and self.max_redshift <= 0.0:
            raise ValueError("max_redshift must be positive.")
        if self.samples < 3:
            raise ValueError("samples must be at least three.")

    @property
    def integration_redshift_max(self) -> float:
        if self.max_redshift is not None:
            return self.max_redshift
        assert self.duration_gyr is not None
        return redshift_at_lookback_gyr(self.duration_gyr)

    def rate_at_redshift(self, redshift: np.ndarray) -> np.ndarray:
        z = np.asarray(redshift, dtype=float)
        return self.rate_gpc3_yr * (
            (1.0 + z) / (1.0 + self.rate_anchor_redshift)
        ) ** self.rate_evolution_kappa

    def mass_elements(self) -> tuple[np.ndarray, np.ndarray]:
        z = np.linspace(0.0, self.integration_redshift_max, self.samples)
        quadrature = trapezoid_quadrature_weights(z)
        d_rho_dz = (
            self.processing_fraction
            * self.rate_at_redshift(z)
            * 1.0e-9
            * self.mean_mass_msun
            * source_time_per_redshift_yr(z)
        )
        mass = d_rho_dz * quadrature
        if self.rate_evolution_kappa == 0.0 and self.duration_gyr is not None:
            exact_density = (
                self.processing_fraction
                * self.rate_gpc3_yr
                * self.mean_mass_msun
                * self.duration_gyr
            )
            mass *= exact_density / np.sum(mass)
        return z, mass

    @property
    def rho_src_msun_mpc3(self) -> float:
        return float(np.sum(self.mass_elements()[1]))


@dataclass(frozen=True)
class PopIIIFormationHistory:
    """Liu--Bromm Pop-III SFRD with a declared prompt-merger mapping."""

    processing_fraction: float = 0.01
    redshift_min: float = 4.0
    redshift_max: float = 30.0
    sfrd_a: float = 765.7
    sfrd_b: float = -5.92
    sfrd_c: float = 12.83
    sfrd_d: float = -8.55
    samples: int = 1_201
    description: str = (
        "Liu--Bromm SFRD over z=4--30; one-percent prompt-merger mapping"
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.processing_fraction <= 1.0:
            raise ValueError("processing_fraction must lie in [0, 1].")
        if not 0.0 <= self.redshift_min < self.redshift_max:
            raise ValueError("Invalid Pop-III redshift interval.")

    def sfrd_msun_yr_mpc3(self, redshift: np.ndarray) -> np.ndarray:
        z = np.asarray(redshift, dtype=float)
        return self.sfrd_a * (1.0 + z) ** self.sfrd_b / (
            1.0 + ((1.0 + z) / self.sfrd_c) ** self.sfrd_d
        )

    def formation_mass_elements(self) -> tuple[np.ndarray, np.ndarray]:
        z = np.linspace(self.redshift_min, self.redshift_max, self.samples)
        quadrature = trapezoid_quadrature_weights(z)
        d_rho_dz = self.sfrd_msun_yr_mpc3(z) * source_time_per_redshift_yr(z)
        return z, d_rho_dz * quadrature

    def mass_elements(self) -> tuple[np.ndarray, np.ndarray]:
        z, formation_mass = self.formation_mass_elements()
        return z, self.processing_fraction * formation_mass

    @property
    def rho_iii_msun_mpc3(self) -> float:
        return float(np.sum(self.formation_mass_elements()[1]))

    @property
    def rho_src_msun_mpc3(self) -> float:
        return self.processing_fraction * self.rho_iii_msun_mpc3


SourceHistory = FixedRedshiftHistory | RateHistory | PopIIIFormationHistory


@dataclass(frozen=True)
class AGNPipelinePoint:
    """One point in the declared AGN--IMRI mass-flow domain."""

    kappa_im: float
    f_agn: float
    rate_bbh_gpc3_yr: float
    mean_bbh_mass_msun: float
    t_eff_gyr: float

    @property
    def processing_fraction(self) -> float:
        return self.kappa_im * self.f_agn

    @property
    def rho_bbh_proc_msun_mpc3(self) -> float:
        return (
            self.rate_bbh_gpc3_yr
            * self.mean_bbh_mass_msun
            * self.t_eff_gyr
        )

    @property
    def rho_imr_msun_mpc3(self) -> float:
        return self.processing_fraction * self.rho_bbh_proc_msun_mpc3

    def explicit_lookback_history(self) -> RateHistory:
        return RateHistory(
            rate_gpc3_yr=self.rate_bbh_gpc3_yr,
            mean_mass_msun=self.mean_bbh_mass_msun,
            duration_gyr=self.t_eff_gyr,
            rate_anchor_redshift=0.2,
            processing_fraction=self.processing_fraction,
            description=(
                "AGN pipeline with constant source-frame rate over T_eff "
                "interpreted as a lookback interval"
            ),
        )


AGN_BENCHMARK_POINT = AGNPipelinePoint(
    0.3,
    0.1,
    GWTC5_DEFAULT_BBH_RATE_90[1],
    60.0,
    10.0,
)
AGN_CEILING_POINT = AGNPipelinePoint(1.0, 1.0, 49.4, 150.0, 13.0)


@dataclass(frozen=True)
class SpectralChannel:
    key: str
    label: str
    benchmark_or_ceiling: str
    epsilon: float
    history: SourceHistory
    f_ref_hz: float
    shape_kind: str
    f_max_hz: float | None = None
    total_mass_msun: float | None = None
    eta: float | None = None
    chi: float | None = None
    provenance: str = ""

    @property
    def rho_src_msun_mpc3(self) -> float:
        return float(np.sum(self.history.mass_elements()[1]))

    @property
    def mean_energy_dilution(self) -> float:
        redshift, mass = self.history.mass_elements()
        return float(np.sum(mass / (1.0 + redshift)) / np.sum(mass))

    @property
    def present_budget(self) -> float:
        return (
            self.epsilon
            * self.rho_src_msun_mpc3
            * self.mean_energy_dilution
            / RHO_CRIT
        )

    def shape_at_redshift(
        self, f_hz: np.ndarray, redshift: float
    ) -> np.ndarray:
        if self.shape_kind == "inspiral":
            if self.f_max_hz is None:
                raise ValueError(f"{self.key}: f_max is required.")
            return inspiral_shape(f_hz, self.f_max_hz / (1.0 + redshift))
        if self.shape_kind == "phenomb":
            if self.total_mass_msun is None or self.eta is None or self.chi is None:
                raise ValueError(f"{self.key}: IMR parameters are required.")
            return phenomb_shape(
                f_hz,
                self.total_mass_msun,
                self.eta,
                self.chi,
                redshift,
            )
        raise ValueError(f"Unknown shape kind: {self.shape_kind}")

    def omega(self, f_hz: np.ndarray) -> np.ndarray:
        f = np.asarray(f_hz, dtype=float)
        redshift, mass = self.history.mass_elements()
        omega = np.zeros_like(f)
        for z_value, mass_value in zip(redshift, mass):
            omega += (
                self.epsilon
                * mass_value
                / ((1.0 + z_value) * RHO_CRIT)
                * self.shape_at_redshift(f, float(z_value))
            )
        return omega

    def hc_at_reference(self) -> float:
        omega = self.omega(np.asarray([self.f_ref_hz]))[0]
        return float(hc_from_omega(self.f_ref_hz, omega))


def channel_inputs() -> dict[str, SpectralChannel]:
    """Return the Step-3 recommended channel calculations."""

    bbh_weighting = gwtc5_bbh_population_weighting()
    bns_benchmark = RateHistory(
        GWTC5_BNS_FULLPOP_RATE_90[1],
        2.8,
        rate_anchor_redshift=0.0,
        description="GWTC-5 FullPop median; constant rate over 10 Gyr",
    )
    bns_ceiling = RateHistory(
        154.7,
        2.8,
        rate_anchor_redshift=0.0,
        description=(
            "GWTC-5 union-of-model 90-percent endpoint; "
            "constant rate over 10 Gyr"
        ),
    )
    bbh_benchmark = RateHistory(
        GWTC5_DEFAULT_BBH_RATE_90[1],
        bbh_weighting["mean_total_mass_msun"],
        rate_anchor_redshift=0.2,
        description="GWTC-5 Default-BBH median; constant rate over 10 Gyr",
    )
    bbh_ceiling = RateHistory(
        49.4,
        bbh_weighting["mean_total_mass_msun"],
        rate_anchor_redshift=0.2,
        description=(
            "GWTC-5 union-of-model 90-percent endpoint; "
            "constant rate over 10 Gyr"
        ),
    )
    popiii_benchmark = PopIIIFormationHistory()
    popiii_ceiling = replace(
        popiii_benchmark,
        sfrd_a=(
            popiii_benchmark.sfrd_a
            * POPIII_CEILING_RHO_MSUN_MPC3
            / popiii_benchmark.rho_iii_msun_mpc3
        ),
        description=(
            "Liu--Bromm redshift shape normalized to the Planck-calibrated "
            "Pop-III mass endpoint; one-percent prompt-merger mapping"
        ),
    )

    rho_nuc = ETA_NSC_FIDUCIAL * RHO_STAR
    return {
        "emri_benchmark": SpectralChannel(
            key="emri_benchmark",
            label="EMRI",
            benchmark_or_ceiling="benchmark",
            epsilon=0.05,
            history=FixedRedshiftHistory(
                1.0e-4 * rho_nuc,
                0.0,
                (
                    "Scalar benchmark; published all-sky rate lacks the "
                    "redshift distribution needed for a unique conversion"
                ),
            ),
            f_ref_hz=1.0e-2,
            shape_kind="inspiral",
            f_max_hz=1.0e-2,
            provenance="NSC reservoir and Bonetti--Sesana M1-scale processing",
        ),
        "emri_closed_pool_ceiling": SpectralChannel(
            key="emri_closed_pool_ceiling",
            label="EMRI",
            benchmark_or_ceiling="ceiling",
            epsilon=0.05,
            history=FixedRedshiftHistory(
                EMRI_CLOSED_POOL_F_MERGE * rho_nuc,
                0.0,
                (
                    "Unreplenished NSC with the Babak et al. stellar-BH "
                    "mass fraction; other benchmark inputs fixed"
                ),
            ),
            f_ref_hz=1.0e-2,
            shape_kind="inspiral",
            f_max_hz=1.0e-2,
            provenance="Conditional closed-pool NSC inventory ceiling",
        ),
        "agn_imri_benchmark": SpectralChannel(
            key="agn_imri_benchmark",
            label="AGN--IMRI",
            benchmark_or_ceiling="benchmark",
            epsilon=0.05,
            history=AGN_BENCHMARK_POINT.explicit_lookback_history(),
            f_ref_hz=3.0e-3,
            shape_kind="inspiral",
            f_max_hz=4.0e-2,
            provenance="Explicit AGN pipeline benchmark mass delivery",
        ),
        "agn_imri_ceiling": SpectralChannel(
            key="agn_imri_ceiling",
            label="AGN--IMRI",
            benchmark_or_ceiling="ceiling",
            epsilon=0.05,
            history=AGN_CEILING_POINT.explicit_lookback_history(),
            f_ref_hz=3.0e-3,
            shape_kind="inspiral",
            f_max_hz=4.0e-2,
            provenance="Maximum over the declared rectangular domain",
        ),
        "bns_benchmark": SpectralChannel(
            key="bns_benchmark",
            label="BNS",
            benchmark_or_ceiling="benchmark",
            epsilon=0.02,
            history=bns_benchmark,
            f_ref_hz=0.1,
            shape_kind="inspiral",
            f_max_hz=1_500.0,
            provenance="GWTC-5 FullPop median rate; 10-Gyr constant-rate history",
        ),
        "bns_ceiling": SpectralChannel(
            key="bns_ceiling",
            label="BNS",
            benchmark_or_ceiling="ceiling",
            epsilon=0.02,
            history=bns_ceiling,
            f_ref_hz=0.1,
            shape_kind="inspiral",
            f_max_hz=1_500.0,
            provenance=(
                "GWTC-5 union-of-model 90-percent rate upper endpoint; "
                "fixed history"
            ),
        ),
        "popiii_benchmark": SpectralChannel(
            key="popiii_benchmark",
            label="Pop III BBH",
            benchmark_or_ceiling="benchmark",
            epsilon=0.05,
            history=popiii_benchmark,
            f_ref_hz=0.1,
            shape_kind="phenomb",
            total_mass_msun=100.0,
            eta=0.25,
            chi=0.0,
            provenance=(
                "Liu--Bromm formation history; one-percent prompt "
                "50+50-Msun nonspinning merger model"
            ),
        ),
        "popiii_ceiling": SpectralChannel(
            key="popiii_ceiling",
            label="Pop III BBH",
            benchmark_or_ceiling="ceiling",
            epsilon=0.05,
            history=popiii_ceiling,
            f_ref_hz=0.1,
            shape_kind="phenomb",
            total_mass_msun=100.0,
            eta=0.25,
            chi=0.0,
            provenance=(
                "Planck-calibrated Pop-III mass endpoint from Inayoshi et "
                "al.; other benchmark inputs fixed"
            ),
        ),
        "bbh_benchmark": SpectralChannel(
            key="bbh_benchmark",
            label="sBBH",
            benchmark_or_ceiling="benchmark",
            epsilon=0.05,
            history=bbh_benchmark,
            f_ref_hz=25.0,
            shape_kind="phenomb",
            total_mass_msun=bbh_weighting["mean_total_mass_msun"],
            eta=0.25,
            chi=0.0,
            provenance=(
                "GWTC-5 Default-BBH median rate and representative mass "
                "from the median population model; "
                "10-Gyr constant-rate history"
            ),
        ),
        "bbh_ceiling": SpectralChannel(
            key="bbh_ceiling",
            label="sBBH",
            benchmark_or_ceiling="ceiling",
            epsilon=0.0801,
            history=bbh_ceiling,
            f_ref_hz=25.0,
            shape_kind="phenomb",
            total_mass_msun=bbh_weighting["mean_total_mass_msun"],
            eta=0.25,
            chi=0.7,
            provenance=(
                "GWTC-5 union-of-model rate upper endpoint; "
                "q=1, chi1=chi2=+0.7; "
                "10-Gyr fixed history"
            ),
        ),
    }


def rate_only_benchmark_interval_results(
    channels: dict[str, SpectralChannel] | None = None,
) -> dict[str, object]:
    """Propagate the model-specific GWTC-5 rate intervals.

    Every other channel input is fixed.  The low, median, and high rate
    values are evaluated directly rather than combined into a joint
    population posterior.
    """

    channel_map = channel_inputs() if channels is None else channels
    specifications = {
        "agn_imri_benchmark": {
            "rate_interval": GWTC5_DEFAULT_BBH_RATE_90,
            "available_mass_model": "rate_processed_mass",
            "available_mass_definition": (
                "rho_BBH,proc = R_BBH <M_BBH> T_eff"
            ),
            "source_fraction_definition": "f_AGN kappa_IM",
            "rate_model": "GWTC-5 Default-BBH",
        },
        "bns_benchmark": {
            "rate_interval": GWTC5_BNS_FULLPOP_RATE_90,
            "reservoir_msun_mpc3": RHO_STAR,
            "available_mass_model": "fixed_reservoir",
            "available_mass_definition": "rho_star",
            "source_fraction_definition": "rho_src / rho_star",
            "rate_model": "GWTC-5 FullPop BNS",
        },
        "bbh_benchmark": {
            "rate_interval": GWTC5_DEFAULT_BBH_RATE_90,
            "reservoir_msun_mpc3": RHO_STAR,
            "available_mass_model": "fixed_reservoir",
            "available_mass_definition": "rho_star",
            "source_fraction_definition": "rho_src / rho_star",
            "rate_model": "GWTC-5 Default-BBH",
        },
    }
    output: dict[str, object] = {
        "scope": (
            "Model-specific marginal 90-percent merger-rate credible "
            "intervals only; masses, efficiencies, spectra, and 10-Gyr "
            "histories are held fixed."
        ),
        "scaling": (
            "rho_src, Omega_ref, and integrated energy scale as R; h_c "
            "scales as sqrt(R). Available mass and source fraction follow "
            "the channel-specific definitions reported below."
        ),
        "channels": {},
    }
    rows = output["channels"]
    assert isinstance(rows, dict)
    for key, specification in specifications.items():
        channel = channel_map[key]
        if not isinstance(channel.history, RateHistory):
            raise TypeError(f"{key} does not use a RateHistory")
        low_rate, median_rate, high_rate = specification["rate_interval"]
        if not math.isclose(
            channel.history.rate_gpc3_yr,
            median_rate,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"{key} benchmark rate does not match its median")
        endpoint_rows: dict[str, dict[str, float]] = {}
        for endpoint, rate in (
            ("low", low_rate),
            ("median", median_rate),
            ("high", high_rate),
        ):
            endpoint_channel = replace(
                channel,
                history=replace(channel.history, rate_gpc3_yr=rate),
            )
            if specification["available_mass_model"] == "rate_processed_mass":
                available_mass = (
                    endpoint_channel.history.rate_gpc3_yr
                    * endpoint_channel.history.mean_mass_msun
                    * float(endpoint_channel.history.duration_gyr)
                )
                source_fraction = endpoint_channel.history.processing_fraction
            else:
                available_mass = float(specification["reservoir_msun_mpc3"])
                source_fraction = (
                    endpoint_channel.rho_src_msun_mpc3 / available_mass
                )
            endpoint_rows[endpoint] = {
                "rate_gpc3_yr": rate,
                "available_mass_msun_mpc3": available_mass,
                "rho_src_msun_mpc3": endpoint_channel.rho_src_msun_mpc3,
                "source_fraction": source_fraction,
                "hc_at_reference": endpoint_channel.hc_at_reference(),
                "omega_at_reference": float(
                    endpoint_channel.omega(
                        np.asarray([endpoint_channel.f_ref_hz])
                    )[0]
                ),
                "present_integral_omega_dlnf": (
                    endpoint_channel.present_budget
                ),
            }
        rows[key] = {
            "rate_model": specification["rate_model"],
            "available_mass_definition": specification[
                "available_mass_definition"
            ],
            "source_fraction_definition": specification[
                "source_fraction_definition"
            ],
            "coverage": "90-percent marginal credible interval",
            "other_inputs_held_fixed": True,
            "endpoints": endpoint_rows,
        }
    return output


STEP3B_BENCHMARK_CHANNEL_KEYS = (
    "emri_benchmark",
    "agn_imri_benchmark",
    "bns_benchmark",
    "popiii_benchmark",
    "bbh_benchmark",
)

STEP3B_CEILING_CHANNEL_KEYS = (
    "emri_closed_pool_ceiling",
    "agn_imri_ceiling",
    "bns_ceiling",
    "popiii_ceiling",
    "bbh_ceiling",
)


def lm24_full_imr_efficiency_moment(spin: float = 0.0) -> float:
    """Mass-weighted Husa efficiency for the full analytic census."""

    q = np.linspace(SMBHB_Q_MIN, SMBHB_Q_MAX, 20_001)
    weight = q**2
    efficiency = husa_radiated_fraction(
        np.ones_like(q),
        q,
        np.full_like(q, spin),
        np.full_like(q, spin),
    )
    return float(
        np.trapezoid(weight * efficiency, x=q)
        / np.trapezoid(weight, x=q)
    )


def step3b_integrated_energy_results() -> dict[str, object]:
    """Integrated benchmark and one-generation conditional ceiling sums."""

    channels = channel_inputs()
    smbhb = smbhb_population_results()
    participating_rho = smbhb[
        "participant_mass_density_benchmark_msun_mpc3"
    ]
    participating_eta = smbhb["eta_moment_participating_mass"]
    smbhb_efficiency = (
        participating_eta * peters_isco_efficiency_per_eta(0.0)
    )
    smbhb_dilution = lm24_energy_dilution_moment()
    rows: dict[str, dict[str, float | str]] = {
        "smbhb_benchmark": {
            "label": "SMBHB LM24 one-pass benchmark",
            "history_scope": (
                "Full analytic LM24 mass moment with the explicit benchmark "
                "redshift distribution"
            ),
            "rho_src_msun_mpc3": participating_rho,
            "source_frame_efficiency": smbhb_efficiency,
            "mean_energy_dilution": smbhb_dilution,
            "source_frame_energy_over_rho_crit": (
                smbhb_efficiency * participating_rho / RHO_CRIT
            ),
            "present_integral_omega_dlnf": smbhb_inspiral_present_budget(),
        }
    }
    for key in STEP3B_BENCHMARK_CHANNEL_KEYS:
        channel = channels[key]
        rows[key] = {
            "label": channel.label,
            "history_scope": (
                "effective-z=0 scalar benchmark; no unique history supplied"
                if key == "emri_benchmark"
                else "explicit source history with redshift dilution"
            ),
            "rho_src_msun_mpc3": channel.rho_src_msun_mpc3,
            "source_frame_efficiency": channel.epsilon,
            "mean_energy_dilution": channel.mean_energy_dilution,
            "source_frame_energy_over_rho_crit": (
                channel.epsilon
                * channel.rho_src_msun_mpc3
                / RHO_CRIT
            ),
            "present_integral_omega_dlnf": channel.present_budget,
        }

    total = float(
        sum(
            float(row["present_integral_omega_dlnf"])
            for row in rows.values()
        )
    )
    for row in rows.values():
        row["fraction_of_conditional_benchmark_sum"] = (
            float(row["present_integral_omega_dlnf"]) / total
        )

    rate_intervals = rate_only_benchmark_interval_results(channels)
    lm24_plus = float(
        rows["smbhb_benchmark"]["present_integral_omega_dlnf"]
    ) * (0.8 / 1.8)
    lm24_minus = float(
        rows["smbhb_benchmark"]["present_integral_omega_dlnf"]
    ) * (0.5 / 1.8)
    rate_rows = rate_intervals["channels"]
    assert isinstance(rate_rows, dict)
    rate_sensitivity_cases: dict[str, dict[str, object]] = {}
    for label, keys in {
        "shared_default_bbh_rate": (
            "agn_imri_benchmark",
            "bbh_benchmark",
        ),
        "bns_fullpop_rate": ("bns_benchmark",),
    }.items():
        case_plus = 0.0
        case_minus = 0.0
        for key in keys:
            endpoints = rate_rows[key]["endpoints"]
            median_energy = endpoints["median"][
                "present_integral_omega_dlnf"
            ]
            case_plus += (
                endpoints["high"]["present_integral_omega_dlnf"]
                - median_energy
            )
            case_minus += (
                median_energy
                - endpoints["low"]["present_integral_omega_dlnf"]
            )
        rate_sensitivity_cases[label] = {
            "varied_channels": keys,
            "coverage": (
                "90-percent interval induced by the marginal rate posterior"
            ),
            "central_sum_omega_dlnf": total,
            "plus_omega_dlnf": case_plus,
            "minus_omega_dlnf": case_minus,
            "upper_sum_omega_dlnf": total + case_plus,
            "lower_sum_omega_dlnf": total - case_minus,
        }
    rate_plus = float(
        sum(
            float(case["plus_omega_dlnf"])
            for case in rate_sensitivity_cases.values()
        )
    )
    rate_minus = float(
        sum(
            float(case["minus_omega_dlnf"])
            for case in rate_sensitivity_cases.values()
        )
    )
    benchmark_rate_only_sensitivities = {
        "scope": (
            "Each model-specific marginal 90-percent rate interval is "
            "mapped into the benchmark sum with every other input fixed."
        ),
        "statistical_status": (
            "One-at-a-time propagated marginal ranges, not a joint "
            "credible interval for the benchmark sum."
        ),
        "cases": rate_sensitivity_cases,
    }
    benchmark_rate_only_envelope = {
        "scope": (
            "The shared Default-BBH rate used by AGN--IMRI and sBBH and the "
            "FullPop BNS rate are evaluated at their model-specific "
            "90-percent endpoints; all other inputs are held fixed."
        ),
        "statistical_status": (
            "Not a joint credible interval because the BNS and BBH rate "
            "summaries do not define a joint posterior."
        ),
        "plus_omega_dlnf": rate_plus,
        "minus_omega_dlnf": rate_minus,
        "upper_sum_omega_dlnf": total + rate_plus,
        "lower_sum_omega_dlnf": total - rate_minus,
    }
    benchmark_input_envelope = {
        "scope": (
            "Endpoint envelope obtained by varying the LM24 SMBH-density "
            "normalization, the shared Default-BBH rate, and the FullPop "
            "BNS rate; all other inputs are held fixed."
        ),
        "statistical_status": (
            "Not a joint credible interval: the inputs come from separate "
            "analyses and no cross-channel covariance is available."
        ),
        "plus_omega_dlnf": lm24_plus + rate_plus,
        "minus_omega_dlnf": lm24_minus + rate_minus,
        "upper_sum_omega_dlnf": total + lm24_plus + rate_plus,
        "lower_sum_omega_dlnf": total - lm24_minus - rate_minus,
    }

    smbhb_ceiling_budget = smbhb_inspiral_present_budget(
        "conditional_ceiling"
    )
    smbhb_ceiling_efficiency = (
        0.25 * peters_isco_efficiency_per_eta(0.0)
    )
    ceiling_rows: dict[str, dict[str, float | str | bool]] = {
        "smbhb_conditional_ceiling": {
            "label": "SMBHB LM24 one-generation ceiling",
            "history_scope": (
                "One equal-mass merger per present-day remnant at z=0; "
                "the repeated-merger fixed-frequency factor is excluded"
            ),
            "rho_src_msun_mpc3": smbhb[
                "participant_mass_density_conditional_ceiling_msun_mpc3"
            ],
            "source_frame_efficiency": smbhb_ceiling_efficiency,
            "mean_energy_dilution": 1.0,
            "source_frame_energy_over_rho_crit": smbhb_ceiling_budget,
            "present_integral_omega_dlnf": smbhb_ceiling_budget,
            "mass_ratio": 1.0,
            "mergers_per_present_day_remnant": 1.0,
            "repeated_merger_factor_included": False,
        }
    }
    for key in STEP3B_CEILING_CHANNEL_KEYS:
        channel = channels[key]
        output_key = (
            "bbh_ceiling_realizable_chi0p7"
            if key == "bbh_ceiling"
            else key
        )
        ceiling_rows[output_key] = {
            "label": channel.label,
            "history_scope": channel.history.description,
            "rho_src_msun_mpc3": channel.rho_src_msun_mpc3,
            "source_frame_efficiency": channel.epsilon,
            "mean_energy_dilution": channel.mean_energy_dilution,
            "source_frame_energy_over_rho_crit": (
                channel.epsilon
                * channel.rho_src_msun_mpc3
                / RHO_CRIT
            ),
            "present_integral_omega_dlnf": channel.present_budget,
        }
        if key == "bbh_ceiling":
            ceiling_rows[output_key]["aligned_component_spin"] = 0.7
            ceiling_rows[output_key]["spectrum_scope"] = (
                "Single realizable q=1, chi1=chi2=+0.7 spectrum; "
                "not the pointwise spin envelope"
            )

    ceiling_total = float(
        sum(
            float(row["present_integral_omega_dlnf"])
            for row in ceiling_rows.values()
        )
    )
    for row in ceiling_rows.values():
        row["fraction_of_conditional_ceiling_sum"] = (
            float(row["present_integral_omega_dlnf"])
            / ceiling_total
        )

    lm24_plus = smbhb_ceiling_budget * (0.8 / 1.8)
    lm24_minus = smbhb_ceiling_budget * (0.5 / 1.8)
    ceiling_uncertainty = {
        "source": (
            "LM24 SMBH mass-density normalization only; all other "
            "ceiling inputs are held fixed"
        ),
        "plus_omega_dlnf": lm24_plus,
        "minus_omega_dlnf": lm24_minus,
        "upper_sum_omega_dlnf": ceiling_total + lm24_plus,
        "lower_sum_omega_dlnf": ceiling_total - lm24_minus,
        "reported_in_units_of_1e_minus_7": {
            "central": round(ceiling_total / 1.0e-7, 2),
            "plus": round(lm24_plus / 1.0e-7, 2),
            "minus": round(lm24_minus / 1.0e-7, 2),
        },
    }

    support_sensitivities: dict[str, object] = {
        "primary_schwarzschild_inspiral": {
            "source_frame_efficiency": smbhb_efficiency,
            "present_integral_omega_dlnf": (
                smbhb_inspiral_present_budget("benchmark", spin=0.0)
            ),
            "scope": (
                "Full analytic LM24 mass moment; Newtonian circular inspiral "
                "through each binary's source-frame Schwarzschild ISCO."
            ),
        }
    }
    kerr_efficiency = (
        participating_eta * peters_isco_efficiency_per_eta(spin=0.7)
    )
    support_sensitivities["kerr_chi0p7_inspiral"] = {
        "source_frame_efficiency": kerr_efficiency,
        "present_integral_omega_dlnf": smbhb_inspiral_present_budget(
            "benchmark",
            spin=0.7,
        ),
        "ratio_to_primary": (
            kerr_efficiency / smbhb_efficiency
        ),
        "scope": (
            "Sensitivity only; adopting it as primary would require a "
            "declared SMBHB spin domain."
        ),
    }
    for spin_label, spin in (("nonspinning", 0.0), ("chi0p7", 0.7)):
        efficiency = lm24_full_imr_efficiency_moment(spin)
        support_sensitivities[f"full_imr_{spin_label}"] = {
            "source_frame_efficiency": efficiency,
            "present_integral_omega_dlnf": (
                participating_rho
                / RHO_CRIT
                * efficiency
                * smbhb_dilution
            ),
            "scope": (
                "Integrated Husa radiated-fraction sensitivity only; no "
                "SMBHB full-IMR population spectrum is adopted."
            ),
        }

    neff_h2_limit = 5.6e-6 * WRIGHT_DELTA_NEFF_95
    neff_omega_limit = neff_h2_limit / WRIGHT_H**2
    return {
        "included_components": tuple(rows),
        "excluded": {
            "repeated_merger_pta_strain_extension": (
                "The fixed-frequency repeated-merger factor is not applied "
                "to the one-generation integrated ceiling."
            ),
            "bbh_pointwise_envelope": (
                "The frequencywise spin envelope is not one realizable "
                "population and is never integrated."
            ),
            "galactic_dwd_foreground": (
                "Contextual Galactic foreground outside the cosmological ledger."
            ),
            "broadband_smbhb_composite_cumulative": (
                "Retired: only the full-census PTA display segment is emitted; "
                "the scalar SMBHB energies use analytic mass moments."
            ),
        },
        "channels": rows,
        "conditional_benchmark_sum_omega_dlnf": total,
        "benchmark_rate_only_sensitivities": (
            benchmark_rate_only_sensitivities
        ),
        "benchmark_rate_only_envelope": benchmark_rate_only_envelope,
        "benchmark_input_envelope": benchmark_input_envelope,
        "smbhb_support_model": (
            "Full analytic LM24 mass moment; Newtonian circular inspiral to "
            "each source-frame Schwarzschild ISCO"
        ),
        "interpretation": (
            "The benchmark and high-side sums add independently calculated "
            "scalar channel energies. The SMBHB scalars use full analytic "
            "LM24 moments and are not normalized by an emitted broadband "
            "spectrum. The high-side cases need not occur together; their "
            "sum is not a universal or jointly inferred population bound."
        ),
        "additivity": (
            "Distinct GW events add in energy. Shared or nested parent stocks "
            "are not counted as independent reservoirs."
        ),
        "smbhb_conditional_reference_spectrum_integral_omega_dlnf": (
            smbhb_inspiral_present_budget("conditional_ceiling")
        ),
        "conditional_ceiling": {
            "included_components": tuple(ceiling_rows),
            "channels": ceiling_rows,
            "sum_omega_dlnf": ceiling_total,
            "uncertainty_lm24_only": ceiling_uncertainty,
            "scope": (
                "Six conditional channel ceilings. The SMBHB term is one "
                "equal-mass merger per present-day remnant; its repeated-"
                "merger fixed-frequency factor is excluded. The sBBH term "
                "uses the realizable chi1=chi2=+0.7 spectrum."
            ),
        },
        "smbhb_support_sensitivities": support_sensitivities,
        "smbhb_emitted_spectrum": {
            "frequency_min_hz": SMBHB_PTA_DISPLAY_MIN_HZ,
            "frequency_max_hz": SMBHB_PTA_DISPLAY_MAX_HZ,
            "scope": (
                "Display-only full-census PTA segment. It is not integrated "
                "to obtain either scalar SMBHB energy or a composite sum."
            ),
            "broadband_composite_cumulative_status": "retired",
        },
        "cosmological_context": {
            "wright_giblin_hazboun": {
                "delta_neff_95": WRIGHT_DELTA_NEFF_95,
                "h": WRIGHT_H,
                "h2_integral_limit": neff_h2_limit,
                "omega_integral_limit": neff_omega_limit,
                "omega_limit_over_conditional_benchmark_sum": (
                    neff_omega_limit / total
                ),
                "scope": (
                    "Pre-recombination backgrounds only; this is not a bound "
                    "on the late-time astrophysical composite."
                ),
                "provenance": (
                    "Equation and inputs verified against arXiv:2409.15572v1; "
                    "2026 PRL metadata verified at doi:10.1103/clwq-n2vp."
                ),
            },
            "giblin_thrane": {
                "peak_omega_levels": GIBLIN_THRANE_PEAK_LEVELS,
                "scope": (
                    "Pointwise rule-of-thumb levels for radiation-era "
                    "cosmological sources; explicitly not an astrophysical "
                    "bound and not an integrated-energy limit. No ratio to a "
                    "retired broadband composite is emitted."
                ),
            },
        },
    }


def bbh_aligned_spin_ceiling(
    f_hz: np.ndarray,
    spin_samples: int = 141,
) -> np.ndarray:
    """Pointwise sBBH ceiling over the declared aligned-spin domain.

    The table quotes the maximizing endpoint at 25 Hz, q=1 and chi=+0.7.
    Across the full plotted band, however, changing spin moves energy between
    inspiral, merger, and ringdown.  The dashed curve must therefore maximize
    B(chi) p(f|chi) independently at each frequency.  The public Husa
    radiated-energy fit supplies the relative spin dependence and is scaled
    to the adopted endpoint efficiency 0.0801, preserving the tabulated
    25-Hz normalization.

    This envelope is not one realizable spectrum and must not be integrated.
    """

    f = np.asarray(f_hz, dtype=float)
    if spin_samples < 2:
        raise ValueError("spin_samples must be at least two.")
    ceiling_channel = channel_inputs()["bbh_ceiling"]
    spins = np.linspace(0.0, 0.7, spin_samples)
    ones = np.ones_like(spins)
    fit_efficiencies = husa_radiated_fraction(ones, ones, spins, spins)
    efficiencies = (
        ceiling_channel.epsilon
        * fit_efficiencies
        / fit_efficiencies[-1]
    )
    redshift, mass = ceiling_channel.history.mass_elements()

    envelope = np.zeros_like(f)
    for spin, efficiency in zip(spins, efficiencies):
        spectrum = np.zeros_like(f)
        for z_value, mass_value in zip(redshift, mass):
            spectrum += (
                efficiency
                * mass_value
                / ((1.0 + z_value) * RHO_CRIT)
                * phenomb_shape(
                    f,
                    ceiling_channel.total_mass_msun,
                    ceiling_channel.eta,
                    float(spin),
                    float(z_value),
                )
            )
        envelope = np.maximum(envelope, spectrum)
    return envelope


def nsc_density_audit() -> dict[str, float | str]:
    """Integrate the NSC--host relation over the local Baldry GSMF."""

    log_mass = np.linspace(7.0, 12.0, 20_001)
    x = 10.0 ** (log_mass - 10.66)
    gsmf = np.log(10.0) * np.exp(-x) * (
        3.96e-3 * x ** (-0.35 + 1.0)
        + 0.79e-3 * x ** (-1.47 + 1.0)
    )
    nsc_mass = 10.0 ** (0.48 * (log_mass - 9.0) + 6.51)
    stellar_density = float(
        np.trapezoid(10.0**log_mass * gsmf, x=log_mass)
    )
    nsc_density_median = float(np.trapezoid(nsc_mass * gsmf, x=log_mass))
    scatter_boost = math.exp((0.6 * math.log(10.0)) ** 2 / 2.0)
    return {
        "stellar_density_from_gsmf_msun_mpc3": stellar_density,
        "eta_nsc_unit_occupation_median_relation": (
            nsc_density_median / stellar_density
        ),
        "lognormal_scatter_boost_0p6dex": scatter_boost,
        "eta_nsc_unit_occupation_scatter_mean": (
            scatter_boost * nsc_density_median / stellar_density
        ),
        "eta_nsc_adopted_fiducial": ETA_NSC_FIDUCIAL,
        "interpretation": (
            "factor-of-few estimate conditional on the NSC-host relation, "
            "its scatter, galaxy occupation, and integration range; not a "
            "universal hard upper bound"
        ),
    }


BONETTI_SESANA_EMRI_MODELS: dict[str, tuple[float, float]] = {
    "M1": (1_600.0, 10.0),
    "M2": (1_400.0, 10.0),
    "M3": (2_770.0, 10.0),
    "M4": (520.0, 30.0),
    "M5": (140.0, 10.0),
    "M6": (2_080.0, 10.0),
    "M7": (15_800.0, 10.0),
    "M8": (180.0, 10.0),
    "M9": (1_530.0, 10.0),
    "M10": (1_520.0, 10.0),
    "M11": (13.0, 10.0),
    "M12": (20_000.0, 10.0),
}


def comoving_volume_mpc3(redshift: float) -> float:
    """Full-sky comoving volume in the adopted flat cosmology."""

    z = np.linspace(0.0, redshift, 20_001)
    distance_mpc = (
        C_SI / H0_SI / MPC_SI * np.trapezoid(1.0 / expansion_function(z), x=z)
    )
    return 4.0 * math.pi * distance_mpc**3 / 3.0


def emri_rate_proxy_audit() -> dict[str, object]:
    """Convert published all-sky EMRI rates to a transparent mass proxy.

    This deliberately reproduces the old uniform-volume, 10-Gyr conversion
    only to show why f_merge=1e-4 is a motivated scale.  The source paper does
    not publish the redshift distribution needed to turn its observer-time
    all-sky totals into a unique cosmic processed density.
    """

    volume = comoving_volume_mpc3(4.5)
    rho_nsc = ETA_NSC_FIDUCIAL * RHO_STAR
    models: dict[str, dict[str, float]] = {}
    for name, (rate_per_year, compact_mass) in BONETTI_SESANA_EMRI_MODELS.items():
        rho_src = rate_per_year * compact_mass * 1.0e10 / volume
        models[name] = {
            "all_sky_rate_per_observer_year": rate_per_year,
            "compact_object_mass_msun": compact_mass,
            "rho_src_uniform_volume_proxy_msun_mpc3": rho_src,
            "f_merge_uniform_volume_proxy": rho_src / rho_nsc,
        }
    return {
        "z_max": 4.5,
        "comoving_volume_mpc3": volume,
        "duration_proxy_gyr": 10.0,
        "models": models,
        "adopted_f_merge_benchmark": 1.0e-4,
        "closed_pool_f_merge": EMRI_CLOSED_POOL_F_MERGE,
        "status": (
            "rate conversion motivates the benchmark only; the separate "
            "closed-pool ceiling assumes no NSC replenishment"
        ),
    }


def agn_pipeline_audit() -> dict[str, object]:
    """Verify the AGN rectangular-domain maximum by enumerating all corners."""

    ranges = {
        "kappa_im": (0.05, 1.0),
        "f_agn": (0.01, 1.0),
        "rate_bbh_gpc3_yr": (27.5, 49.4),
        "mean_bbh_mass_msun": (20.0, 150.0),
        "t_eff_gyr": (1.0, 13.0),
    }
    corners = [
        AGNPipelinePoint(*values)
        for values in product(
            ranges["kappa_im"],
            ranges["f_agn"],
            ranges["rate_bbh_gpc3_yr"],
            ranges["mean_bbh_mass_msun"],
            ranges["t_eff_gyr"],
        )
    ]
    maximum = max(corners, key=lambda point: point.rho_imr_msun_mpc3)
    return {
        "benchmark_point": asdict(AGN_BENCHMARK_POINT),
        "benchmark_rho_bbh_proc_msun_mpc3": (
            AGN_BENCHMARK_POINT.rho_bbh_proc_msun_mpc3
        ),
        "benchmark_source_fraction": AGN_BENCHMARK_POINT.processing_fraction,
        "benchmark_rho_imr_msun_mpc3": AGN_BENCHMARK_POINT.rho_imr_msun_mpc3,
        "domain": ranges,
        "corner_count": len(corners),
        "maximizing_point": asdict(maximum),
        "maximum_rho_imr_msun_mpc3": maximum.rho_imr_msun_mpc3,
        "maximum_over_central_rho_bh": (
            maximum.rho_imr_msun_mpc3 / RHO_BH_LM24
        ),
        "broad_census_guardrail_satisfied": (
            maximum.rho_imr_msun_mpc3 <= RHO_BH_LM24
        ),
        "status": (
            "maximum over declared Cartesian domain; not a joint-posterior "
            "maximum"
        ),
    }


def fixed_z0_copy(channel: SpectralChannel) -> SpectralChannel:
    """Return the prior no-dilution convention for traceability."""

    return replace(
        channel,
        history=FixedRedshiftHistory(
            channel.rho_src_msun_mpc3,
            0.0,
            "Traceability-only no-dilution convention",
        ),
    )


def history_sensitivity_audit(
    channels: dict[str, SpectralChannel],
) -> dict[str, object]:
    """Compare adopted explicit histories with traceability/sensitivity cases."""

    traceability: dict[str, dict[str, float]] = {}
    for key, channel in channels.items():
        if isinstance(channel.history, RateHistory):
            local = fixed_z0_copy(channel)
            traceability[key] = {
                "rho_src_msun_mpc3": local.rho_src_msun_mpc3,
                "hc_at_reference": local.hc_at_reference(),
                "explicit_history_hc_at_reference": channel.hc_at_reference(),
                "explicit_over_no_dilution_amplitude_ratio": (
                    channel.hc_at_reference() / local.hc_at_reference()
                ),
            }

    weighting = gwtc5_bbh_population_weighting()
    bbh_base = channels["bbh_benchmark"]

    def bbh_evolving_case(rate: float, kappa: float) -> SpectralChannel:
        return replace(
            bbh_base,
            history=RateHistory(
                rate,
                weighting["mean_total_mass_msun"],
                duration_gyr=10.0,
                rate_anchor_redshift=0.2,
                rate_evolution_kappa=kappa,
                description=(
                    "GWTC-5 rate-evolution sensitivity over 10 Gyr"
                ),
            ),
        )

    def bbh_case_outputs(case: SpectralChannel) -> dict[str, float]:
        """Return every reported scalar for one evolving-rate case."""

        return {
            "rho_src_msun_mpc3": case.rho_src_msun_mpc3,
            "source_fraction": case.rho_src_msun_mpc3 / RHO_STAR,
            "mean_energy_dilution": case.mean_energy_dilution,
            "hc_at_reference": case.hc_at_reference(),
            "omega_at_reference": float(
                case.omega(np.asarray([case.f_ref_hz]))[0]
            ),
            "present_integral_omega_dlnf": case.present_budget,
        }

    bbh_kappa_10gyr = bbh_evolving_case(
        GWTC5_DEFAULT_BBH_RATE_90[1],
        GWTC5_DEFAULT_BBH_KAPPA_90[1],
    )
    bbh_kappa_z3 = replace(
        bbh_base,
        history=RateHistory(
            30.4,
            weighting["mean_total_mass_msun"],
            duration_gyr=None,
            max_redshift=3.0,
            rate_anchor_redshift=0.2,
            rate_evolution_kappa=2.5,
            description="GWTC-5 central kappa_z extrapolated to z=3",
        ),
    )

    popiii = channels["popiii_benchmark"]
    popiii_ceiling = channels["popiii_ceiling"]
    popiii_delta_7e4 = replace(
        popiii,
        history=FixedRedshiftHistory(
            700.0,
            10.0,
            "Sensitivity: 7e4 inventory, one-percent mergers at z=10",
        ),
    )
    popiii_delta_1e5 = replace(
        popiii,
        history=FixedRedshiftHistory(
            1_000.0,
            10.0,
            "Sensitivity: 1e5 inventory, one-percent mergers at z=10",
        ),
    )

    bbh_base_omega = float(
        bbh_base.omega(np.asarray([bbh_base.f_ref_hz]))[0]
    )
    bbh_base_fraction = bbh_base.rho_src_msun_mpc3 / RHO_STAR
    stochastic_limit_mappings: dict[str, dict[str, float | str]] = {}
    for prior, omega_limit in {
        "log_uniform_amplitude": LVK_O4A_OMEGA_25_LOG_UNIFORM_95,
        "uniform_amplitude": LVK_O4A_OMEGA_25_UNIFORM_95,
    }.items():
        scale = omega_limit / bbh_base_omega
        stochastic_limit_mappings[prior] = {
            "coverage": "95-percent credible observational upper limit",
            "omega_25_upper_limit": omega_limit,
            "scale_relative_to_fixed_history_benchmark": scale,
            "source_fraction_upper_limit": bbh_base_fraction * scale,
            "rho_src_msun_mpc3_upper_limit": (
                bbh_base.rho_src_msun_mpc3 * scale
            ),
            "rate_gpc3_yr_upper_limit": (
                GWTC5_DEFAULT_BBH_RATE_90[1] * scale
            ),
        }

    return {
        "traceability_no_dilution": traceability,
        "bbh_rate_evolution": {
            "rate_interval_gpc3_yr": GWTC5_DEFAULT_BBH_RATE_90,
            "kappa_z_interval": GWTC5_DEFAULT_BBH_KAPPA_90,
            "ten_gyr": bbh_case_outputs(bbh_kappa_10gyr),
            "rate_only_at_median_kappa": {
                endpoint: {
                    "rate_gpc3_yr": rate,
                    **bbh_case_outputs(
                        bbh_evolving_case(
                            rate,
                            GWTC5_DEFAULT_BBH_KAPPA_90[1],
                        )
                    ),
                }
                for endpoint, rate in zip(
                    ("low", "median", "high"),
                    GWTC5_DEFAULT_BBH_RATE_90,
                )
            },
            "kappa_only_at_median_rate": {
                endpoint: {
                    "kappa_z": kappa,
                    **bbh_case_outputs(
                        bbh_evolving_case(
                            GWTC5_DEFAULT_BBH_RATE_90[1],
                            kappa,
                        )
                    ),
                }
                for endpoint, kappa in zip(
                    ("low", "median", "high"),
                    GWTC5_DEFAULT_BBH_KAPPA_90,
                )
            },
            "to_z3": {
                "rho_src_msun_mpc3": bbh_kappa_z3.rho_src_msun_mpc3,
                "hc_at_reference": bbh_kappa_z3.hc_at_reference(),
            },
            "status": (
                "One-at-a-time sensitivity ranges only; the marginal rate "
                "and kappa endpoints are not combined into a credible "
                "interval because their joint posterior is not propagated."
            ),
        },
        "bbh_stochastic_limit_inversion": {
            "scope": (
                "The full allowed f^(2/3) component is assigned to sBBHs "
                "and mapped with the representative binary, fixed spectrum, "
                "efficiency, and 10-Gyr constant-rate history."
            ),
            "statistical_status": (
                "Each row is one prior-specific 95-percent observational "
                "upper limit expressed in three equivalent units."
            ),
            "priors": stochastic_limit_mappings,
        },
        "popiii_inventory": {
            "liu_bromm_history": {
                "rho_iii_msun_mpc3": (
                    popiii.history.rho_iii_msun_mpc3
                    if isinstance(popiii.history, PopIIIFormationHistory)
                    else math.nan
                ),
                "rho_src_msun_mpc3": popiii.rho_src_msun_mpc3,
                "hc_at_reference": popiii.hc_at_reference(),
            },
            "planck_calibrated_ceiling": {
                "rho_iii_msun_mpc3": (
                    popiii_ceiling.history.rho_iii_msun_mpc3
                    if isinstance(
                        popiii_ceiling.history,
                        PopIIIFormationHistory,
                    )
                    else math.nan
                ),
                "rho_src_msun_mpc3": popiii_ceiling.rho_src_msun_mpc3,
                "hc_at_reference": popiii_ceiling.hc_at_reference(),
                "present_integral_omega_dlnf": popiii_ceiling.present_budget,
            },
            "delta_z10_rho_iii_7e4": {
                "rho_src_msun_mpc3": popiii_delta_7e4.rho_src_msun_mpc3,
                "hc_at_reference": popiii_delta_7e4.hc_at_reference(),
            },
            "delta_z10_rho_iii_1e5": {
                "rho_src_msun_mpc3": popiii_delta_1e5.rho_src_msun_mpc3,
                "hc_at_reference": popiii_delta_1e5.hc_at_reference(),
            },
        },
    }


def one_pass_audit(channels: dict[str, SpectralChannel]) -> dict[str, object]:
    """Record where the scalar one-pass bound is load-bearing."""

    adopted_input_fractions = {
        "smbhb_fixed_shape_benchmark_f_merge": 1.0,
        "emri_nsc_f_merge": 1.0e-4,
        "emri_nsc_closed_pool_f_merge": EMRI_CLOSED_POOL_F_MERGE,
        "agn_benchmark_kappa_times_fagn": (
            AGN_BENCHMARK_POINT.processing_fraction
        ),
        "agn_ceiling_kappa_times_fagn": AGN_CEILING_POINT.processing_fraction,
        "agn_ceiling_rho_over_broad_smbh_guardrail": (
            AGN_CEILING_POINT.rho_imr_msun_mpc3 / RHO_BH_LM24
        ),
        "bns_benchmark_rho_over_rho_star": (
            channels["bns_benchmark"].rho_src_msun_mpc3 / RHO_STAR
        ),
        "bns_high_side_rho_over_rho_star": (
            channels["bns_ceiling"].rho_src_msun_mpc3 / RHO_STAR
        ),
        "popiii_f_merge": 0.01,
        "bbh_benchmark_rho_over_rho_star": (
            channels["bbh_benchmark"].rho_src_msun_mpc3 / RHO_STAR
        ),
        "bbh_high_side_rho_over_rho_star": (
            channels["bbh_ceiling"].rho_src_msun_mpc3 / RHO_STAR
        ),
    }
    smbhb = smbhb_population_results()
    cnm_central_fmerge = smbhb["fmerge_cnm_central_fixed_shape"]
    cnm_high_fmerge = smbhb["fmerge_cnm_high_fixed_shape"]
    return {
        **adopted_input_fractions,
        "all_adopted_input_fractions_at_or_below_one": all(
            0.0 <= value <= 1.0 for value in adopted_input_fractions.values()
        ),
        "cnm_central_shape_conditional_f_merge": cnm_central_fmerge,
        "cnm_central_inside_one_pass_family": cnm_central_fmerge <= 1.0,
        "cnm_central_status": (
            "the central fixed-shape inference is approximately one merger "
            "per present-day remnant and lies inside the one-pass family"
        ),
        "cnm_high_shape_conditional_f_merge": cnm_high_fmerge,
        "cnm_high_inside_one_pass_family": cnm_high_fmerge <= 1.0,
        "cnm_high_status": (
            "the inferred high endpoint requires f_merge>1 and therefore "
            "lies outside the fixed-shape one-pass family"
        ),
        "hierarchy_scope": (
            "repeated generations are outside the one-pass construction; "
            "the SP24 SMBHB factor remains a separate extension"
        ),
    }


def spectral_diagnostics(
    channels: dict[str, SpectralChannel],
) -> dict[str, object]:
    """Per-channel checks used by the table and later figure regeneration."""

    comparison_frequency = 0.3
    comparison = {
        key: float(channel.omega(np.asarray([comparison_frequency]))[0])
        for key, channel in channels.items()
        if key in {"bns_benchmark", "popiii_benchmark", "bbh_benchmark"}
    }
    popiii_frequency = np.logspace(-3.0, 4.0, 10_000)
    popiii_omega = channels["popiii_benchmark"].omega(popiii_frequency)
    peak_index = int(np.argmax(popiii_omega))
    return {
        "omega_at_0p3_hz_by_channel": comparison,
        "popiii_peak_frequency_hz": float(popiii_frequency[peak_index]),
        "popiii_peak_omega": float(popiii_omega[peak_index]),
        "scope": (
            "inclusive total spectra; no post-subtraction residual is inferred"
        ),
    }


def build_results() -> dict[str, object]:
    """Build all Step-3 scalars from the approved channel ledger."""

    smbhb = smbhb_population_results()
    szq24_audit = szq24_population_moment_audit()
    channels = channel_inputs()
    rows: dict[str, object] = {}
    for key, channel in channels.items():
        redshift, mass = channel.history.mass_elements()
        mean_redshift = float(np.sum(redshift * mass) / np.sum(mass))
        frequencies = {}
        if channel.shape_kind == "phenomb":
            source_transitions = phenomb_frequencies(
                channel.total_mass_msun,
                channel.eta,
                channel.chi,
                0.0,
            )
            frequencies = {
                f"source_{name}": value
                for name, value in zip(
                    ("f_merg_hz", "f_ring_hz", "sigma_hz", "f_cut_hz"),
                    source_transitions,
                )
            }
            frequencies.update(
                {
                    f"at_mean_redshift_{name}": value / (1.0 + mean_redshift)
                    for name, value in zip(
                        ("f_merg_hz", "f_ring_hz", "sigma_hz", "f_cut_hz"),
                        source_transitions,
                    )
                }
            )
        rows[key] = {
            **asdict(channel),
            "history_type": type(channel.history).__name__,
            "rho_src_msun_mpc3": channel.rho_src_msun_mpc3,
            "redshift_min": float(np.min(redshift)),
            "redshift_max": float(np.max(redshift)),
            "mass_weighted_mean_redshift": mean_redshift,
            "mean_energy_dilution": channel.mean_energy_dilution,
            "present_budget": channel.present_budget,
            "hc_at_reference": channel.hc_at_reference(),
            "omega_at_reference": float(
                channel.omega(np.asarray([channel.f_ref_hz]))[0]
            ),
            **frequencies,
        }

    # Independent efficiency checks.
    bbh_weighting = gwtc5_bbh_population_weighting()
    bns_eps_exact = peters_inspiral_efficiency(
        total_mass_msun=2.8,
        eta=0.25,
        source_cutoff_hz=1_500.0,
    )
    uib_nonspinning_equal_mass = float(
        jimenez_forteza_nonspinning_radiated_fraction(0.25)
    )
    isco_support = smbhb_isco_support_audit(0.0)

    return {
        "constants": {
            "H0_km_s_Mpc": H0_KM_S_MPC,
            "Omega_m": OMEGA_M,
            "Omega_Lambda": OMEGA_LAMBDA,
            "rho_crit_msun_mpc3": RHO_CRIT,
            "rho_star_msun_mpc3": RHO_STAR,
            "rho_bh_lm24_msun_mpc3": RHO_BH_LM24,
            "rho_star_over_rho_bh_lm24": RHO_STAR / RHO_BH_LM24,
        },
        "core_physics_audit": {
            "peters_bns_equal_mass_to_1500_hz_efficiency": bns_eps_exact,
            "schwarzschild_isco_radius_gm_c2": kerr_isco_radius(0.0),
            "kerr_chi0p7_isco_radius_gm_c2": kerr_isco_radius(0.7),
            "kerr_chi0p7_mass_for_400_nhz_msun": (
                mass_for_kerr_isco_frequency(400.0e-9, 0.7)
            ),
            "smbhb_schwarzschild_isco_support_at_1yr": isco_support,
        },
        "smbhb": smbhb,
        "szq24_population_moment_audit": szq24_audit,
        "channels": rows,
        "efficiency_checks": {
            "bns_inspiral_to_1500_hz": bns_eps_exact,
            "bbh_population_weighting": bbh_weighting,
            "bbh_benchmark_adopted": 0.05,
            "jimenez_forteza_nonspinning_equal_mass": (
                uib_nonspinning_equal_mass
            ),
            "aligned_spin_nr_fit": (
                "Husa et al. 2016 dependency-free IMRPhenomD fit; "
                "the Jimenez-Forteza et al. 2016 nonspinning polynomial "
                "is retained as an independent equal-mass cross-check"
            ),
            "bbh_q1_chi0p7_NR": bbh_weighting[
                "equal_mass_chi0p7_efficiency"
            ],
            "capture_schwarzschild_test_particle": (
                1.0 - math.sqrt(8.0 / 9.0)
            ),
            "capture_adopted": 0.05,
        },
        "nsc_density_audit": nsc_density_audit(),
        "emri_rate_proxy_audit": emri_rate_proxy_audit(),
        "agn_pipeline_audit": agn_pipeline_audit(),
        "rate_only_benchmark_intervals": (
            rate_only_benchmark_interval_results(channels)
        ),
        "history_sensitivity_audit": history_sensitivity_audit(channels),
        "one_pass_audit": one_pass_audit(channels),
        "spectral_diagnostics": spectral_diagnostics(channels),
        "detector_context_audit": detector_context_audit(),
        "step3b_integrated_energy": step3b_integrated_energy_results(),
        "interpretation": {
            "aggregation": (
                "Step 3B emits both the declared benchmark sum and a sum "
                "formed from the five theoretical-upper-limit cases plus "
                "one equal-mass SMBHB merger per present-day remnant. The "
                "sum excludes the repeated-merger fixed-frequency factor. "
                "The cases were calculated separately and need not occur "
                "together, so the sum is not a physical ceiling."
            ),
            "smbhb_interval": (
                "CNM-to-fmerge mapping is conditional on the central LM24 "
                "full-census shape, mass-ratio distribution, and redshift "
                "kernel; no additional primary-mass selection is imposed"
            ),
            "smbhb_provenance": (
                "2.11e-15 is derived from the full convergent moment of the "
                "erratum-corrected analytic LM24 fit normalized to the "
                "1.8e6 Msun/Mpc^3 census, with no additional primary-mass cut; "
                "it is consistent with LM24's published 2.0e-15 forward result"
            ),
            "lvk_history": (
                "constant comoving source-frame rate over a 10-Gyr lookback "
                "interval, with redshift dilution and shifted support explicit"
            ),
            "popiii": (
                "Liu--Bromm SFRD plus a declared prompt one-percent merger "
                "mapping; the ceiling uses the Planck-calibrated mass "
                "endpoint with other benchmark inputs fixed"
            ),
            "population_scope": (
                "inclusive total populations; no unresolved/subtraction "
                "forecast is computed without a selection model"
            ),
            "bbh_population_proxy": (
                "21.687 Msun and 0.04708 come from a deterministic "
                "GWTC-5-inspired approximation, not posterior propagation"
            ),
        },
    }


def build_spectra(
    f_grid: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return the common grid and all emitted astrophysical spectra."""

    if f_grid is None:
        f_grid = np.logspace(-12.0, 4.3, 12_000)
    f = np.asarray(f_grid, dtype=float)
    if np.any(f <= 0.0) or np.any(np.diff(f) <= 0.0):
        raise ValueError("Frequency grid must be positive and strictly increasing.")

    spectra: dict[str, np.ndarray] = {}
    for key, channel in channel_inputs().items():
        output_key = (
            "bbh_ceiling_realizable_chi0p7"
            if key == "bbh_ceiling"
            else key
        )
        spectra[output_key] = channel.omega(f)
    # The chi=0.7 channel fixes the tabulated 25-Hz endpoint.  The plotted
    # conditional ceiling is the frequencywise envelope over 0 <= chi <= 0.7.
    pointwise_envelope = bbh_aligned_spin_ceiling(f)
    spectra["bbh_ceiling_pointwise_envelope"] = pointwise_envelope
    # Compatibility alias for the existing figure source.  It is explicitly
    # not the realizable chi=0.7 spectrum and must not be energy-integrated.
    spectra["bbh_ceiling"] = pointwise_envelope

    # Only the declared PTA display segment of the full-census SMBHB model is
    # emitted.  The zeros outside this interval are an archive/display boundary,
    # not physical source support.  In particular, these arrays must never be
    # integrated to recover the analytic scalar SMBHB energies.
    smbhb_display = np.zeros_like(f)
    smbhb_ceiling_display = np.zeros_like(f)
    display_mask = (
        (f >= SMBHB_PTA_DISPLAY_MIN_HZ)
        & (f <= SMBHB_PTA_DISPLAY_MAX_HZ)
    )
    if np.any(display_mask):
        smbhb_display[display_mask] = smbhb_population_omega(
            f[display_mask],
            "benchmark",
        )
        smbhb_ceiling_display[display_mask] = smbhb_population_omega(
            f[display_mask],
            "conditional_ceiling",
        )
    spectra["smbhb_benchmark"] = smbhb_display
    spectra["smbhb_conditional_ceiling"] = smbhb_ceiling_display

    # Broadband SMBHB aliases, cross-channel composites, and cumulative
    # products are deliberately retired.  Their scalar normalizations require
    # the full analytic LM24 moments, while the archive exposes only the PTA
    # display segment.

    return f, spectra


def verify_normalizations() -> dict[str, float]:
    """Numerically verify each realizable channel's integrated energy."""

    f = np.logspace(-14.0, 5.0, 50_000)
    residuals: dict[str, float] = {}
    for key, channel in channel_inputs().items():
        omega_integral = trapz_log(channel.omega(f), f)
        budget_residual = omega_integral / channel.present_budget - 1.0
        if abs(budget_residual) > 2.5e-3:
            raise RuntimeError(
                f"{key} normalization failed: "
                f"integral/budget={omega_integral/channel.present_budget}"
            )
        residuals[f"{key}_omega_over_B_minus_1"] = budget_residual

    # SMBHB scalar energies are analytic full-census moments.  The emitted
    # SMBHB arrays are deliberately PTA-display-only, so neither they nor a
    # cross-channel composite belong in this numerical normalization check.
    return residuals


def print_report(results: dict[str, object], residuals: dict[str, float]) -> None:
    """Print the concise decision ledger used to update the manuscript."""

    constants = results["constants"]
    smbhb = results["smbhb"]
    szq24 = results["szq24_population_moment_audit"]
    channels = results["channels"]
    checks = results["efficiency_checks"]
    weighting = checks["bbh_population_weighting"]

    print("PAPER-I ENERGETIC CEILINGS: STEP-3 VERIFICATION RESULTS")
    print("=" * 78)
    print(f"rho_c       = {constants['rho_crit_msun_mpc3']:.6e} Msun Mpc^-3")
    print(f"rho_star    = {constants['rho_star_msun_mpc3']:.6e} Msun Mpc^-3")
    print(f"rho_BH LM24 = {constants['rho_bh_lm24_msun_mpc3']:.6e} Msun Mpc^-3")
    print()
    print("SMBHB population moment (fixed central LM24 shape)")
    print(f"  A_benchmark(1/yr)              = {smbhb['benchmark_A_1yr']:.6e}")
    print(
        f"  A_equal_mass_z0(q={smbhb['conditional_ceiling_mass_ratio']:.1f})"
        "      = "
        f"{smbhb['equal_mass_latest_generation_A_1yr']:.6e}"
    )
    print(
        "  A_repeated_merger_ceiling       = "
        f"{smbhb['repeated_merger_ceiling_A_1yr']:.6e} "
        f"(h_c^2 boost={smbhb['szq_hierarchy_hc2_boost']:.6f})"
    )
    print(
        f"  CNM central -> f_merge          = "
        f"{smbhb['fmerge_cnm_central_fixed_shape']:.4f}"
    )
    print(
        f"  CNM 90% band -> f_merge         = "
        f"[{smbhb['fmerge_cnm_low_fixed_shape']:.3f}, "
        f"{smbhb['fmerge_cnm_high_fixed_shape']:.3f}] (fixed shape)"
    )
    print(
        f"  f_merge rho_BH correspondence   = "
        f"{smbhb['shape_conditional_product_msun_mpc3']:.4e} Msun Mpc^-3"
    )
    print()
    print("SZQ24 independent population-moment audit")
    print(
        "  A_fiducial(1/yr)               = "
        f"{szq24['fiducial_log_uniform_one_pass_A_1yr']:.6e}"
    )
    print(
        "  A_equal_mass_z0(1/yr)          = "
        f"{szq24['equal_mass_z0_one_pass_A_1yr']:.6e}"
    )
    print(
        "  A_equal_mass_z0_plus_hierarchy = "
        f"{szq24['equal_mass_z0_infinite_A_1yr']:.6e}"
    )
    print(
        "  literal Eq. (21) / direct       = "
        f"{szq24['literal_eq21_over_direct_strain_ratio']:.4f} in strain"
    )
    print()
    print("Other channels")
    print(
        f"{'key':<24s} {'kind':<9s} {'rho_src':>11s} {'eps':>7s} "
        f"{'<1/(1+z)>':>11s} {'h_c(fref)':>13s}"
    )
    for key, row in channels.items():
        print(
            f"{key:<24s} "
            f"{'benchmark' if row['benchmark_or_ceiling'] == 'benchmark' else 'upper':<9s} "
            f"{row['rho_src_msun_mpc3']:11.4e} {row['epsilon']:7.4f} "
            f"{row['mean_energy_dilution']:11.4f} "
            f"{row['hc_at_reference']:13.4e}"
        )
    print()
    print("Efficiency checks")
    print(f"  BNS to 1500 Hz              = {checks['bns_inspiral_to_1500_hz']:.5f}")
    print(
        f"  BBH rate-weighted <Mtot>    = "
        f"{weighting['mean_total_mass_msun']:.3f} Msun"
    )
    print(
        f"  BBH mass-weighted efficiency= "
        f"{weighting['mass_weighted_mean_efficiency']:.5f}"
    )
    print(
        f"  q=1 nonspinning NR fit      = "
        f"{weighting['equal_mass_nonspinning_efficiency']:.5f}"
    )
    print(f"  q=1, chi=0.7 NR fit         = {checks['bbh_q1_chi0p7_NR']:.5f}")
    print(
        f"  Schwarzschild capture       = "
        f"{checks['capture_schwarzschild_test_particle']:.5f}"
    )
    print()
    nsc = results["nsc_density_audit"]
    print("Deferred-source checks")
    print(
        f"  eta_NSC unit-occ., median   = "
        f"{nsc['eta_nsc_unit_occupation_median_relation']:.4e}"
    )
    print(
        f"  eta_NSC with scatter mean   = "
        f"{nsc['eta_nsc_unit_occupation_scatter_mean']:.4e}"
    )
    agn = results["agn_pipeline_audit"]
    print(
        f"  AGN pipeline benchmark/max  = "
        f"{agn['benchmark_rho_imr_msun_mpc3']:.1f} / "
        f"{agn['maximum_rho_imr_msun_mpc3']:.1f} Msun Mpc^-3"
    )
    history = results["history_sensitivity_audit"]
    print(
        f"  Pop-III Liu--Bromm rho_III  = "
        f"{history['popiii_inventory']['liu_bromm_history']['rho_iii_msun_mpc3']:.4e} "
        "Msun Mpc^-3"
    )
    print(
        f"  BBH kappa_z=2.5, z<=3 hc    = "
        f"{history['bbh_rate_evolution']['to_z3']['hc_at_reference']:.4e}"
    )
    print()
    step3b = results["step3b_integrated_energy"]
    print("Step 3B integrated-energy diagnostic")
    print(
        "  SMBHB LM24 inspiral integral = "
        f"{step3b['channels']['smbhb_benchmark']['present_integral_omega_dlnf']:.6e}"
    )
    print(
        "  benchmark sum                = "
        f"{step3b['conditional_benchmark_sum_omega_dlnf']:.6e}"
    )
    print(
        "  stated-input upper-limit sum = "
        f"{step3b['conditional_ceiling']['sum_omega_dlnf']:.6e}"
    )
    print(
        "  status: one SMBHB generation; repeated-merger factor excluded"
    )
    print(
        f"Maximum absolute shape/integral residual = "
        f"{max(abs(value) for value in residuals.values()):.3e}"
    )


def json_default(value: object) -> object:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="Write scalar results and checks.")
    parser.add_argument("--npz", type=Path, help="Write the common spectrum arrays.")
    args = parser.parse_args()

    results = build_results()
    residuals = verify_normalizations()
    print_report(results, residuals)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = {**results, "normalization_residuals": residuals}
        args.json.write_text(
            json.dumps(payload, indent=2, default=json_default) + "\n",
            encoding="utf-8",
        )
        print(f"[wrote {args.json}]")

    if args.npz:
        args.npz.parent.mkdir(parents=True, exist_ok=True)
        f, spectra = build_spectra()
        np.savez(args.npz, frequency_hz=f, **spectra)
        print(f"[wrote {args.npz}]")


if __name__ == "__main__":
    main()
