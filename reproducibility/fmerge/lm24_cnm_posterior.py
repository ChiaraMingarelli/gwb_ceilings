#!/usr/bin/env python3
"""Propagate LM24 population uncertainty and the NG15 CNM amplitude to f_merge.

This script is the reproducible SMBHB merger-factor calculation of
Mingarelli (2026), "Energetic Ceilings and Benchmarks of Astrophysical
Gravitational-Wave Backgrounds" (arXiv:2601.18859).  Its output underlies
Eq. (15) of the paper, f_merge = 1.11 (+1.35, -0.63) at 90% credibility,
and the distribution shown in Figure 2 (fmerge-posterior.pdf).

It reconstructs the *approximate* Liepold & Ma (2024; LM24) galaxy stellar
mass-function posterior described in their Appendix B, propagates every draw
through the McConnell & Ma (2013) black-hole--bulge relation, and evaluates
the full-census Phinney population moment at f = 1/yr.  It then maps
independent draws of the NANOGrav 15-year customized-noise-model (CNM)
amplitude through Eq. (14) of the paper,

    f_merge = (A_CNM / A_model[f_merge=1])**2.

Here f_merge (dimensionless) rescales the processed SMBHB mass density
while the population shape is held fixed; the amplitudes A are
characteristic-strain amplitudes at f = 1/yr, and mass densities are in
Msun Mpc^-3.

The manuscript calculation uses only published summary information.  The
LM24 input is its Appendix-B approximation, and the CNM input is its published
median and 90% bounds, treated as the 5th, 50th, and 95th percentiles.  The
default output is therefore an *approximate* posterior under an explicitly
declared log-uniform prior on f_merge.  Optional one-dimensional input
distributions, including importance weights, can be supplied for other uses.

Primary references
------------------
LM24 paper and Appendix B: https://doi.org/10.3847/2041-8213/ad66b8
LM24 author helper code:   https://github.com/EmilyLiepold/LM24_helpers
LM24 erratum:              https://doi.org/10.3847/2041-8213/ae5f96
McConnell & Ma (2013):     https://doi.org/10.1088/0004-637X/764/2/184
NANOGrav CNM analysis:     https://arxiv.org/abs/2606.28554

The implementation below is derived directly from the published equations
and tabulated parameters; it does not import or copy the LM24 helper package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


# Exact/standard SI constants used by the authoritative calculation.
G_SI = 6.67430e-11
C_SI = 299_792_458.0
MSUN_SI = 1.98847e30
MPC_SI = 3.0856775814913673e22
YEAR_SI = 365.25 * 86_400.0
F_YEAR_HZ = 1.0 / YEAR_SI

LN10 = math.log(10.0)
PHI_FLOOR_MPC3 = 1.0e-10
CALCULATION_VERSION = "0.4.0"


def normal_ppf(probabilities: np.ndarray | float) -> np.ndarray:
    """Vectorized inverse standard-normal CDF (Acklam approximation)."""

    p = np.asarray(probabilities, dtype=float)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("normal-quantile probabilities must lie in (0, 1)")

    a = np.asarray(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )
    b = np.asarray(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )
    c = np.asarray(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )
    d = np.asarray(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    result = np.empty_like(p)
    lower_limit = 0.02425
    upper_limit = 1.0 - lower_limit

    lower = p < lower_limit
    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        result[lower] = (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )

    central = (p >= lower_limit) & (p <= upper_limit)
    if np.any(central):
        q = p[central] - 0.5
        r = q**2
        result[central] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    upper = p > upper_limit
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        result[upper] = -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    return result


@dataclass(frozen=True)
class GaussianParameter:
    """Mean and one-standard-deviation width of a Gaussian parameter."""

    mean: float
    sigma: float


@dataclass(frozen=True)
class GSMFBranch:
    """LM24 Appendix-B approximation for one stellar-mass calibration."""

    name: str
    log10_phi_low_mpc3_dex: GaussianParameter
    log10_phi_high_mpc3_dex: GaussianParameter
    alpha_high_mass: GaussianParameter
    alpha_low_mass: GaussianParameter
    log10_mscale_msun: GaussianParameter

    def means(self) -> np.ndarray:
        return np.asarray(
            [
                self.log10_phi_low_mpc3_dex.mean,
                self.log10_phi_high_mpc3_dex.mean,
                self.alpha_high_mass.mean,
                self.alpha_low_mass.mean,
                self.log10_mscale_msun.mean,
            ],
            dtype=float,
        )

    def sigmas(self) -> np.ndarray:
        return np.asarray(
            [
                self.log10_phi_low_mpc3_dex.sigma,
                self.log10_phi_high_mpc3_dex.sigma,
                self.alpha_high_mass.sigma,
                self.alpha_low_mass.sigma,
                self.log10_mscale_msun.sigma,
            ],
            dtype=float,
        )


# Table B1, separate dynamical-Mstar and SPS-Mstar columns.  LM24 define the
# fiducial population as an equal mixture of these two posterior branches.
LM24_DYNAMICAL = GSMFBranch(
    name="dynamical",
    log10_phi_low_mpc3_dex=GaussianParameter(-2.01, 0.03),
    log10_phi_high_mpc3_dex=GaussianParameter(-4.45, 0.09),
    alpha_high_mass=GaussianParameter(1.08, 0.77),
    alpha_low_mass=GaussianParameter(-1.38, 0.04),
    log10_mscale_msun=GaussianParameter(11.34, 0.06),
)

LM24_SPS = GSMFBranch(
    name="sps",
    log10_phi_low_mpc3_dex=GaussianParameter(-2.01, 0.03),
    log10_phi_high_mpc3_dex=GaussianParameter(-4.59, 0.11),
    alpha_high_mass=GaussianParameter(0.84, 0.96),
    alpha_low_mass=GaussianParameter(-1.37, 0.04),
    log10_mscale_msun=GaussianParameter(11.31, 0.06),
)


@dataclass(frozen=True)
class PropagationConfig:
    """All numerical and astrophysical choices needed for the calculation."""

    sample_count: int = 262_144
    seed: int = 20_240_724
    batch_size: int = 512
    stellar_quadrature_nodes: int = 96
    mass_ratio_quadrature_nodes: int = 40
    redshift_quadrature_nodes: int = 96
    log10_stellar_mass_min: float = 8.5
    log10_stellar_mass_max: float = 12.0
    mass_ratio_min: float = 0.1
    mass_ratio_max: float = 1.0
    scaling_intercept_mean: float = 8.46
    scaling_intercept_sigma: float = 0.08
    scaling_slope_mean: float = 1.05
    scaling_slope_sigma: float = 0.11
    # LM24 publish 0.34 dex; the later author helper rounds this to 0.35 dex.
    scaling_intrinsic_scatter_dex: float = 0.34
    cnm_amplitude_median: float = 2.10e-15
    cnm_amplitude_lower_90: float = 1.60e-15
    cnm_amplitude_upper_90: float = 2.70e-15
    cnm_log10_amplitude_prior_min: float = -18.0
    cnm_log10_amplitude_prior_max: float = -14.0
    log10_f_prior_min: float = -3.0
    log10_f_prior_max: float = 1.0

    def validate(self) -> None:
        if self.sample_count < 2:
            raise ValueError("sample_count must be at least two")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if min(
            self.stellar_quadrature_nodes,
            self.mass_ratio_quadrature_nodes,
            self.redshift_quadrature_nodes,
        ) < 2:
            raise ValueError("every quadrature node count must be at least two")
        if self.log10_stellar_mass_min >= self.log10_stellar_mass_max:
            raise ValueError("stellar-mass integration bounds are reversed")
        if not (0.0 < self.mass_ratio_min < self.mass_ratio_max <= 1.0):
            raise ValueError("mass-ratio bounds must satisfy 0 < qmin < qmax <= 1")
        if self.scaling_intrinsic_scatter_dex <= 0.0:
            raise ValueError("intrinsic scatter must be positive")
        if not (
            0.0
            < self.cnm_amplitude_lower_90
            < self.cnm_amplitude_median
            < self.cnm_amplitude_upper_90
        ):
            raise ValueError("CNM amplitude quantiles must be positive and ordered")
        if (
            self.cnm_log10_amplitude_prior_min
            >= self.cnm_log10_amplitude_prior_max
        ):
            raise ValueError("CNM amplitude prior bounds are reversed")
        amplitude_prior_min = 10.0**self.cnm_log10_amplitude_prior_min
        amplitude_prior_max = 10.0**self.cnm_log10_amplitude_prior_max
        if not (
            amplitude_prior_min
            < self.cnm_amplitude_lower_90
            < self.cnm_amplitude_upper_90
            < amplitude_prior_max
        ):
            raise ValueError("CNM amplitude quantiles lie outside its prior")
        if self.log10_f_prior_min >= self.log10_f_prior_max:
            raise ValueError("f_merge prior bounds are reversed")


@dataclass(frozen=True)
class PopulationDraws:
    """Independent draws of the LM24 GSMF and black-hole scaling relation."""

    branch: np.ndarray
    phi1_mpc3: np.ndarray
    phi2_mpc3: np.ndarray
    alpha1: np.ndarray
    alpha2: np.ndarray
    log10_mscale_msun: np.ndarray
    scaling_intercept: np.ndarray
    scaling_slope: np.ndarray
    intrinsic_scatter_dex: np.ndarray
    phi1_floored: np.ndarray
    phi2_floored: np.ndarray


@dataclass(frozen=True)
class PopulationMoments:
    """Paired outputs from every reconstructed LM24 population realization."""

    rho_total_msun_mpc3: np.ndarray
    amplitude_uncut: np.ndarray
    redshift_moment: float
    eta_moment: float


@dataclass(frozen=True)
class ExternalAmplitudeSamples:
    """Validated amplitude samples and optional importance weights."""

    values: np.ndarray
    weights: np.ndarray
    sha256: str
    value_key: str
    weight_key: str | None


def _mapped_legendre_nodes(
    lower: float, upper: float, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss--Legendre nodes and weights mapped to [lower, upper]."""

    if count < 2:
        raise ValueError("quadrature node count must be at least two")
    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(count)
    half_width = 0.5 * (upper - lower)
    midpoint = 0.5 * (upper + lower)
    return midpoint + half_width * raw_nodes, half_width * raw_weights


def solve_double_schechter_amplitudes(
    phi_low_per_dex: np.ndarray,
    phi_high_per_dex: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    log10_mscale_msun: np.ndarray,
    log10_mass_low: float = 10.0,
    log10_mass_high: float = 12.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Map GSMF values at two masses to the double-Schechter amplitudes.

    LM24 write the GSMF as dn/dln(Mstar).  Their reference amplitudes phi_L
    and phi_H are dn/dlog10(Mstar), which supplies the factor ln(10) below.
    """

    t_low = 10.0 ** (log10_mass_low - log10_mscale_msun)
    t_high = 10.0 ** (log10_mass_high - log10_mscale_msun)

    l1_low = np.exp(-t_low) * t_low ** (alpha1 + 1.0)
    l1_high = np.exp(-t_high) * t_high ** (alpha1 + 1.0)
    l2_low = np.exp(-t_low) * t_low ** (alpha2 + 1.0)
    l2_high = np.exp(-t_high) * t_high ** (alpha2 + 1.0)

    denominator = LN10 * (l1_low * l2_high - l1_high * l2_low)
    phi1 = (l2_high * phi_low_per_dex - l2_low * phi_high_per_dex) / denominator
    phi2 = (l1_low * phi_high_per_dex - l1_high * phi_low_per_dex) / denominator
    return phi1, phi2


def evaluate_gsmf_per_dex(
    log10_stellar_mass: np.ndarray,
    phi1_mpc3: np.ndarray,
    phi2_mpc3: np.ndarray,
    alpha1: np.ndarray,
    alpha2: np.ndarray,
    log10_mscale_msun: np.ndarray,
) -> np.ndarray:
    """Evaluate dn/dlog10(Mstar) for batched double-Schechter draws."""

    log_x = LN10 * (
        log10_stellar_mass[None, :] - log10_mscale_msun[:, None]
    )
    x = np.exp(log_x)
    term1 = phi1_mpc3[:, None] * np.exp((alpha1[:, None] + 1.0) * log_x)
    term2 = phi2_mpc3[:, None] * np.exp((alpha2[:, None] + 1.0) * log_x)
    return LN10 * (term1 + term2) * np.exp(-x)


def draw_lm24_population(
    config: PropagationConfig,
    gsmf_rng: np.random.Generator,
    scaling_rng: np.random.Generator,
) -> PopulationDraws:
    """Draw the LM24 Appendix-B approximation and independent MM13 parameters."""

    count = config.sample_count
    branch = np.arange(count, dtype=np.int8) % 2
    transformed = np.empty((count, 5), dtype=float)

    for branch_index, definition in enumerate((LM24_DYNAMICAL, LM24_SPS)):
        mask = branch == branch_index
        transformed[mask] = gsmf_rng.normal(
            loc=definition.means(),
            scale=definition.sigmas(),
            size=(int(np.count_nonzero(mask)), 5),
        )

    phi1, phi2 = solve_double_schechter_amplitudes(
        10.0 ** transformed[:, 0],
        10.0 ** transformed[:, 1],
        transformed[:, 2],
        transformed[:, 3],
        transformed[:, 4],
    )
    if np.any(~np.isfinite(phi1)) or np.any(~np.isfinite(phi2)):
        raise FloatingPointError("non-finite Schechter amplitude in LM24 draw")

    phi1_floored = phi1 < 0.0
    phi2_floored = phi2 < 0.0
    phi1 = np.where(phi1_floored, PHI_FLOOR_MPC3, phi1)
    phi2 = np.where(phi2_floored, PHI_FLOOR_MPC3, phi2)

    scaling_intercept = scaling_rng.normal(
        config.scaling_intercept_mean,
        config.scaling_intercept_sigma,
        size=count,
    )
    scaling_slope = scaling_rng.normal(
        config.scaling_slope_mean,
        config.scaling_slope_sigma,
        size=count,
    )

    return PopulationDraws(
        branch=branch,
        phi1_mpc3=phi1,
        phi2_mpc3=phi2,
        alpha1=transformed[:, 2],
        alpha2=transformed[:, 3],
        log10_mscale_msun=transformed[:, 4],
        scaling_intercept=scaling_intercept,
        scaling_slope=scaling_slope,
        intrinsic_scatter_dex=np.full(
            count, config.scaling_intrinsic_scatter_dex, dtype=float
        ),
        phi1_floored=phi1_floored,
        phi2_floored=phi2_floored,
    )


def lm24_redshift_moment(config: PropagationConfig) -> float:
    """Return <(1+z)^(-1/3)> for p(z) proportional to z exp[-(z/0.5)^2]."""

    z, weights = _mapped_legendre_nodes(
        0.0, 6.0, config.redshift_quadrature_nodes
    )
    density = z * np.exp(-((z / 0.5) ** 2))
    return float(
        np.sum(weights * density * (1.0 + z) ** (-1.0 / 3.0))
        / np.sum(weights * density)
    )


def strain_from_weighted_i53(
    weighted_i53_msun53_mpc3: np.ndarray,
    redshift_moment: float,
    frequency_hz: float = F_YEAR_HZ,
) -> np.ndarray:
    """Evaluate the circular-inspiral Phinney population moment."""

    i53_si = (
        np.asarray(weighted_i53_msun53_mpc3)
        * MSUN_SI ** (5.0 / 3.0)
        / MPC_SI**3
    )
    hc2 = (
        4.0
        / (3.0 * math.pi ** (1.0 / 3.0))
        * G_SI ** (5.0 / 3.0)
        / C_SI**2
        * frequency_hz ** (-4.0 / 3.0)
        * redshift_moment
        * i53_si
    )
    return np.sqrt(hc2)


def compute_population_moments(
    draws: PopulationDraws, config: PropagationConfig
) -> PopulationMoments:
    """Propagate every LM24 draw through the full-census population moments."""

    log_mstar, stellar_weights = _mapped_legendre_nodes(
        config.log10_stellar_mass_min,
        config.log10_stellar_mass_max,
        config.stellar_quadrature_nodes,
    )
    q, q_weights = _mapped_legendre_nodes(
        config.mass_ratio_min,
        config.mass_ratio_max,
        config.mass_ratio_quadrature_nodes,
    )
    q_normalization = (
        config.mass_ratio_max**3 - config.mass_ratio_min**3
    ) / 3.0
    probability_q_weights = q_weights * q**2 / q_normalization
    eta = q / (1.0 + q) ** 2
    eta_moment = float(np.sum(probability_q_weights * eta))

    count = draws.branch.size
    rho_total = np.empty(count, dtype=float)
    i53_uncut_weighted = np.empty(count, dtype=float)

    for start in range(0, count, config.batch_size):
        stop = min(start + config.batch_size, count)
        batch = slice(start, stop)
        dndlog_mstar = evaluate_gsmf_per_dex(
            log_mstar,
            draws.phi1_mpc3[batch],
            draws.phi2_mpc3[batch],
            draws.alpha1[batch],
            draws.alpha2[batch],
            draws.log10_mscale_msun[batch],
        )
        mean_ln_bh_mass = LN10 * (
            draws.scaling_intercept[batch, None]
            + draws.scaling_slope[batch, None] * (log_mstar[None, :] - 11.0)
        )
        sigma_ln_bh_mass = (
            LN10 * draws.intrinsic_scatter_dex[batch, None]
        )

        def integrate_power(power: float) -> np.ndarray:
            sigma2 = sigma_ln_bh_mass**2
            complete = np.exp(
                power * mean_ln_bh_mass
                + 0.5 * power**2 * sigma2
            )
            return np.einsum(
                "bp,bp,p->b",
                dndlog_mstar,
                complete,
                stellar_weights,
                optimize=True,
            )

        rho_total[batch] = integrate_power(1.0)
        i53_uncut_weighted[batch] = eta_moment * integrate_power(5.0 / 3.0)

    redshift_moment = lm24_redshift_moment(config)
    return PopulationMoments(
        rho_total_msun_mpc3=rho_total,
        amplitude_uncut=strain_from_weighted_i53(
            i53_uncut_weighted, redshift_moment
        ),
        redshift_moment=redshift_moment,
        eta_moment=eta_moment,
    )


def split_lognormal_quantile_draws(
    lower_90: float,
    median: float,
    upper_90: float,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a log-space two-piece normal matching 5/50/95% quantiles.

    Midpoint-stratified probabilities reduce Monte Carlo noise and cannot hit
    the singular endpoints of the inverse CDF.  The final shuffle prevents an
    artificial rank correlation when paired with model draws.
    """

    if not (0.0 < lower_90 < median < upper_90):
        raise ValueError("split-lognormal quantiles must be positive and ordered")
    probabilities = (np.arange(count, dtype=float) + 0.5) / count
    log_median = math.log(median)
    normal_95 = float(normal_ppf(0.95))
    sigma_lower = math.log(median / lower_90) / normal_95
    sigma_upper = math.log(upper_90 / median) / normal_95
    sigma = np.where(probabilities < 0.5, sigma_lower, sigma_upper)
    samples = np.exp(log_median + sigma * normal_ppf(probabilities))
    rng.shuffle(samples)
    return samples


def sha256_file(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_weighted_amplitude_samples(
    path: Path, preferred_key: str
) -> ExternalAmplitudeSamples:
    """Load a positive 1-D amplitude chain and optional importance weights.

    NPZ archives may contain ``preferred_key`` (or ``amplitude``) and an
    optional ``weight``/``weights`` array.  Other formats are interpreted as
    unweighted amplitudes.  Invalid entries are rejected rather than silently
    removed, so problems in a source chain surface immediately and can be
    traced back to the input file.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    value_key = "array"
    weight_key: str | None = None
    if path.suffix == ".npy":
        values = np.load(path, allow_pickle=False)
        weights = None
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            candidates = [preferred_key, "amplitude", "amplitude_yr"]
            value_key = next(
                (key for key in candidates if key in archive.files), ""
            )
            weight_keys = [
                key for key in ("weight", "weights") if key in archive.files
            ]
            nonweight_keys = [
                key for key in archive.files if key not in {"weight", "weights"}
            ]
            if not value_key and len(nonweight_keys) == 1:
                value_key = nonweight_keys[0]
            if not value_key:
                raise ValueError(
                    f"{path} has multiple arrays; expected key {preferred_key!r}, "
                    "'amplitude', or 'amplitude_yr'"
                )
            if len(weight_keys) > 1:
                raise ValueError(f"{path} contains both 'weight' and 'weights'")
            values = np.asarray(archive[value_key])
            if weight_keys:
                weight_key = weight_keys[0]
                weights = np.asarray(archive[weight_key])
            else:
                weights = None
    else:
        delimiter = "," if path.suffix == ".csv" else None
        values = np.loadtxt(path, delimiter=delimiter)
        weights = None

    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError(
            f"{path} must contain a one-dimensional amplitude chain; "
            "free-spectrum tables are not valid inputs"
        )
    if values.size < 2:
        raise ValueError(f"{path} does not contain at least two samples")
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{path} amplitudes must all be finite and positive")

    if weights is None:
        validated_weights = np.ones(values.size, dtype=float)
    else:
        validated_weights = np.asarray(weights, dtype=float)
        if validated_weights.ndim != 1 or validated_weights.size != values.size:
            raise ValueError(
                f"{path} weights must be one-dimensional and match amplitudes"
            )
        if (
            np.any(~np.isfinite(validated_weights))
            or np.any(validated_weights < 0.0)
            or not np.any(validated_weights > 0.0)
        ):
            raise ValueError(
                f"{path} weights must be finite, nonnegative, and not all zero"
            )

    return ExternalAmplitudeSamples(
        values=values,
        weights=validated_weights,
        sha256=sha256_file(path),
        value_key=value_key,
        weight_key=weight_key,
    )


def load_positive_samples(path: Path, preferred_key: str) -> np.ndarray:
    """Backward-compatible unweighted view used by small validation tests."""

    return load_weighted_amplitude_samples(path, preferred_key).values


def resample_to_count(
    values: np.ndarray,
    count: int,
    rng: np.random.Generator,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Independently resample an external chain to the requested draw count."""

    if weights is None:
        indices = rng.integers(0, values.size, size=count)
    else:
        probabilities = np.asarray(weights, dtype=float)
        probabilities = probabilities / np.sum(probabilities)
        indices = rng.choice(values.size, size=count, replace=True, p=probabilities)
    return values[indices]


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    """Return the central equal-tailed 90% summary used in the manuscript."""

    lower, median, upper = np.quantile(values, [0.05, 0.5, 0.95])
    return {
        "lower_90": float(lower),
        "median": float(median),
        "upper_90": float(upper),
        "minus": float(median - lower),
        "plus": float(upper - median),
    }


def weighted_quantile_summary(
    values: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    """Return weighted 5/50/95% quantiles and their asymmetric widths."""

    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.ndim != 1 or values.size != weights.size:
        raise ValueError("weighted quantiles require equal-length 1-D arrays")
    if np.any(~np.isfinite(values)) or np.any(~np.isfinite(weights)):
        raise ValueError("weighted quantile inputs must be finite")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("weights must be nonnegative with a positive sum")

    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cumulative = (cumulative - 0.5 * sorted_weights) / cumulative[-1]
    lower, median, upper = np.interp(
        [0.05, 0.5, 0.95], cumulative, sorted_values
    )
    return {
        "lower_90": float(lower),
        "median": float(median),
        "upper_90": float(upper),
        "minus": float(median - lower),
        "plus": float(upper - median),
    }


def weighted_probability(mask: np.ndarray, weights: np.ndarray) -> float:
    """Return the normalized weighted probability of a Boolean condition."""

    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if mask.ndim != 1 or weights.ndim != 1 or mask.size != weights.size:
        raise ValueError("weighted probability requires equal-length 1-D arrays")
    if (
        np.any(~np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.any(weights > 0.0)
    ):
        raise ValueError(
            "probability weights must be finite, nonnegative, and not all zero"
        )
    return float(np.sum(weights[mask]) / np.sum(weights))


def run_propagation(
    config: PropagationConfig,
    cnm_sample_path: Path | None = None,
    lm24_amplitude_sample_path: Path | None = None,
    confirm_external_cnm_semantics: bool = False,
    confirm_external_lm24_full_census_semantics: bool = False,
    cnm_source_id: str | None = None,
    lm24_source_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run the deterministic propagation and return summary plus raw samples."""

    config.validate()
    if cnm_sample_path is not None and not confirm_external_cnm_semantics:
        raise ValueError(
            "external CNM samples require explicit confirmation that they "
            "represent the fixed-gamma, Hellings-Downs amplitude posterior "
            "with its log-uniform amplitude prior"
        )
    if cnm_sample_path is not None and not (cnm_source_id or "").strip():
        raise ValueError(
            "external CNM samples require a source DOI, release ID, or URL"
        )
    if (
        lm24_amplitude_sample_path is not None
        and not confirm_external_lm24_full_census_semantics
    ):
        raise ValueError(
            "external LM24 samples require explicit confirmation that they "
            "are full-census one-pass amplitudes at 1/yr with the declared "
            "mass-ratio and redshift kernels"
        )
    if lm24_amplitude_sample_path is not None and not (
        lm24_source_id or ""
    ).strip():
        raise ValueError(
            "external LM24 samples require a source DOI, release ID, or URL"
        )
    seed_sequence = np.random.SeedSequence(config.seed)
    gsmf_seed, scaling_seed, cnm_seed, lm_external_seed = seed_sequence.spawn(4)
    gsmf_rng = np.random.Generator(np.random.PCG64(gsmf_seed))
    scaling_rng = np.random.Generator(np.random.PCG64(scaling_seed))
    cnm_rng = np.random.Generator(np.random.PCG64(cnm_seed))
    lm_external_rng = np.random.Generator(np.random.PCG64(lm_external_seed))

    draws = draw_lm24_population(config, gsmf_rng, scaling_rng)
    moments = compute_population_moments(draws, config)
    input_provenance: dict[str, Any] = {}

    if cnm_sample_path is None:
        cnm_amplitude = split_lognormal_quantile_draws(
            config.cnm_amplitude_lower_90,
            config.cnm_amplitude_median,
            config.cnm_amplitude_upper_90,
            config.sample_count,
            cnm_rng,
        )
        cnm_source = "split-lognormal surrogate of reported CNM 5/50/95% quantiles"
    else:
        external_cnm = load_weighted_amplitude_samples(
            cnm_sample_path, "amplitude"
        )
        cnm_amplitude = resample_to_count(
            external_cnm.values,
            config.sample_count,
            cnm_rng,
            external_cnm.weights if external_cnm.weight_key else None,
        )
        cnm_source = (
            f"caller-confirmed external samples: {Path(cnm_sample_path).name}"
        )
        input_provenance["cnm"] = {
            "file": Path(cnm_sample_path).name,
            "sha256": external_cnm.sha256,
            "sample_count": int(external_cnm.values.size),
            "value_key": external_cnm.value_key,
            "weight_key": external_cnm.weight_key,
            "weight_effective_sample_size": float(
                np.sum(external_cnm.weights) ** 2
                / np.sum(external_cnm.weights**2)
            ),
            "semantics_confirmed_by_caller": True,
            "source_id": cnm_source_id,
        }

    amplitude_prior_min = 10.0**config.cnm_log10_amplitude_prior_min
    amplitude_prior_max = 10.0**config.cnm_log10_amplitude_prior_max
    if np.any(
        (cnm_amplitude < amplitude_prior_min)
        | (cnm_amplitude > amplitude_prior_max)
    ):
        raise ValueError(
            "CNM amplitude samples extend outside the declared amplitude prior"
        )

    if lm24_amplitude_sample_path is None:
        model_amplitude = moments.amplitude_uncut
        lm24_source = (
            "LM24 Appendix-B approximate GSMF reconstruction with independent "
            "McConnell-Ma scaling draws over the full census"
        )
    else:
        external_lm24 = load_weighted_amplitude_samples(
            lm24_amplitude_sample_path, "amplitude"
        )
        model_amplitude = resample_to_count(
            external_lm24.values,
            config.sample_count,
            lm_external_rng,
            external_lm24.weights if external_lm24.weight_key else None,
        )
        lm24_source = (
            "caller-confirmed external full-census one-pass model amplitudes: "
            f"{Path(lm24_amplitude_sample_path).name}"
        )
        input_provenance["lm24"] = {
            "file": Path(lm24_amplitude_sample_path).name,
            "sha256": external_lm24.sha256,
            "sample_count": int(external_lm24.values.size),
            "value_key": external_lm24.value_key,
            "weight_key": external_lm24.weight_key,
            "weight_effective_sample_size": float(
                np.sum(external_lm24.weights) ** 2
                / np.sum(external_lm24.weights**2)
            ),
            "semantics_confirmed_by_caller": True,
            "source_id": lm24_source_id,
        }

    f_merge = (cnm_amplitude / model_amplitude) ** 2
    f_prior_min = 10.0**config.log10_f_prior_min
    f_prior_max = 10.0**config.log10_f_prior_max
    within_f_prior = (f_merge >= f_prior_min) & (f_merge <= f_prior_max)
    if np.count_nonzero(within_f_prior) < 2:
        raise RuntimeError("f_merge samples do not overlap the declared prior")
    f_merge_supported = f_merge[within_f_prior]
    # The published CNM amplitude summary was obtained with a log-uniform
    # amplitude prior.  The quantile-matched surrogate inherits that prior, so
    # its transformation corresponds to a log-uniform f prior.  A flat-f prior
    # is obtained by importance reweighting with one factor of f.
    flat_f_weights = f_merge_supported.copy()
    flat_f_effective_sample_size = float(
        np.sum(flat_f_weights) ** 2 / np.sum(flat_f_weights**2)
    )
    log_rho = np.log(moments.rho_total_msun_mpc3)
    log_amplitude = np.log(moments.amplitude_uncut)

    source_status = (
        f"posterior propagation from {lm24_source} and {cnm_source}; "
        "file identity and caller confirmation are recorded but not "
        "independently verified"
    )
    if cnm_sample_path is None and lm24_amplitude_sample_path is None:
        source_status = (
            "quantile-based reconstruction from the published LM24 "
            "Appendix-B approximation and CNM amplitude summary"
        )

    limitations = [
        "Mass-ratio and redshift distributions are held fixed; their "
        "population-model uncertainty is not included.",
        "The posterior interpretation assumes the fixed-gamma CNM log-uniform "
        "amplitude prior and the declared log-uniform f_merge prior bounds.",
        "The LM24 mass-ratio and redshift kernels were chosen using an earlier "
        "PTA population analysis, so this is not a fully independent joint "
        "population inference.",
        "The host-mass integral follows the LM24 helper's declared "
        f"{config.log10_stellar_mass_min:.1f}--"
        f"{config.log10_stellar_mass_max:.1f} log10(Mstar/Msun) bounds; "
        "extrapolation beyond that domain is excluded.",
    ]
    if lm24_amplitude_sample_path is None:
        limitations.append(
            "The public LM24 generator approximates the full correlated GSMF "
            "distribution."
        )
    else:
        limitations.append(
            "The scientific semantics of the external LM24 chain are declared "
            "by the caller and cannot be inferred from its numeric arrays."
        )
    if cnm_sample_path is None:
        limitations.append(
            "The default CNM distribution is a piecewise-lognormal surrogate "
            "fixed only by three reported amplitude quantiles; its tail shape "
            "is not determined by those quantiles."
        )
    else:
        limitations.append(
            "The scientific semantics of the external CNM chain are declared "
            "by the caller and cannot be inferred from its numeric arrays."
        )
    summary: dict[str, Any] = {
        "status": source_status,
        "method": {
            "lm24": lm24_source,
            "cnm": cnm_source,
            "mapping": "f_merge = (A_CNM / A_model[f_merge=1])^2",
            "random_generator": (
                "NumPy PCG64 with independent spawned streams; this deliberately "
                "avoids the same-seed coupling in the public helper example"
            ),
            "prior_interpretation": (
                "The transformed samples correspond to a log-uniform f_merge "
                "prior because the fixed-gamma CNM analysis used a log-uniform "
                "amplitude prior. A flat-f sensitivity is importance weighted."
            ),
        },
        "implementation": {
            "version": CALCULATION_VERSION,
            "file": Path(__file__).name,
            "sha256": sha256_file(Path(__file__).resolve()),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "external_inputs": input_provenance,
        "configuration": asdict(config),
        "references": {
            "lm24": "doi:10.3847/2041-8213/ad66b8",
            "lm24_erratum": "doi:10.3847/2041-8213/ae5f96",
            "lm24_helper_commit": (
                "https://github.com/EmilyLiepold/LM24_helpers/commit/"
                "2092837e91197df16d5231d5845628638afd62ea"
            ),
            "mcconnell_ma_2013": "doi:10.1088/0004-637X/764/2/184",
            "massive_survey": "doi:10.1088/0004-637X/795/2/158",
            "cnm_fixed_gamma": "arXiv:2606.28554",
        },
        "diagnostics": {
            "dynamical_draw_count": int(np.count_nonzero(draws.branch == 0)),
            "sps_draw_count": int(np.count_nonzero(draws.branch == 1)),
            "phi1_floor_fraction": float(np.mean(draws.phi1_floored)),
            "phi2_floor_fraction": float(np.mean(draws.phi2_floored)),
            "redshift_moment": moments.redshift_moment,
            "eta_moment": moments.eta_moment,
            "corr_log_rho_total_log_A_uncut": float(
                np.corrcoef(log_rho, log_amplitude)[0, 1]
            ),
            "f_samples_outside_prior_fraction": float(
                1.0 - np.mean(within_f_prior)
            ),
            "flat_f_effective_sample_size": flat_f_effective_sample_size,
        },
        "results": {
            "lm24_reconstructed_rho_total_msun_mpc3": quantile_summary(
                moments.rho_total_msun_mpc3
            ),
            "A_lm24_reconstructed_uncut": quantile_summary(
                moments.amplitude_uncut
            ),
            "A_lm24_used_for_f_merge": quantile_summary(model_amplitude),
            "A_cnm": quantile_summary(cnm_amplitude),
            "f_merge_log_uniform_prior": quantile_summary(f_merge_supported),
            "probability_f_merge_le_1_log_uniform_prior": float(
                np.mean(f_merge_supported <= 1.0)
            ),
            "probability_f_merge_gt_1_log_uniform_prior": float(
                np.mean(f_merge_supported > 1.0)
            ),
            "f_merge_flat_prior_sensitivity": weighted_quantile_summary(
                f_merge_supported, flat_f_weights
            ),
            "probability_f_merge_le_1_flat_prior": weighted_probability(
                f_merge_supported <= 1.0, flat_f_weights
            ),
            "probability_f_merge_gt_1_flat_prior": weighted_probability(
                f_merge_supported > 1.0, flat_f_weights
            ),
        },
        "published_cross_checks": {
            "lm24_A_uncut_90": {
                "lower": 1.5e-15,
                "central": 2.0e-15,
                "upper": 2.7e-15,
            },
            "lm24_rho_total_90_msun_mpc3": {
                "lower": 1.3e6,
                "central": 1.8e6,
                "upper": 2.6e6,
            },
            "cnm_A_90": {
                "lower": config.cnm_amplitude_lower_90,
                "central": config.cnm_amplitude_median,
                "upper": config.cnm_amplitude_upper_90,
            },
        },
        "limitations": limitations,
    }

    samples = {
        "lm24_reconstruction_branch": draws.branch,
        "lm24_reconstructed_rho_total_msun_mpc3": (
            moments.rho_total_msun_mpc3
        ),
        "A_lm24_reconstructed_uncut": moments.amplitude_uncut,
        "A_lm24_used_for_f_merge": model_amplitude,
        "A_cnm": cnm_amplitude,
        "f_merge": f_merge,
        "log10_f_merge": np.log10(f_merge),
        "within_f_prior": within_f_prior,
        "flat_f_prior_weight": np.where(within_f_prior, f_merge, 0.0),
    }
    return summary, samples


def write_outputs(
    summary: dict[str, Any],
    samples: dict[str, np.ndarray],
    summary_path: Path,
    sample_path: Path | None = None,
    protected_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Write linked JSON/NPZ outputs and return the serialized summary."""

    summary_path = Path(summary_path)
    if sample_path is not None:
        sample_path = Path(sample_path)
        if sample_path.suffix != ".npz":
            raise ValueError("sample output path must end in .npz")
        if summary_path.resolve() == sample_path.resolve():
            raise ValueError("summary and sample output paths must be different")
    output_paths = {summary_path.resolve()}
    if sample_path is not None:
        output_paths.add(sample_path.resolve())
    for protected_path in protected_paths:
        if Path(protected_path).resolve() in output_paths:
            raise ValueError("an output path would overwrite an input chain")

    serialized_summary = json.loads(json.dumps(summary))
    if sample_path is not None:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            sample_path,
            **samples,
            metadata_json=np.asarray(
                json.dumps(serialized_summary, sort_keys=True)
            ),
        )
        serialized_summary["sample_archive"] = {
            "path_relative_to_summary": os.path.relpath(
                sample_path.resolve(), start=summary_path.parent.resolve()
            ),
            "sha256": sha256_file(sample_path),
            "sample_count": int(np.asarray(samples["f_merge"]).size),
        }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(serialized_summary, indent=2) + "\n", encoding="utf-8"
    )
    return serialized_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct LM24 population uncertainty and combine it with the "
            "fixed-slope CNM amplitude to infer f_merge."
        )
    )
    parser.add_argument("--samples", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=20_240_724)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--log10-f-prior-min", type=float, default=-3.0)
    parser.add_argument("--log10-f-prior-max", type=float, default=1.0)
    parser.add_argument(
        "--intrinsic-scatter-dex",
        type=float,
        default=0.34,
        help="fixed intrinsic scatter in the black-hole--bulge relation",
    )
    parser.add_argument(
        "--cnm-samples",
        type=Path,
        help="optional positive CNM amplitude samples (txt, csv, npy, or npz)",
    )
    parser.add_argument(
        "--confirm-external-cnm-semantics",
        action="store_true",
        help=(
            "confirm that --cnm-samples are fixed-gamma=13/3, Hellings-Downs "
            "posterior amplitudes at 1/yr with the log-uniform amplitude prior"
        ),
    )
    parser.add_argument(
        "--cnm-source-id",
        help="source DOI, release identifier, or URL for --cnm-samples",
    )
    parser.add_argument(
        "--lm24-amplitude-samples",
        type=Path,
        help=(
            "optional positive full-census one-pass model-amplitude samples "
            "(txt, csv, npy, or npz)"
        ),
    )
    parser.add_argument(
        "--confirm-external-lm24-full-census-semantics",
        action="store_true",
        help=(
            "confirm that --lm24-amplitude-samples are full-census one-pass "
            "amplitudes at 1/yr with the LM24 mass-ratio/redshift kernels"
        ),
    )
    parser.add_argument(
        "--lm24-source-id",
        help=(
            "source DOI, release identifier, or URL for "
            "--lm24-amplitude-samples"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lm24-cnm-fmerge.json"),
        help="JSON summary path",
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        help="optional compressed NPZ file for the propagated samples",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = PropagationConfig(
        sample_count=args.samples,
        seed=args.seed,
        batch_size=args.batch_size,
        scaling_intrinsic_scatter_dex=args.intrinsic_scatter_dex,
        log10_f_prior_min=args.log10_f_prior_min,
        log10_f_prior_max=args.log10_f_prior_max,
    )
    summary, samples = run_propagation(
        config,
        cnm_sample_path=args.cnm_samples,
        lm24_amplitude_sample_path=args.lm24_amplitude_samples,
        confirm_external_cnm_semantics=args.confirm_external_cnm_semantics,
        confirm_external_lm24_full_census_semantics=(
            args.confirm_external_lm24_full_census_semantics
        ),
        cnm_source_id=args.cnm_source_id,
        lm24_source_id=args.lm24_source_id,
    )
    summary = write_outputs(
        summary,
        samples,
        args.output,
        sample_path=args.sample_output,
        protected_paths=tuple(
            path
            for path in (args.cnm_samples, args.lm24_amplitude_samples)
            if path is not None
        ),
    )

    f_result = summary["results"]["f_merge_log_uniform_prior"]
    lm_result = summary["results"]["A_lm24_used_for_f_merge"]
    flat_result = summary["results"]["f_merge_flat_prior_sensitivity"]
    print(
        "One-pass model amplitude used for f_merge: "
        f"{lm_result['median']:.6e} "
        f"(-{lm_result['minus']:.2e}, +{lm_result['plus']:.2e})"
    )
    print(
        "Manuscript f_merge (log-uniform prior): "
        f"{f_result['median']:.4f} "
        f"(-{f_result['minus']:.4f}, +{f_result['plus']:.4f})"
    )
    print(
        "Flat-f prior sensitivity: "
        f"{flat_result['median']:.4f} "
        f"(-{flat_result['minus']:.4f}, +{flat_result['plus']:.4f})"
    )
    print(f"Wrote {args.output}")
    if args.sample_output is not None:
        print(f"Wrote {args.sample_output}")


if __name__ == "__main__":
    main()
