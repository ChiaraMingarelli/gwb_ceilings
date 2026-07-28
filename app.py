import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as path_effects
from matplotlib.ticker import LogLocator, FixedLocator, FuncFormatter, NullFormatter
import hashlib
import io
from pathlib import Path

# Unicode superscript map for tick labels (avoids mathtext parser entirely)
_SUP = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')

def _log_fmt(x, pos):
    """Format log-scale tick labels using Unicode superscripts."""
    if x <= 0:
        return ''
    exp = round(np.log10(x))
    if abs(x - 10**exp) / max(x, 1e-30) < 0.01:
        return f'10{str(exp).translate(_SUP)}'
    return ''

# =============================================================================
# Energetic Ceilings on Astrophysical Gravitational-Wave Backgrounds
# 
# Interactive visualization tool for GWB ceiling constraints
#
# Author: Chiara Mingarelli (Yale University)
# Code development assisted by Claude (Anthropic)
#
# Reference: Mingarelli (2026), "Energetic Ceilings on Astrophysical
#            Gravitational-Wave Backgrounds", arXiv:2601.18859
#
# If you use figures from this tool, please cite:
#   Mingarelli, C. M. F. (2026), arXiv:2601.18859
#   https://arxiv.org/abs/2601.18859
#
# All conventions, benchmark amplitudes, and detector curves are synchronized
# with the paper's verification suite (verify_ceilings.py, step3-numbers.json,
# make_fig2_corrected.py).  Omega_gw(f) = (2 pi^2 / 3 H0^2) f^3 S_h(f)
# throughout; detector curves are single-natural-log-bin, SNR-one orientation
# curves, Omega_n(f)/sqrt(T f).  Official detector data products are
# hash-pinned in data/detectors/ (see the README there for provenance).
# =============================================================================

# Physical Constants (synchronized with verify_ceilings.py)
H0_km_s_Mpc = 67.4
MPC_SI = 3.0856775814913673e22
H0 = H0_km_s_Mpc * 1000.0 / MPC_SI
h = H0_km_s_Mpc / 100.0
YR_SI = 365.25 * 86_400.0
F_YR = 1.0 / YR_SI  # 1 yr^-1 in Hz

# Hash-pinned official detector data products (see data/detectors/README.md)
DATA_DIR = Path(__file__).resolve().parent / "data" / "detectors"
ALIGO_DESIGN_PI = DATA_DIR / "Figures_3_and_4_PICurve_Design.dat"
ALIGO_DESIGN_PI_SHA256 = (
    "7d32bbf49db02a653da043266c74efce7761490c978b9f1e1ae92d06711f5ef4"
)
BBO_PLIS = DATA_DIR / "BBO_PLIS_Schmitz2021.dat"
BBO_PLIS_SHA256 = (
    "e2a005bc0d57090c7dea355f4fd3e869d1da31cab011488cf8d29991a91b61ae"
)
CE_40KM_STRAIN = DATA_DIR / "cosmic_explorer_strain_T2000017-v9.txt"
CE_40KM_STRAIN_SHA256 = (
    "ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c"
)

# Pre-compute frequency grid (reduced from 3000 to 1000 points for speed)
F_GRID = np.logspace(-9.5, 3.5, 1000)

# Fiducial reservoir densities (Msun/Mpc^3), step3-numbers.json constants
RHO_SMBH_FID = 1.8e6  # Liepold & Ma (2024), ApJL 971, L29
RHO_STELLAR_FID = 630372082.0092556  # 5e-3 x rho_crit
RHO_NSC_FID = 630372.0820092556  # ETA_NSC_FIDUCIAL = 1e-3 x rho_star

# Integrated benchmark energy budget (step3b): the conditional sum of the
# declared benchmark source-event spectra, Sum_i Omega_i dlnf.  Not a
# universal ceiling, jointly inferred population, or baryonic wall.
INTEGRATED_BUDGET_OMEGA = 1.3907155534322894e-07

# Declared benchmark channels, anchored to the paper's audited scalars
# (step3-numbers.json).  A_bench is the characteristic strain h_c at f_ref;
# each band is drawn as the f^{-2/3} inspiral law over [f_min, f_max].  For
# the phenomenological-shape channels (Pop III, Stellar BBH) f_max is set at
# the audited spectral peak so the power-law band ends where the true
# spectrum turns over.  rho_src is the processed source density
# rho_src = f_merge * rho_res for the table below.
POPULATIONS = {
    'SMBHB': {
        'reservoir': 'SMBH',
        'f_ref': F_YR,
        'A_bench': 2.0572436926091547e-15,  # benchmark_A_1yr, M1 >= 1e8 Msun
        'f_min': 1e-9,
        'f_max': 4e-7,
        'epsilon_gw': None,  # one-pass population moment, not eps*rho
        'rho_src': 1225283.3222939118,  # participating mass density
        'color': '#0072B2',
    },
    'AGN-IMRI': {
        'reservoir': 'SMBH',
        'f_ref': 3e-3,
        'A_bench': 1.337115866150623e-21,  # agn_imri_benchmark
        'f_min': 1e-5,
        'f_max': 4e-2,
        'epsilon_gw': 0.05,
        'rho_src': 540.0,
        'color': '#D55E00',
    },
    'EMRI': {
        'reservoir': 'NSC',
        'f_ref': 1e-2,
        'A_bench': 3.4763979324143e-22,  # emri_benchmark
        'f_min': 1e-5,
        'f_max': 1e-2,
        'epsilon_gw': 0.05,
        'rho_src': 63.03720820092557,
        'color': '#009E73',
    },
    'BNS': {
        'reservoir': 'STELLAR',
        'f_ref': 0.1,
        'A_bench': 4.277390831170633e-24,  # bns_benchmark
        'f_min': 1e-3,
        'f_max': 1500.0,
        'epsilon_gw': 0.02,
        'rho_src': 1660.4,
        'color': '#CC79A7',
    },
    'Pop III': {
        'reservoir': 'STELLAR',
        'f_ref': 0.1,
        'A_bench': 6.803213932893254e-24,  # popiii_benchmark
        'f_min': 1e-3,
        'f_max': 15.4,  # audited spectral peak (phenomb shape)
        'epsilon_gw': 0.05,
        'rho_src': 733.2211228627597,
        'color': '#56B4E9',
    },
    'Stellar BBH': {
        'reservoir': 'STELLAR',
        'f_ref': 25.0,
        'A_bench': 4.031480337458788e-25,  # bbh_benchmark
        'f_min': 3.1622776601683795e-5,
        'f_max': 476.0,  # audited spectral peak (phenomb shape)
        'epsilon_gw': 0.05,
        'rho_src': 6592.729111209701,
        'color': '#E69F00',
    },
}

# Tuned label positions for h_c mode (carefully positioned to avoid overlaps)
labels_pos_hc = {
    'SMBHB': (3e-9, 8e-15),
    'AGN-IMRI': (2e-2, 1e-22),
    'EMRI': (2e-5, 1e-21),
    'Pop III': (0.05, 8e-25),
    'BNS': (0.5, 3e-24),
    'Stellar BBH': (80, 8e-25)
}

# Detector label positions for Omega_gw mode (paper Figure-3 placements)
detector_labels_omega = {
    'muAres': (1.6e-6, 1e-15),
    'BBO': (1.2e-1, 4e-17),
    'LISA': (2e-3, 1e-13),
    'aLIGO': (17, 5.5e-9),
    'CE': (300, 7e-13),
}

# Paper multiband-figure visual grammar (matches make_fig2_corrected.py)
CONTEXT_GRAY = '#8A8A8A'
CONTEXT_LABEL_GRAY = '#666666'
CONTEXT_ALPHA = 0.38
BACKGROUND_FILL_ALPHA = 0.10
PLOT_OMEGA_MIN = 1e-18
DWD_GRAY = '#696969'
BUDGET_RED = '#B2182B'
NANOGRAV_FACE_ALPHA = 0.30
NANOGRAV_EDGE_ALPHA = 0.58
DETECTOR_PLOT_MAX = 2.0e-7
LOW_FREQUENCY_CONTINUATION_DEX = 0.50
EMRI_MODEL_CUTOFF_HZ = 1.0e-2
EMRI_HIGH_FREQUENCY_CONTINUATION_DEX = 0.50
CUMULATIVE_QUANTILES = (0.50, 0.90, 0.99)

WHITE_STROKE = [
    path_effects.Stroke(linewidth=3.0, foreground='white', alpha=0.80),
    path_effects.Normal(),
]
LABEL_STROKE = [
    path_effects.withStroke(linewidth=2.2, foreground='white', alpha=0.95),
]

# Audited paper spectra and NANOGrav free-spectrum context (hash-pinned
# copies of the paper's step3-spectra.npz / step3-figure-context.npz)
SPECTRA_FILE = Path(__file__).resolve().parent / "data" / "step3-spectra.npz"
SPECTRA_SHA256 = (
    "a146133de1e39bb335a1b5900f8d6ad39ea06fde88b19ccbd8fedba173936688"
)
PTA_CONTEXT_FILE = (
    Path(__file__).resolve().parent / "data" / "step3-figure-context.npz"
)
PTA_CONTEXT_SHA256 = (
    "26afed590123475fc82fd0a2a863f8d00665a52cec728f5cc6fdd329e1338e46"
)

# Population key -> audited spectrum key in step3-spectra.npz
SPECTRUM_KEYS = {
    'SMBHB': 'smbhb_benchmark',
    'AGN-IMRI': 'agn_imri_benchmark',
    'EMRI': 'emri_benchmark',
    'BNS': 'bns_benchmark',
    'Pop III': 'popiii_benchmark',
    'Stellar BBH': 'bbh_benchmark',
}

# Declared plot-only low-frequency interpretation boundaries (paper values)
BENCHMARK_DISPLAY_MINIMUM_HZ = {
    'smbhb_benchmark': 1.0e-9,
    'emri_benchmark': 1.0e-5,
    'agn_imri_benchmark': 1.0e-5,
    'popiii_benchmark': 1.0e-3,
    'bbh_benchmark': 3.1622776601683795e-5,
    'bns_benchmark': 1.0e-3,
}

# Detector label positions for h_c mode
detector_labels_hc = {
    'muAres': (3e-5, 1e-17),
    'BBO': (3, 1e-24),
    'LISA': (5e-4, 1e-17),
    'aLIGO': (50, 3e-22),
    'CE': (20, 3e-25),
}

# PTA label positions for Omega_gw mode: (x, y, ha)
# Long-baseline PTAs labelled at left edge where curves are well-separated;
# short-baseline PTAs (MPTA, CPTA) labelled near their minima offset to the right.
pta_label_pos_omega = {
    'NANOGrav 15yr': (2e-9, 5e-10, 'left'),
    'EPTA DR2': (1.5e-9, 2e-11, 'left'),
    'PPTA DR3': (2e-9, 1.2e-10, 'left'),
    'MPTA': (5e-9, 5e-9, 'right'),
    'CPTA': (7e-9, 1.5e-9, 'right'),
    'IPTA DR3 (proj.)': (1.5e-9, 3e-12, 'left'),
    'SKA-era': (3e-9, 5e-14, 'left'),
}

display_names = {
    'SMBHB': 'SMBHBs',
    'AGN-IMRI': 'AGN-IMRI',
    'EMRI': 'EMRI',
    'Pop III': 'Pop III',
    'BNS': 'BNS',
    'Stellar BBH': 'sBBHs'
}


def get_omega_gw(f, A, f_ref, f_min, f_max):
    """Compute Omega_gw from characteristic strain amplitude."""
    omega = np.zeros_like(f)
    mask = (f >= f_min) & (f <= f_max)
    if np.sum(mask) == 0:
        return omega
    f_band = f[mask]
    hc = A * (f_band / f_ref)**(-2/3)
    prefac = 2 * np.pi**2 / (3 * H0**2)
    omega[mask] = prefac * f_band**2 * hc**2
    return omega


# All detector curves below are ported verbatim from the paper's
# make_fig2_corrected.py (same conventions, same data products).

@st.cache_data
def get_lisa_sensitivity(f_tuple, T_yrs=4.0):
    """RCL19 LISA noise in a single-log-bin, SNR-one orientation curve."""
    f = np.array(f_tuple)
    L = 2.5e9
    f_star = 19.09e-3
    P_oms = (1.5e-11)**2 * (1 + (2e-3/f)**4)
    P_acc = (3e-15)**2 * (1 + (0.4e-3/f)**2) * (1 + (f/8e-3)**4)
    Sn = (
        10.0 / (3.0 * L**2)
        * (P_oms + 4.0 * P_acc / (2.0 * np.pi * f)**4)
        * (1.0 + 0.6 * (f / f_star)**2)
    )
    omega_n = (2.0 * np.pi**2 / (3.0 * H0**2)) * f**3 * Sn
    return omega_n / np.sqrt(T_yrs * YR_SI * f)


@st.cache_data
def get_muares_sensitivity(f_tuple, T_yrs=10.0):
    """Proposal-based muAres single-log-bin orientation sensitivity.

    Sesana et al. (2021) strawman: 395 Gm arms, 50 pm/rtHz total readout,
    flat 1e-15 m s^-2/rtHz acceleration noise down to 1e-7 Hz, with the
    sky-averaged Robson response form.
    """
    f = np.array(f_tuple)
    omega = np.full_like(f, np.nan)
    mask = (f >= 1e-7) & (f <= 1.0)
    fm = f[mask]
    L = 3.95e11
    f_star = 299_792_458.0 / (2.0 * np.pi * L)
    S_pos = (50.0e-12)**2
    S_acc = (1.0e-15)**2
    Sn = (
        10.0 / (3.0 * L**2)
        * (4.0 * S_acc / (2.0 * np.pi * fm)**4 + S_pos)
        * (1.0 + 0.6 * (fm / f_star)**2)
    )
    omega[mask] = (
        (2.0 * np.pi**2 / (3.0 * H0**2)) * fm**3 * Sn
        / np.sqrt(T_yrs * YR_SI * fm)
    )
    return omega


@st.cache_data
def get_ce_sensitivity(f_tuple, T_yrs=1.0):
    """Official CE 40-km strain ASD in a single-log-bin orientation curve."""
    digest = hashlib.sha256(CE_40KM_STRAIN.read_bytes()).hexdigest()
    if digest != CE_40KM_STRAIN_SHA256:
        raise RuntimeError("Cosmic Explorer source hash changed.")
    source = np.loadtxt(CE_40KM_STRAIN)
    source_frequency = np.asarray(source[:, 0], dtype=float)
    source_asd = np.asarray(source[:, 1], dtype=float)

    f = np.array(f_tuple)
    omega = np.full_like(f, np.nan)
    support = (f >= source_frequency[0]) & (f <= source_frequency[-1])
    fm = f[support]
    asd = np.exp(
        np.interp(np.log(fm), np.log(source_frequency), np.log(source_asd))
    )
    omega[support] = (
        (2.0 * np.pi**2 / (3.0 * H0**2)) * fm**3 * asd**2
        / np.sqrt(T_yrs * YR_SI * fm)
    )
    return omega


@st.cache_data
def get_aligo_design_pi(f_tuple):
    """Published two-sigma Advanced-LIGO/Virgo design PI sensitivity."""
    digest = hashlib.sha256(ALIGO_DESIGN_PI.read_bytes()).hexdigest()
    if digest != ALIGO_DESIGN_PI_SHA256:
        raise RuntimeError("Advanced-LIGO design PI source hash changed.")
    design = np.loadtxt(ALIGO_DESIGN_PI)
    source_frequency = np.asarray(design[:, 0], dtype=float)
    one_sigma_omega = np.asarray(design[:, 1], dtype=float)

    f = np.array(f_tuple)
    omega = np.full_like(f, np.nan)
    support = (f >= source_frequency[0]) & (f <= source_frequency[-1])
    omega[support] = 2.0 * np.exp(
        np.interp(
            np.log(f[support]),
            np.log(source_frequency),
            np.log(one_sigma_omega),
        )
    )
    return omega


@st.cache_data
def get_bbo_sensitivity(f_tuple, T_yrs=5.0):
    """Published BBO SNR-one power-law-integrated curve (Schmitz 2021).

    The source file is the one-year curve; observing time rescales it by
    1/sqrt(T_yrs).  Converted from h^2 Omega with the manuscript h.
    """
    digest = hashlib.sha256(BBO_PLIS.read_bytes()).hexdigest()
    if digest != BBO_PLIS_SHA256:
        raise RuntimeError("BBO PLIS source hash changed.")
    source = np.loadtxt(BBO_PLIS)
    source_log_frequency = np.asarray(source[:, 0], dtype=float)
    source_log_h2omega = np.asarray(source[:, 1], dtype=float)

    f = np.array(f_tuple)
    omega = np.full_like(f, np.nan)
    log_frequency = np.full_like(f, np.nan)
    positive = f > 0.0
    log_frequency[positive] = np.log10(f[positive])
    support = (
        positive
        & (log_frequency >= source_log_frequency[0])
        & (log_frequency <= source_log_frequency[-1])
    )
    omega[support] = 10.0 ** np.interp(
        log_frequency[support],
        source_log_frequency,
        source_log_h2omega,
    ) / (h**2 * np.sqrt(T_yrs))
    return omega


@st.cache_data
def get_dwd_foreground(f_tuple):
    """Robson et al. (2019) four-year Galactic-DWD confusion fit."""
    f = np.array(f_tuple)
    amplitude = 9.0e-45
    alpha = 0.138
    beta = -221.0
    kappa = 521.0
    gamma = 1_680.0
    f_knee = 0.00113
    omega = np.zeros_like(f)
    mask = (f >= 10.0**-5.5) & (f <= 0.1)
    fm = f[mask]
    sh = (
        amplitude
        * fm**(-7.0 / 3.0)
        * np.exp(-fm**alpha + beta * fm * np.sin(kappa * fm))
        * (1.0 + np.tanh(gamma * (f_knee - fm)))
    )
    omega[mask] = (2.0 * np.pi**2 / (3.0 * H0**2)) * fm**3 * sh
    return omega


@st.cache_data
def load_benchmark_spectra():
    """Load the audited paper spectra (hash-pinned step3-spectra.npz)."""
    digest = hashlib.sha256(SPECTRA_FILE.read_bytes()).hexdigest()
    if digest != SPECTRA_SHA256:
        raise RuntimeError("Benchmark spectra archive hash changed.")
    with np.load(SPECTRA_FILE) as archive:
        frequency = np.asarray(archive["frequency_hz"], dtype=float)
        spectra = {
            key: np.asarray(archive[key], dtype=float)
            for key in archive.files
            if key != "frequency_hz"
        }
    return frequency, spectra


@st.cache_data
def load_nanograv_violins():
    """Load the archived NANOGrav free-spectrum marginal posteriors."""
    digest = hashlib.sha256(PTA_CONTEXT_FILE.read_bytes()).hexdigest()
    if digest != PTA_CONTEXT_SHA256:
        raise RuntimeError("NANOGrav context archive hash changed.")
    with np.load(PTA_CONTEXT_FILE) as context:
        center_frequency = np.asarray(context["pta_frequency_hz"], dtype=float)
        log10_omega = np.asarray(context["pta_log10_omega_grid"], dtype=float)
        half_width = np.asarray(context["pta_half_width_hz"], dtype=float)
    return center_frequency, log10_omega, half_width


def display_curve(values):
    """Return a plotting copy that ends cleanly at physical support."""
    curve = np.where(values > 0.0, values, np.nan).astype(float, copy=True)
    valid = np.isfinite(curve[:-1]) & np.isfinite(curve[1:])
    log_step = np.full(curve.size - 1, np.nan)
    log_step[valid] = np.log10(curve[1:][valid] / curve[:-1][valid])
    abrupt = np.flatnonzero(log_step < -0.05)
    if abrupt.size:
        curve[abrupt[0] + 1:] = np.nan
    return curve


def benchmark_display_curve(frequency, values, spectrum_key):
    """Apply the declared plot-only low-frequency interpretation boundary."""
    curve = display_curve(values)
    curve[frequency < BENCHMARK_DISPLAY_MINIMUM_HZ[spectrum_key]] = np.nan
    return curve


def benchmark_continuation_curve(frequency, values, spectrum_key):
    """Show a half-decade analytic continuation below the solid boundary."""
    curve = display_curve(values)
    solid_minimum = BENCHMARK_DISPLAY_MINIMUM_HZ[spectrum_key]
    continuation_minimum = solid_minimum * 10.0 ** (
        -LOW_FREQUENCY_CONTINUATION_DEX
    )
    support = (frequency >= continuation_minimum) & (frequency <= solid_minimum)
    curve[~support] = np.nan
    return curve


def emri_high_frequency_continuation(frequency, values):
    """Continue the EMRI inspiral power law above its declared model cutoff."""
    f = np.asarray(frequency, dtype=float)
    omega = np.asarray(values, dtype=float)
    anchor_support = (
        (f > 0.0) & (f <= EMRI_MODEL_CUTOFF_HZ)
        & np.isfinite(omega) & (omega > 0.0)
    )
    anchor_index = int(np.flatnonzero(anchor_support)[-1])
    anchor_omega = omega[anchor_index] * (
        EMRI_MODEL_CUTOFF_HZ / f[anchor_index]
    ) ** (2.0 / 3.0)
    continuation_frequency = np.geomspace(
        EMRI_MODEL_CUTOFF_HZ,
        EMRI_MODEL_CUTOFF_HZ * 10.0**EMRI_HIGH_FREQUENCY_CONTINUATION_DEX,
        120,
    )
    continuation_omega = anchor_omega * (
        continuation_frequency / EMRI_MODEL_CUTOFF_HZ
    ) ** (2.0 / 3.0)
    return continuation_frequency, continuation_omega


def cumulative_budget_fraction(frequency, omega_sum):
    """Cumulative fraction of the integrated budget, Int Omega dlnf below f."""
    lnf = np.log(frequency)
    contributions = np.zeros_like(omega_sum)
    contributions[1:] = (
        0.5 * (omega_sum[1:] + omega_sum[:-1]) * np.diff(lnf)
    )
    cumulative = np.cumsum(contributions)
    total = cumulative[-1]
    return cumulative / total, total


def scale_amplitude(A_bench, reservoir, rho_smbh, rho_stellar, rho_nsc):
    """Scale amplitude based on reservoir density relative to fiducial."""
    if reservoir == 'SMBH':
        return A_bench * np.sqrt(rho_smbh / RHO_SMBH_FID)
    elif reservoir == 'STELLAR':
        return A_bench * np.sqrt(rho_stellar / RHO_STELLAR_FID)
    elif reservoir == 'NSC':
        return A_bench * np.sqrt(rho_nsc / RHO_NSC_FID)
    return A_bench


def reservoir_ratio(reservoir, rho_smbh, rho_stellar, rho_nsc):
    """Linear Omega scaling of a channel with its reservoir density."""
    if reservoir == 'SMBH':
        return rho_smbh / RHO_SMBH_FID
    elif reservoir == 'STELLAR':
        return rho_stellar / RHO_STELLAR_FID
    elif reservoir == 'NSC':
        return rho_nsc / RHO_NSC_FID
    return 1.0


@st.cache_data
def get_pta_sensitivity_analytic(n_pulsars=67, timespan=15.0, sigma_ns=300, cadence=26, preset='NANOGrav 15yr'):
    """
    PTA sensitivity curve in Omega_gw, independently calibrated to each array's published results.
    
    Uses the formalism of Hazboun, Romano & Smith (2019), PRD 100, 104028.
    https://github.com/Hazboun6/hasasia
    
    Each PTA with a published detection is calibrated to its own reported amplitude.
    Projections (IPTA DR3, SKA-era) are scaled from the most similar existing array.
    
    Parameters:
    -----------
    n_pulsars : int
        Number of pulsars in the array
    timespan : float
        Observation timespan in years
    sigma_ns : float
        RMS timing residual in nanoseconds
    cadence : int
        Observations per year
    preset : str
        PTA name for independent calibration
    
    Returns:
    --------
    freqs, omega_gw : arrays
    """
    # Frequency array
    f_yr = 1.0 / (365.25 * 24 * 3600)  # 1/year in Hz
    T_sec = timespan * 365.25 * 24 * 3600
    f_min = 1.0 / T_sec
    f_max = cadence / (2 * 365.25 * 24 * 3600)
    freqs = np.logspace(np.log10(f_min * 0.5), np.log10(f_max), 100)
    
    # Independent calibrations based on published detections at FIXED gamma=13/3
    # Each array's sensitivity is set to match their detection threshold
    # (sensitivity ~ 0.8-0.9 × detected amplitude for a ~3-5 sigma detection)
    calibrations = {
        # Detected signals - calibrated to each array's published amplitude at gamma=13/3
        'NANOGrav 15yr': {'h_c_min': 2.0e-15, 'n': 67, 'T': 15.0, 'sigma': 300, 'cad': 26},  # A=2.4e-15 (Agazie+ 2023)
        'EPTA DR2': {'h_c_min': 2.1e-15, 'n': 25, 'T': 24.0, 'sigma': 500, 'cad': 20},       # A=2.5e-15 (EPTA+ 2023)
        'PPTA DR3': {'h_c_min': 1.7e-15, 'n': 30, 'T': 18.0, 'sigma': 400, 'cad': 26},       # A=2.0e-15 (Reardon+ 2023)
        'CPTA': {'h_c_min': 1.7e-15, 'n': 57, 'T': 3.4, 'sigma': 100, 'cad': 26},            # A=2.0e-15 (Xu+ 2023, fixed alpha)
        'MPTA': {'h_c_min': 4.0e-15, 'n': 83, 'T': 4.5, 'sigma': 200, 'cad': 26},            # A=4.8e-15 (Miles+ 2025, fixed alpha)
        'IPTA DR3 (proj.)': {'h_c_min': 8.0e-16, 'n': 115, 'T': 25.0, 'sigma': 200, 'cad': 26},
        'SKA-era': {'h_c_min': 7.0e-17, 'n': 200, 'T': 20.0, 'sigma': 50, 'cad': 52},
    }
    
    # Get calibration for this preset
    if preset in calibrations and preset != 'Custom':
        h_c_min = calibrations[preset]['h_c_min']
    else:
        # Custom: scale from NANOGrav 15yr based on user parameters
        n_ref, T_ref, sigma_ref, cad_ref = 67, 15.0, 300.0, 26
        h_c_ref = 2.0e-15
        
        N_pairs_ref = n_ref * (n_ref - 1) / 2
        N_pairs = n_pulsars * (n_pulsars - 1) / 2
        
        # Sensitivity scales as: sigma / sqrt(N_pairs * T * cadence)
        scaling = (sigma_ns / sigma_ref) * \
                  np.sqrt(N_pairs_ref / max(N_pairs, 1)) * \
                  np.sqrt(T_ref / max(timespan, 0.1)) * \
                  np.sqrt(cad_ref / max(cadence, 1))
        h_c_min = h_c_ref * scaling
    
    # Frequency-dependent sensitivity shape (from PTA physics)
    f_low = 1.5 / T_sec  # Timing model cutoff
    f_high = cadence * f_yr / 3  # White noise takeover
    
    # Shape function: minimum near geometric mean of f_low and f_high
    low_f_rise = (f_low / freqs)**4
    high_f_rise = (freqs / f_high)**2
    shape = np.sqrt(1 + low_f_rise + high_f_rise)
    
    # Normalize so minimum = h_c_min
    h_c = h_c_min * shape / np.min(shape)
    
    # Convert to Omega_gw: Omega = (2π²/3H₀²) f² h_c²
    prefac = 2 * np.pi**2 / (3 * H0**2)
    omega_gw = prefac * freqs**2 * h_c**2
    
    return freqs, omega_gw


def omega_to_hc(freqs, omega_gw):
    """Convert Omega_gw to characteristic strain h_c."""
    prefac = 2 * np.pi**2 / (3 * H0**2)
    h_c = np.sqrt(omega_gw / (prefac * freqs**2))
    return h_c


# Streamlit app
st.set_page_config(page_title="GW Background Ceilings", layout="wide")
st.title("Energetic Ceilings on Astrophysical Gravitational-Wave Backgrounds")

st.markdown("""
Interactive visualization of astrophysical gravitational wave background ceilings 
based on energy reservoir constraints. Adjust the mass density reservoirs to see 
how the GWB amplitudes scale.

**Reference:** Mingarelli (2026), *Energetic Ceilings on Astrophysical Gravitational-Wave Backgrounds* — [arXiv:2601.18859](https://arxiv.org/abs/2601.18859)

If you use figures from this tool, please cite [Mingarelli (2026)](https://arxiv.org/abs/2601.18859).
""")

# Sidebar controls
st.sidebar.header("Mass Reservoirs")
st.sidebar.markdown("Adjust reservoir densities (M☉/Mpc³)")

# Initialize session state with fiducial values (step3-numbers.json)
if 'rho_smbh_val' not in st.session_state:
    st.session_state.rho_smbh_val = 1.8  # ×10^6, L&M (2024)
if 'rho_stellar_val' not in st.session_state:
    st.session_state.rho_stellar_val = 6.3  # ×10^8, 5e-3 x rho_crit
if 'rho_nsc_val' not in st.session_state:
    st.session_state.rho_nsc_val = 0.63  # ×10^6, 1e-3 x rho_star

# Reset button
if st.sidebar.button("Reset to paper fiducials"):
    st.session_state.rho_smbh_val = 1.8  # L&M (2024)
    st.session_state.rho_stellar_val = 6.3
    st.session_state.rho_nsc_val = 0.63
    st.rerun()

rho_smbh = st.sidebar.slider(
    "ρ_SMBH (×10⁶)",
    min_value=0.5, max_value=5.0,
    step=0.1,
    key='rho_smbh_val'
) * 1e6

rho_stellar = st.sidebar.slider(
    "ρ_★ (×10⁸)",
    min_value=1.0, max_value=10.0,
    step=0.1,
    key='rho_stellar_val'
) * 1e8

rho_nsc = st.sidebar.slider(
    "ρ_NSC (×10⁶)",
    min_value=0.1, max_value=5.0,
    step=0.01,
    key='rho_nsc_val'
) * 1e6

st.sidebar.header("Display Options")
# Y-axis toggle temporarily disabled - always use Omega_gw
# y_axis_unit = st.sidebar.radio("Y-axis", ["Ω_gw", "h_c (characteristic strain)"], index=0, horizontal=True)
y_axis_unit = "Ω_gw"  # Fixed to Omega_gw for now
show_dwd = st.sidebar.checkbox("Show DWD foreground", value=True)
show_nanograv = st.sidebar.checkbox("Show NANOGrav free-spectrum", value=True)
show_ceiling = st.sidebar.checkbox("Show integrated benchmark strip", value=True)

# Individual detector toggles
with st.sidebar.expander("Detectors", expanded=True):
    show_lisa = st.checkbox("LISA", value=True)
    show_muares = st.checkbox("muAres", value=True)
    show_bbo = st.checkbox("BBO", value=True)
    show_aligo = st.checkbox("aLIGO", value=True)
    show_ce = st.checkbox("Cosmic Explorer", value=True)
    show_ptas = st.checkbox("PTA sensitivity curves", value=True)

# Observation time sliders for space-based and next-generation detectors
with st.sidebar.expander("Detector Observation Times", expanded=False):
    lisa_obs_years = st.slider("LISA (years)", min_value=1, max_value=10, value=4, step=1)
    muares_obs_years = st.slider("muAres (years)", min_value=1, max_value=10, value=10, step=1)
    bbo_obs_years = st.slider("BBO (years)", min_value=1, max_value=10, value=5, step=1)
    ce_obs_years = st.slider("CE (years)", min_value=1, max_value=10, value=1, step=1)

# PTA presets
PTA_PRESETS = {
    'NANOGrav 15yr': {'n_pulsars': 67, 'timespan': 15.0, 'sigma_ns': 300, 'cadence': 26},
    'EPTA DR2': {'n_pulsars': 25, 'timespan': 24.0, 'sigma_ns': 500, 'cadence': 20},
    'PPTA DR3': {'n_pulsars': 30, 'timespan': 18.0, 'sigma_ns': 400, 'cadence': 26},
    'MPTA': {'n_pulsars': 83, 'timespan': 4.5, 'sigma_ns': 200, 'cadence': 26},
    'CPTA': {'n_pulsars': 57, 'timespan': 3.4, 'sigma_ns': 100, 'cadence': 26},
    'IPTA DR3 (proj.)': {'n_pulsars': 115, 'timespan': 25.0, 'sigma_ns': 200, 'cadence': 26},
    'SKA-era': {'n_pulsars': 200, 'timespan': 20.0, 'sigma_ns': 50, 'cadence': 52},
    'Custom': None
}

# PTA parameters (sensitivity curves are optional; the paper figure shows
# the NANOGrav free-spectrum violins instead)
with st.sidebar.expander("PTA Parameters", expanded=False):
    pta_presets = st.multiselect(
        "Select PTAs",
        [k for k in PTA_PRESETS.keys() if k != 'Custom'],
        default=[]
    )
    
    # Custom PTA option
    show_custom_pta = st.checkbox("Add custom PTA")
    if show_custom_pta:
        pta_npsr_custom = st.slider("Number of pulsars", 10, 300, 67)
        pta_timespan_custom = st.slider("Timespan (years)", 5.0, 30.0, 15.0, step=0.5)
        pta_sigma_custom = st.select_slider(
            "Timing precision (ns)", 
            options=[30, 50, 100, 200, 300, 500, 1000],
            value=300
        )
        pta_cadence_custom = st.slider("Cadence (obs/year)", 12, 52, 26)
    else:
        pta_npsr_custom, pta_timespan_custom, pta_sigma_custom, pta_cadence_custom = 67, 15.0, 300, 26

selected_pops = st.sidebar.multiselect(
    "Select populations",
    list(POPULATIONS.keys()),
    default=list(POPULATIONS.keys())
)

# =============================================================================
# MAIN FIGURE
# =============================================================================

# Set axis based on y-axis unit choice
use_hc = (y_axis_unit == "h_c (characteristic strain)")

# Create figure (with the cumulative benchmark strip below, as in the paper)
show_budget_strip = show_ceiling and not use_hc
if show_budget_strip:
    fig, (ax, ax_budget) = plt.subplots(
        2, 1, figsize=(14, 8.2), sharex=True,
        gridspec_kw={'height_ratios': (4.0, 0.75), 'hspace': 0.04},
    )
else:
    fig, ax = plt.subplots(figsize=(14, 7))
    ax_budget = None
fig.patch.set_facecolor('white')

f_grid = F_GRID  # Use pre-computed grid
f_grid_tuple = tuple(f_grid)  # For caching
omega_cutoff = DETECTOR_PLOT_MAX

ax.set_xlim(10.0**-9.5, 3e3)
if use_hc:
    ax.set_ylim(1e-26, 1e-12)
    ax.set_ylabel(r'$h_c(f)$', fontsize=14)
else:
    ax.set_ylim(1e-18, 1e-6)
    ax.set_ylabel(r'$\Omega_{\mathrm{gw}}(f)$', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')

# NOTE: tick locators and formatters are set after all loglog() calls,
# right before tight_layout(), because loglog() resets them.

# Detectors - individual toggles
det_labels = detector_labels_hc if use_hc else detector_labels_omega

if show_muares:
    muares = get_muares_sensitivity(f_grid_tuple, T_yrs=float(muares_obs_years))
    mask_mu = (f_grid > 1e-7) & (f_grid < 1e-1) & (muares < omega_cutoff)
    plot_mu = omega_to_hc(f_grid, muares) if use_hc else muares
    ax.loglog(f_grid[mask_mu], plot_mu[mask_mu], color=CONTEXT_GRAY, ls='-.', alpha=CONTEXT_ALPHA, lw=0.9, zorder=0)
    lx, ly = det_labels['muAres']
    ax.text(lx, ly, '\u03bcAres ({0}yr)'.format(muares_obs_years), fontsize=10, color=CONTEXT_LABEL_GRAY, path_effects=LABEL_STROKE, ha='left')

if show_bbo:
    bbo = get_bbo_sensitivity(f_grid_tuple, T_yrs=float(bbo_obs_years))
    mask_bbo = (bbo > 0) & (bbo < omega_cutoff)
    plot_bbo = omega_to_hc(f_grid, bbo) if use_hc else bbo
    ax.loglog(f_grid[mask_bbo], plot_bbo[mask_bbo], color=CONTEXT_GRAY, ls='-', alpha=CONTEXT_ALPHA, lw=0.9, zorder=0)
    lx, ly = det_labels['BBO']
    ax.text(lx, ly, f'BBO ({bbo_obs_years}yr)', fontsize=10, color=CONTEXT_LABEL_GRAY, path_effects=LABEL_STROKE, ha='center')

if show_lisa:
    lisa = get_lisa_sensitivity(f_grid_tuple, T_yrs=float(lisa_obs_years))
    mask_lisa = lisa < omega_cutoff
    plot_lisa = omega_to_hc(f_grid, lisa) if use_hc else lisa
    ax.loglog(f_grid[mask_lisa], plot_lisa[mask_lisa], color=CONTEXT_GRAY, ls='--', alpha=CONTEXT_ALPHA, lw=0.9, zorder=0)
    lx, ly = det_labels['LISA']
    ax.text(lx, ly, f'LISA ({lisa_obs_years}yr)', fontsize=10, color=CONTEXT_LABEL_GRAY, path_effects=LABEL_STROKE, ha='center')

if show_aligo:
    aligo = get_aligo_design_pi(f_grid_tuple)
    mask_aligo = (aligo < 1e-4) & (aligo < omega_cutoff)
    plot_aligo = omega_to_hc(f_grid, aligo) if use_hc else aligo
    ax.loglog(f_grid[mask_aligo], plot_aligo[mask_aligo], color=CONTEXT_GRAY, ls=(0, (5.0, 2.0)), alpha=CONTEXT_ALPHA, lw=0.9, zorder=0)
    lx, ly = det_labels['aLIGO']
    ax.text(lx, ly, 'aLIGO design', fontsize=10, color=CONTEXT_LABEL_GRAY, path_effects=LABEL_STROKE, ha='center')

if show_ce:
    ce = get_ce_sensitivity(f_grid_tuple, T_yrs=float(ce_obs_years))
    mask_ce = (ce < 1e-4) & (ce < omega_cutoff)
    plot_ce = omega_to_hc(f_grid, ce) if use_hc else ce
    ax.loglog(f_grid[mask_ce], plot_ce[mask_ce], color=CONTEXT_GRAY, ls=':', alpha=CONTEXT_ALPHA, lw=0.9, zorder=0)
    lx, ly = det_labels['CE']
    ax.text(lx, ly, f'CE ({ce_obs_years}yr)', fontsize=10, color=CONTEXT_LABEL_GRAY, path_effects=LABEL_STROKE, ha='center')

# PTA sensitivity curves
if show_ptas:
    # Distinct colors and line styles for each PTA (all broken lines)
    pta_styles = {
        'NANOGrav 15yr': {'color': '#E41A1C', 'ls': '--'},     # red, dashed
        'EPTA DR2': {'color': '#377EB8', 'ls': '-.'},          # blue, dash-dot
        'PPTA DR3': {'color': '#4DAF4A', 'ls': ':'},           # green, dotted
        'MPTA': {'color': '#984EA3', 'ls': '--'},              # purple, dashed
        'CPTA': {'color': '#FF7F00', 'ls': '-.'},              # orange, dash-dot
        'IPTA DR3 (proj.)': {'color': '#A65628', 'ls': ':'},   # brown, dotted
        'SKA-era': {'color': '#F781BF', 'ls': '--'},           # pink, dashed
        'Custom': {'color': '#999999', 'ls': ':'},             # gray, dotted
    }
    
    # Plot each selected PTA
    for i, pta_name in enumerate(pta_presets):
        preset = PTA_PRESETS[pta_name]
        pta_freqs, pta_omega = get_pta_sensitivity_analytic(
            n_pulsars=preset['n_pulsars'],
            timespan=preset['timespan'],
            sigma_ns=preset['sigma_ns'],
            cadence=preset['cadence'],
            preset=pta_name
        )
        mask_pta = (pta_omega > 1e-18) & (pta_omega < 1e-5) & (pta_freqs > 1e-10) & (pta_freqs < 1e-6)
        if show_ceiling:
            mask_pta &= pta_omega < omega_cutoff
        if np.any(mask_pta):
            style = pta_styles.get(pta_name, {'color': 'gray', 'ls': '-'})
            plot_pta = omega_to_hc(pta_freqs, pta_omega) if use_hc else pta_omega
            ax.loglog(pta_freqs[mask_pta], plot_pta[mask_pta],
                     color=style['color'], ls=style['ls'], alpha=0.9, lw=1.5)
            label_text = pta_name.replace(' (proj.)', '*').replace('NANOGrav ', 'NG').replace('yr', '')
            if not use_hc and pta_name in pta_label_pos_omega:
                lx, ly, ha_lbl = pta_label_pos_omega[pta_name]
                ax.text(lx, ly, label_text, fontsize=9, color=style['color'],
                        ha=ha_lbl, va='center', fontweight='bold')
            else:
                min_idx = np.argmin(plot_pta[mask_pta])
                label_x = pta_freqs[mask_pta][min_idx]
                label_y = plot_pta[mask_pta][min_idx] * (3 if use_hc else 0.3)
                va = 'bottom' if use_hc else 'top'
                ax.text(label_x, label_y, label_text, fontsize=9, color=style['color'],
                        ha='center', va=va, fontweight='bold')
    
    # Custom PTA if enabled
    if show_custom_pta:
        pta_freqs, pta_omega = get_pta_sensitivity_analytic(
            n_pulsars=pta_npsr_custom,
            timespan=pta_timespan_custom,
            sigma_ns=pta_sigma_custom,
            cadence=pta_cadence_custom,
            preset='Custom'
        )
        mask_pta = (pta_omega > 1e-18) & (pta_omega < 1e-5) & (pta_freqs > 1e-10) & (pta_freqs < 1e-6)
        if show_ceiling:
            mask_pta &= pta_omega < omega_cutoff
        if np.any(mask_pta):
            style = pta_styles['Custom']
            plot_pta = omega_to_hc(pta_freqs, pta_omega) if use_hc else pta_omega
            ax.loglog(pta_freqs[mask_pta], plot_pta[mask_pta],
                     color=style['color'], ls=style['ls'], alpha=0.9, lw=1.5)
            min_idx = np.argmin(plot_pta[mask_pta])
            label_x = pta_freqs[mask_pta][min_idx]
            if use_hc:
                label_y = plot_pta[mask_pta][min_idx] * 3
                va = 'bottom'
            else:
                label_y = plot_pta[mask_pta][min_idx] * 0.3
                va = 'top'
            ax.text(label_x, label_y, 'Custom', fontsize=10, color=style['color'], ha='center', va=va)

# DWD foreground (Figure-2 style: thin line, dashed continuation, light fill)
if show_dwd:
    omega_wd = get_dwd_foreground(f_grid_tuple)
    solid_wd = (f_grid >= 1e-5) & (f_grid < 2e-2) & (omega_wd > PLOT_OMEGA_MIN)
    cont_wd = (f_grid >= 10.0**-5.5) & (f_grid <= 1e-5) & (omega_wd > PLOT_OMEGA_MIN)
    fill_wd = (f_grid >= 10.0**-5.5) & (f_grid < 2e-2) & (omega_wd > PLOT_OMEGA_MIN)
    if np.any(fill_wd):
        if use_hc:
            hc_wd = omega_to_hc(f_grid, omega_wd)
            ax.loglog(f_grid[solid_wd], hc_wd[solid_wd], color=DWD_GRAY, alpha=0.72, lw=1.2, zorder=2)
            ax.loglog(f_grid[cont_wd], hc_wd[cont_wd], color=DWD_GRAY, ls=(0, (3.0, 2.0)), alpha=0.66, lw=1.1, zorder=2)
            ax.fill_between(f_grid[fill_wd], 1e-26, hc_wd[fill_wd], color=DWD_GRAY, alpha=BACKGROUND_FILL_ALPHA, linewidth=0, zorder=1)
            ax.text(2e-3, 5e-18, 'DWD', fontsize=12, color='dimgray', ha='center', fontweight='bold')
        else:
            ax.loglog(f_grid[solid_wd], omega_wd[solid_wd], color=DWD_GRAY, alpha=0.72, lw=1.2, zorder=2)
            ax.loglog(f_grid[cont_wd], omega_wd[cont_wd], color=DWD_GRAY, ls=(0, (3.0, 2.0)), alpha=0.66, lw=1.1, zorder=2)
            ax.fill_between(f_grid[fill_wd], PLOT_OMEGA_MIN, omega_wd[fill_wd], color=DWD_GRAY, alpha=BACKGROUND_FILL_ALPHA, linewidth=0, zorder=1)
            ax.text(5.5e-4, 1.65e-10, 'Galactic DWD', fontsize=11, color='#5F5F5F',
                    ha='center', va='bottom', fontweight='bold', path_effects=LABEL_STROKE)

# NANOGrav free-spectrum marginal posteriors (violins, as in the paper)
if show_nanograv and not use_hc:
    ng_centers, ng_log10_omega, ng_half_width = load_nanograv_violins()
    for center, log10_omega, half_width in zip(ng_centers, ng_log10_omega, ng_half_width):
        omega_grid = 10.0**log10_omega
        ax.fill_betweenx(
            omega_grid,
            center - half_width,
            center + half_width,
            facecolor=mcolors.to_rgba(CONTEXT_GRAY, NANOGRAV_FACE_ALPHA),
            edgecolor=mcolors.to_rgba(CONTEXT_GRAY, NANOGRAV_EDGE_ALPHA),
            linewidth=0.40,
            zorder=12,
        )
    ax.text(1.45e-9, 2.5e-7, 'NANOGrav', color=CONTEXT_LABEL_GRAY,
            fontsize=11, ha='left', va='bottom', fontweight='bold',
            path_effects=LABEL_STROKE, zorder=13)

# Populations: audited paper spectra, scaled linearly with reservoir density
spec_f, spec_curves = load_benchmark_spectra()
scaled_selected_sum = np.zeros_like(spec_f)


def label_curve(curve, x_value, label, offset, color, horizontal='center', vertical='center'):
    """Annotate a spectrum at x_value with an offset-points label (paper style)."""
    finite = np.isfinite(curve)
    y_value = 10.0 ** np.interp(
        np.log10(x_value),
        np.log10(spec_f[finite]),
        np.log10(curve[finite]),
    )
    ax.annotate(
        label, xy=(x_value, y_value), xytext=offset,
        textcoords='offset points', color=color, fontsize=13,
        fontweight='bold', ha=horizontal, va=vertical,
        path_effects=LABEL_STROKE, annotation_clip=False, zorder=9,
    )


# (x anchor, label text, offset points, ha, va) — paper Figure-3 placements
POP_LABEL_SPECS = {
    'SMBHB': (5.0e-8, 'SMBHBs', (0, -20), 'center', 'top'),
    'EMRI': (2.0e-3, 'EMRI', (-10, -16), 'right', 'top'),
    'AGN-IMRI': (1.5e-2, 'IMRI', (2, 13), 'center', 'bottom'),
    'Pop III': (2.0, 'POPIII', (-4, -16), 'right', 'top'),
    'Stellar BBH': (70.0, 'sBBHs', (0, 13), 'center', 'bottom'),
    'BNS': (300.0, 'BNS', (12, -2), 'left', 'center'),
}

for name in selected_pops:
    params = POPULATIONS[name]
    key = SPECTRUM_KEYS[name]
    ratio = reservoir_ratio(params['reservoir'], rho_smbh, rho_stellar, rho_nsc)
    omega_spec = spec_curves[key] * ratio
    scaled_selected_sum += omega_spec

    curve = benchmark_display_curve(spec_f, omega_spec, key)
    continuation = benchmark_continuation_curve(spec_f, omega_spec, key)
    plot_curve = omega_to_hc(spec_f, curve) if use_hc else curve
    plot_cont = omega_to_hc(spec_f, continuation) if use_hc else continuation
    floor = 1e-26 if use_hc else PLOT_OMEGA_MIN

    ax.loglog(spec_f, plot_cont, color=params['color'],
              linestyle=(0, (3.0, 2.0)), lw=1.45, alpha=0.82, zorder=5)
    fill_curve = np.where(np.isfinite(plot_curve), plot_curve, plot_cont)
    support = np.isfinite(fill_curve) & (fill_curve >= floor)
    ax.fill_between(spec_f, floor, fill_curve, where=support,
                    color=params['color'], alpha=BACKGROUND_FILL_ALPHA,
                    linewidth=0, zorder=1)
    (pop_line,) = ax.loglog(spec_f, plot_curve, color=params['color'],
                            lw=2.05, alpha=0.98, zorder=5)
    pop_line.set_path_effects(WHITE_STROKE)

    if name == 'EMRI':
        emri_f, emri_omega = emri_high_frequency_continuation(spec_f, omega_spec)
        emri_plot = omega_to_hc(emri_f, emri_omega) if use_hc else emri_omega
        ax.fill_between(emri_f, floor, emri_plot, color=params['color'],
                        alpha=BACKGROUND_FILL_ALPHA, linewidth=0, zorder=4)
        ax.loglog(emri_f, emri_plot, color=params['color'],
                  linestyle=(0, (3.0, 2.0)), lw=1.45, alpha=0.82, zorder=5)

    if not use_hc:
        x_anchor, text, offset, ha_lbl, va_lbl = POP_LABEL_SPECS[name]
        label_curve(curve, x_anchor, text, offset, params['color'], ha_lbl, va_lbl)
    else:
        lx, ly = labels_pos_hc.get(name, (1e-4, 1e-20))
        ax.text(lx, ly, display_names.get(name, name), fontsize=14,
                color=params['color'], fontweight='bold', ha='left', va='bottom')

# Cumulative benchmark strip (paper style)
if show_budget_strip and np.any(scaled_selected_sum > 0.0):
    cumulative_fraction, e_comp = cumulative_budget_fraction(spec_f, scaled_selected_sum)
    ax_budget.fill_between(spec_f, 0.0, cumulative_fraction, color=BUDGET_RED,
                           alpha=0.08, linewidth=0.0, zorder=1)
    ax_budget.plot(spec_f, cumulative_fraction, color=BUDGET_RED, lw=1.9, zorder=3)
    ax_budget.axhline(1.0, color=BUDGET_RED, linestyle=(0, (3.0, 2.0)),
                      lw=0.8, alpha=0.52, zorder=2)
    for quantile in CUMULATIVE_QUANTILES:
        idx = int(np.searchsorted(cumulative_fraction, quantile))
        idx = min(idx, cumulative_fraction.size - 1)
        qf = spec_f[idx]
        ax_budget.axvline(qf, color=CONTEXT_LABEL_GRAY,
                          linestyle=(0, (1.0, 2.0)), lw=0.8, alpha=0.62, zorder=2)
        ax_budget.annotate(f"{quantile:.0%}", xy=(qf, 0.055), xytext=(4.0, 0.0),
                           textcoords='offset points', color=CONTEXT_LABEL_GRAY,
                           fontsize=9, ha='left', va='bottom', zorder=4)
    exponent = int(np.floor(np.log10(e_comp)))
    mantissa = e_comp / 10.0**exponent
    e_comp_text = (
        rf"$\mathcal{{E}}_{{\mathrm{{comp}}}} = "
        rf"{mantissa:.1f} \times 10^{{{exponent}}}$"
    )
    ax_budget.text(3e-3, 0.40,
                   "Integrated benchmark,  " + e_comp_text,
                   color=BUDGET_RED, fontsize=13, ha='left', va='center')
    ax_budget.set_ylim(0.0, 1.12)
    ax_budget.set_ylabel('cumulative\nfraction', fontsize=11)
    ax_budget.set_xscale('log')

ax.tick_params(axis='both', which='major', labelsize=12, length=6)
ax.tick_params(axis='both', which='minor', length=3)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# Set tick locators and formatters AFTER all loglog() calls, because loglog() resets them
bottom_ax = ax_budget if ax_budget is not None else ax
bottom_ax.set_xlabel('Frequency f [Hz]', fontsize=14)
for axes_obj in ([ax, ax_budget] if ax_budget is not None else [ax]):
    axes_obj.xaxis.set_major_locator(FixedLocator([10**i for i in range(-9, 4)]))
    axes_obj.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    axes_obj.xaxis.set_major_formatter(FuncFormatter(_log_fmt))
    axes_obj.xaxis.set_minor_formatter(NullFormatter())
if use_hc:
    ax.yaxis.set_major_locator(FixedLocator([10**i for i in range(-26, -11)]))
else:
    ax.yaxis.set_major_locator(FixedLocator([10**i for i in range(-18, -5)]))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.yaxis.set_major_formatter(FuncFormatter(_log_fmt))
ax.yaxis.set_minor_formatter(NullFormatter())
if ax_budget is not None:
    ax_budget.set_yticks([0.0, 0.5, 1.0])
    ax_budget.tick_params(axis='both', which='major', labelsize=12, length=6)
    for spine in ax_budget.spines.values():
        spine.set_linewidth(1.2)

plt.tight_layout()
st.pyplot(fig)

# Add citation to figure for downloads
fig.text(0.99, 0.01, 'Mingarelli (2026) arXiv:2601.18859',
         fontsize=8, color='gray', ha='right', va='bottom',
         transform=fig.transFigure)

# Download button for PDF
img = io.BytesIO()
try:
    fig.savefig(img, format='pdf', dpi=300, bbox_inches='tight')
    img.seek(0)
    st.download_button(
        label="Download Figure as PDF",
        data=img,
        file_name="gw_ceiling.pdf",
        mime="application/pdf"
    )
except ValueError:
    # Fallback: PDF mathtext rendering can fail in some matplotlib versions.
    # Save as PNG instead.
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300, bbox_inches='tight')
    img.seek(0)
    st.download_button(
        label="Download Figure as PNG",
        data=img,
        file_name="gw_ceiling.png",
        mime="image/png"
    )

# =============================================================================
# TABLE I - Population Parameters
# =============================================================================
st.markdown("---")
st.subheader("Declared Benchmark Channels")

table1 = """
| Channel | Reservoir | ρ_res (M☉/Mpc³) | ρ_src (M☉/Mpc³) | ε_gw | f_ref (Hz) | h_c(f_ref) | Band |
|------------|-----------|-------------|---------|------|------------|-----------|------|
| **SMBHBs** | SMBH (M₁≥10⁸) | 1.8×10⁶ | 1.2×10⁶ | moment | 1 yr⁻¹ | 2.06×10⁻¹⁵ | PTA |
| **AGN-IMRI** | SMBH | 1.8×10⁶ | 5.4×10² | 0.05 | 3×10⁻³ | 1.34×10⁻²¹ | LISA |
| **EMRI** | NSC | 6.3×10⁵ | 6.3×10¹ | 0.05 | 10⁻² | 3.48×10⁻²² | LISA |
| **BNS** | Stellar | 6.3×10⁸ | 1.7×10³ | 0.02 | 0.1 | 4.28×10⁻²⁴ | Ground |
| **Pop III BBH** | Stellar | 6.3×10⁸ | 7.3×10² | 0.05 | 0.1 | 6.80×10⁻²⁴ | Ground |
| **Stellar BBH** | Stellar | 6.3×10⁸ | 6.6×10³ | 0.05 | 25 | 4.03×10⁻²⁵ | Ground |
"""
st.markdown(table1)
st.caption("""
**ρ_res**: Reservoir mass density (SMBH from Liepold & Ma 2024, ApJL 971, L29;
stellar = 5×10⁻³ ρ_crit; NSC = 10⁻³ ρ_★).
**ρ_src**: Processed source density, ρ_src = f_merge × ρ_res.
**ε_gw**: Radiative efficiency. **h_c(f_ref)**: Benchmark characteristic strain at f_ref.
The SMBHB benchmark is a one-pass population moment over the erratum-corrected
dynamical mass function (Newtonian circular inspiral to each source-frame
Schwarzschild ISCO), not an ε·ρ product.
A benchmark is one observationally or theoretically motivated choice; a ceiling
is the maximum over a stated allowed domain. Amplitudes scale as A ∝ √ρ
relative to fiducial values. All values from the paper's audited
step3-numbers.json.
""")

# =============================================================================
# Current Amplitude Values
# =============================================================================
st.markdown("---")
st.subheader("Current Amplitude Values")
cols = st.columns(3)
for i, name in enumerate(selected_pops):
    params = POPULATIONS[name]
    A_current = scale_amplitude(params['A_bench'], params['reservoir'], rho_smbh, rho_stellar, rho_nsc)
    col_idx = i % 3
    cols[col_idx].metric(
        display_names[name],
        f"A = {A_current:.2e}",
        f"f_ref = {params['f_ref']:.2e} Hz"
    )

# =============================================================================
# SMBHB Ceiling Comparison
# =============================================================================
st.markdown("---")
st.subheader("SMBHB Benchmark vs PTA Measurements")
st.markdown("""
**SMBHB benchmark and conditional ceiling (erratum-corrected dynamical mass
function, participating domain M₁ ≥ 10⁸ M☉):**

- One-pass population-moment benchmark: **A_bench = 2.06 × 10⁻¹⁵** at f_ref = 1 yr⁻¹
- Conditional reference-frequency ceiling: **A_ceil = 2.23 × 10⁻¹⁵**
- NANOGrav customized-noise (CNM) amplitude: 2.1 (range 1.6–2.7) × 10⁻¹⁵

| PTA | A (×10⁻¹⁵) at γ=13/3 | A / A_bench |
|-----|------------|------------------------|
| CPTA | 2.0 +0.9/-1.9 dex (95%) | 0.97 |
| PPTA DR3 | 2.04 ± 0.24 | 0.99 |
| NANOGrav 15yr | 2.4 +0.7/-0.6 | 1.17 |
| EPTA DR2 | 2.5 ± 0.7 | 1.22 |
| MPTA | 4.8 +0.8/-0.9 | 2.33 |

Within this fixed population family, the central NANOGrav CNM amplitude maps to
a merged mass fraction **f_merge = 1.04 ≃ 1**: the measured background saturates
the one-pass energetic budget of the M₁ ≥ 10⁸ M☉ population. Amplitudes above
the conditional ceiling require conditions outside the stated domain (e.g.
scatter in scaling relations, additional participating mass, cosmic variance
from nearby massive binaries, or mis-modeled pulsar noise).
""")

# =============================================================================
# PTA SECTION (moved to bottom)
# =============================================================================
if show_ptas and (len(pta_presets) > 0 or show_custom_pta):
    st.markdown("---")
    st.subheader("PTA Sensitivity Curves")
    st.markdown("""
    PTA sensitivity curves are calibrated to each array's published GWB amplitude at fixed γ=13/3.
    Projections (IPTA DR3, SKA-era) are scaled from similar existing arrays.
    """)
    
    pta_table = """
| PTA | N_psr | Timespan | σ_RMS | Cadence | A (γ=13/3) | Reference |
|-----|-------|----------|-------|---------|------------|-----------|
| NANOGrav 15yr | 67 | 15 yr | 300 ns | 26/yr | 2.4×10⁻¹⁵ | [Agazie et al. (2023)](https://arxiv.org/abs/2306.16213) |
| EPTA DR2 | 25 | 24 yr | 500 ns | 20/yr | 2.5×10⁻¹⁵ | [EPTA Collab. (2023)](https://arxiv.org/abs/2306.16214) |
| PPTA DR3 | 30 | 18 yr | 400 ns | 26/yr | 2.0×10⁻¹⁵ | [Reardon et al. (2023)](https://arxiv.org/abs/2306.16215) |
| CPTA | 57 | 3.4 yr | 100 ns | 26/yr | 2.0×10⁻¹⁵ | [Xu et al. (2023)](https://arxiv.org/abs/2306.16216) |
| MPTA | 83 | 4.5 yr | 200 ns | 26/yr | 4.8×10⁻¹⁵ | [Miles et al. (2025)](https://arxiv.org/abs/2412.01153) |
| IPTA DR3 (proj.) | ~115 | 25 yr | 200 ns | 26/yr | — | ~2.5×: h_c ∝ 1/√(N_pairs × T) |
| SKA-era | 200 | 20 yr | 50 ns | 52/yr | — | [Shannon et al. (2025)](https://arxiv.org/abs/2512.16163) |
"""
    st.markdown(pta_table)
    st.caption("All amplitudes A are at **fixed γ=13/3** (α=-2/3). σ_RMS values are approximate array-averaged timing precisions.")
    st.caption("IPTA DR3 scaling: h_c ∝ 1/√(N_pairs × T), where N_pairs = N(N-1)/2, see e.g. Siemens et al. (2013). With ~115 pulsars (6555 pairs vs NANOGrav's 2211) and 25-year baseline, improvement ≈ √(3.0 × 1.7) ≈ 2.2×, and with additional gains from combined noise modeling 2.5x is reasonable..")
    st.caption("PTA sensitivity curves use the formalism of [Hazboun, Romano & Smith (2019)](https://arxiv.org/abs/1907.04341), implemented in [hasasia](https://github.com/Hazboun6/hasasia).")

# =============================================================================
# CITATION
# =============================================================================
st.markdown("---")
st.markdown("""
**Citation:** If you use figures from this tool, please cite:

> Mingarelli, C. M. F. (2026), "Energetic Ceilings on Astrophysical Gravitational-Wave Backgrounds",
> [arXiv:2601.18859](https://arxiv.org/abs/2601.18859)
""")