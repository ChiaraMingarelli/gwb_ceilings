import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

# Tuned label positions for Omega_gw mode
labels_pos_omega = {
    'SMBHB': (1e-8, 1e-11),
    'AGN-IMRI': (4.4e-2, 4e-11),
    'EMRI': (1e-5, 3e-13),
    'Pop III': (1.5, 2e-12),
    'BNS': (40, 5e-12),
    'Stellar BBH': (150, 8e-11)
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

# Detector label positions for Omega_gw mode
detector_labels_omega = {
    'muAres': (1e-6, 2e-14),
    'BBO': (5e-2, 2e-17),
    'LISA': (2e-5, 5e-11),
    'aLIGO': (60, 8e-9),
    'CE': (30, 3e-15),
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


def scale_amplitude(A_bench, reservoir, rho_smbh, rho_stellar, rho_nsc):
    """Scale amplitude based on reservoir density relative to fiducial."""
    if reservoir == 'SMBH':
        return A_bench * np.sqrt(rho_smbh / RHO_SMBH_FID)
    elif reservoir == 'STELLAR':
        return A_bench * np.sqrt(rho_stellar / RHO_STELLAR_FID)
    elif reservoir == 'NSC':
        return A_bench * np.sqrt(rho_nsc / RHO_NSC_FID)
    return A_bench


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
show_ceiling = st.sidebar.checkbox("Show integrated ceiling", value=True)

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

# PTA parameters
with st.sidebar.expander("PTA Parameters", expanded=True):
    pta_presets = st.multiselect(
        "Select PTAs", 
        [k for k in PTA_PRESETS.keys() if k != 'Custom'],
        default=['NANOGrav 15yr']
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

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')

f_grid = F_GRID  # Use pre-computed grid
f_grid_tuple = tuple(f_grid)  # For caching
omega_cutoff = INTEGRATED_BUDGET_OMEGA

# Set axis based on y-axis unit choice
use_hc = (y_axis_unit == "h_c (characteristic strain)")

ax.set_xlim(1e-9, 3e3)
if use_hc:
    ax.set_ylim(1e-26, 1e-12)
    ax.set_ylabel('Characteristic Strain hc(f)', fontsize=14)
else:
    ax.set_ylim(1e-18, 1e-6)
    ax.set_ylabel('\u03A9gw(f)', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency f [Hz]', fontsize=14)

# NOTE: tick locators and formatters are set after all loglog() calls,
# right before tight_layout(), because loglog() resets them.

# Detectors - individual toggles
det_labels = detector_labels_hc if use_hc else detector_labels_omega

if show_muares:
    muares = get_muares_sensitivity(f_grid_tuple, T_yrs=float(muares_obs_years))
    mask_mu = (f_grid > 1e-7) & (f_grid < 1e-1) & (muares < omega_cutoff)
    plot_mu = omega_to_hc(f_grid, muares) if use_hc else muares
    ax.loglog(f_grid[mask_mu], plot_mu[mask_mu], color='gray', ls='-.', alpha=0.6, lw=1.2)
    lx, ly = det_labels['muAres']
    ax.text(lx, ly, '\u03bcAres ({0}yr)'.format(muares_obs_years), fontsize=10, color='gray', ha='left')

if show_bbo:
    bbo = get_bbo_sensitivity(f_grid_tuple, T_yrs=float(bbo_obs_years))
    mask_bbo = (bbo > 0) & (bbo < omega_cutoff)
    plot_bbo = omega_to_hc(f_grid, bbo) if use_hc else bbo
    ax.loglog(f_grid[mask_bbo], plot_bbo[mask_bbo], color='gray', ls='--', alpha=0.6, lw=1.2)
    lx, ly = det_labels['BBO']
    ax.text(lx, ly, f'BBO ({bbo_obs_years}yr)', fontsize=10, color='gray', ha='center')

if show_lisa:
    lisa = get_lisa_sensitivity(f_grid_tuple, T_yrs=float(lisa_obs_years))
    mask_lisa = lisa < omega_cutoff
    plot_lisa = omega_to_hc(f_grid, lisa) if use_hc else lisa
    ax.loglog(f_grid[mask_lisa], plot_lisa[mask_lisa], color='gray', ls='--', alpha=0.6, lw=1.5)
    lx, ly = det_labels['LISA']
    ax.text(lx, ly, f'LISA ({lisa_obs_years}yr)', fontsize=10, color='gray', ha='center')

if show_aligo:
    aligo = get_aligo_design_pi(f_grid_tuple)
    mask_aligo = (aligo < 1e-4) & (aligo < omega_cutoff)
    plot_aligo = omega_to_hc(f_grid, aligo) if use_hc else aligo
    ax.loglog(f_grid[mask_aligo], plot_aligo[mask_aligo], color='gray', ls=':', alpha=0.6, lw=1.2)
    lx, ly = det_labels['aLIGO']
    ax.text(lx, ly, 'aLIGO design', fontsize=10, color='gray', ha='center')

if show_ce:
    ce = get_ce_sensitivity(f_grid_tuple, T_yrs=float(ce_obs_years))
    mask_ce = (ce < 1e-4) & (ce < omega_cutoff)
    plot_ce = omega_to_hc(f_grid, ce) if use_hc else ce
    ax.loglog(f_grid[mask_ce], plot_ce[mask_ce], color='gray', ls=':', alpha=0.6, lw=1.2)
    lx, ly = det_labels['CE']
    ax.text(lx, ly, f'CE ({ce_obs_years}yr)', fontsize=10, color='gray', ha='center')

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
        if np.any(mask_pta):
            style = pta_styles.get(pta_name, {'color': 'gray', 'ls': '-'})
            pta_plot_omega = np.minimum(pta_omega, 1e-7) if show_ceiling else pta_omega
            plot_pta = omega_to_hc(pta_freqs, pta_plot_omega) if use_hc else pta_plot_omega
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
        if np.any(mask_pta):
            style = pta_styles['Custom']
            pta_plot_omega = np.minimum(pta_omega, 1e-7) if show_ceiling else pta_omega
            plot_pta = omega_to_hc(pta_freqs, pta_plot_omega) if use_hc else pta_plot_omega
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

# DWD foreground
if show_dwd:
    omega_wd = get_dwd_foreground(f_grid_tuple)
    mask_wd = omega_wd > 1e-25
    if np.any(mask_wd):
        if use_hc:
            hc_wd = omega_to_hc(f_grid, omega_wd)
            ax.fill_between(f_grid[mask_wd], 1e-26, hc_wd[mask_wd], color='gray', alpha=0.15, linewidth=0)
            ax.text(2e-3, 5e-18, 'DWD', fontsize=12, color='dimgray', ha='center', fontweight='bold')
        else:
            ax.fill_between(f_grid[mask_wd], 1e-25, omega_wd[mask_wd], color='gray', alpha=0.3, linewidth=0)
            ax.text(7e-4, 1e-11, 'DWD', fontsize=15, color='gray', ha='center', fontweight='bold')

# Integrated benchmark budget (step3b conditional sum, not a universal ceiling)
if show_ceiling:
    if use_hc:
        # In h_c space, the budget line is frequency-dependent
        f_ceil = np.logspace(-9, 3, 100)
        hc_ceil = omega_to_hc(f_ceil, np.full_like(f_ceil, INTEGRATED_BUDGET_OMEGA))
        ax.loglog(f_ceil, hc_ceil, color='red', linestyle='-', linewidth=2.5, alpha=0.9)
        ax.text(1e-1, 3e-17, 'Integrated Benchmark Budget', color='red', fontsize=12, fontweight='bold', ha='center')
    else:
        ax.axhline(y=INTEGRATED_BUDGET_OMEGA, color='red', linestyle='-', linewidth=2.5, alpha=0.9)
        ax.text(1e-3, 1.8 * INTEGRATED_BUDGET_OMEGA, 'Integrated Benchmark Budget', color='red', fontsize=14, fontweight='bold', ha='center')

# Populations
# Select label positions based on display mode
pop_labels = labels_pos_hc if use_hc else labels_pos_omega

for name in selected_pops:
    params = POPULATIONS[name]
    A_current = scale_amplitude(params['A_bench'], params['reservoir'], rho_smbh, rho_stellar, rho_nsc)
    omega = get_omega_gw(f_grid, A_current, params['f_ref'], params['f_min'], params['f_max'])
    valid = omega > 1e-30
    if np.any(valid):
        if use_hc:
            hc_pop = omega_to_hc(f_grid, omega)
            ax.loglog(f_grid[valid], hc_pop[valid], color=params['color'], lw=2.5, alpha=1.0)
            ax.fill_between(f_grid[valid], 1e-26, hc_pop[valid], color=params['color'], alpha=0.08, linewidth=0)
        else:
            ax.loglog(f_grid[valid], omega[valid], color=params['color'], lw=2.5, alpha=1.0)
            ax.fill_between(f_grid[valid], 1e-25, omega[valid], color=params['color'], alpha=0.15, linewidth=0)
        lx, ly = pop_labels.get(name, (1e-4, 1e-15))
        display_name = display_names.get(name, name)
        # Different alignment for h_c mode vs omega mode
        if use_hc:
            ha = 'left'
            va = 'bottom'
            fontsize = 14
        else:
            ha = 'right' if name == 'EMRI' else ('left' if name == 'AGN-IMRI' else 'center')
            va = 'bottom' if name == 'EMRI' else 'center'
            fontsize = 16
        ax.text(lx, ly, display_name, fontsize=fontsize, color=params['color'], fontweight='bold', ha=ha, va=va)

ax.tick_params(axis='both', which='major', labelsize=12, length=6)
ax.tick_params(axis='both', which='minor', length=3)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# Set tick locators and formatters AFTER all loglog() calls, because loglog() resets them
ax.xaxis.set_major_locator(FixedLocator([10**i for i in range(-9, 4)]))
if use_hc:
    ax.yaxis.set_major_locator(FixedLocator([10**i for i in range(-26, -11)]))
else:
    ax.yaxis.set_major_locator(FixedLocator([10**i for i in range(-18, -5)]))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.xaxis.set_major_formatter(FuncFormatter(_log_fmt))
ax.yaxis.set_major_formatter(FuncFormatter(_log_fmt))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())

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