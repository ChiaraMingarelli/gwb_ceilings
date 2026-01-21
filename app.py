import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import io

# Physical Constants
H0_km_s_Mpc = 67.4
H0 = H0_km_s_Mpc * 1000.0 / 3.08567758e22
h = H0_km_s_Mpc / 100.0

# Fiducial reservoir densities (Msun/Mpc^3)
RHO_SMBH_FID = 4.2e5
RHO_STELLAR_FID = 5.9e8
RHO_NSC_FID = 1.4e6

# Population parameters from Table I
POPULATIONS = {
    'SMBHB': {
        'reservoir': 'SMBH',
        'f_ref': 3.2e-8,
        'A_bench': 1.0e-15,
        'f_min': 1e-9,
        'f_max': 4e-7,
        'f_merge_fid': 0.1,
        'epsilon_gw': 0.02,
        'color': '#0072B2',
    },
    'IMBH-SMBH': {
        'reservoir': 'SMBH',
        'f_ref': 3e-3,
        'A_bench': 1.1e-20,
        'f_min': 1e-5,
        'f_max': 4e-2,
        'f_merge_fid': 0.05,
        'epsilon_gw': 0.05,
        'color': '#D55E00',
    },
    'EMRI': {
        'reservoir': 'NSC',
        'f_ref': 1e-2,
        'A_bench': 1.1e-20,
        'f_min': 1e-5,
        'f_max': 1e-2,
        'f_merge_fid': 0.1,
        'epsilon_gw': 0.05,
        'color': '#009E73',
    },
    'BNS': {
        'reservoir': 'STELLAR',
        'f_ref': 0.1,
        'A_bench': 6.7e-24,
        'f_min': 1e-2,
        'f_max': 1500.0,
        'f_merge_fid': 1.1e-5,
        'epsilon_gw': 0.01,
        'color': '#CC79A7',
    },
    'Pop III': {
        'reservoir': 'STELLAR',
        'f_ref': 0.1,
        'A_bench': 4.9e-24,
        'f_min': 1e-2,
        'f_max': 200.0,
        'f_merge_fid': 3.0e-7,
        'epsilon_gw': 0.05,
        'color': '#56B4E9',
    },
    'Stellar BBH': {
        'reservoir': 'STELLAR',
        'f_ref': 25.0,
        'A_bench': 9.5e-25,
        'f_min': 5.0,
        'f_max': 200.0,
        'f_merge_fid': 1.8e-5,
        'epsilon_gw': 0.05,
        'color': '#E69F00',
    },
}

# Tuned label positions
labels_pos = {
    'SMBHB': (1e-8, 1e-11),
    'IMBH-SMBH': (4.4e-2, 1e-9),
    'EMRI': (1e-5, 1e-10 * 10**0.1),
    'Pop III': (0.1 * 10**0.3, 5e-14 * 10**(-0.6)),
    'BNS': (1e-1 * 10**0.2, 1e-11 * 10**(-0.5)),
    'Stellar BBH': (10 * 10**0.3, 3e-15)
}

display_names = {
    'SMBHB': 'SMBHBs',
    'IMBH-SMBH': 'IMBH',
    'EMRI': 'EMRI',
    'Pop III': 'POPIII',
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


def get_lisa_sensitivity(f, T_yrs=10.0):
    """LISA sensitivity curve in Omega_gw."""
    L = 2.5e9
    f_star = 19.09e-3
    P_oms = (1.5e-11)**2 * (1 + (2e-3/f)**4)
    P_acc = (3e-15)**2 * (1 + (0.4e-3/f)**2) * (1 + (f/8e-3)**4)
    Sn = 10/(3*L**2) * (P_oms + 4*P_acc/((2*np.pi*f)**4) * (1 + 0.6*(f/f_star)**2))
    omega_n = (4 * np.pi**2 / (3 * H0**2)) * f**3 * Sn
    T_sec = T_yrs * 365.25 * 24 * 3600
    return omega_n / np.sqrt(T_sec * f)


def get_muares_sensitivity(f, T_yrs=10.0):
    """mu-Ares sensitivity curve in Omega_gw."""
    L = 3.95e11
    f_star = 3e8 / (2 * np.pi * L)
    S_pos = 1e-24
    S_acc = 9e-30 * (1 + (1e-4/f)**2)
    Sn = (20/3) * (1/L**2) * (4 * S_acc / (2 * np.pi * f)**4 + S_pos) * (1 + (f/f_star)**2)
    omega_n = (4 * np.pi**2 / (3 * H0**2)) * f**3 * Sn
    T_sec = T_yrs * 365.25 * 24 * 3600
    return omega_n / np.sqrt(T_sec * f)


def get_ce_sensitivity(f, T_yrs=1.0):
    """Cosmic Explorer approximate sensitivity in Omega_gw."""
    omega = np.full_like(f, 1e-1)
    mask = (f > 5) & (f < 4000)
    if np.sum(mask) == 0:
        return omega
    f_band = f[mask]
    Sn_proxy = 2e-51 * ((f_band/40)**(-4) + 0.5 + (f_band/200)**2)
    omega_n = (4 * np.pi**2 / (3 * H0**2)) * f_band**3 * Sn_proxy
    T_sec = T_yrs * 365.25 * 24 * 3600
    omega[mask] = omega_n / np.sqrt(T_sec * f_band)
    return omega


def get_aligo_approx(f):
    """aLIGO approximate sensitivity in Omega_gw."""
    omega = np.full_like(f, 1e-5)
    mask = (f > 10) & (f < 2000)
    f_band = f[mask]
    omega[mask] = 1e-9 * ((f_band/50.0)**(-2) + (f_band/50.0)**3)
    return omega


def get_bbo_approx(f):
    """BBO approximate sensitivity in Omega_gw."""
    mask = (f > 1e-3) & (f < 100)
    L_bbo = 5.0e7
    S_pos, S_acc = 2.0e-34, 9.0e-34
    f_star = 3.0e8 / (2.0 * np.pi * L_bbo)
    fm = f[mask]
    if len(fm) == 0:
        return np.zeros_like(f)
    x = fm / f_star
    Sx = (4.0 * S_pos / L_bbo**2) * (1 + np.cos(x)**2) + (16.0 * S_acc / (L_bbo**2 * (2*np.pi*fm)**4)) * (1 + np.cos(x)**2)
    T_obs = 5.0 * 3.15e7
    omega = np.zeros_like(f)
    omega[mask] = (2 * np.pi**2 / (3 * H0**2)) * fm**3 * np.sqrt(Sx**2 / (2 * T_obs))
    return omega


def get_dwd_foreground(f):
    """Double white dwarf foreground in Omega_gw."""
    A_wd, f_knee = 3e-10, 3e-3
    mask = (f > 1e-4) & (f < 2e-2)
    omega = np.zeros_like(f)
    if np.sum(mask) == 0:
        return omega
    ff = f[mask]
    omega[mask] = A_wd * (ff / 1e-3)**(2/3) * np.exp(-(ff / f_knee)**2)
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
    
    # Cap at integrated astrophysical ceiling (Omega_gw < 1e-7)
    omega_gw = np.minimum(omega_gw, 1e-7)
    
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

**Reference:** Mingarelli (2026), *Energetic Ceilings on Astrophysical Gravitational-Wave Backgrounds*
""")

# Sidebar controls
st.sidebar.header("Mass Reservoirs")
st.sidebar.markdown("Adjust reservoir densities (M☉/Mpc³)")

# Initialize session state with fiducial values
if 'rho_smbh_val' not in st.session_state:
    st.session_state.rho_smbh_val = 4.2
if 'rho_stellar_val' not in st.session_state:
    st.session_state.rho_stellar_val = 5.9
if 'rho_nsc_val' not in st.session_state:
    st.session_state.rho_nsc_val = 1.4

# Reset button
if st.sidebar.button("Reset to Table I values"):
    st.session_state.rho_smbh_val = 4.2
    st.session_state.rho_stellar_val = 5.9
    st.session_state.rho_nsc_val = 1.4
    st.rerun()

rho_smbh = st.sidebar.slider(
    "ρ_SMBH (×10⁵)",
    min_value=1.0, max_value=10.0,
    value=st.session_state.rho_smbh_val,
    step=0.1,
    key='rho_smbh_val'
) * 1e5

rho_stellar = st.sidebar.slider(
    "ρ_★ (×10⁸)",
    min_value=1.0, max_value=10.0,
    value=st.session_state.rho_stellar_val,
    step=0.1,
    key='rho_stellar_val'
) * 1e8

rho_nsc = st.sidebar.slider(
    "ρ_NSC (×10⁶)",
    min_value=0.5, max_value=5.0,
    value=st.session_state.rho_nsc_val,
    step=0.1,
    key='rho_nsc_val'
) * 1e6

st.sidebar.header("Display Options")
y_axis_unit = st.sidebar.radio("Y-axis", ["Ω_gw", "h_c (characteristic strain)"], index=0, horizontal=True)
show_detectors = st.sidebar.checkbox("Show detector curves", value=True)
show_pta = st.sidebar.checkbox("Show PTA sensitivity", value=True)
show_dwd = st.sidebar.checkbox("Show DWD foreground", value=True)
show_ceiling = st.sidebar.checkbox("Show integrated ceiling", value=True)

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
    pta_preset = st.selectbox("Preset", list(PTA_PRESETS.keys()), index=0)
    
    if pta_preset != 'Custom':
        preset = PTA_PRESETS[pta_preset]
        pta_npsr = preset['n_pulsars']
        pta_timespan = preset['timespan']
        pta_sigma = preset['sigma_ns']
        pta_cadence = preset['cadence']
        st.caption(f"N={pta_npsr}, T={pta_timespan}yr, σ={pta_sigma}ns, cad={pta_cadence}/yr")
    else:
        pta_npsr = st.slider("Number of pulsars", 10, 300, 67)
        pta_timespan = st.slider("Timespan (years)", 5.0, 30.0, 15.0, step=0.5)
        pta_sigma = st.select_slider(
            "Timing precision (ns)", 
            options=[30, 50, 100, 200, 300, 500, 1000],
            value=300
        )
        pta_cadence = st.slider("Cadence (obs/year)", 12, 52, 26)

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

f_grid = np.logspace(-9.5, 3.5, 3000)
omega_cutoff = 1e-7

# Set axis based on y-axis unit choice
use_hc = (y_axis_unit == "h_c (characteristic strain)")

ax.set_xlim(1e-9, 3e3)
if use_hc:
    ax.set_ylim(1e-26, 1e-12)
    ax.set_ylabel(r'Characteristic Strain $h_c(f)$', fontsize=14)
else:
    ax.set_ylim(1e-18, 1e-6)
    ax.set_ylabel(r'$\Omega_{\mathrm{gw}}(f)$', fontsize=14)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency f [Hz]', fontsize=14)

ax.xaxis.set_major_locator(LogLocator(base=10, numticks=20))
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))

# Detectors
if show_detectors:
    muares = get_muares_sensitivity(f_grid, T_yrs=10.0)
    mask_mu = (f_grid > 1e-7) & (f_grid < 1e-1) & (muares < omega_cutoff)
    plot_mu = omega_to_hc(f_grid, muares) if use_hc else muares
    ax.loglog(f_grid[mask_mu], plot_mu[mask_mu], color='gray', ls='-.', alpha=0.6, lw=1.2)
    ax.text(1e-6, omega_to_hc(np.array([1e-6]), np.array([1e-11]))[0] if use_hc else 1e-11, 'μAres', fontsize=10, color='gray', ha='left')

    bbo = get_bbo_approx(f_grid)
    mask_bbo = (bbo > 0) & (bbo < omega_cutoff)
    plot_bbo = omega_to_hc(f_grid, bbo) if use_hc else bbo
    ax.loglog(f_grid[mask_bbo], plot_bbo[mask_bbo], color='gray', ls='-', alpha=0.6, lw=1.2)
    ax.text(5e-2, omega_to_hc(np.array([5e-2]), np.array([2e-17]))[0] if use_hc else 2e-17, 'BBO', fontsize=10, color='gray', ha='center')

    lisa = get_lisa_sensitivity(f_grid)
    mask_lisa = lisa < omega_cutoff
    plot_lisa = omega_to_hc(f_grid, lisa) if use_hc else lisa
    ax.loglog(f_grid[mask_lisa], plot_lisa[mask_lisa], color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.text(2e-5, omega_to_hc(np.array([2e-5]), np.array([8e-10]))[0] if use_hc else 8e-10, 'LISA', fontsize=10, color='gray', ha='center')

    aligo = get_aligo_approx(f_grid)
    mask_aligo = (aligo < 1e-4) & (aligo < omega_cutoff)
    plot_aligo = omega_to_hc(f_grid, aligo) if use_hc else aligo
    ax.loglog(f_grid[mask_aligo], plot_aligo[mask_aligo], color='gray', ls=':', alpha=0.6, lw=1.2)
    ax.text(70 * 10 * 10**(-0.4), omega_to_hc(np.array([700]), np.array([2e-9 * 10**0.5]))[0] if use_hc else 2e-9 * 10**0.5, 'aLIGO', fontsize=10, color='gray', ha='center')

    ce = get_ce_sensitivity(f_grid, T_yrs=1.0)
    mask_ce = (ce < 1e-4) & (ce < omega_cutoff)
    plot_ce = omega_to_hc(f_grid, ce) if use_hc else ce
    ax.loglog(f_grid[mask_ce], plot_ce[mask_ce], color='gray', ls=':', alpha=0.6, lw=1.2)
    ax.text(100, omega_to_hc(np.array([100]), np.array([1e-14]))[0] if use_hc else 1e-14, 'CE', fontsize=10, color='gray', ha='center')

# PTA sensitivity (analytic approximation)
if show_pta:
    pta_freqs, pta_omega = get_pta_sensitivity_analytic(
        n_pulsars=pta_npsr,
        timespan=pta_timespan,
        sigma_ns=pta_sigma,
        cadence=pta_cadence,
        preset=pta_preset
    )
    # Less restrictive mask - show curve even if above ceiling
    mask_pta = (pta_omega > 1e-18) & (pta_omega < 1e-5) & (pta_freqs > 1e-10) & (pta_freqs < 1e-6)
    if np.any(mask_pta):
        plot_pta = omega_to_hc(pta_freqs, pta_omega) if use_hc else pta_omega
        ax.loglog(pta_freqs[mask_pta], plot_pta[mask_pta], 
                 color='purple', ls='--', alpha=0.8, lw=1.5)
        # Position label near the curve minimum - show preset name
        label_text = pta_preset.replace(' (projected)', '').replace('yr', '')
        label_y = omega_to_hc(np.array([5e-8]), np.array([5e-10]))[0] if use_hc else 5e-10
        ax.text(5e-8, label_y, label_text, fontsize=9, color='purple', ha='center')

# DWD foreground
if show_dwd:
    omega_wd = get_dwd_foreground(f_grid)
    mask_wd = omega_wd > 1e-25
    if np.any(mask_wd):
        if use_hc:
            hc_wd = omega_to_hc(f_grid, omega_wd)
            ax.fill_between(f_grid[mask_wd], 1e-26, hc_wd[mask_wd], color='gray', alpha=0.3, linewidth=0)
            ax.text(3e-3 / 10 * 10**0.4, 1e-19, 'DWD', fontsize=15, color='gray', ha='center', fontweight='bold')
        else:
            ax.fill_between(f_grid[mask_wd], 1e-25, omega_wd[mask_wd], color='gray', alpha=0.3, linewidth=0)
            ax.text(3e-3 / 10 * 10**0.4, 1e-12, 'DWD', fontsize=15, color='white', ha='center', fontweight='bold')

# Ceiling
if show_ceiling:
    if use_hc:
        # In h_c space, the ceiling is frequency-dependent: h_c = sqrt(Omega / (prefac * f^2))
        # At f=1e-8 Hz, Omega=1e-7 -> h_c ~ 3e-14
        # At f=1 Hz, Omega=1e-7 -> h_c ~ 3e-22
        # Draw a line showing where Omega = 1e-7 in h_c space
        f_ceil = np.logspace(-9, 3, 100)
        hc_ceil = omega_to_hc(f_ceil, np.full_like(f_ceil, 1e-7))
        ax.loglog(f_ceil, hc_ceil, color='red', linestyle='-', linewidth=2.5, alpha=0.9)
        ax.text(1e-3, omega_to_hc(np.array([1e-3]), np.array([2e-7]))[0], 'Integrated Ceiling', color='red', fontsize=14, fontweight='bold', ha='center')
    else:
        ax.axhline(y=1e-7, color='red', linestyle='-', linewidth=2.5, alpha=0.9)
        ax.text(1e-3, 1.8e-7, 'Integrated Astrophysical Ceiling', color='red', fontsize=18, fontweight='bold', ha='center')

# Populations
for name in selected_pops:
    params = POPULATIONS[name]
    A_current = scale_amplitude(params['A_bench'], params['reservoir'], rho_smbh, rho_stellar, rho_nsc)
    omega = get_omega_gw(f_grid, A_current, params['f_ref'], params['f_min'], params['f_max'])
    valid = omega > 1e-30
    if np.any(valid):
        if use_hc:
            hc_pop = omega_to_hc(f_grid, omega)
            ax.loglog(f_grid[valid], hc_pop[valid], color=params['color'], lw=2.5, alpha=1.0)
            ax.fill_between(f_grid[valid], 1e-26, hc_pop[valid], color=params['color'], alpha=0.15, linewidth=0)
        else:
            ax.loglog(f_grid[valid], omega[valid], color=params['color'], lw=2.5, alpha=1.0)
            ax.fill_between(f_grid[valid], 1e-25, omega[valid], color=params['color'], alpha=0.15, linewidth=0)
        lx, ly = labels_pos.get(name, (1e-4, 1e-15))
        # Convert label y position if using h_c
        if use_hc:
            ly = omega_to_hc(np.array([lx]), np.array([ly]))[0]
        display_name = display_names.get(name, name)
        ha = 'right' if name == 'EMRI' else ('left' if name == 'IMBH-SMBH' else 'center')
        va = 'bottom' if name == 'EMRI' else 'center'
        ax.text(lx, ly, display_name, fontsize=18, color=params['color'], fontweight='bold', ha=ha, va=va)

ax.tick_params(axis='both', which='major', labelsize=12, length=6)
ax.tick_params(axis='both', which='minor', length=3)
ax.grid(False)
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

plt.tight_layout()
st.pyplot(fig)

# Download button for PDF
img = io.BytesIO()
fig.savefig(img, format='pdf', dpi=300, bbox_inches='tight')
img.seek(0)
st.download_button(
    label="Download Figure as PDF",
    data=img,
    file_name="gw_ceiling.pdf",
    mime="application/pdf"
)

# =============================================================================
# TABLE I - Population Parameters
# =============================================================================
st.markdown("---")
st.subheader("Table I: GWB Population Parameters")

table1 = """
| Population | Reservoir | ρ (M☉/Mpc³) | f_merge | ε_gw | f_ref (Hz) | A_ceiling | Band |
|------------|-----------|-------------|---------|------|------------|-----------|------|
| **SMBHBs** | SMBH | 4.2×10⁵ | 0.1 | 0.02 | 3.2×10⁻⁸ | 1.0×10⁻¹⁵ | PTA |
| **IMBH-SMBH** | SMBH | 4.2×10⁵ | 0.05 | 0.05 | 3×10⁻³ | 1.1×10⁻²⁰ | LISA |
| **EMRI** | NSC | 1.4×10⁶ | 0.1 | 0.05 | 10⁻² | 1.1×10⁻²⁰ | LISA |
| **BNS** | Stellar | 5.9×10⁸ | 1.1×10⁻⁵ | 0.01 | 0.1 | 6.7×10⁻²⁴ | Ground |
| **Pop III BBH** | Stellar | 5.9×10⁸ | 3×10⁻⁷ | 0.05 | 0.1 | 4.9×10⁻²⁴ | Ground |
| **Stellar BBH** | Stellar | 5.9×10⁸ | 1.8×10⁻⁵ | 0.05 | 25 | 9.5×10⁻²⁵ | Ground |
"""
st.markdown(table1)
st.caption("""
**ρ**: Mass density reservoir. **f_merge**: Fraction of reservoir that merges within a Hubble time. 
**ε_gw**: Radiative efficiency. **A_ceiling**: Maximum characteristic strain amplitude at f_ref.
Amplitudes scale as A ∝ √ρ relative to fiducial values.
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
# SMBHB Ceiling Tension
# =============================================================================
st.markdown("---")
st.subheader("SMBHB Ceiling Tension")
st.markdown("""
**Key result:** All PTA-measured GWB amplitudes **exceed** the SMBHB energetic ceiling (A ≤ 1.0×10⁻¹⁵).

The SMBHB ceiling (cyan curve) is *not* a sensitivity limit—it is the **maximum amplitude** 
that SMBHBs can produce given the available mass budget from the Kormendy & Ho (2013) scaling relations.

This tension suggests one or more of:
1. Unmodeled pulsar noise contaminating the GWB measurement
2. Underestimated SMBH demographics (high-mass tail, intrinsic scatter)
3. Contributions from exotic physics
""")

# =============================================================================
# PTA SECTION (moved to bottom)
# =============================================================================
if show_pta:
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
    st.caption("IPTA DR3 scaling: h_c ∝ 1/√(N_pairs × T), where N_pairs = N(N-1)/2. With ~115 pulsars (6555 pairs vs NANOGrav's 2211) and 25-year baseline, improvement ≈ √(3.0 × 1.7) ≈ 2.2×, with additional gains from combined noise modeling.")
    st.caption("PTA sensitivity curves use the formalism of [Hazboun, Romano & Smith (2019)](https://arxiv.org/abs/1907.04341), implemented in [hasasia](https://github.com/Hazboun6/hasasia).")