import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import io

# hasasia for PTA sensitivity curves
import hasasia.sensitivity as hsen
import hasasia.sim as hsim

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


@st.cache_data
def get_pta_sensitivity_hasasia(n_pulsars=34, timespan=15.0, sigma=1e-7, cadence=14):
    """
    Compute PTA sensitivity curve using hasasia.
    
    Parameters:
    -----------
    n_pulsars : int
        Number of pulsars in the array
    timespan : float
        Observation timespan in years
    sigma : float
        RMS timing residual in seconds (e.g., 1e-7 = 100 ns)
    cadence : int
        Observations per year
    
    Returns:
    --------
    freqs, omega_gw : arrays
        Frequency array and corresponding Omega_gw sensitivity
    """
    # Random sky positions for pulsars
    np.random.seed(42)  # For reproducibility
    phi = np.random.uniform(0, 2*np.pi, size=n_pulsars)
    theta = np.arccos(np.random.uniform(-1, 1, size=n_pulsars))
    
    # Simulate PTA
    psrs = hsim.sim_pta(timespan=timespan, cad=cadence, sigma=sigma,
                        phi=phi, theta=theta, Npsrs=n_pulsars)
    
    # Frequency array for PTA band
    fyr = 1.0 / (365.25 * 24 * 3600)
    freqs = np.logspace(np.log10(fyr/5), np.log10(1e-6), 100)
    
    # Build spectra for each pulsar
    spectra = []
    for p in psrs:
        sp = hsen.Spectrum(p, freqs=freqs)
        sp.NcalInv  # Trigger calculation
        spectra.append(sp)
    
    # Create GWB sensitivity curve
    sc = hsen.GWBSensitivityCurve(spectra)
    
    return sc.freqs, sc.Omega_gw


# Streamlit app
st.set_page_config(page_title="GW Background Ceilings", layout="wide")
st.title("Gravitational Wave Background Ceilings")

st.markdown("""
Interactive visualization of astrophysical gravitational wave background ceilings 
based on energy reservoir constraints. Adjust the mass density reservoirs to see 
how the GWB amplitudes scale.
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
show_detectors = st.sidebar.checkbox("Show detector curves", value=True)
show_pta = st.sidebar.checkbox("Show PTA sensitivity (hasasia)", value=True)
show_dwd = st.sidebar.checkbox("Show DWD foreground", value=True)
show_ceiling = st.sidebar.checkbox("Show integrated ceiling", value=True)

# PTA parameters (in expander)
with st.sidebar.expander("PTA Parameters"):
    pta_npsr = st.slider("Number of pulsars", 10, 100, 34)
    pta_timespan = st.slider("Timespan (years)", 5.0, 20.0, 15.0, step=0.5)
    pta_sigma = st.select_slider(
        "Timing precision (ns)", 
        options=[50, 100, 200, 500, 1000],
        value=100
    )
    pta_cadence = st.slider("Cadence (obs/year)", 12, 52, 26)

selected_pops = st.sidebar.multiselect(
    "Select populations",
    list(POPULATIONS.keys()),
    default=list(POPULATIONS.keys())
)

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))
fig.patch.set_facecolor('white')

f_grid = np.logspace(-9.5, 3.5, 3000)
omega_cutoff = 1e-7

ax.set_xlim(1e-9, 3e3)
ax.set_ylim(1e-18, 1e-6)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Frequency f [Hz]', fontsize=14)
ax.set_ylabel(r'$\Omega_{\mathrm{gw}}(f)$', fontsize=14)

ax.xaxis.set_major_locator(LogLocator(base=10, numticks=20))
ax.yaxis.set_major_locator(LogLocator(base=10, numticks=20))
ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))

# Detectors
if show_detectors:
    muares = get_muares_sensitivity(f_grid, T_yrs=10.0)
    mask_mu = (f_grid > 1e-7) & (f_grid < 1e-1) & (muares < omega_cutoff)
    ax.loglog(f_grid[mask_mu], muares[mask_mu], color='gray', ls='-.', alpha=0.6, lw=1.2)
    ax.text(1e-6, 1e-11, 'muAres', fontsize=10, color='gray', ha='left')

    bbo = get_bbo_approx(f_grid)
    mask_bbo = (bbo > 0) & (bbo < omega_cutoff)
    ax.loglog(f_grid[mask_bbo], bbo[mask_bbo], color='gray', ls='-', alpha=0.6, lw=1.2)
    ax.text(5e-2, 2e-17, 'BBO', fontsize=10, color='gray', ha='center')

    lisa = get_lisa_sensitivity(f_grid)
    mask_lisa = lisa < omega_cutoff
    ax.loglog(f_grid[mask_lisa], lisa[mask_lisa], color='gray', ls='--', alpha=0.6, lw=1.5)
    ax.text(2e-5, 8e-10, 'LISA', fontsize=10, color='gray', ha='center')

    aligo = get_aligo_approx(f_grid)
    mask_aligo = (aligo < 1e-4) & (aligo < omega_cutoff)
    ax.loglog(f_grid[mask_aligo], aligo[mask_aligo], color='gray', ls=':', alpha=0.6, lw=1.2)
    ax.text(70 * 10 * 10**(-0.4), 2e-9 * 10**0.5, 'aLIGO', fontsize=10, color='gray', ha='center')

    ce = get_ce_sensitivity(f_grid, T_yrs=1.0)
    mask_ce = (ce < 1e-4) & (ce < omega_cutoff)
    ax.loglog(f_grid[mask_ce], ce[mask_ce], color='gray', ls=':', alpha=0.6, lw=1.2)
    ax.text(100, 1e-14, 'CE', fontsize=10, color='gray', ha='center')

# PTA sensitivity from hasasia
if show_pta:
    try:
        pta_freqs, pta_omega = get_pta_sensitivity_hasasia(
            n_pulsars=pta_npsr,
            timespan=pta_timespan,
            sigma=pta_sigma * 1e-9,  # Convert ns to seconds
            cadence=pta_cadence
        )
        mask_pta = (pta_omega > 1e-20) & (pta_omega < omega_cutoff)
        if np.any(mask_pta):
            ax.loglog(pta_freqs[mask_pta], pta_omega[mask_pta], 
                     color='purple', ls='--', alpha=0.8, lw=1.5)
            ax.text(3e-9, 3e-9, 'PTA', fontsize=10, color='purple', ha='center')
    except Exception as e:
        st.sidebar.warning(f"Could not compute PTA curve: {e}")

# DWD foreground
if show_dwd:
    omega_wd = get_dwd_foreground(f_grid)
    mask_wd = omega_wd > 1e-25
    if np.any(mask_wd):
        ax.fill_between(f_grid[mask_wd], 1e-25, omega_wd[mask_wd], color='gray', alpha=0.3, linewidth=0)
        ax.text(3e-3 / 10 * 10**0.4, 1e-12, 'DWD', fontsize=15, color='white', ha='center', fontweight='bold')

# Ceiling
if show_ceiling:
    ax.axhline(y=1e-7, color='red', linestyle='-', linewidth=2.5, alpha=0.9)
    ax.text(1e-3, 1.8e-7, 'Integrated Astrophysical Ceiling', color='red', fontsize=18, fontweight='bold', ha='center')

# Populations
for name in selected_pops:
    params = POPULATIONS[name]
    A_current = scale_amplitude(params['A_bench'], params['reservoir'], rho_smbh, rho_stellar, rho_nsc)
    omega = get_omega_gw(f_grid, A_current, params['f_ref'], params['f_min'], params['f_max'])
    valid = omega > 1e-30
    if np.any(valid):
        ax.loglog(f_grid[valid], omega[valid], color=params['color'], lw=2.5, alpha=1.0)
        ax.fill_between(f_grid[valid], 1e-25, omega[valid], color=params['color'], alpha=0.15, linewidth=0)
        lx, ly = labels_pos.get(name, (1e-4, 1e-15))
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

# Info panel
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

st.markdown("---")
st.markdown("""
**Reference:** Mingarelli (2026), *Energetic Ceilings of Astrophysical Gravitational-Wave Backgrounds*

Amplitudes scale as A proportional to sqrt(rho_reservoir) relative to fiducial values from Table I.
""")
