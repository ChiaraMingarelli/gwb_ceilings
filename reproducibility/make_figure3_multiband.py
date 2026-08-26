"""Build the paper's Figure 3, the multiband Omega_gw(f) benchmark map.

Figure 3 of Mingarelli (2026), "Energetic Ceilings and Benchmarks of
Astrophysical Gravitational-Wave Backgrounds" (arXiv:2601.18859), places the
six benchmark gravitational-wave-background spectra of Table 2 across the
nanohertz-to-kilohertz band (Sec. IV of the paper).  Omega_gw(f) is the
dimensionless energy-density spectrum obtained from the characteristic
strain h_c(f) through Eq. (1) of the paper; all frequencies are in Hz.

The full-census SMBHB curve (Sec. III A 1) is solid from 1e-9 to 1e-6 Hz,
with short dashed guide extensions on both sides; its frequency-integrated
energy is calculated separately from the global mass budget.  Only the six
benchmark spectra are drawn; the theoretical upper limits and integrated
energies appear in the paper's tables, not in this figure.  Five detector
sensitivity curves (muAres, LISA, BBO, aLIGO design, and Cosmic Explorer)
and the Galactic double-white-dwarf confusion foreground (Sec. III C)
provide orientation.  Unless a published power-law-integrated curve is used
directly, the detector curves are single-natural-log-bin SNR-one
orientation curves, Omega_n(f)/sqrt(T f); see the individual function
docstrings.  An optional local archive of NANOGrav free-spectrum marginals
adds light-gray context violins when it is present; the public build does
not require it.

The manuscript asset for Figure 3 is figure3-final.pdf; this script writes
the fully code-generated equivalent, figures/gwb-ceilings-reproducible.pdf.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from verify_ceilings import (  # noqa: E402
    H0_KM_S_MPC,
    H0_SI,
    smbhb_population_omega,
)


FIGURES = ROOT / "figures"
NUMBERS = ROOT / "results" / "step3-numbers.json"
SPECTRA = ROOT / "results" / "step3-spectra.npz"
DEFAULT_PTA_CONTEXT = ROOT.parent / "step3-figure-context.npz"
ALIGO_DESIGN_PI = (
    ROOT / "data" / "detectors" / "Figures_3_and_4_PICurve_Design.dat"
)
ALIGO_DESIGN_PI_SHA256 = (
    "7d32bbf49db02a653da043266c74efce7761490c978b9f1e1ae92d06711f5ef4"
)
BBO_PLIS = ROOT / "data" / "detectors" / "BBO_PLIS_Schmitz2021.dat"
BBO_PLIS_SHA256 = (
    "e2a005bc0d57090c7dea355f4fd3e869d1da31cab011488cf8d29991a91b61ae"
)
CE_40KM_STRAIN = (
    ROOT
    / "data"
    / "detectors"
    / "cosmic_explorer_strain_T2000017-v9.txt"
)
CE_40KM_STRAIN_SHA256 = (
    "ebc9145dc9079b9f8839730ba8ce6642dc25542b1fe63c22db90982abf61c29c"
)
OUTPUT = FIGURES / "gwb-ceilings-reproducible.pdf"
PNG_OUTPUT = FIGURES / "gwb-ceilings-reproducible.png"
# The published manuscript uses an author-finished layout asset
# (figure3-final.pdf).  This code-generated version is the public, fully
# reproducible scientific plot with identical numerical content.
SPECTRA_ONLY_OUTPUT = FIGURES / "figure2-spectra-only.pdf"
SPECTRA_ONLY_PNG_OUTPUT = (
    FIGURES / "figure2-spectra-only.png"
)
NO_LABELS_OUTPUT = FIGURES / "figure2-no-labels.pdf"
NO_LABELS_PNG_OUTPUT = FIGURES / "figure2-no-labels.png"
SPECTRA_ONLY_NO_LABELS_OUTPUT = (
    FIGURES / "figure2-spectra-only-no-labels.pdf"
)
SPECTRA_ONLY_NO_LABELS_PNG_OUTPUT = (
    FIGURES / "figure2-spectra-only-no-labels.png"
)
PLOT_OMEGA_MIN = 1.0e-18
BACKGROUND_FILL_ALPHA = 0.10
CONTEXT_GRAY = "#8A8A8A"
CONTEXT_LABEL_GRAY = "#666666"
CONTEXT_ALPHA = 0.38
DETECTOR_PLOT_MAX = 2.0e-7
NANOGRAV_FACE_ALPHA = 0.30
NANOGRAV_EDGE_ALPHA = 0.58
DWD_DISPLAY_MINIMUM_HZ = 1.0e-5
LOW_FREQUENCY_CONTINUATION_DEX = 0.50
SMBHB_HIGH_FREQUENCY_CONTINUATION_DEX = 0.50
EMRI_MODEL_CUTOFF_HZ = 1.0e-2
EMRI_HIGH_FREQUENCY_CONTINUATION_DEX = 0.50
DWD_CONTINUATION_MINIMUM_HZ = DWD_DISPLAY_MINIMUM_HZ * 10.0 ** (
    -LOW_FREQUENCY_CONTINUATION_DEX
)

COLORS = {
    "smbhb": "#0072B2",
    "agn": "#D55E00",
    "emri": "#009E73",
    "bns": "#CC79A7",
    "popiii": "#56B4E9",
    "bbh": "#E69F00",
}

# These are plotting/model-validity boundaries, not physical formation
# cutoffs.  The energetic spectra retain their full idealized normalization.
BENCHMARK_DISPLAY_MINIMUM_HZ = {
    "smbhb_benchmark": 1.0e-9,
    "emri_benchmark": 1.0e-5,
    "agn_imri_benchmark": 1.0e-5,
    "popiii_benchmark": 1.0e-3,
    # For the representative 21.687-Msun equal-mass binary, the dashed
    # half-decade continuation now ends at 1e-5 Hz, where the Peters
    # time to merger is approximately the declared 10-Gyr history.
    "bbh_benchmark": 3.1622776601683795e-5,
    "bns_benchmark": 1.0e-3,
}

# The full LM24 mass moment is used for the SMBHB normalization.  The solid
# curve ends below the LISA band, with short guide extensions on either side.
# These are display boundaries, not physical termination frequencies.
SMBHB_DISPLAY_MAXIMUM_HZ = 1.0e-6


def display_curve(values: np.ndarray) -> np.ndarray:
    """Return a plotting copy that ends cleanly at physical support.

    Non-positive values become NaN, and any abrupt drop steeper than
    0.05 dex per grid step truncates the curve so that a spectrum's
    physical cutoff is not rendered as a vertical cliff.
    """

    curve = np.where(values > 0.0, values, np.nan).astype(float, copy=True)
    valid = np.isfinite(curve[:-1]) & np.isfinite(curve[1:])
    log_step = np.full(curve.size - 1, np.nan)
    log_step[valid] = np.log10(curve[1:][valid] / curve[:-1][valid])
    abrupt = np.flatnonzero(log_step < -0.05)
    if abrupt.size:
        curve[abrupt[0] + 1 :] = np.nan
    return curve


def benchmark_display_curve(
    frequency: np.ndarray,
    values: np.ndarray,
    spectrum_key: str,
) -> np.ndarray:
    """Apply the plot-only frequency boundaries to one benchmark spectrum.

    The boundaries in BENCHMARK_DISPLAY_MINIMUM_HZ (and, for the SMBHB
    curve, SMBHB_DISPLAY_MAXIMUM_HZ) mark where each idealized spectrum is
    drawn solid; they do not alter the underlying energetics.
    """

    curve = display_curve(values)
    curve[frequency < BENCHMARK_DISPLAY_MINIMUM_HZ[spectrum_key]] = np.nan
    if spectrum_key == "smbhb_benchmark":
        curve[frequency > SMBHB_DISPLAY_MAXIMUM_HZ] = np.nan
    return curve


def benchmark_continuation_curve(
    frequency: np.ndarray,
    values: np.ndarray,
    spectrum_key: str,
) -> np.ndarray:
    """Return the dashed half-decade guide segments outside the solid range."""

    curve = display_curve(values)
    solid_minimum = BENCHMARK_DISPLAY_MINIMUM_HZ[spectrum_key]
    continuation_minimum = solid_minimum * 10.0 ** (
        -LOW_FREQUENCY_CONTINUATION_DEX
    )
    support = (frequency >= continuation_minimum) & (
        frequency <= solid_minimum
    )
    if spectrum_key == "smbhb_benchmark":
        continuation_maximum = SMBHB_DISPLAY_MAXIMUM_HZ * 10.0 ** (
            SMBHB_HIGH_FREQUENCY_CONTINUATION_DEX
        )
        support |= (
            (frequency >= SMBHB_DISPLAY_MAXIMUM_HZ)
            & (frequency <= continuation_maximum)
        )
    curve[~support] = np.nan
    return curve


def emri_high_frequency_continuation(
    frequency: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Continue the EMRI inspiral power law above its declared model cutoff.

    The EMRI benchmark spectrum (Sec. III B 1 of the paper) is evaluated up
    to 1e-2 Hz; this dashed guide extends the circular-inspiral
    Omega_gw proportional to f^(2/3) slope half a decade beyond that
    cutoff for display only.
    """

    f = np.asarray(frequency, dtype=float)
    omega = np.asarray(values, dtype=float)
    if f.shape != omega.shape:
        raise ValueError("EMRI frequency and spectrum arrays must share a shape.")
    anchor_support = (
        (f > 0.0)
        & (f <= EMRI_MODEL_CUTOFF_HZ)
        & np.isfinite(omega)
        & (omega > 0.0)
    )
    anchor_indices = np.flatnonzero(anchor_support)
    if not anchor_indices.size:
        raise ValueError("EMRI spectrum does not reach its declared cutoff.")
    anchor_index = int(anchor_indices[-1])
    anchor_omega = omega[anchor_index] * (
        EMRI_MODEL_CUTOFF_HZ / f[anchor_index]
    ) ** (2.0 / 3.0)
    continuation_frequency = np.geomspace(
        EMRI_MODEL_CUTOFF_HZ,
        EMRI_MODEL_CUTOFF_HZ
        * 10.0**EMRI_HIGH_FREQUENCY_CONTINUATION_DEX,
        120,
    )
    continuation_omega = anchor_omega * (
        continuation_frequency / EMRI_MODEL_CUTOFF_HZ
    ) ** (2.0 / 3.0)
    return continuation_frequency, continuation_omega


def lisa_sensitivity(frequency: np.ndarray, years: float = 4.0) -> np.ndarray:
    """Illustrative single-log-bin LISA stochastic sensitivity.

    Analytic LISA instrument noise (2.5 Gm arms, standard optical-metrology
    and acceleration noise terms) converted to a dimensionless Omega_gw
    noise level and divided by sqrt(T f), giving the SNR-one orientation
    curve for one natural-log frequency bin over a `years`-long observation.
    """

    length = 2.5e9
    f_star = 19.09e-3
    p_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / frequency) ** 4)
    p_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / frequency) ** 2) * (
        1.0 + (frequency / 8.0e-3) ** 4
    )
    noise = (
        10.0
        / (3.0 * length**2)
        * (p_oms + 4.0 * p_acc / (2.0 * np.pi * frequency) ** 4)
        * (1.0 + 0.6 * (frequency / f_star) ** 2)
    )
    omega = 2.0 * np.pi**2 * frequency**3 * noise / (3.0 * H0_SI**2)
    return omega / np.sqrt(years * 365.25 * 86_400.0 * frequency)


def muares_sensitivity(
    frequency: np.ndarray,
    years: float = 10.0,
) -> np.ndarray:
    """Proposal-based muAres single-log-bin SNR-one orientation sensitivity.

    Uses the muAres proposal noise parameters (3.95e11-m arms) between
    1e-7 and 1 Hz, converted to Omega_gw and divided by sqrt(T f) as for
    the LISA curve above.
    """

    omega = np.full_like(frequency, np.nan, dtype=float)
    mask = (frequency >= 1.0e-7) & (frequency <= 1.0)
    fm = frequency[mask]
    length = 3.95e11
    f_star = 299_792_458.0 / (2.0 * np.pi * length)
    s_pos = (50.0e-12) ** 2
    s_acc = (1.0e-15) ** 2
    noise = (
        10.0
        / (3.0 * length**2)
        * (4.0 * s_acc / (2.0 * np.pi * fm) ** 4 + s_pos)
        * (1.0 + 0.6 * (fm / f_star) ** 2)
    )
    omega[mask] = (
        2.0
        * np.pi**2
        * fm**3
        * noise
        / (3.0 * H0_SI**2)
        / np.sqrt(years * 365.25 * 86_400.0 * fm)
    )
    return omega


def bbo_sensitivity(frequency: np.ndarray, years: float = 5.0) -> np.ndarray:
    """Published BBO SNR-one power-law-integrated orientation curve.

    Interpolates the tabulated Schmitz (2021) BBO power-law-integrated
    sensitivity, log10(h^2 Omega_gw) versus log10(f/Hz), removes the
    reduced-Hubble factor h^2, and rescales by 1/sqrt(years).  The input
    file is verified against its pinned SHA-256 hash before use.
    """

    if years <= 0.0:
        raise ValueError("BBO observation time must be positive.")
    digest = hashlib.sha256(BBO_PLIS.read_bytes()).hexdigest()
    if digest != BBO_PLIS_SHA256:
        raise RuntimeError("BBO PLIS source hash changed.")

    source = np.loadtxt(BBO_PLIS)
    if source.shape != (16_001, 4):
        raise RuntimeError("BBO PLIS source must have 16,001 rows and four columns.")
    source_log_frequency = np.asarray(source[:, 0], dtype=float)
    source_log_h2omega = np.asarray(source[:, 1], dtype=float)
    if (
        not np.isfinite(source).all()
        or np.any(np.diff(source_log_frequency) <= 0.0)
        or not np.isclose(source_log_frequency[0], -3.0)
        or not np.isclose(source_log_frequency[-1], 1.0)
    ):
        raise RuntimeError("BBO PLIS source is malformed.")

    omega = np.full_like(frequency, np.nan, dtype=float)
    log_frequency = np.full_like(frequency, np.nan, dtype=float)
    positive = frequency > 0.0
    log_frequency[positive] = np.log10(frequency[positive])
    support = (
        positive
        & (log_frequency >= source_log_frequency[0])
        & (log_frequency <= source_log_frequency[-1])
    )
    little_h = H0_KM_S_MPC / 100.0
    omega[support] = 10.0 ** np.interp(
        log_frequency[support],
        source_log_frequency,
        source_log_h2omega,
    ) / (
        little_h**2 * np.sqrt(years)
    )
    return omega


def cosmic_explorer_sensitivity(
    frequency: np.ndarray,
    years: float = 1.0,
) -> np.ndarray:
    """Official CE 40-km strain ASD as a single-log-bin orientation curve.

    Interpolates the Cosmic Explorer 40-km amplitude spectral density
    (units Hz^-1/2, 5-5000 Hz), squares it to a one-sided strain noise
    power spectral density, converts to Omega_gw, and divides by
    sqrt(T f) for the SNR-one single-natural-log-bin orientation.  The
    input file is verified against its pinned SHA-256 hash before use.
    """

    if years <= 0.0:
        raise ValueError("Cosmic Explorer observation time must be positive.")
    digest = hashlib.sha256(CE_40KM_STRAIN.read_bytes()).hexdigest()
    if digest != CE_40KM_STRAIN_SHA256:
        raise RuntimeError("Cosmic Explorer source hash changed.")

    source = np.loadtxt(CE_40KM_STRAIN)
    if source.shape != (1_000, 2):
        raise RuntimeError("Cosmic Explorer source must have 1,000 rows.")
    source_frequency = np.asarray(source[:, 0], dtype=float)
    source_asd = np.asarray(source[:, 1], dtype=float)
    if (
        not np.isfinite(source).all()
        or np.any(source_asd <= 0.0)
        or np.any(np.diff(source_frequency) <= 0.0)
        or not np.isclose(source_frequency[0], 5.0)
        or not np.isclose(source_frequency[-1], 5_000.0)
    ):
        raise RuntimeError("Cosmic Explorer source is malformed.")

    omega = np.full_like(frequency, np.nan, dtype=float)
    support = (frequency >= source_frequency[0]) & (
        frequency <= source_frequency[-1]
    )
    fm = frequency[support]
    asd = np.exp(
        np.interp(
            np.log(fm),
            np.log(source_frequency),
            np.log(source_asd),
        )
    )
    noise = asd**2
    omega[support] = (
        2.0
        * np.pi**2
        * fm**3
        * noise
        / (3.0 * H0_SI**2)
        / np.sqrt(years * 365.25 * 86_400.0 * fm)
    )
    return omega


def aligo_design_pi_sensitivity(frequency: np.ndarray) -> np.ndarray:
    """Published Advanced-LIGO/Virgo design power-law-integrated sensitivity.

    Interpolates the published one-sigma design PI curve (Omega_gw versus
    frequency, 10-2000 Hz) and doubles it to the two-sigma level shown in
    the figure.  The input file is verified against its pinned SHA-256
    hash before use.
    """

    digest = hashlib.sha256(ALIGO_DESIGN_PI.read_bytes()).hexdigest()
    if digest != ALIGO_DESIGN_PI_SHA256:
        raise RuntimeError("Advanced-LIGO design PI source hash changed.")

    design = np.loadtxt(ALIGO_DESIGN_PI)
    if design.ndim != 2 or design.shape[1] != 2:
        raise RuntimeError("Advanced-LIGO design PI source must have two columns.")
    source_frequency = np.asarray(design[:, 0], dtype=float)
    one_sigma_omega = np.asarray(design[:, 1], dtype=float)
    if (
        np.any(np.diff(source_frequency) <= 0.0)
        or np.any(one_sigma_omega <= 0.0)
        or not np.isclose(source_frequency[0], 10.0)
        or not np.isclose(source_frequency[-1], 2_000.0)
    ):
        raise RuntimeError("Advanced-LIGO design PI source is malformed.")

    omega = np.full_like(frequency, np.nan, dtype=float)
    support = (frequency >= source_frequency[0]) & (
        frequency <= source_frequency[-1]
    )
    omega[support] = 2.0 * np.exp(
        np.interp(
            np.log(frequency[support]),
            np.log(source_frequency),
            np.log(one_sigma_omega),
        )
    )
    return omega


def dwd_confusion_omega(frequency: np.ndarray) -> np.ndarray:
    """Robson et al. (2019) four-year Galactic-DWD confusion fit.

    Evaluates the analytic fit to the unresolved Galactic double-white-dwarf
    confusion strain noise S_h(f) after four years of LISA observation and
    converts it to the dimensionless Omega_gw(f) foreground discussed in
    Sec. III C of the paper.
    """

    amplitude = 9.0e-45
    alpha = 0.138
    beta = -221.0
    kappa = 521.0
    gamma = 1_680.0
    f_knee = 0.00113
    sh = np.zeros_like(frequency)
    mask = (frequency >= DWD_CONTINUATION_MINIMUM_HZ) & (frequency <= 0.1)
    fm = frequency[mask]
    sh[mask] = (
        amplitude
        * fm ** (-7.0 / 3.0)
        * np.exp(-fm**alpha + beta * fm * np.sin(kappa * fm))
        * (1.0 + np.tanh(gamma * (f_knee - fm)))
    )
    return 2.0 * np.pi**2 * frequency**3 * sh / (3.0 * H0_SI**2)


def load_authoritative_inputs() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load and validate the frozen benchmark scalars and spectra.

    The frozen results archive keeps its historical file names
    (results/step3-numbers.json and results/step3-spectra.npz) so that
    pinned hashes remain valid.  The archive holds the shared frequency
    grid (Hz) and the six dimensionless benchmark Omega_gw(f) spectra of
    Table 2 of the paper; the SMBHB benchmark strain amplitude at
    f = 1/yr is checked against its published value before plotting.
    """

    results = json.loads(NUMBERS.read_text(encoding="utf-8"))
    smbhb = results["smbhb"]
    expected_scalars = {"benchmark_A_1yr": 2.1078708082857504e-15}
    for key, expected in expected_scalars.items():
        if not np.isclose(float(smbhb[key]), expected, rtol=5.0e-12):
            raise RuntimeError(
                f"authoritative SMBHB {key} changed to {smbhb[key]!r}; "
                "re-verify the multiband figure against the paper"
            )

    with np.load(SPECTRA) as archive:
        frequency = np.asarray(archive["frequency_hz"], dtype=float)
        spectra = {
            key: np.asarray(archive[key], dtype=float)
            for key in archive.files
            if key != "frequency_hz"
        }
    required = {
        "smbhb_benchmark",
        "emri_benchmark",
        "agn_imri_benchmark",
        "popiii_benchmark",
        "bbh_benchmark",
        "bns_benchmark",
    }
    missing = required.difference(spectra)
    if missing:
        raise RuntimeError(f"frozen spectrum archive is missing {sorted(missing)}")
    if np.any(frequency <= 0.0) or np.any(np.diff(frequency) <= 0.0):
        raise RuntimeError("frozen-archive frequency grid is not positive and increasing.")
    if any(spectra[key].shape != frequency.shape for key in required):
        raise RuntimeError("frozen benchmark spectra do not share one frequency grid.")
    return frequency, spectra


def parse_args() -> argparse.Namespace:
    """Parse the optional output-path and labeling controls."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spectra-only",
        action="store_true",
        help="deprecated alias; the figure is always a one-panel spectrum",
    )
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help=(
            "suppress manual curve annotations while retaining axes and ticks"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="override the PDF output path",
    )
    parser.add_argument(
        "--png-output",
        type=Path,
        help="override the PNG output path",
    )
    parser.add_argument(
        "--pta-context",
        type=Path,
        help=(
            "optional validated local NPZ archive for the five PTA "
            "free-spectrum marginals; when omitted, use the repository-level "
            "archive if it is present"
        ),
    )
    return parser.parse_args()


def load_pta_context(
    context_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate an NPZ archive of NANOGrav free-spectrum marginals.

    The archive supplies, for each of five frequency bins, the bin-center
    frequency (Hz), a grid of log10 Omega_gw values, and the corresponding
    violin half-widths (Hz).  The embedded provenance hashes must match the
    source table from which the archive was derived.  These marginals are
    display context only; see data/pta/README.md.
    """

    expected_hashes = {
        "source_log10_frequency_sha256": (
            "144fd81d0ccfc01057f5fe865de5430eb496360932510734aac345b2f3ef538e"
        ),
        "source_log10_omega_sha256": (
            "20f9ecd23735f8fe506bcaa200581c80de5e4128b38ed9ba0a615e9a529e1939"
        ),
    }
    with np.load(context_path) as context:
        center_frequency = np.asarray(context["pta_frequency_hz"], dtype=float)
        log10_omega = np.asarray(
            context["pta_log10_omega_grid"],
            dtype=float,
        )
        half_width = np.asarray(context["pta_half_width_hz"], dtype=float)
        for key, expected in expected_hashes.items():
            if str(context[key]) != expected:
                raise RuntimeError(f"archived NANOGrav provenance changed: {key}")
    if center_frequency.shape != (5,):
        raise RuntimeError("NANOGrav context must contain exactly five bins.")
    if log10_omega.shape != (5, 180) or half_width.shape != (5, 180):
        raise RuntimeError("NANOGrav violin grids have unexpected shapes.")
    if np.any(np.diff(center_frequency) <= 0.0) or np.any(half_width < 0.0):
        raise RuntimeError("NANOGrav violin coordinates are invalid.")
    return center_frequency, log10_omega, half_width


def main() -> None:
    args = parse_args()
    frequency, spectra = load_authoritative_inputs()
    smbhb_plot_minimum = BENCHMARK_DISPLAY_MINIMUM_HZ[
        "smbhb_benchmark"
    ] * 10.0 ** (-LOW_FREQUENCY_CONTINUATION_DEX)
    smbhb_plot_maximum = SMBHB_DISPLAY_MAXIMUM_HZ * 10.0 ** (
        SMBHB_HIGH_FREQUENCY_CONTINUATION_DEX
    )
    smbhb_plot_support = (
        (frequency >= smbhb_plot_minimum)
        & (frequency <= smbhb_plot_maximum)
    )
    smbhb_plot_curve = np.zeros_like(frequency)
    smbhb_plot_curve[smbhb_plot_support] = smbhb_population_omega(
        frequency[smbhb_plot_support],
        "benchmark",
    )
    spectra["smbhb_benchmark"] = smbhb_plot_curve

    pta_context_path = args.pta_context
    if pta_context_path is None and DEFAULT_PTA_CONTEXT.is_file():
        pta_context_path = DEFAULT_PTA_CONTEXT
    pta_context = (
        load_pta_context(pta_context_path)
        if pta_context_path is not None
        else None
    )
    benchmark_specs = (
        ("smbhb_benchmark", COLORS["smbhb"], 3),
        ("emri_benchmark", COLORS["emri"], 5),
        ("agn_imri_benchmark", COLORS["agn"], 5),
        ("popiii_benchmark", COLORS["popiii"], 6),
        ("bbh_benchmark", COLORS["bbh"], 6),
        ("bns_benchmark", COLORS["bns"], 6),
    )
    benchmark_keys = {key for key, _, _ in benchmark_specs}
    if benchmark_keys != set(BENCHMARK_DISPLAY_MINIMUM_HZ):
        raise RuntimeError(
            "Multiband-figure display boundaries do not match its spectra."
        )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 9.2,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 4.15))

    context_specs = (
        (r"$\mu$Ares", muares_sensitivity(frequency), "-."),
        ("LISA", lisa_sensitivity(frequency), "--"),
        ("BBO", bbo_sensitivity(frequency), "-"),
        (
            "aLIGO design",
            aligo_design_pi_sensitivity(frequency),
            (0, (5.0, 2.0)),
        ),
        ("CE", cosmic_explorer_sensitivity(frequency), ":"),
    )
    for _, curve, line_style in context_specs:
        mask = (
            np.isfinite(curve)
            & (curve > 1.0e-20)
            & (curve <= DETECTOR_PLOT_MAX)
        )
        ax.loglog(
            frequency[mask],
            curve[mask],
            color=CONTEXT_GRAY,
            linestyle=line_style,
            linewidth=0.85,
            alpha=CONTEXT_ALPHA,
            zorder=0,
        )

    dwd = dwd_confusion_omega(frequency)
    dwd_solid_mask = (
        (frequency >= DWD_DISPLAY_MINIMUM_HZ)
        & (frequency < 2.0e-2)
        & (dwd > PLOT_OMEGA_MIN)
    )
    dwd_continuation_mask = (
        (frequency >= DWD_CONTINUATION_MINIMUM_HZ)
        & (frequency <= DWD_DISPLAY_MINIMUM_HZ)
        & (dwd > PLOT_OMEGA_MIN)
    )
    dwd_fill_mask = (
        (frequency >= DWD_CONTINUATION_MINIMUM_HZ)
        & (frequency < 2.0e-2)
        & (dwd > PLOT_OMEGA_MIN)
    )
    ax.loglog(
        frequency[dwd_continuation_mask],
        dwd[dwd_continuation_mask],
        color="#696969",
        linestyle=(0, (3.0, 2.0)),
        alpha=0.66,
        linewidth=1.1,
        zorder=2,
    )
    ax.loglog(
        frequency[dwd_solid_mask],
        dwd[dwd_solid_mask],
        color="#696969",
        alpha=0.72,
        linewidth=1.2,
        zorder=2,
    )
    ax.fill_between(
        frequency[dwd_fill_mask],
        PLOT_OMEGA_MIN,
        dwd[dwd_fill_mask],
        color="#696969",
        alpha=BACKGROUND_FILL_ALPHA,
        linewidth=0.0,
        zorder=1,
    )

    for key, color, zorder in benchmark_specs:
        curve = benchmark_display_curve(frequency, spectra[key], key)
        continuation = benchmark_continuation_curve(
            frequency,
            spectra[key],
            key,
        )
        ax.loglog(
            frequency,
            continuation,
            color=color,
            linestyle=(0, (3.0, 2.0)),
            linewidth=1.45,
            alpha=0.82,
            zorder=zorder,
        )
        fill_curve = np.where(np.isfinite(curve), curve, continuation)
        support = np.isfinite(fill_curve) & (fill_curve >= PLOT_OMEGA_MIN)
        ax.fill_between(
            frequency,
            PLOT_OMEGA_MIN,
            fill_curve,
            where=support,
            color=color,
            alpha=BACKGROUND_FILL_ALPHA,
            linewidth=0.0,
            zorder=zorder - 1,
        )
        (line,) = ax.loglog(
            frequency,
            curve,
            color=color,
            linewidth=2.05,
            alpha=0.98,
            zorder=zorder,
        )
        line.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=3.0,
                    foreground="white",
                    alpha=0.80,
                ),
                path_effects.Normal(),
            ]
        )

    emri_high_frequency, emri_high_omega = emri_high_frequency_continuation(
        frequency,
        spectra["emri_benchmark"],
    )
    ax.fill_between(
        emri_high_frequency,
        PLOT_OMEGA_MIN,
        emri_high_omega,
        color=COLORS["emri"],
        alpha=BACKGROUND_FILL_ALPHA,
        linewidth=0.0,
        zorder=4,
    )
    ax.loglog(
        emri_high_frequency,
        emri_high_omega,
        color=COLORS["emri"],
        linestyle=(0, (3.0, 2.0)),
        linewidth=1.45,
        alpha=0.82,
        zorder=5,
    )

    if pta_context is not None:
        pta_frequency, pta_log10_omega, pta_half_width = pta_context
        for center, log10_omega, half_width in zip(
            pta_frequency,
            pta_log10_omega,
            pta_half_width,
        ):
            omega_grid = 10.0**log10_omega
            ax.fill_betweenx(
                omega_grid,
                center - half_width,
                center + half_width,
                facecolor=matplotlib.colors.to_rgba(
                    CONTEXT_GRAY,
                    NANOGRAV_FACE_ALPHA,
                ),
                edgecolor=matplotlib.colors.to_rgba(
                    CONTEXT_GRAY,
                    NANOGRAV_EDGE_ALPHA,
                ),
                linewidth=0.40,
                zorder=12,
            )
    if not args.no_labels and pta_context is not None:
        ax.text(
            1.45e-9,
            2.5e-7,
            "NANOGrav",
            color=CONTEXT_LABEL_GRAY,
            fontsize=8.7,
            ha="left",
            va="bottom",
            path_effects=[
                path_effects.withStroke(
                    linewidth=1.8,
                    foreground="white",
                    alpha=0.95,
                )
            ],
            zorder=13,
        )

    def label_curve(
        spectrum_key: str,
        x_value: float,
        label: str,
        offset: tuple[float, float],
        horizontal: str = "center",
        vertical: str = "center",
    ) -> None:
        curve = benchmark_display_curve(
            frequency,
            spectra[spectrum_key],
            spectrum_key,
        )
        finite = np.isfinite(curve)
        y_value = 10.0 ** np.interp(
            np.log10(x_value),
            np.log10(frequency[finite]),
            np.log10(curve[finite]),
        )
        ax.annotate(
            label,
            xy=(x_value, y_value),
            xytext=offset,
            textcoords="offset points",
            color="#202020",
            fontsize=10.0,
            fontweight="semibold",
            ha=horizontal,
            va=vertical,
            path_effects=[
                path_effects.withStroke(
                    linewidth=2.2,
                    foreground="white",
                    alpha=0.95,
                )
            ],
            annotation_clip=False,
            zorder=9,
        )

    if not args.no_labels:
        label_curve(
            "smbhb_benchmark",
            5.0e-8,
            "SMBHB",
            (0, -18),
            vertical="top",
        )
        label_curve(
            "emri_benchmark",
            2.0e-3,
            "EMRI",
            (-10, -14),
            horizontal="right",
            vertical="top",
        )
        label_curve(
            "agn_imri_benchmark",
            1.5e-2,
            "AGN-IMRI",
            (0, 12),
            vertical="bottom",
        )
        label_curve(
            "popiii_benchmark",
            2.0,
            "Pop III",
            (-4, -14),
            horizontal="right",
            vertical="top",
        )
        label_curve(
            "bbh_benchmark",
            70.0,
            "sBBHs",
            (0, 12),
            vertical="bottom",
        )
        label_curve(
            "bns_benchmark",
            300.0,
            "BNS",
            (10, -2),
            horizontal="left",
            vertical="center",
        )

    context_labels = (
        (1.6e-6, 8.0e-13, r"$\mu$Ares"),
        (2.0e-3, 1.0e-13, "LISA"),
        (1.2e-1, 4.0e-17, "BBO"),
        (1.7e1, 5.5e-9, "aLIGO"),
        (3.0e2, 7.0e-13, "CE"),
    )
    if not args.no_labels:
        for x_value, y_value, label in context_labels:
            ax.text(
                x_value,
                y_value,
                label,
                color=CONTEXT_LABEL_GRAY,
                fontsize=8.7,
                path_effects=[
                    path_effects.withStroke(
                        linewidth=1.8,
                        foreground="white",
                        alpha=0.90,
                    )
                ],
                zorder=8,
            )

        ax.text(
            5.5e-4,
            1.65e-10,
            "Galactic DWD",
            color="#5F5F5F",
            fontsize=8.7,
            ha="center",
            va="bottom",
            path_effects=[
                path_effects.withStroke(
                    linewidth=1.8,
                    foreground="white",
                    alpha=0.78,
                )
            ],
            zorder=8,
        )

    ax.set_xlim(
        BENCHMARK_DISPLAY_MINIMUM_HZ["smbhb_benchmark"]
        * 10.0 ** (-LOW_FREQUENCY_CONTINUATION_DEX),
        3.0e3,
    )
    ax.set_xticks(10.0 ** np.arange(-9, 4, 2))
    ax.set_ylim(PLOT_OMEGA_MIN, 1.0e-6)
    ax.set_ylabel(r"$\Omega_{\rm gw}(f)$")
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.grid(False)

    ax.set_xlabel(r"frequency $f$ (Hz)")

    output = args.output
    png_output = args.png_output
    if output is None:
        if args.spectra_only:
            output = (
                SPECTRA_ONLY_NO_LABELS_OUTPUT
                if args.no_labels
                else SPECTRA_ONLY_OUTPUT
            )
        else:
            output = NO_LABELS_OUTPUT if args.no_labels else OUTPUT
    if png_output is None:
        if args.spectra_only:
            png_output = (
                SPECTRA_ONLY_NO_LABELS_PNG_OUTPUT
                if args.no_labels
                else SPECTRA_ONLY_PNG_OUTPUT
            )
        else:
            png_output = (
                NO_LABELS_PNG_OUTPUT if args.no_labels else PNG_OUTPUT
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    png_output.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout(pad=0.5)
    fig.savefig(output)
    fig.savefig(png_output, dpi=300)
    plt.close(fig)

    print("saved", output)
    print("saved", png_output)
    print("included", [key for key, _, _ in benchmark_specs])


if __name__ == "__main__":
    main()
