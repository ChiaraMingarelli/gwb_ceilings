"""Build the paper's Figure 1 (PTA amplitude forest) and a diagnostic figure.

Figure 1 of Mingarelli (2026), "Energetic Ceilings and Benchmarks of
Astrophysical Gravitational-Wave Backgrounds" (arXiv:2601.18859), compares
the SMBHB benchmark amplitude A_bench (the f_merge=1 case of Sec. III A 1)
and the repeated-merger ceiling A_ceil (Eq. (12) of the paper) with
published PTA power-law amplitudes at f = 1/yr and fixed gamma=13/3.  All
amplitudes are dimensionless characteristic-strain amplitudes A_yr.  A
forest plot is used because the quoted intervals carry different statistical
meanings across collaborations; interval and estimator semantics follow the
source papers and are restated in the figure caption.  The row labels are
rendered by matplotlib at final column size, so the exported PDF
(figures/pta-amplitude-forest.pdf, the manuscript asset for Figure 1) drops
into the manuscript column with no trimming and no external label pass.

The second output is a diagnostic SMBHB inference plane, not included in
the published paper.  It maps the merger factor f_merge (Eq. (13) of the
paper) onto the SMBHB mass density available after the adopted
M1>=1e8-Msun population selection, in Msun Mpc^-3.  The blue band is
the Liepold--Ma (2024; LM24) 90% density interval after the same selection.
The orange band is the NANOGrav 15-year customized-chromatic-noise (CNM) 90%
credible interval mapped through the fixed population shape.  Values above
f_merge=1 represent repeated mergers rather than a fraction of distinct black
holes.  The SZQ24 fixed-frequency h_c^2 hierarchy boost (App. A of the
paper) is shown at the same numerical ordinate for comparison; it is not an
f_merge limit.

The authoritative SMBHB calculation values are imported from
``verify_ceilings.py``.  Observational values are transcribed from:

* PPTA DR3: Reardon et al. (2023), 68% Bayesian CURN credible interval;
* NANOGrav 15-year CNM: Agarwal et al. (2026), 90% credible interval;
* EPTA DR2: Antoniadis et al. (2023), 90% credible interval;
* MPTA: Miles et al. (2025), quoted 1-sigma optimal-statistic uncertainty;
* CPTA DR1: Xu et al. (2023), maximum-likelihood estimate and authors'
  quoted 95% confidence level.

Run from the reproducibility directory:

    python3 make_figure1_plane_forest.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from verify_ceilings import (  # noqa: E402
    RHO_BH_LM24,
    SZQ_HIERARCHY_HC2_BOOST,
    smbhb_population_results,
)


FIGURES = ROOT / "figures"
DEFAULT_FOREST_STEM = FIGURES / "pta-amplitude-forest"
DEFAULT_PLANE_STEM = FIGURES / "smbhb-inference-plane"

COLORS = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#E69F00",
    "purple": "#CC79A7",
    "black": "#111111",
    "gray": "#777777",
}

# LM24 quote for the local SMBH mass density in Msun Mpc^-3
# (90% confidence interval); see Eq. (9) of the paper.
LM24_RHO_LOW = 1.3e6
LM24_RHO_HIGH = 2.6e6

INFERENCE_Y_MAX = 4.0


@dataclass(frozen=True)
class ForestPoint:
    """One row of the PTA comparison; amplitudes A_yr in units of 1e-15."""

    label: str
    central: float
    lower: float | None
    upper: float | None
    color: str
    marker: str = "o"
    filled: bool = True
    unbounded_right: bool = False


def pta_forest_points(smbhb: dict[str, float]) -> tuple[ForestPoint, ...]:
    """Return model and observational rows with their stated intervals."""

    cpta_central = 10.0 ** (-14.7) * 1.0e15
    cpta_lower = 10.0 ** (-14.7 - 1.9) * 1.0e15
    cpta_upper = 10.0 ** (-14.7 + 0.9) * 1.0e15
    benchmark = smbhb["benchmark_A_1yr"] * 1.0e15
    ceiling = smbhb["repeated_merger_ceiling_A_1yr"] * 1.0e15
    # These are fixed-shape normalization sensitivities, not full model
    # confidence intervals: LM24 did not publish the BHMF covariance required
    # to propagate shape and high-mass-tail uncertainty through the M1 cut.
    normalization_scale_low = np.sqrt(LM24_RHO_LOW / RHO_BH_LM24)
    normalization_scale_high = np.sqrt(LM24_RHO_HIGH / RHO_BH_LM24)
    points = (
        ForestPoint(
            r"$A_{\rm bench}$",
            benchmark,
            benchmark * normalization_scale_low,
            benchmark * normalization_scale_high,
            COLORS["blue"],
            marker="*",
        ),
        ForestPoint(
            r"$A_{\rm ceil}$",
            ceiling,
            ceiling * normalization_scale_low,
            ceiling * normalization_scale_high,
            COLORS["black"],
            marker="D",
            filled=False,
        ),
        ForestPoint(
            "PPTA DR3",
            2.04,
            2.04 - 0.22,
            2.04 + 0.25,
            COLORS["green"],
        ),
        ForestPoint(
            "NG15 CNM",
            2.10,
            2.10 - 0.50,
            2.10 + 0.60,
            COLORS["orange"],
        ),
        ForestPoint(
            "EPTA DR2",
            2.50,
            2.50 - 0.70,
            2.50 + 0.70,
            COLORS["blue"],
        ),
        ForestPoint(
            "MPTA",
            4.80,
            4.80 - 0.90,
            4.80 + 0.80,
            COLORS["yellow"],
        ),
        ForestPoint(
            "CPTA DR1",
            cpta_central,
            cpta_lower,
            cpta_upper,
            COLORS["purple"],
            unbounded_right=True,
        ),
    )
    labels = [point.label for point in points]
    if len(labels) != len(set(labels)):
        raise RuntimeError("Forest-plot row labels must be unique.")
    for point in points:
        if point.central <= 0.0:
            raise RuntimeError(f"Non-positive forest central value: {point}")
        if (point.lower is None) != (point.upper is None):
            raise RuntimeError(f"Incomplete forest interval: {point}")
        if point.lower is not None and not (
            0.0 < point.lower <= point.central <= point.upper
        ):
            raise RuntimeError(f"Invalid forest interval ordering: {point}")
    return points


def configure_style() -> None:
    """Apply the compact serif style used by the public manuscript."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["cmr10"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 7.4,
            "axes.linewidth": 0.75,
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def draw_inference_plane(ax: plt.Axes, smbhb: dict[str, float]) -> None:
    """Draw the diagnostic plane of merger factor versus available mass.

    The abscissa is the SMBHB mass density retained after the population
    selection (Msun Mpc^-3); the ordinate is the merger factor f_merge of
    Eq. (13) of the paper (dimensionless).
    """

    retained_fraction = smbhb["retained_mass_density_fraction_benchmark"]
    rho_res_central = smbhb["participant_mass_density_benchmark_msun_mpc3"]
    rho_res_low = LM24_RHO_LOW * retained_fraction
    rho_res_high = LM24_RHO_HIGH * retained_fraction
    scale_x = lambda rho_bullet: rho_bullet * retained_fraction
    rho_res = np.logspace(np.log10(scale_x(7.0e5)), np.log10(scale_x(3.5e6)), 800)

    def fmerge_for_amplitude(amplitude: float) -> np.ndarray:
        return (
            (amplitude / smbhb["benchmark_A_1yr"]) ** 2
            * rho_res_central
            / rho_res
        )

    f_low = fmerge_for_amplitude(smbhb["cnm_low_A_1yr"])
    f_mid = fmerge_for_amplitude(smbhb["cnm_central_A_1yr"])
    f_high = fmerge_for_amplitude(smbhb["cnm_high_A_1yr"])

    one_pass = 1.0
    ax.axhspan(
        one_pass,
        SZQ_HIERARCHY_HC2_BOOST,
        facecolor="#F1F1F1",
        linewidth=0.0,
        zorder=0,
    )
    ax.fill_between(
        rho_res,
        f_low,
        f_high,
        color=COLORS["orange"],
        alpha=0.22,
        linewidth=0,
        zorder=1,
    )
    ax.plot(
        rho_res,
        f_mid,
        color=COLORS["orange"],
        linewidth=1.3,
        linestyle="--",
        zorder=3,
    )

    ax.axhspan(
        SZQ_HIERARCHY_HC2_BOOST,
        INFERENCE_Y_MAX,
        facecolor="#E5E5E5",
        edgecolor=COLORS["gray"],
        hatch="///",
        linewidth=0.0,
        alpha=0.62,
        zorder=0,
    )
    ax.axvspan(
        rho_res_low,
        rho_res_high,
        color=COLORS["blue"],
        alpha=0.12,
        linewidth=0,
    )
    ax.axvline(rho_res_central, color=COLORS["blue"], linewidth=1.25)
    ax.axhline(one_pass, color=COLORS["black"], linewidth=1.1)
    ax.axhline(
        SZQ_HIERARCHY_HC2_BOOST,
        color=COLORS["black"],
        linewidth=0.9,
        linestyle=(0, (4.0, 2.0)),
    )
    ax.plot(
        rho_res_central,
        1.0,
        marker="*",
        markersize=9.5,
        markerfacecolor=COLORS["sky"],
        markeredgecolor=COLORS["black"],
        markeredgewidth=0.55,
        zorder=6,
    )

    ax.text(
        scale_x(3.35e6),
        0.955,
        r"$f_{\rm merge}=1$",
        ha="right",
        va="top",
        fontsize=7.1,
    )
    ax.text(
        scale_x(3.35e6),
        SZQ_HIERARCHY_HC2_BOOST * 0.965,
        r"SZQ24 fixed-$f$ boost: $B_{\rm SZQ}=2.70$",
        ha="right",
        va="top",
        fontsize=6.45,
    )
    ax.text(
        scale_x(2.85e6),
        1.62,
        "SZQ24 hierarchy\ncomparison",
        ha="center",
        va="center",
        fontsize=6.7,
        color="#555555",
        linespacing=1.02,
    )
    ax.text(
        scale_x(7.35e5),
        0.36,
        (
            "LM24 family\n"
            r"$A_{\rm bench}="
            f"{smbhb['benchmark_A_1yr'] * 1.0e15:.2f}"
            r"\times10^{-15}$"
        ),
        ha="left",
        va="bottom",
        fontsize=6.55,
        color=COLORS["blue"],
        linespacing=1.02,
    )
    ax.text(
        scale_x(2.68e6),
        0.50,
        "NG15 CNM\n90% CrI",
        color=COLORS["orange"],
        fontsize=6.8,
        ha="center",
        va="center",
        linespacing=1.05,
    )
    ax.text(
        scale_x(1.315e6),
        0.235,
        "LM24 90% confidence interval",
        color=COLORS["blue"],
        fontsize=6.75,
        rotation=90,
        ha="left",
        va="bottom",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(scale_x(7.0e5), scale_x(3.5e6))
    ax.set_ylim(0.22, INFERENCE_Y_MAX)
    ax.set_yticks([0.25, 0.5, 1.0, SZQ_HIERARCHY_HC2_BOOST])
    ax.set_yticklabels([r"$0.25$", r"$0.5$", r"$1$", r"$2.7$"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(r"$\rho_{\rm res}^{\rm SMBHB}\ [M_\odot\,{\rm Mpc}^{-3}]$")
    ax.set_ylabel(r"merger factor $f_{\rm merge}$")
    ax.tick_params(which="both", direction="in", top=True, right=True)


def draw_forest(ax: plt.Axes, smbhb: dict[str, float]) -> None:
    """Draw model points and PTA power-law amplitude intervals."""

    points = pta_forest_points(smbhb)
    y_positions = np.arange(len(points) - 1, -1, -1, dtype=float)
    x_min, x_max = 1.0, 6.4
    theory_rows = 2
    separator_y = y_positions[theory_rows - 1] - 0.38

    ax.axhspan(
        separator_y,
        y_positions[0] + 0.62,
        color=COLORS["sky"],
        alpha=0.06,
        linewidth=0,
    )
    ax.axhline(separator_y, color="#BBBBBB", linewidth=0.7)
    ax.text(
        x_min + 0.22,
        0.5 * (separator_y + y_positions[0] + 0.62),
        "This work",
        rotation=90,
        ha="center",
        va="center",
        fontsize=7.8,
        color=COLORS["blue"],
    )

    for point, y in zip(points, y_positions, strict=True):
        facecolor = point.color if point.filled else "white"
        if point.label.startswith("$"):
            # Model rows: math label to the right of the interval end.
            ax.text(
                (point.upper if point.upper is not None else point.central)
                + 0.17,
                y,
                point.label,
                ha="left",
                va="center",
                fontsize=9.0,
                color=point.color,
            )
        else:
            # Observational rows: colored label above the interval center.
            ax.text(
                point.central,
                y + 0.29,
                point.label,
                ha="center",
                va="center",
                fontsize=7.3,
                color=point.color,
            )
        if point.lower is None or point.upper is None:
            ax.plot(
                point.central,
                y,
                marker=point.marker,
                markersize=8.0 if point.marker == "*" else 5.4,
                markerfacecolor=facecolor,
                markeredgecolor=point.color,
                markeredgewidth=1.0,
                linestyle="none",
                zorder=4,
            )
            continue

        if point.unbounded_right and point.upper > x_max:
            arrow_start = x_max - 0.65
            arrow_end = x_max - 0.08
            if point.lower < x_min:
                # Lower endpoint also lies beyond the plotted range: mirror
                # the arrow instead of drawing a clipped interval cap.
                left_start = x_min + 0.65
                left_end = x_min + 0.08
                ax.hlines(
                    y, left_start, arrow_start, color=point.color, linewidth=1.25
                )
                ax.annotate(
                    "",
                    xy=(left_end, y),
                    xytext=(left_start, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": point.color,
                        "lw": 1.25,
                        "shrinkA": 0,
                        "shrinkB": 0,
                    },
                )
            else:
                ax.hlines(
                    y, point.lower, arrow_start, color=point.color, linewidth=1.25
                )
                ax.vlines(
                    point.lower,
                    y - 0.10,
                    y + 0.10,
                    color=point.color,
                    linewidth=0.9,
                )
            ax.annotate(
                "",
                xy=(arrow_end, y),
                xytext=(arrow_start, y),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": point.color,
                    "lw": 1.25,
                    "shrinkA": 0,
                    "shrinkB": 0,
                },
            )
            ax.plot(
                point.central,
                y,
                marker=point.marker,
                markersize=5.2,
                markerfacecolor=facecolor,
                markeredgecolor=point.color,
                markeredgewidth=0.85,
                linestyle="none",
                zorder=4,
            )
            continue

        ax.errorbar(
            point.central,
            y,
            xerr=np.asarray(
                [
                    [point.central - point.lower],
                    [point.upper - point.central],
                ]
            ),
            fmt=point.marker,
            markersize=5.2,
            markerfacecolor=facecolor,
            markeredgecolor=point.color,
            markeredgewidth=0.85,
            ecolor=point.color,
            elinewidth=1.25,
            capsize=2.8,
            capthick=0.9,
            zorder=4,
        )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.82, y_positions[0] + 0.62)
    ax.set_yticks([])
    ax.set_xticks(np.arange(1.0, 6.1, 1.0))
    ax.set_xlabel(r"strain amplitude $A_{\rm yr}\ [10^{-15}]$")
    ax.tick_params(axis="x", which="both", direction="in", top=True)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.55, zorder=0)


def save_figure(fig: plt.Figure, output_stem: Path) -> tuple[Path, Path]:
    """Save one production figure as PDF and PNG."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_stem.with_suffix(".pdf")
    png_path = output_stem.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"saved {pdf_path}")
    print(f"saved {png_path}")
    return pdf_path, png_path


def build_figures(
    forest_output_stem: Path,
    plane_output_stem: Path,
) -> tuple[Path, Path, Path, Path]:
    """Render and save the forest first, then the inference plane."""

    configure_style()
    smbhb = smbhb_population_results()

    expected = {
        "benchmark_A_1yr": 2.1078708082857504e-15,
        "conditional_ceiling_A_1yr": 2.2878239901827213e-15,
        "repeated_merger_ceiling_A_1yr": 3.7609588494480605e-15,
    }
    for key, expected_value in expected.items():
        if not np.isclose(smbhb[key], expected_value, rtol=5.0e-13, atol=0.0):
            raise RuntimeError(
                f"authoritative {key} changed: {smbhb[key]:.16e}; "
                f"expected {expected_value:.16e}. Re-audit Figure 1."
            )

    forest_fig, forest_ax = plt.subplots(figsize=(3.42, 2.75))
    forest_fig.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.175,
        top=0.97,
    )
    draw_forest(forest_ax, smbhb)
    forest_pdf, forest_png = save_figure(forest_fig, forest_output_stem)

    plane_fig, plane_ax = plt.subplots(figsize=(3.35, 3.15))
    plane_fig.subplots_adjust(
        left=0.20,
        right=0.985,
        bottom=0.18,
        top=0.965,
    )
    draw_inference_plane(plane_ax, smbhb)
    plane_pdf, plane_png = save_figure(plane_fig, plane_output_stem)
    print(
        "SMBHB amplitudes at 1/yr: "
        f"benchmark={smbhb['benchmark_A_1yr']:.9e}, "
        "equal-mass z=0 base="
        f"{smbhb['equal_mass_latest_generation_A_1yr']:.9e}, "
        "repeated-merger ceiling="
        f"{smbhb['repeated_merger_ceiling_A_1yr']:.9e}"
    )
    print(
        "fixed-shape CNM mapping: "
        f"central={smbhb['fmerge_cnm_central_fixed_shape']:.6f}, "
        f"90% CrI=[{smbhb['fmerge_cnm_low_fixed_shape']:.6f}, "
        f"{smbhb['fmerge_cnm_high_fixed_shape']:.6f}]"
    )
    return forest_pdf, forest_png, plane_pdf, plane_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forest-output-stem",
        type=Path,
        default=DEFAULT_FOREST_STEM,
        help=(
            "forest output path without extension "
            f"(default: {DEFAULT_FOREST_STEM.relative_to(ROOT)})"
        ),
    )
    parser.add_argument(
        "--plane-output-stem",
        type=Path,
        default=DEFAULT_PLANE_STEM,
        help=(
            "inference-plane output path without extension "
            f"(default: {DEFAULT_PLANE_STEM.relative_to(ROOT)})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forest_stem = args.forest_output_stem
    plane_stem = args.plane_output_stem
    if not forest_stem.is_absolute():
        forest_stem = ROOT / forest_stem
    if not plane_stem.is_absolute():
        plane_stem = ROOT / plane_stem
    build_figures(forest_stem, plane_stem)


if __name__ == "__main__":
    main()
