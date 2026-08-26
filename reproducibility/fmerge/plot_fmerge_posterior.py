#!/usr/bin/env python3
"""Build the paper's Figure 2, the reconstructed SMBHB f_merge posterior.

Reads the propagated draws written by ``lm24_cnm_posterior.py`` and renders
the probability density of the dimensionless merger factor f_merge with its
median and equal-tailed 90% interval, f_merge = 1.11 (+1.35, -0.63)
(Eq. (15) of Mingarelli 2026, arXiv:2601.18859).  The default PDF output,
figures/fmerge-posterior.pdf, is the manuscript asset for Figure 2.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


HERE = Path(__file__).resolve().parent
REPRO_ROOT = HERE.parent
DEFAULT_SAMPLES = REPRO_ROOT / "results" / "fmerge" / "posterior_draws.npz"
DEFAULT_OUTPUT_STEM = REPRO_ROOT / "figures" / "fmerge-posterior"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=DEFAULT_OUTPUT_STEM,
        help="output path without an extension",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.samples) as data:
        f_merge = data["f_merge"]
        f_merge = f_merge[data["within_f_prior"]]

    lower, median, upper = np.quantile(f_merge, [0.05, 0.50, 0.95])

    # Use all saved draws in a kernel-density estimate.  A modestly broadened
    # Scott bandwidth suppresses sampling-scale ripples without changing the
    # reported quantiles.
    x = np.linspace(0.0, 4.0, 600)
    kde = gaussian_kde(f_merge, bw_method=lambda estimate: 1.5 * estimate.scotts_factor())
    density = kde(x)

    blue = "#0072B2"
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8.5,
            "axes.labelsize": 10,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )

    # APS/PRD single-column figures are 3.375 inches wide.
    fig, ax = plt.subplots(figsize=(3.375, 2.35))
    ax.plot(x, density, color=blue, lw=2.0)
    interval = (x >= lower) & (x <= upper)
    ax.fill_between(
        x[interval], density[interval], color=blue, alpha=0.24, lw=0
    )
    ax.axvline(1.0, color="0.70", ls="--", lw=1.3)

    median_density = np.interp(median, x, density)
    ax.plot(median, median_density, "o", color=blue, ms=5.5, zorder=4)

    ax.text(
        0.97,
        0.94,
        rf"$f_{{\rm merge}}={median:.2f}^{{+{upper-median:.2f}}}_{{-{median-lower:.2f}}}$",
        ha="right",
        va="top",
        fontsize=9,
        transform=ax.transAxes,
    )

    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel(r"$f_{\rm merge}$")
    ax.set_ylabel("Probability density")

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"))
    fig.savefig(args.output_stem.with_suffix(".png"))
    plt.close(fig)

    print(
        f"f_merge = {median:.6f} [{lower:.6f}, {upper:.6f}]\n"
        f"wrote {args.output_stem.with_suffix('.pdf')}\n"
        f"wrote {args.output_stem.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
