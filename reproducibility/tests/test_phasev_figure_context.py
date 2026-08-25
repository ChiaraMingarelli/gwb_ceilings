"""Regression tests for the Phase-V detector and figure-context audit."""

from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent
sys.path.insert(0, str(ROOT))

import make_figure1_plane_forest as fig1  # noqa: E402
import verify_ceilings as vc  # noqa: E402


FIGURE_SOURCE = ROOT / "make_fig2_corrected.py"
PLANE_SOURCE = ROOT / "make_figure1_plane_forest.py"
MANUSCRIPT = Path(
    os.environ.get(
        "GWB_CEILINGS_MANUSCRIPT",
        REPOSITORY / "article" / "LISA-IMBH-GWB" / "main-resubmit.tex",
    )
)
PRIVATE_CONTEXT_ARCHIVE = Path(
    os.environ.get(
        "GWB_CEILINGS_PTA_CONTEXT",
        REPOSITORY / "step3-figure-context.npz",
    )
)
SPECTRA = ROOT / "results" / "step3-spectra.npz"
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


def load_display_minimums() -> dict[str, float]:
    """Read the Figure-2 plot-only frequency boundaries without importing it."""

    tree = ast.parse(FIGURE_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name)
            and target.id == "BENCHMARK_DISPLAY_MINIMUM_HZ"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise RuntimeError("Could not locate Figure-2 display-frequency boundaries.")


def load_detector_functions() -> dict[str, object]:
    """Load only pure detector functions without executing figure writes."""

    wanted = {
        "lisa_sensitivity",
        "muares_sensitivity",
        "bbo_sensitivity",
        "aligo_design_pi_sensitivity",
        "cosmic_explorer_sensitivity",
        "dwd_confusion_omega",
        "display_curve",
        "benchmark_display_curve",
        "benchmark_continuation_curve",
        "emri_high_frequency_continuation",
    }
    tree = ast.parse(FIGURE_SOURCE.read_text(encoding="utf-8"))
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    if {node.name for node in functions} != wanted:
        raise RuntimeError("Could not locate every detector-context function.")
    module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
    namespace: dict[str, object] = {
        "np": np,
        "hashlib": hashlib,
        "H0_KM_S_MPC": vc.H0_KM_S_MPC,
        "H0_SI": vc.H0_SI,
        "ALIGO_DESIGN_PI": ALIGO_DESIGN_PI,
        "ALIGO_DESIGN_PI_SHA256": ALIGO_DESIGN_PI_SHA256,
        "BBO_PLIS": BBO_PLIS,
        "BBO_PLIS_SHA256": BBO_PLIS_SHA256,
        "CE_40KM_STRAIN": CE_40KM_STRAIN,
        "CE_40KM_STRAIN_SHA256": CE_40KM_STRAIN_SHA256,
        "DWD_CONTINUATION_MINIMUM_HZ": 10.0**-5.5,
        "BENCHMARK_DISPLAY_MINIMUM_HZ": {
            "smbhb_benchmark": 1.0e-9,
        },
        "LOW_FREQUENCY_CONTINUATION_DEX": 0.50,
        "SMBHB_DISPLAY_MAXIMUM_HZ": 1.0e-6,
        "SMBHB_HIGH_FREQUENCY_CONTINUATION_DEX": 0.50,
        "EMRI_MODEL_CUTOFF_HZ": 1.0e-2,
        "EMRI_HIGH_FREQUENCY_CONTINUATION_DEX": 0.50,
    }
    exec(compile(module, str(FIGURE_SOURCE), "exec"), namespace)
    return namespace


def manuscript_text(test_case: unittest.TestCase) -> str:
    """Return the sibling manuscript or skip repository-integration checks."""

    if not MANUSCRIPT.is_file():
        test_case.skipTest(
            "manuscript integration requires article/LISA-IMBH-GWB"
        )
    return MANUSCRIPT.read_text(encoding="utf-8")


class PhaseVFigureContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.functions = load_detector_functions()

    def test_lisa_transfer_parenthesization_and_omega_conversion(self) -> None:
        frequency = np.asarray([20.0e-3, 40.0e-3])
        duration_seconds = 4.0 * 365.25 * 86_400.0
        omega = self.functions["lisa_sensitivity"](frequency)
        recovered_noise = (
            omega
            * np.sqrt(duration_seconds * frequency)
            * 3.0
            * vc.H0_SI**2
            / (2.0 * math.pi**2 * frequency**3)
        )

        length = 2.5e9
        f_star = 19.09e-3
        p_oms = (1.5e-11) ** 2 * (1.0 + (2.0e-3 / frequency) ** 4)
        p_acc = (3.0e-15) ** 2 * (1.0 + (0.4e-3 / frequency) ** 2) * (
            1.0 + (frequency / 8.0e-3) ** 4
        )
        transfer = 1.0 + 0.6 * (frequency / f_star) ** 2
        legacy_noise = (
            10.0
            / (3.0 * length**2)
            * (
                p_oms
                + 4.0
                * p_acc
                / (2.0 * math.pi * frequency) ** 4
                * transfer
            )
        )
        np.testing.assert_allclose(
            recovered_noise / legacy_noise,
            [1.63162944542, 3.41400701151],
            rtol=2.0e-11,
        )

    def test_bbo_uses_published_five_year_rescaled_plis(self) -> None:
        self.assertEqual(
            hashlib.sha256(BBO_PLIS.read_bytes()).hexdigest(),
            BBO_PLIS_SHA256,
        )
        frequency = np.asarray(
            [0.00099, 0.001, 0.01, 0.1, 0.3, 1.0, 10.0, 10.01]
        )
        omega = self.functions["bbo_sensitivity"](frequency)
        self.assertTrue(np.isnan(omega[0]))
        self.assertTrue(np.isnan(omega[-1]))
        np.testing.assert_allclose(
            omega[1:-1],
            [
                1.21611097e-13,
                2.12236075e-16,
                1.12247636e-17,
                1.12877649e-17,
                2.43588738e-16,
                2.35023454e-10,
            ],
            rtol=4.0e-9,
        )
        source = np.loadtxt(BBO_PLIS)
        minimum_index = int(np.argmin(source[:, 1]))
        minimum_frequency = 10.0 ** source[minimum_index, 0]
        minimum = self.functions["bbo_sensitivity"](
            np.asarray([minimum_frequency])
        )[0]
        self.assertAlmostEqual(minimum_frequency, 0.1833369491, places=9)
        self.assertAlmostEqual(minimum / 8.32461591e-18, 1.0, places=8)
        np.testing.assert_allclose(
            self.functions["bbo_sensitivity"](frequency[1:-1], years=20.0),
            omega[1:-1] / 2.0,
            rtol=2.0e-14,
        )

    def test_muares_uses_proposal_noise_and_one_sided_psd_conversion(
        self,
    ) -> None:
        frequency = np.asarray(
            [0.99e-7, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0, 1.01]
        )
        duration_seconds = 10.0 * 365.25 * 86_400.0
        omega = self.functions["muares_sensitivity"](frequency)
        self.assertTrue(np.isnan(omega[0]))
        self.assertTrue(np.isnan(omega[-1]))
        supported_frequency = frequency[1:-1]
        supported_omega = omega[1:-1]
        recovered_noise = (
            supported_omega
            * np.sqrt(duration_seconds * supported_frequency)
            * 3.0
            * vc.H0_SI**2
            / (2.0 * math.pi**2 * supported_frequency**3)
        )
        length = 3.95e11
        f_star = vc.C_SI / (2.0 * math.pi * length)
        s_pos = (50.0e-12) ** 2
        s_acc = (1.0e-15) ** 2
        expected = (
            10.0
            / (3.0 * length**2)
            * (
                4.0
                * s_acc
                / (2.0 * math.pi * supported_frequency) ** 4
                + s_pos
            )
            * (1.0 + 0.6 * (supported_frequency / f_star) ** 2)
        )
        np.testing.assert_allclose(recovered_noise, expected, rtol=2.0e-14)

    def test_ce_uses_official_40km_strain_data(self) -> None:
        self.assertEqual(
            hashlib.sha256(CE_40KM_STRAIN.read_bytes()).hexdigest(),
            CE_40KM_STRAIN_SHA256,
        )
        source = np.loadtxt(CE_40KM_STRAIN)
        self.assertEqual(source.shape, (1_000, 2))
        np.testing.assert_allclose(source[[0, -1], 0], [5.0, 5_000.0])
        frequency = np.asarray(
            [4.99, 5.0, 10.0, 20.0, 100.0, 1_000.0, 5_000.0, 5_000.1]
        )
        omega = self.functions["cosmic_explorer_sensitivity"](frequency)
        self.assertTrue(np.isnan(omega[0]))
        self.assertTrue(np.isnan(omega[-1]))
        np.testing.assert_allclose(
            omega[[2, 3, 4, 5]],
            [
                3.58835e-14,
                3.08790e-14,
                1.03348e-12,
                8.35304e-10,
            ],
            rtol=7.0e-6,
        )
        grid = np.geomspace(5.0, 5_000.0, 20_000)
        curve = self.functions["cosmic_explorer_sensitivity"](grid)
        minimum_index = int(np.nanargmin(curve))
        self.assertAlmostEqual(grid[minimum_index] / 14.2044, 1.0, places=3)
        self.assertAlmostEqual(curve[minimum_index] / 2.28617e-14, 1.0, places=3)

    def test_aligo_uses_published_two_sigma_design_pi_curve(self) -> None:
        self.assertEqual(
            hashlib.sha256(ALIGO_DESIGN_PI.read_bytes()).hexdigest(),
            ALIGO_DESIGN_PI_SHA256,
        )
        frequency = np.asarray([9.75, 10.0, 25.25, 100.0, 2_000.0, 2_000.25])
        omega = self.functions["aligo_design_pi_sensitivity"](frequency)
        self.assertTrue(np.isnan(omega[0]))
        self.assertTrue(np.isnan(omega[-1]))
        np.testing.assert_allclose(
            omega[1:-1],
            [
                2.0 * 1.9487915e-8,
                2.0 * 9.56176551e-10,
                2.0 * 1.45154677e-8,
                2.0 * 1.17171849e-2,
            ],
            rtol=2.0e-14,
        )
        self.assertEqual(float(frequency[np.nanargmin(omega)]), 25.25)

    def test_figure_scope_keeps_conditional_ceilings_out(self) -> None:
        figure_source = FIGURE_SOURCE.read_text(encoding="utf-8")
        plane_source = PLANE_SOURCE.read_text(encoding="utf-8")
        for key in (
            "smbhb_benchmark",
            "emri_benchmark",
            "agn_imri_benchmark",
            "bns_benchmark",
            "popiii_benchmark",
            "bbh_benchmark",
        ):
            self.assertIn(key, figure_source)
        self.assertIn("muares_sensitivity", figure_source)
        self.assertIn("BBO_PLIS_Schmitz2021.dat", figure_source)
        self.assertIn("cosmic_explorer_strain_T2000017-v9.txt", figure_source)
        self.assertIn("aligo_design_pi_sensitivity", figure_source)
        self.assertIn('"aLIGO design"', figure_source)
        self.assertIn("DETECTOR_PLOT_MAX = 2.0e-7", figure_source)
        self.assertIn("curve <= DETECTOR_PLOT_MAX", figure_source)
        self.assertNotIn("curve < 1.0e-6", figure_source)
        self.assertIn('"NANOGrav"', figure_source)
        self.assertIn("pta_log10_omega_grid", figure_source)
        self.assertIn("ax.fill_betweenx(", figure_source)
        self.assertIn("--pta-context", figure_source)
        self.assertIn("if pta_context is not None:", figure_source)
        self.assertNotIn("ax.errorbar(", figure_source)
        self.assertNotIn("conditional_benchmark_composite", figure_source)
        self.assertNotIn("conditional_ceiling", figure_source)
        self.assertNotIn("Integrated Astrophysical Ceiling", figure_source)
        self.assertIn("--spectra-only", figure_source)
        self.assertIn("--no-labels", figure_source)
        self.assertIn("figure2-no-labels.pdf", figure_source)
        self.assertIn("if not args.no_labels:", figure_source)
        self.assertIn(r'frequency $f$ (Hz)', figure_source)
        self.assertNotIn(r'frequency $f$ [Hz]', figure_source)
        self.assertNotIn("Integrated benchmark sum", figure_source)
        self.assertNotIn("CUMULATIVE_QUANTILES", figure_source)
        self.assertNotIn("cumulative_quantile_frequencies", figure_source)
        self.assertIn("SMBHB_DISPLAY_MAXIMUM_HZ = 1.0e-6", figure_source)
        self.assertIn(
            "SMBHB_HIGH_FREQUENCY_CONTINUATION_DEX = 0.50",
            figure_source,
        )
        self.assertIn("smbhb_population_omega", figure_source)
        self.assertIn("DEFAULT_PTA_CONTEXT.is_file()", figure_source)
        self.assertNotIn(r"\mathcal{B}_{\rm comp}", figure_source)
        manuscript = manuscript_text(self)
        self.assertIn(r"\mathcal E_{\rm bench}^{(6)}", manuscript)
        self.assertNotIn(r"\mathcal{B}_{\rm comp}", manuscript)
        self.assertIn(
            r"\includegraphics[width=\linewidth]{figure3-final.pdf}",
            manuscript,
        )
        self.assertIn("not the filled areas here", manuscript)
        self.assertNotIn("The lower panel shows the cumulative energy fraction", manuscript)
        self.assertIn(r"\label{eq:omega-strain}", manuscript)
        self.assertIn("LM24 90% confidence interval", plane_source)

    def test_figure1_restores_szq_hierarchy_extension(self) -> None:
        plane_source = PLANE_SOURCE.read_text(encoding="utf-8")
        self.assertIn(
            "SZQ_HIERARCHY_HC2_BOOST,",
            plane_source,
        )
        self.assertIn("INFERENCE_Y_MAX = 4.0", plane_source)
        self.assertIn("SZQ_HIERARCHY_HC2_BOOST,", plane_source)
        self.assertIn('hatch="///"', plane_source)
        self.assertIn(
            r"SZQ24 fixed-$f$ boost: $B_{\rm SZQ}=2.70$",
            plane_source,
        )
        self.assertIn(
            r"merger factor $f_{\rm merge}$",
            plane_source,
        )
        self.assertNotIn("np.minimum(f_high, one_pass)", plane_source)
        manuscript = manuscript_text(self)
        self.assertNotIn(r"B_{\rm SZQ}", manuscript)
        self.assertIn(
            r"A_{\rm ceil}",
            manuscript,
        )
        self.assertIn(
            r"$1.05\times1.12\times2.70=3.18$",
            manuscript,
        )
        self.assertIn(r"\label{eq:smbhb-repeated-ceiling}", manuscript)
        self.assertIn(r"\citet{satopolito2024}", manuscript)
        self.assertIn(
            "include the complete SZQ24 merger hierarchy",
            manuscript,
        )
        smbhb = vc.smbhb_population_results()
        self.assertGreater(smbhb["fmerge_cnm_high_fixed_shape"], 1.0)
        self.assertLess(
            smbhb["fmerge_cnm_high_fixed_shape"],
            1.64**2,
        )

    def test_figure1_forest_uses_plain_labels_and_density_ranges(self) -> None:
        smbhb = vc.smbhb_population_results()
        points = fig1.pta_forest_points(smbhb)
        self.assertEqual(
            tuple(point.label for point in points),
            (
                r"$A_{\rm bench}$",
                r"$A_{\rm ceil}$",
                "PPTA DR3",
                "NG15 CNM",
                "EPTA DR2",
                "MPTA",
                "CPTA DR1",
            ),
        )
        benchmark, ceiling = points[:2]
        self.assertAlmostEqual(ceiling.central, 3.7609588494480605, places=12)
        density_scales = np.sqrt(
            np.asarray([fig1.LM24_RHO_LOW, fig1.LM24_RHO_HIGH])
            / vc.RHO_BH_LM24
        )
        np.testing.assert_allclose(
            [benchmark.lower, benchmark.upper],
            benchmark.central * density_scales,
            rtol=1.0e-14,
        )
        np.testing.assert_allclose(
            [ceiling.lower, ceiling.upper],
            ceiling.central * density_scales,
            rtol=1.0e-14,
        )
        manuscript = manuscript_text(self)
        self.assertIn(
            r"\includegraphics[width=0.9\columnwidth]{pta-amplitude-forest.pdf}",
            manuscript,
        )
        self.assertIn(r"\textbf{PTA amplitude comparison.}", manuscript)
        self.assertIn(r"fixed-$\gamma=13/3$", manuscript)
        self.assertIn(
            "and all fit comfortably within the ceiling.",
            " ".join(manuscript.split()),
        )
        self.assertIn(
            "The benchmark overlaps measurements from the Parkes PTA third data",
            " ".join(manuscript.split()),
        )
        self.assertIn(
            r"\includegraphics[width=\columnwidth]{fmerge-posterior.pdf}",
            manuscript,
        )
        self.assertIn(r"\label{fig:fmerge-posterior}", manuscript)
        self.assertIn(
            r"Posterior for $f_{\rm merge}$ from Eq.~\eqref{eq:fmerge-propagation}",
            " ".join(manuscript.split()),
        )
        self.assertIn(
            r"=1.11^{+1.35}_{-0.63}\qquad(90\%).",
            manuscript,
        )
        self.assertNotIn(
            r"\includegraphics[width=\columnwidth]{smbhb-inference-plane.pdf}",
            manuscript,
        )

    def test_smbhb_spectrum_is_restricted_to_the_pta_display_window(self) -> None:
        with np.load(SPECTRA) as spectra:
            frequency = np.asarray(spectra["frequency_hz"], dtype=float)
            smbhb = np.asarray(spectra["smbhb_benchmark"], dtype=float)
            self.assertNotIn(
                "conditional_benchmark_cumulative_below_frequency",
                spectra.files,
            )
        in_window = (frequency >= 1.0e-9) & (frequency <= 1.0e-7)
        self.assertTrue(np.any(smbhb[in_window] > 0.0))
        self.assertTrue(np.all(smbhb[~in_window] == 0.0))

    def test_benchmark_low_frequency_boundaries_are_plot_only(self) -> None:
        self.assertEqual(
            load_display_minimums(),
            {
                "smbhb_benchmark": 1.0e-9,
                "emri_benchmark": 1.0e-5,
                "agn_imri_benchmark": 1.0e-5,
                "popiii_benchmark": 1.0e-3,
                "bbh_benchmark": 3.1622776601683795e-5,
                "bns_benchmark": 1.0e-3,
            },
        )
        self.assertAlmostEqual(
            load_display_minimums()["bbh_benchmark"]
            * 10.0 ** (-0.50),
            1.0e-5,
            places=18,
        )
        source = FIGURE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("plotting/model-validity boundaries", source)
        self.assertIn("energetic spectra retain", source)
        self.assertIn("time to merger is approximately the declared 10-Gyr", source)
        self.assertIn("LOW_FREQUENCY_CONTINUATION_DEX = 0.50", source)
        self.assertIn("benchmark_continuation_curve", source)
        self.assertIn("linestyle=(0, (3.0, 2.0))", source)
        self.assertIn("DWD_CONTINUATION_MINIMUM_HZ", source)
        self.assertIn("dwd_continuation_mask", source)

    def test_emri_has_display_only_high_frequency_continuation(self) -> None:
        frequency = np.geomspace(1.0e-5, 1.0e-2, 1000)
        omega = 2.5e-12 * (frequency / 1.0e-2) ** (2.0 / 3.0)
        continuation_frequency, continuation_omega = self.functions[
            "emri_high_frequency_continuation"
        ](frequency, omega)
        self.assertAlmostEqual(continuation_frequency[0], 1.0e-2)
        self.assertAlmostEqual(
            continuation_frequency[-1],
            1.0e-2 * 10.0**0.5,
        )
        self.assertAlmostEqual(continuation_omega[0], 2.5e-12)
        self.assertAlmostEqual(
            continuation_omega[-1] / continuation_omega[0],
            10.0 ** (1.0 / 3.0),
        )
        source = FIGURE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("EMRI_HIGH_FREQUENCY_CONTINUATION_DEX = 0.50", source)
        self.assertIn("emri_high_frequency_continuation", source)
        self.assertIn("emri_high_omega", source)

    def test_smbhb_has_solid_segment_and_two_dashed_guides(self) -> None:
        frequency = np.geomspace(1.0e-10, 1.0e-5, 1001)
        omega = 2.0e-9 * (frequency / 1.0e-8) ** (2.0 / 3.0)
        solid = self.functions["benchmark_display_curve"](
            frequency,
            omega,
            "smbhb_benchmark",
        )
        guide = self.functions["benchmark_continuation_curve"](
            frequency,
            omega,
            "smbhb_benchmark",
        )
        solid_frequency = frequency[np.isfinite(solid)]
        guide_frequency = frequency[np.isfinite(guide)]
        self.assertAlmostEqual(solid_frequency[0], 1.0e-9)
        self.assertAlmostEqual(solid_frequency[-1], 1.0e-6)
        self.assertAlmostEqual(guide_frequency[0], 10.0**-9.5)
        self.assertAlmostEqual(guide_frequency[-1], 10.0**-5.5)
        self.assertFalse(
            np.any(
                np.isfinite(guide)
                & (frequency > 1.0e-9)
                & (frequency < 1.0e-6)
            )
        )
    def test_background_fills_are_subtle_and_share_the_plot_floor(self) -> None:
        source = FIGURE_SOURCE.read_text(encoding="utf-8")
        self.assertIn("BACKGROUND_FILL_ALPHA = 0.10", source)
        self.assertIn("PLOT_OMEGA_MIN = 1.0e-18", source)
        self.assertIn("DWD_DISPLAY_MINIMUM_HZ = 1.0e-5", source)
        self.assertIn(
            "fill_curve = np.where(np.isfinite(curve), curve, continuation)",
            source,
        )
        self.assertIn("frequency[dwd_fill_mask]", source)
        self.assertGreaterEqual(source.count("ax.fill_between("), 2)
        self.assertIn("ax.set_ylim(PLOT_OMEGA_MIN, 1.0e-6)", source)

    def test_nanograv_violins_share_detector_gray(self) -> None:
        source = FIGURE_SOURCE.read_text(encoding="utf-8")
        self.assertIn('CONTEXT_GRAY = "#8A8A8A"', source)
        self.assertIn("color=CONTEXT_GRAY", source)
        self.assertGreaterEqual(source.count("CONTEXT_GRAY,"), 3)
        self.assertIn("NANOGRAV_FACE_ALPHA = 0.30", source)
        self.assertIn("NANOGRAV_EDGE_ALPHA = 0.58", source)
        self.assertNotIn('color="#4A4A4A"', source)
        self.assertNotIn('edgecolor="#363636"', source)

    def test_archived_pta_context_is_self_contained_and_provenanced(self) -> None:
        if not PRIVATE_CONTEXT_ARCHIVE.is_file():
            self.skipTest(
                "private PTA plotting context is intentionally absent from "
                "the public package"
            )
        with np.load(PRIVATE_CONTEXT_ARCHIVE) as context:
            self.assertEqual(context["pta_frequency_hz"].shape, (5,))
            self.assertEqual(context["pta_log10_omega_grid"].shape, (5, 180))
            self.assertEqual(context["pta_half_width_hz"].shape, (5, 180))
            self.assertTrue(np.all(context["pta_half_width_hz"] >= 0.0))
            np.testing.assert_allclose(
                context["pta_half_width_hz"].max(axis=1),
                0.22 * context["pta_frequency_hz"],
                rtol=2.0e-14,
            )
            self.assertEqual(
                str(context["source_log10_frequency_sha256"]),
                "144fd81d0ccfc01057f5fe865de5430eb496360932510734aac345b2f3ef538e",
            )
            self.assertEqual(
                str(context["source_log10_omega_sha256"]),
                "20f9ecd23735f8fe506bcaa200581c80de5e4128b38ed9ba0a615e9a529e1939",
            )
            self.assertEqual(int(context["source_sample_count_per_bin"]), 50_000)


if __name__ == "__main__":
    unittest.main()
