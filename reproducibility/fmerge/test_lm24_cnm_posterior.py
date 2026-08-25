"""Numerical and statistical tests for the LM24--CNM propagation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

try:
    from fmerge import lm24_cnm_posterior as inference
except ImportError:  # Support direct execution from this directory.
    import lm24_cnm_posterior as inference


class LM24CNMPosteriorTests(unittest.TestCase):
    def assertRelative(self, actual: float, expected: float, rtol: float) -> None:
        self.assertLessEqual(
            abs(actual - expected),
            rtol * abs(expected),
            msg=f"{actual!r} != {expected!r} within rtol={rtol}",
        )

    def test_reference_amplitudes_round_trip(self) -> None:
        for definition in (inference.LM24_DYNAMICAL, inference.LM24_SPS):
            with self.subTest(branch=definition.name):
                means = definition.means()[None, :]
                phi_low = 10.0 ** means[:, 0]
                phi_high = 10.0 ** means[:, 1]
                phi1, phi2 = inference.solve_double_schechter_amplitudes(
                    phi_low,
                    phi_high,
                    means[:, 2],
                    means[:, 3],
                    means[:, 4],
                )
                evaluated = inference.evaluate_gsmf_per_dex(
                    np.asarray([10.0, 12.0]),
                    phi1,
                    phi2,
                    means[:, 2],
                    means[:, 3],
                    means[:, 4],
                )[0]
                np.testing.assert_allclose(
                    evaluated,
                    np.asarray([phi_low[0], phi_high[0]]),
                    rtol=2.0e-13,
                )

    def test_population_quadrature_is_converged(self) -> None:
        config = inference.PropagationConfig(
            sample_count=48,
            seed=811,
            batch_size=48,
        )
        sequence = np.random.SeedSequence(config.seed)
        gsmf_seed, scaling_seed = sequence.spawn(2)
        draws = inference.draw_lm24_population(
            config,
            np.random.Generator(np.random.PCG64(gsmf_seed)),
            np.random.Generator(np.random.PCG64(scaling_seed)),
        )
        baseline = inference.compute_population_moments(draws, config)
        refined = inference.compute_population_moments(
            draws,
            replace(
                config,
                stellar_quadrature_nodes=144,
                mass_ratio_quadrature_nodes=64,
                redshift_quadrature_nodes=144,
            ),
        )
        np.testing.assert_allclose(
            baseline.amplitude_uncut,
            refined.amplitude_uncut,
            rtol=2.0e-6,
        )
        np.testing.assert_allclose(
            baseline.rho_total_msun_mpc3,
            refined.rho_total_msun_mpc3,
            rtol=2.0e-6,
        )

    def test_population_moments_do_not_depend_on_batch_size(self) -> None:
        config = inference.PropagationConfig(
            sample_count=40,
            seed=314,
            batch_size=40,
            stellar_quadrature_nodes=48,
            mass_ratio_quadrature_nodes=24,
            redshift_quadrature_nodes=48,
        )
        sequence = np.random.SeedSequence(config.seed)
        gsmf_seed, scaling_seed = sequence.spawn(2)
        draws = inference.draw_lm24_population(
            config,
            np.random.Generator(np.random.PCG64(gsmf_seed)),
            np.random.Generator(np.random.PCG64(scaling_seed)),
        )
        one_batch = inference.compute_population_moments(draws, config)
        many_batches = inference.compute_population_moments(
            draws, replace(config, batch_size=7)
        )
        np.testing.assert_allclose(
            one_batch.amplitude_uncut,
            many_batches.amplitude_uncut,
            rtol=1.0e-14,
        )
        np.testing.assert_allclose(
            one_batch.rho_total_msun_mpc3,
            many_batches.rho_total_msun_mpc3,
            rtol=1.0e-14,
        )

    def test_split_lognormal_matches_reported_quantiles(self) -> None:
        samples = inference.split_lognormal_quantile_draws(
            1.6e-15,
            2.1e-15,
            2.7e-15,
            100_000,
            np.random.Generator(np.random.PCG64(1234)),
        )
        summary = inference.quantile_summary(samples)
        self.assertRelative(summary["lower_90"], 1.6e-15, 3.0e-4)
        self.assertRelative(summary["median"], 2.1e-15, 3.0e-4)
        self.assertRelative(summary["upper_90"], 2.7e-15, 3.0e-4)

    def test_reconstructed_population_and_mapping_regression(self) -> None:
        config = inference.PropagationConfig(
            sample_count=2_000,
            seed=20_240_724,
            batch_size=256,
        )
        summary, samples = inference.run_propagation(config)
        results = summary["results"]

        self.assertRelative(
            results["lm24_reconstructed_rho_total_msun_mpc3"]["median"],
            1.733386048710455e6,
            1.0e-8,
        )
        self.assertRelative(
            results["A_lm24_reconstructed_uncut"]["median"],
            1.9839902807862163e-15,
            1.0e-8,
        )
        self.assertRelative(
            results["f_merge_log_uniform_prior"]["median"],
            1.1138033774572433,
            1.0e-8,
        )
        self.assertNotIn("manuscript_centering", summary["method"])
        self.assertNotIn("f_merge_manuscript", samples)
        np.testing.assert_array_equal(
            samples["A_lm24_used_for_f_merge"],
            samples["A_lm24_reconstructed_uncut"],
        )
        self.assertLessEqual(
            results["f_merge_log_uniform_prior"]["lower_90"],
            1.0,
        )
        self.assertGreaterEqual(
            results["f_merge_log_uniform_prior"]["upper_90"],
            1.0,
        )
        np.testing.assert_allclose(
            samples["log10_f_merge"],
            2.0
            * (
                np.log10(samples["A_cnm"])
                - np.log10(samples["A_lm24_used_for_f_merge"])
            ),
            rtol=0.0,
            atol=5.0e-15,
        )
        self.assertEqual(summary["diagnostics"]["dynamical_draw_count"], 1_000)
        self.assertEqual(summary["diagnostics"]["sps_draw_count"], 1_000)
        self.assertRelative(
            results["probability_f_merge_le_1_log_uniform_prior"]
            + results["probability_f_merge_gt_1_log_uniform_prior"],
            1.0,
            1.0e-14,
        )
        self.assertGreater(
            results["f_merge_flat_prior_sensitivity"]["median"],
            results["f_merge_log_uniform_prior"]["median"],
        )
        self.assertLessEqual(
            results["f_merge_log_uniform_prior"]["lower_90"], 1.0
        )
        self.assertGreaterEqual(
            results["f_merge_log_uniform_prior"]["upper_90"], 1.0
        )
        self.assertLessEqual(
            results["A_lm24_reconstructed_uncut"]["lower_90"], 2.0e-15
        )
        self.assertGreaterEqual(
            results["A_lm24_reconstructed_uncut"]["upper_90"], 2.0e-15
        )
        self.assertRelative(
            results["lm24_reconstructed_rho_total_msun_mpc3"]["lower_90"],
            1.3e6,
            0.12,
        )
        self.assertRelative(
            results["lm24_reconstructed_rho_total_msun_mpc3"]["upper_90"],
            2.6e6,
            0.12,
        )
        self.assertRelative(
            results["A_lm24_reconstructed_uncut"]["lower_90"],
            1.5e-15,
            0.12,
        )
        self.assertRelative(
            results["A_lm24_reconstructed_uncut"]["upper_90"],
            2.7e-15,
            0.12,
        )

    def test_weighted_npz_chain_is_loaded_and_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weighted_amplitudes.npz"
            np.savez(
                path,
                amplitude=np.asarray([1.0e-15, 2.0e-15, 3.0e-15]),
                weight=np.asarray([0.1, 0.2, 0.7]),
            )
            loaded = inference.load_weighted_amplitude_samples(
                path, "amplitude"
            )
            np.testing.assert_array_equal(
                loaded.values, np.asarray([1.0e-15, 2.0e-15, 3.0e-15])
            )
            np.testing.assert_array_equal(
                loaded.weights, np.asarray([0.1, 0.2, 0.7])
            )
            self.assertEqual(loaded.weight_key, "weight")
            self.assertEqual(len(loaded.sha256), 64)

    def test_invalid_external_chain_entries_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            negative_path = root / "negative.npy"
            np.save(negative_path, np.asarray([1.0e-15, -2.0e-15]))
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                inference.load_weighted_amplitude_samples(
                    negative_path, "amplitude"
                )

            zero_weight_path = root / "zero_weights.npz"
            np.savez(
                zero_weight_path,
                amplitude=np.asarray([1.0e-15, 2.0e-15]),
                weight=np.asarray([0.0, 0.0]),
            )
            with self.assertRaisesRegex(ValueError, "not all zero"):
                inference.load_weighted_amplitude_samples(
                    zero_weight_path, "amplitude"
                )

            mismatch_path = root / "mismatch.npz"
            np.savez(
                mismatch_path,
                amplitude=np.asarray([1.0e-15, 2.0e-15]),
                weight=np.asarray([1.0]),
            )
            with self.assertRaisesRegex(ValueError, "match amplitudes"):
                inference.load_weighted_amplitude_samples(
                    mismatch_path, "amplitude"
                )

    def test_external_weighted_chains_use_saved_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cnm_path = root / "cnm.npz"
            lm24_path = root / "lm24.npz"
            np.savez(
                cnm_path,
                amplitude=np.asarray([1.7e-15, 2.0e-15, 2.4e-15]),
                weight=np.asarray([0.2, 0.5, 0.3]),
            )
            np.savez(
                lm24_path,
                amplitude=np.asarray([1.5e-15, 1.9e-15, 2.3e-15]),
                weight=np.asarray([0.3, 0.4, 0.3]),
            )
            config = inference.PropagationConfig(
                sample_count=32,
                seed=71,
                batch_size=16,
                stellar_quadrature_nodes=32,
                mass_ratio_quadrature_nodes=20,
                redshift_quadrature_nodes=32,
            )
            summary, samples = inference.run_propagation(
                config,
                cnm_sample_path=cnm_path,
                lm24_amplitude_sample_path=lm24_path,
                confirm_external_cnm_semantics=True,
                confirm_external_lm24_full_census_semantics=True,
                cnm_source_id="test:cnm-chain",
                lm24_source_id="test:lm24-chain",
            )
            np.testing.assert_allclose(
                samples["f_merge"],
                (
                    samples["A_cnm"]
                    / samples["A_lm24_used_for_f_merge"]
                )
                ** 2,
                rtol=2.0e-16,
            )
            self.assertTrue(
                summary["external_inputs"]["cnm"][
                    "semantics_confirmed_by_caller"
                ]
            )
            self.assertIn("caller-confirmed", summary["status"])

    def test_external_chain_requires_semantic_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "amplitudes.npy"
            np.save(path, np.asarray([1.8e-15, 2.2e-15]))
            config = inference.PropagationConfig(sample_count=4, batch_size=4)
            with self.assertRaisesRegex(ValueError, "explicit confirmation"):
                inference.run_propagation(config, cnm_sample_path=path)
            with self.assertRaisesRegex(ValueError, "source DOI"):
                inference.run_propagation(
                    config,
                    cnm_sample_path=path,
                    confirm_external_cnm_semantics=True,
                )

    def test_output_archive_is_linked_to_json_metadata(self) -> None:
        config = inference.PropagationConfig(
            sample_count=16,
            seed=19,
            batch_size=8,
            stellar_quadrature_nodes=24,
            mass_ratio_quadrature_nodes=16,
            redshift_quadrature_nodes=24,
        )
        summary, samples = inference.run_propagation(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary_path = root / "summary.json"
            sample_path = root / "draws.npz"
            written = inference.write_outputs(
                summary, samples, summary_path, sample_path
            )
            loaded_summary = json.loads(summary_path.read_text())
            digest = hashlib.sha256(sample_path.read_bytes()).hexdigest()
            self.assertEqual(written, loaded_summary)
            self.assertEqual(loaded_summary["sample_archive"]["sha256"], digest)
            with np.load(sample_path, allow_pickle=False) as archive:
                self.assertIn("A_lm24_used_for_f_merge", archive.files)
                self.assertIn("metadata_json", archive.files)
                embedded = json.loads(str(archive["metadata_json"]))
            self.assertEqual(
                embedded["implementation"]["sha256"],
                loaded_summary["implementation"]["sha256"],
            )

            cross_summary = root / "summaries" / "summary.json"
            cross_samples = root / "samples" / "draws.npz"
            cross_written = inference.write_outputs(
                summary, samples, cross_summary, cross_samples
            )
            self.assertEqual(
                cross_written["sample_archive"]["path_relative_to_summary"],
                str(Path("..") / "samples" / "draws.npz"),
            )

            with self.assertRaisesRegex(ValueError, "end in .npz"):
                inference.write_outputs(
                    summary, samples, root / "other.json", root / "draws.dat"
                )
            with self.assertRaisesRegex(ValueError, "end in .npz"):
                inference.write_outputs(
                    summary, samples, root / "other.json", root / "draws.NPZ"
                )
            with self.assertRaisesRegex(ValueError, "must be different"):
                inference.write_outputs(
                    summary, samples, root / "collision.npz", root / "collision.npz"
                )
            protected = root / "source.npz"
            protected.write_bytes(b"source chain")
            with self.assertRaisesRegex(ValueError, "overwrite an input"):
                inference.write_outputs(
                    summary,
                    samples,
                    root / "other.json",
                    protected,
                    protected_paths=(protected,),
                )

    def test_package_and_calculation_versions_match(self) -> None:
        pyproject = Path(__file__).with_name("pyproject.toml").read_text()
        match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), inference.CALCULATION_VERSION)
        citation = Path(__file__).with_name("CITATION.cff").read_text()
        citation_match = re.search(r"^version: (\S+)$", citation, re.MULTILINE)
        self.assertIsNotNone(citation_match)
        self.assertEqual(citation_match.group(1), inference.CALCULATION_VERSION)

    def test_flat_prior_reweighting_identity(self) -> None:
        values = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0])
        weights = values.copy()
        self.assertRelative(
            inference.weighted_probability(values <= 1.0, weights),
            (0.25 + 0.5 + 1.0) / np.sum(values),
            1.0e-14,
        )

    def test_population_rescaling_identity(self) -> None:
        observed = np.asarray([1.8e-15, 2.1e-15, 2.6e-15])
        model = np.asarray([1.5e-15, 2.0e-15, 2.7e-15])
        rho = np.asarray([1.0e6, 1.5e6, 2.2e6])
        scale = 3.7
        f_original = (observed / model) ** 2
        f_rescaled = (observed / (math.sqrt(scale) * model)) ** 2
        np.testing.assert_allclose(f_rescaled, f_original / scale, rtol=1e-15)
        np.testing.assert_allclose(
            f_rescaled * scale * rho,
            f_original * rho,
            rtol=1e-15,
        )

    def test_invalid_prior_and_scatter_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "intrinsic scatter"):
            inference.PropagationConfig(
                scaling_intrinsic_scatter_dex=0.0
            ).validate()
        with self.assertRaisesRegex(ValueError, "outside its prior"):
            inference.PropagationConfig(
                cnm_log10_amplitude_prior_max=-15.0
            ).validate()

    def test_free_spectrum_table_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "free_spectrum.npy"
            np.save(path, np.ones((30, 50)))
            with self.assertRaisesRegex(ValueError, "one-dimensional"):
                inference.load_positive_samples(path, "amplitude")


if __name__ == "__main__":
    unittest.main()
