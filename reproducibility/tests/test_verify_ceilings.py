"""Regression tests for the Step-3 authoritative calculation."""

from __future__ import annotations

import inspect
import math
from pathlib import Path
import sys
import unittest

import numpy as np

REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT))

import verify_ceilings as vc  # noqa: E402


class Step3CalculationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = vc.build_results()

    def assertRelative(self, actual: float, expected: float, rtol: float) -> None:
        self.assertLessEqual(
            abs(actual - expected),
            rtol * abs(expected),
            msg=f"{actual!r} != {expected!r} within rtol={rtol}",
        )

    def test_lm24_population_moment_regression(self) -> None:
        smbhb = self.results["smbhb"]
        self.assertRelative(
            smbhb["benchmark_A_1yr"], 2.10787080829e-15, 2.0e-10
        )
        self.assertRelative(
            smbhb["conditional_ceiling_A_1yr"], 2.28782399018e-15, 2.0e-10
        )
        self.assertRelative(
            smbhb["repeated_merger_ceiling_A_1yr"],
            3.76095884945e-15,
            2.0e-10,
        )
        self.assertRelative(
            smbhb["szq_hierarchy_hc2_boost"],
            2.70241438392,
            2.0e-10,
        )
        self.assertRelative(
            smbhb["conditional_ceiling_mass_ratio"],
            1.0,
            2.0e-8,
        )
        self.assertRelative(
            smbhb["fmerge_cnm_central_fixed_shape"], 0.99254592509, 2.0e-10
        )
        self.assertIsNone(smbhb["primary_minimum_mass_msun"])
        self.assertIn("full analytic", smbhb["primary_mass_selection"])
        self.assertIsNone(smbhb["minimum_total_mass_benchmark_msun"])
        self.assertIsNone(
            smbhb["minimum_total_mass_conditional_ceiling_msun"]
        )
        self.assertRelative(
            smbhb["retained_mass_density_fraction_benchmark"],
            1.0,
            2.0e-10,
        )
        self.assertRelative(
            smbhb["participant_mass_density_benchmark_msun_mpc3"],
            1.8e6,
            2.0e-10,
        )
        self.assertRelative(
            smbhb["retained_i53_eta_factor_benchmark"]
            / smbhb["eta_moment_unrestricted"],
            1.0,
            2.0e-10,
        )

    def test_smbhb_path_has_no_lumped_efficiency_or_cutoff(self) -> None:
        parameters = inspect.signature(vc.smbhb_population_results).parameters
        self.assertNotIn("epsilon", parameters)
        self.assertNotIn("spin", parameters)
        self.assertNotIn("f_max_hz", parameters)
        self.assertNotIn("primary_minimum_mass_msun", parameters)

    def test_szq24_population_moment_audit(self) -> None:
        audit = self.results["szq24_population_moment_audit"]
        expected = {
            "mass_scale_msun": 5.85189206526e7,
            "rho_bh_msun_mpc3": 4.54460346830e5,
            "i53_msun53_mpc3": 5.24703469666e11,
            "eta_log_uniform": 0.177665924415,
            "redshift_moment": 0.894118527245,
            "fiducial_log_uniform_one_pass_A_1yr": 9.94781585808e-16,
            "equal_mass_fiducial_redshift_one_pass_A_1yr": 1.18003712802e-15,
            "equal_mass_z0_one_pass_A_1yr": 1.24795269964e-15,
            "fiducial_plus_hierarchy_A_1yr": 1.63532361950e-15,
            "equal_mass_fiducial_redshift_infinite_A_1yr": 1.93986560957e-15,
            "equal_mass_z0_infinite_A_1yr": 2.05151216594e-15,
            "literal_eq21_A_1yr": 1.11112266163e-15,
        }
        for key, value in expected.items():
            self.assertRelative(audit[key], value, 2.0e-10)

        self.assertRelative(
            (
                audit["equal_mass_fiducial_redshift_one_pass_A_1yr"]
                / audit["fiducial_log_uniform_one_pass_A_1yr"]
            )
            ** 2,
            0.25 / audit["eta_log_uniform"],
            2.0e-14,
        )
        self.assertRelative(
            (
                audit["equal_mass_z0_one_pass_A_1yr"]
                / audit["equal_mass_fiducial_redshift_one_pass_A_1yr"]
            )
            ** 2,
            1.0 / audit["redshift_moment"],
            2.0e-14,
        )
        self.assertRelative(
            (
                audit["equal_mass_z0_infinite_A_1yr"]
                / audit["equal_mass_z0_one_pass_A_1yr"]
            )
            ** 2,
            vc.SZQ_HIERARCHY_HC2_BOOST,
            2.0e-14,
        )
        self.assertRelative(
            audit["literal_eq21_over_direct_strain_ratio"],
            1.11695137654,
            2.0e-10,
        )
        self.assertRelative(
            audit["literal_eq21_over_direct_hc2_ratio"],
            1.24758037756,
            2.0e-10,
        )
        self.assertIn("unresolved", audit["normalization_status"])
        self.assertIn("no primary-mass", audit["scope"])

    def test_agn_pipeline_factorization_and_rectangle(self) -> None:
        audit = self.results["agn_pipeline_audit"]
        self.assertRelative(
            audit["benchmark_rho_bbh_proc_msun_mpc3"], 18_240.0, 1.0e-14
        )
        self.assertRelative(
            audit["benchmark_source_fraction"], 0.03, 1.0e-14
        )
        self.assertRelative(
            audit["benchmark_rho_imr_msun_mpc3"], 547.2, 1.0e-14
        )
        self.assertRelative(
            audit["maximum_rho_imr_msun_mpc3"], 96_330.0, 1.0e-14
        )
        self.assertEqual(audit["corner_count"], 32)
        self.assertRelative(
            audit["maximum_over_central_rho_bh"], 0.05351666666666667, 1.0e-14
        )
        self.assertTrue(audit["broad_census_guardrail_satisfied"])

    def test_constant_rate_history_and_redshift_kernel(self) -> None:
        history = vc.RateHistory(
            59.3,
            2.8,
            duration_gyr=10.0,
            rate_anchor_redshift=0.0,
        )
        self.assertRelative(history.rho_src_msun_mpc3, 1_660.4, 2.0e-14)
        self.assertRelative(history.integration_redshift_max, 1.66183, 2.0e-5)
        channel = self.results["channels"]["bns_benchmark"]
        self.assertRelative(channel["mean_energy_dilution"], 0.68346, 2.0e-5)

    def test_rate_only_benchmark_intervals(self) -> None:
        audit = self.results["rate_only_benchmark_intervals"]
        expected = {
            "agn_imri_benchmark": {
                "available_mass": (15_000.0, 18_240.0, 22_140.0),
                "rho": (450.0, 547.2, 664.2),
                "fraction": (0.03, 0.03, 0.03),
                "hc": (1.22061420315e-21, 1.34600045469e-21, 1.48293323268e-21),
                "omega": (
                    1.84922234000796e-11,
                    2.24865436544968e-11,
                    2.72945217385175e-11,
                ),
                "energy": (1.21974184992e-10, 1.48320608950e-10, 1.80033897048e-10),
            },
            "bns_benchmark": {
                "rho": (431.2, 1660.4, 4331.6),
                "fraction": (6.84040446e-7, 2.63399990e-6, 6.87149721e-6),
                "hc": (2.17977640826e-24, 4.27739083117e-24, 6.90870651949e-24),
                "omega": (
                    6.55260033217032e-14,
                    2.52317662141364e-13,
                    6.58238487913474e-13,
                ),
                "energy": (4.67513498385e-11, 1.80023054898e-10, 4.69638559742e-10),
            },
            "bbh_benchmark": {
                "rho": (5421.65222961, 6592.72911121, 8002.35869091),
                "fraction": (
                    8.60071755134238e-6,
                    1.04584725424323e-5,
                    1.26946591057813e-5,
                ),
                "hc": (3.65592904701e-25, 4.03148033746e-25, 4.44161526727e-25),
                "omega": (1.15203405690e-10, 1.40087341319e-10, 1.70040226799e-10),
                "energy": (1.46955913781e-9, 1.78698391158e-9, 2.16906928741e-9),
            },
        }
        for key, values in expected.items():
            self.assertEqual(
                audit["channels"][key]["coverage"],
                "90-percent marginal credible interval",
            )
            self.assertTrue(
                audit["channels"][key]["other_inputs_held_fixed"]
            )
            endpoints = audit["channels"][key]["endpoints"]
            for index, endpoint in enumerate(("low", "median", "high")):
                if "available_mass" in values:
                    self.assertRelative(
                        endpoints[endpoint]["available_mass_msun_mpc3"],
                        values["available_mass"][index],
                        2.0e-11,
                    )
                self.assertRelative(
                    endpoints[endpoint]["rho_src_msun_mpc3"],
                    values["rho"][index],
                    2.0e-11,
                )
                self.assertRelative(
                    endpoints[endpoint]["source_fraction"],
                    values["fraction"][index],
                    2.0e-9,
                )
                self.assertRelative(
                    endpoints[endpoint]["hc_at_reference"],
                    values["hc"][index],
                    2.0e-11,
                )
                self.assertRelative(
                    endpoints[endpoint]["omega_at_reference"],
                    values["omega"][index],
                    2.0e-10,
                )
                self.assertRelative(
                    endpoints[endpoint]["present_integral_omega_dlnf"],
                    values["energy"][index],
                    2.0e-11,
                )
        self.assertIn("held fixed", audit["scope"])

    def test_evolving_rate_one_at_a_time_ranges(self) -> None:
        audit = self.results["history_sensitivity_audit"][
            "bbh_rate_evolution"
        ]
        rate_only = audit["rate_only_at_median_kappa"]
        kappa_only = audit["kappa_only_at_median_rate"]
        expected = {
            "rate": {
                "low": (12352.4638, 1.95955121e-5, 0.566231834012, 5.31592139e-25, 2.43571741149e-10, 2.77390069e-9),
                "median": (15020.5959, 2.38281427e-5, 0.566231834012, 5.86199356e-25, 2.96183237237e-10, 3.37306324e-9),
                "high": (18232.2365, 2.89229759e-5, 0.566231834012, 6.45835225e-25, 3.59511889936e-10, 4.09427742e-9),
            },
            "kappa": {
                "low": (11371.8344, 1.80398763e-5, 0.596924194754, 5.15320466e-25, 2.28888821379e-10, 2.69210966e-9),
                "median": (15020.5959, 2.38281427e-5, 0.566231834012, 5.86199356e-25, 2.96183237237e-10, 3.37306324e-9),
                "high": (20528.8013, 3.25661651e-5, 0.538835448276, 6.78820964e-25, 3.97173527160e-10, 4.38695105e-9),
            },
        }
        for rows, label in ((rate_only, "rate"), (kappa_only, "kappa")):
            for endpoint in ("low", "median", "high"):
                rho, fraction, dilution, hc, omega, energy = expected[label][endpoint]
                self.assertRelative(rows[endpoint]["rho_src_msun_mpc3"], rho, 2.0e-8)
                self.assertRelative(rows[endpoint]["source_fraction"], fraction, 2.0e-8)
                self.assertRelative(rows[endpoint]["mean_energy_dilution"], dilution, 2.0e-8)
                self.assertRelative(rows[endpoint]["hc_at_reference"], hc, 2.0e-8)
                self.assertRelative(rows[endpoint]["omega_at_reference"], omega, 2.0e-8)
                self.assertRelative(rows[endpoint]["present_integral_omega_dlnf"], energy, 2.0e-8)
        self.assertIn("not combined", audit["status"])

    def test_bbh_stochastic_limit_inversion(self) -> None:
        audit = self.results["history_sensitivity_audit"][
            "bbh_stochastic_limit_inversion"
        ]
        log_prior = audit["priors"]["log_uniform_amplitude"]
        self.assertRelative(
            log_prior["source_fraction_upper_limit"],
            1.49313598844e-4,
            2.0e-10,
        )
        self.assertRelative(
            log_prior["rho_src_msun_mpc3_upper_limit"],
            9.41231241755e4,
            2.0e-10,
        )
        self.assertRelative(
            log_prior["rate_gpc3_yr_upper_limit"],
            4.34014946871e2,
            2.0e-10,
        )
        uniform_prior = audit["priors"]["uniform_amplitude"]
        self.assertRelative(
            uniform_prior["omega_25_upper_limit"],
            6.3e-9,
            1.0e-14,
        )
        self.assertIn("three equivalent units", audit["statistical_status"])

    def test_peters_prefactor_and_kerr_isco(self) -> None:
        audit = self.results["core_physics_audit"]
        self.assertRelative(
            audit["peters_bns_equal_mass_to_1500_hz_efficiency"],
            0.02020619596,
            2.0e-10,
        )
        self.assertRelative(
            audit["schwarzschild_isco_radius_gm_c2"], 6.0, 1.0e-14
        )
        self.assertRelative(
            audit["kerr_chi0p7_isco_radius_gm_c2"],
            3.39312847018,
            2.0e-12,
        )
        self.assertRelative(
            audit["kerr_chi0p7_mass_for_400_nhz_msun"],
            2.3244718672e10,
            2.0e-10,
        )
        self.assertRelative(
            vc.kerr_isco_frequency(
                audit["kerr_chi0p7_mass_for_400_nhz_msun"], 0.7
            ),
            400.0e-9,
            2.0e-14,
        )
        peters_spectrum = vc.peters_energy_spectrum(
            np.asarray([100.0, 800.0]), 1.2
        )
        self.assertRelative(
            peters_spectrum[1] / peters_spectrum[0],
            0.5,
            2.0e-14,
        )
        support = audit["smbhb_schwarzschild_isco_support_at_1yr"]
        self.assertIsNone(support["primary_minimum_mass_msun"])
        self.assertRelative(
            support["fractional_hc_change"],
            -4.51796405e-5,
            2.0e-5,
        )

    def test_observed_cutoff_is_redshifted(self) -> None:
        channel = vc.channel_inputs()["bns_benchmark"]
        values = channel.shape_at_redshift(
            np.asarray([749.0, 751.0]), redshift=1.0
        )
        self.assertGreater(values[0], 0.0)
        self.assertEqual(values[1], 0.0)

    def test_nsc_fiducial_is_an_estimate_not_hard_bound(self) -> None:
        audit = self.results["nsc_density_audit"]
        self.assertRelative(
            audit["eta_nsc_unit_occupation_median_relation"],
            8.7934e-4,
            2.0e-5,
        )
        self.assertRelative(
            audit["eta_nsc_unit_occupation_scatter_mean"],
            2.2836e-3,
            2.0e-5,
        )
        self.assertIn("not a universal hard upper bound", audit["interpretation"])

    def test_emri_rate_proxy_motivates_one_e_minus_four(self) -> None:
        audit = self.results["emri_rate_proxy_audit"]
        m1 = audit["models"]["M1"]
        self.assertRelative(
            m1["f_merge_uniform_volume_proxy"], 1.2857e-4, 2.0e-3
        )
        self.assertEqual(audit["adopted_f_merge_benchmark"], 1.0e-4)
        self.assertEqual(audit["closed_pool_f_merge"], 0.03)
        self.assertIn("benchmark only", audit["status"])
        ceiling = self.results["channels"]["emri_closed_pool_ceiling"]
        self.assertRelative(
            ceiling["hc_at_reference"], 6.02129784627e-21, 2.0e-10
        )

    def test_popiii_history_and_reference_strain(self) -> None:
        audit = self.results["history_sensitivity_audit"]["popiii_inventory"]
        adopted = audit["liu_bromm_history"]
        self.assertRelative(
            adopted["rho_iii_msun_mpc3"], 7.33221e4, 2.0e-5
        )
        self.assertRelative(
            adopted["rho_src_msun_mpc3"], 7.33221e2, 2.0e-5
        )
        self.assertRelative(adopted["hc_at_reference"], 6.8032e-24, 3.0e-5)
        ceiling = audit["planck_calibrated_ceiling"]
        self.assertRelative(
            ceiling["rho_iii_msun_mpc3"], 2.0e5, 2.0e-11
        )
        self.assertRelative(ceiling["rho_src_msun_mpc3"], 2.0e3, 2.0e-11)
        self.assertRelative(
            ceiling["hc_at_reference"], 1.12359977e-23, 2.0e-7
        )
        self.assertRelative(
            ceiling["present_integral_omega_dlnf"],
            9.07747145e-11,
            2.0e-7,
        )
        self.assertGreater(
            audit["delta_z10_rho_iii_1e5"]["hc_at_reference"],
            adopted["hc_at_reference"],
        )

    def test_efficiency_checks(self) -> None:
        checks = self.results["efficiency_checks"]
        self.assertRelative(checks["bns_inspiral_to_1500_hz"], 0.02021, 3.0e-4)
        self.assertRelative(
            checks["bbh_population_weighting"][
                "equal_mass_nonspinning_efficiency"
            ],
            0.04839,
            2.0e-4,
        )
        self.assertRelative(
            checks["jimenez_forteza_nonspinning_equal_mass"],
            checks["bbh_population_weighting"][
                "equal_mass_nonspinning_efficiency"
            ],
            5.0e-4,
        )
        self.assertRelative(checks["bbh_q1_chi0p7_NR"], 0.08051, 2.0e-4)
        self.assertRelative(
            checks["capture_schwarzschild_test_particle"],
            1.0 - math.sqrt(8.0 / 9.0),
            1.0e-14,
        )

    def test_stellar_to_smbh_density_ratio_uses_locked_definition(self) -> None:
        constants = self.results["constants"]
        self.assertRelative(
            constants["rho_star_over_rho_bh_lm24"],
            350.2067,
            2.0e-7,
        )

    def test_recommended_channel_values(self) -> None:
        expected = {
            "emri_benchmark": (63.0372, 3.4764e-22),
            "agn_imri_benchmark": (547.2, 1.3460e-21),
            "agn_imri_ceiling": (96_330.0, 1.7300e-20),
            "bns_benchmark": (1_660.4, 4.2774e-24),
            "bns_ceiling": (4_331.6, 6.9087e-24),
            "popiii_benchmark": (733.221, 6.8032e-24),
            "popiii_ceiling": (2_000.0, 1.1236e-23),
            "bbh_benchmark": (6_592.73, 4.0315e-25),
            "bbh_ceiling": (10_713.2, 5.1431e-25),
        }
        for key, (rho_src, strain) in expected.items():
            row = self.results["channels"][key]
            self.assertRelative(row["rho_src_msun_mpc3"], rho_src, 2.0e-5)
            self.assertRelative(row["hc_at_reference"], strain, 5.0e-5)

    def test_scope_and_one_pass_flags(self) -> None:
        self.assertIn(
            "inclusive total",
            self.results["interpretation"]["population_scope"],
        )
        self.assertTrue(
            self.results["one_pass_audit"][
                "all_adopted_input_fractions_at_or_below_one"
            ]
        )
        self.assertFalse(
            self.results["one_pass_audit"][
                "cnm_high_inside_one_pass_family"
            ]
        )
        self.assertTrue(
            self.results["one_pass_audit"][
                "cnm_central_inside_one_pass_family"
            ]
        )
        self.assertIn(
            "inside the one-pass family",
            self.results["one_pass_audit"]["cnm_central_status"],
        )
        self.assertGreater(
            self.results["one_pass_audit"][
                "cnm_high_shape_conditional_f_merge"
            ],
            1.0,
        )
        self.assertNotIn("aggregate_budget", self.results["interpretation"])
        diagnostics = self.results["spectral_diagnostics"]
        self.assertRelative(
            diagnostics["popiii_peak_frequency_hz"], 15.44, 2.0e-3
        )
        self.assertRelative(
            diagnostics["popiii_peak_omega"], 1.649e-11, 5.0e-4
        )
        self.assertIn("no post-subtraction", diagnostics["scope"])

    def test_integrated_energy_normalizations(self) -> None:
        residuals = vc.verify_normalizations()
        self.assertLess(max(abs(value) for value in residuals.values()), 2.5e-3)
        self.assertFalse(any("smbhb" in key for key in residuals))
        self.assertFalse(any("composite" in key for key in residuals))

    def test_step3b_smbhb_integral_and_conditional_benchmark_sum(self) -> None:
        step3b = self.results["step3b_integrated_energy"]
        smbhb = step3b["channels"]["smbhb_benchmark"]
        self.assertRelative(
            smbhb["source_frame_efficiency"],
            0.0198748981922,
            2.0e-11,
        )
        self.assertRelative(
            smbhb["mean_energy_dilution"],
            0.710220271015,
            2.0e-11,
        )
        self.assertRelative(
            smbhb["present_integral_omega_dlnf"],
            2.01531768062e-7,
            2.0e-11,
        )
        self.assertRelative(
            step3b[
                "smbhb_conditional_reference_spectrum_integral_omega_dlnf"
            ],
            2.97443375668e-7,
            2.0e-11,
        )
        self.assertRelative(
            step3b["conditional_benchmark_sum_omega_dlnf"],
            2.03705374606e-7,
            2.0e-11,
        )
        envelope = step3b["benchmark_input_envelope"]
        self.assertRelative(
            envelope["lower_sum_omega_dlnf"],
            1.47247285019e-7,
            2.0e-11,
        )
        self.assertRelative(
            envelope["upper_sum_omega_dlnf"],
            2.93978463469e-7,
            2.0e-11,
        )
        self.assertIn("Not a joint", envelope["statistical_status"])
        rate_envelope = step3b["benchmark_rate_only_envelope"]
        self.assertRelative(
            rate_envelope["lower_sum_omega_dlnf"],
            2.03228331703e-7,
            2.0e-11,
        )
        self.assertRelative(
            rate_envelope["upper_sum_omega_dlnf"],
            2.04408788775e-7,
            2.0e-11,
        )
        rate_cases = step3b["benchmark_rate_only_sensitivities"]
        self.assertIn("not a joint", rate_cases["statistical_status"])
        shared = rate_cases["cases"]["shared_default_bbh_rate"]
        self.assertIn("90-percent", shared["coverage"])
        self.assertRelative(
            shared["lower_sum_omega_dlnf"],
            2.03361603408e-7,
            2.0e-11,
        )
        self.assertRelative(
            shared["upper_sum_omega_dlnf"],
            2.04119173270e-7,
            2.0e-11,
        )
        bns = rate_cases["cases"]["bns_fullpop_rate"]
        self.assertIn("90-percent", bns["coverage"])
        self.assertRelative(
            bns["lower_sum_omega_dlnf"],
            2.03572102901e-7,
            2.0e-11,
        )
        self.assertRelative(
            bns["upper_sum_omega_dlnf"],
            2.03994990111e-7,
            2.0e-11,
        )
        self.assertEqual(
            step3b["included_components"],
            ("smbhb_benchmark",) + vc.STEP3B_BENCHMARK_CHANNEL_KEYS,
        )
        self.assertGreater(
            smbhb["fraction_of_conditional_benchmark_sum"],
            0.989,
        )
        self.assertNotIn("agn_imri_ceiling", step3b["included_components"])
        self.assertNotIn("bns_ceiling", step3b["included_components"])
        self.assertNotIn("popiii_ceiling", step3b["included_components"])
        self.assertNotIn("bbh_ceiling", step3b["included_components"])
        self.assertIn(
            "broadband_smbhb_composite_cumulative",
            step3b["excluded"],
        )
        self.assertIn("not a universal", step3b["interpretation"])
        self.assertNotIn("spectral_diagnostics", step3b)
        self.assertEqual(
            step3b["smbhb_emitted_spectrum"][
                "broadband_composite_cumulative_status"
            ],
            "retired",
        )
        self.assertIn(
            "not integrated",
            step3b["smbhb_emitted_spectrum"]["scope"],
        )

    def test_step3b_one_generation_conditional_ceiling_sum(self) -> None:
        step3b = self.results["step3b_integrated_energy"]
        ceiling = step3b["conditional_ceiling"]
        expected_energies = {
            "smbhb_conditional_ceiling": 2.974433756684786e-7,
            "emri_closed_pool_ceiling": 7.500000000000001e-9,
            "agn_imri_ceiling": 2.245294420725684e-8,
            "bns_ceiling": 4.696385597416180e-10,
            "popiii_ceiling": 9.077471454910992e-11,
            "bbh_ceiling_realizable_chi0p7": 4.651965867813408e-9,
        }
        self.assertEqual(
            ceiling["included_components"],
            tuple(expected_energies),
        )
        self.assertRelative(
            ceiling["sum_omega_dlnf"],
            3.32608699018e-7,
            2.0e-12,
        )
        for key, expected in expected_energies.items():
            row = ceiling["channels"][key]
            self.assertRelative(
                row["present_integral_omega_dlnf"],
                expected,
                2.0e-11,
            )
            self.assertRelative(
                row["fraction_of_conditional_ceiling_sum"],
                expected / 3.326086990178396e-7,
                2.0e-11,
            )

        smbhb = ceiling["channels"]["smbhb_conditional_ceiling"]
        self.assertEqual(smbhb["mass_ratio"], 1.0)
        self.assertEqual(smbhb["mergers_per_present_day_remnant"], 1.0)
        self.assertFalse(smbhb["repeated_merger_factor_included"])
        self.assertIn("repeated-merger", ceiling["scope"])

        bbh = ceiling["channels"]["bbh_ceiling_realizable_chi0p7"]
        self.assertEqual(bbh["aligned_component_spin"], 0.7)
        self.assertIn("not the pointwise", bbh["spectrum_scope"])

        uncertainty = ceiling["uncertainty_lm24_only"]
        self.assertRelative(
            uncertainty["plus_omega_dlnf"],
            1.321970558526572e-7,
            2.0e-11,
        )
        self.assertRelative(
            uncertainty["minus_omega_dlnf"],
            8.262315990791073e-8,
            2.0e-11,
        )
        self.assertEqual(
            uncertainty["reported_in_units_of_1e_minus_7"],
            {"central": 3.33, "plus": 1.32, "minus": 0.83},
        )

    def test_step3b_smbhb_support_and_context_are_scoped(self) -> None:
        step3b = self.results["step3b_integrated_energy"]
        omega_at_reference = vc.smbhb_population_omega(
            np.asarray([vc.F_YEAR_HZ])
        )[0]
        physical_hc = float(
            vc.hc_from_omega(vc.F_YEAR_HZ, omega_at_reference)
        )
        expected_hc = self.results["smbhb"]["benchmark_A_1yr"]
        self.assertRelative(physical_hc, expected_hc, 5.0e-5)

        sensitivities = step3b["smbhb_support_sensitivities"]
        self.assertGreater(
            sensitivities["kerr_chi0p7_inspiral"]["ratio_to_primary"],
            1.6,
        )
        self.assertIn(
            "Sensitivity only",
            sensitivities["kerr_chi0p7_inspiral"]["scope"],
        )
        self.assertNotIn("primary_mass_floor", sensitivities)

        context = step3b["cosmological_context"]
        wright = context["wright_giblin_hazboun"]
        self.assertRelative(wright["h2_integral_limit"], 1.5904e-6, 1.0e-14)
        self.assertRelative(
            wright["omega_integral_limit"],
            3.50095536634e-6,
            2.0e-12,
        )
        self.assertIn("not a bound", wright["scope"])
        giblin_thrane = context["giblin_thrane"]
        self.assertRelative(
            giblin_thrane["peak_omega_levels"]["realistic"],
            1.49e-12,
            1.0e-14,
        )
        self.assertIn("not an astrophysical", giblin_thrane["scope"])

    def test_spectrum_keys_distinguish_bbh_endpoint_and_envelope(self) -> None:
        frequencies = np.asarray(
            [5.0e-10, 1.0e-9, vc.F_YEAR_HZ, 1.0e-7, 2.0e-7, 10.0, 25.0, 100.0]
        )
        _, spectra = vc.build_spectra(frequencies)
        self.assertIn("popiii_ceiling", spectra)
        self.assertIn("bbh_ceiling_realizable_chi0p7", spectra)
        self.assertIn("bbh_ceiling_pointwise_envelope", spectra)
        np.testing.assert_array_equal(
            spectra["bbh_ceiling"],
            spectra["bbh_ceiling_pointwise_envelope"],
        )
        self.assertTrue(
            np.all(
                spectra["bbh_ceiling_pointwise_envelope"]
                >= spectra["bbh_ceiling_realizable_chi0p7"]
            )
        )
        for key in ("smbhb_benchmark", "smbhb_conditional_ceiling"):
            self.assertEqual(spectra[key][0], 0.0)
            self.assertTrue(np.all(spectra[key][1:4] > 0.0))
            self.assertTrue(np.all(spectra[key][4:] == 0.0))

        retired_fragments = ("integrable", "composite", "cumulative")
        self.assertFalse(
            any(
                fragment in key
                for key in spectra
                for fragment in retired_fragments
            )
        )

        display_frequency = np.geomspace(
            vc.SMBHB_PTA_DISPLAY_MIN_HZ,
            vc.SMBHB_PTA_DISPLAY_MAX_HZ,
            2_000,
        )
        _, display_spectra = vc.build_spectra(display_frequency)
        display_integral = vc.trapz_log(
            display_spectra["smbhb_benchmark"],
            display_frequency,
        )
        full_scalar = self.results["step3b_integrated_energy"]["channels"][
            "smbhb_benchmark"
        ]["present_integral_omega_dlnf"]
        self.assertGreater(display_integral, 0.0)
        self.assertLess(display_integral, 0.11 * full_scalar)


if __name__ == "__main__":
    unittest.main()
