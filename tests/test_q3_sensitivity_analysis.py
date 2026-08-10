from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src._internal.q3_sensitivity_analysis import (
    DURATION_ASSUMPTION_TEXT,
    DURATION_LIMITATION_REASON,
    _duration_noncalibration_gate,
    run_q3_sensitivity_analysis,
)
from src._internal.q3_figures import run_q3_figures


ROOT = Path(__file__).resolve().parents[1]


class Q3SensitivityAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = run_q3_sensitivity_analysis()
        cls.summary = pd.read_csv(ROOT / "outputs/q3/tables/q3_sensitivity_summary.csv", encoding="utf-8-sig")
        cls.stability = pd.read_csv(ROOT / "outputs/q3/tables/q3_enterprise_strategy_stability.csv", encoding="utf-8-sig")
        cls.long = pd.read_csv(ROOT / "outputs/q3/tables/q3_sensitivity_enterprise_strategies.csv", encoding="utf-8-sig")
        cls.industry = pd.read_csv(ROOT / "outputs/q3/tables/q3_sensitivity_industry_shocks.csv", encoding="utf-8-sig")

    def test_contract_has_unique_scenarios_and_all_solver_gates(self) -> None:
        self.assertEqual(self.contract["status"], "PASS")
        self.assertEqual(len(self.summary), 13)
        self.assertEqual(self.summary["scenario_id"].nunique(), 13)
        self.assertTrue((self.summary["solve_status"] == "optimal").all())
        self.assertTrue(self.summary["is_optimal"].all())
        self.assertTrue(self.summary["solver_called"].all())
        self.assertTrue((~self.summary["fallback_used"]).all())
        self.assertTrue((self.summary["validation_status"] == "PASS").all())
        self.assertTrue(np.allclose(self.summary["nominal_offered_amount_10k"], 10000.0, atol=1e-6, rtol=0))
        self.assertTrue(self.contract["checks"]["duration_noncalibration_disclosed"])
        self.assertFalse(self.contract["limitations"]["duration_calibration"]["available"])

    def test_single_factor_values_and_actual_p50(self) -> None:
        centre = self.summary.loc[self.summary["scenario_id"] == "formal_robust"].iloc[0]
        self.assertAlmostEqual(float(centre["c_pp"]), 40.0)
        self.assertAlmostEqual(float(centre["rho"]), 0.25)
        p50 = self.summary.loc[self.summary["scenario_id"] == "sensitivity_unknown_policy_p50_known_S", "unknown_S_value"].iloc[0]
        known_s = self.industry.loc[(self.industry["scenario_id"] == "sensitivity_unknown_policy_p50_known_S") & (self.industry["industry_code"] != "unknown"), "S_h"].astype(float)
        self.assertAlmostEqual(float(p50), float(np.percentile(known_s.to_numpy(), 50)), places=12)
        self.assertEqual(self.summary.loc[self.summary["scenario_id"] == "sensitivity_unknown_policy_p50_known_S", "unknown_S_source"].iloc[0], "actual_P50_of_11_known_S")
        for axis in ["rho", "c_pp", "lambda_severe", "unknown_policy", "lgd", "funding_cost_rate"]:
            rows = self.summary.loc[self.summary["sensitivity_axis"] == axis]
            self.assertTrue((rows["changed_parameter_count"] == 1).all())

    def test_stability_and_long_table_are_recomputable(self) -> None:
        self.assertEqual(len(self.long), 13 * 302)
        self.assertEqual(len(self.stability), 302)
        first = self.long.loc[(self.long["enterprise_id"] == self.long["enterprise_id"].iloc[0]) & (self.long["scenario_kind"] != "exploratory_combined_adverse")]
        expected_frequency = (first["offered_loan_amount_10k"] > 1e-6).mean()
        got_frequency = float(self.stability.loc[self.stability["enterprise_id"] == first["enterprise_id"].iloc[0], "loan_frequency"].iloc[0])
        self.assertAlmostEqual(float(expected_frequency), got_frequency, places=12)
        self.assertTrue((self.stability["scenario_count"] == 12).all())
        self.assertTrue((self.stability["excluded_scenario"] == "exploratory_combined_adverse").all())
        self.assertEqual(set(self.stability["stability_label"]), {"robust_selected_core", "selection_sensitive", "consistently_not_selected"})
        self.assertTrue(np.isfinite(self.summary.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all())
        self.assertTrue(np.isfinite(self.long.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all())

    def test_deterministic_repeat_and_q2_hashes(self) -> None:
        self.assertTrue(self.contract["checks"]["deterministic_repeat_all"])
        self.assertTrue(self.contract["checks"]["centre_reproduces_stage4_robust"])
        self.assertEqual(self.contract["q2_artifact_hashes_before"], self.contract["q2_artifact_hashes_after"])
        path = ROOT / "outputs/q3/tables/q3_sensitivity_summary.csv"
        first_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        second = run_q3_sensitivity_analysis()
        self.assertEqual(second["status"], "PASS")
        self.assertEqual(first_hash, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_duration_disclosure_gate_rejects_tampered_text(self) -> None:
        specs = [{"duration_assumption": DURATION_ASSUMPTION_TEXT}]
        long_table = pd.DataFrame({"duration_assumption": [DURATION_ASSUMPTION_TEXT]})
        self.assertTrue(_duration_noncalibration_gate(specs, long_table, DURATION_LIMITATION_REASON))
        tampered = long_table.copy()
        tampered.loc[0, "duration_assumption"] = "calibrated annual recovery duration"
        self.assertFalse(_duration_noncalibration_gate(specs, tampered, DURATION_LIMITATION_REASON))


class Q3FigureQATests(unittest.TestCase):
    def test_python_figure_contract_and_exports_pass(self) -> None:
        contract = run_q3_figures(sensitivity_contract={"status": "PASS"})
        self.assertEqual(contract["status"], "PASS")
        self.assertEqual(contract["backend"], "python")
        self.assertEqual(contract["figure_count"], 3)
        for row in contract["figures"].values():
            self.assertEqual(row["automated_qa"]["fail_count"], 0)
            for suffix in ("png", "svg", "pdf", "tiff"):
                self.assertTrue(Path(row["outputs"][suffix]).is_file())
