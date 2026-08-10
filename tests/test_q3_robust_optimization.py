from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src._internal.q3_robust_optimization import run_q3_robust_optimization


ROOT = Path(__file__).resolve().parents[1]


class Q3RobustOptimizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = run_q3_robust_optimization()
        cls.long = pd.read_csv(ROOT / "outputs/q3/tables/q3_scenario_credit_strategies.csv", encoding="utf-8-sig")
        cls.summary = pd.read_csv(ROOT / "outputs/q3/tables/q3_portfolio_scenario_summary.csv", encoding="utf-8-sig")
        cls.stress = pd.read_csv(ROOT / "data/processed/q3_scenario_risk_scores.csv", encoding="utf-8-sig")
        cls.q2 = pd.read_csv(ROOT / "data/processed/q2_credit_strategy.csv", encoding="utf-8-sig")

    def _scenario(self, name: str) -> pd.DataFrame:
        return self.long.loc[self.long["q3_scenario_id"] == name].sort_values("enterprise_id").reset_index(drop=True)

    def test_risk_override_join_and_robust_max(self) -> None:
        stress = self.stress.set_index("enterprise_id")
        robust = self._scenario("robust").set_index("enterprise_id")
        self.assertEqual(len(robust), 302)
        expected = stress[["identity_risk", "light_risk", "medium_risk", "severe_risk"]].max(axis=1)
        self.assertTrue(np.allclose(stress["robust_risk"], expected, atol=1e-12, rtol=0))
        self.assertTrue(np.allclose(robust["risk_used"], stress.loc[robust.index, "robust_risk"], atol=1e-9, rtol=0))

    def test_identity_reproduces_q2_key_decisions(self) -> None:
        identity = self._scenario("identity").set_index("enterprise_id")
        q2 = self.q2.set_index("enterprise_id")
        for column in ["risk_used", "selected_rate", "offered_loan_amount_10k", "expected_net_return_10k"]:
            self.assertTrue(
                np.allclose(
                    pd.to_numeric(identity[column], errors="coerce").fillna(-999),
                    pd.to_numeric(q2[column], errors="coerce").fillna(-999),
                    atol=1e-8,
                    rtol=0,
                ),
                column,
            )
        self.assertTrue(identity["decision"].astype(str).equals(q2.loc[identity.index, "decision"].astype(str)))

    def test_solver_budget_and_validation_gates(self) -> None:
        self.assertEqual(self.contract["status"], "PASS")
        self.assertTrue(all(self.summary["solve_status"] == "optimal"))
        self.assertTrue(all(self.summary["solver_called"]))
        self.assertTrue(all(~self.summary["fallback_used"]))
        self.assertTrue(all(self.summary["validation_status"] == "PASS"))
        self.assertTrue(np.allclose(self.summary["nominal_offered_amount_10k"], 10000.0, atol=1e-6, rtol=0))
        self.assertEqual(set(self.summary["scenario_id"]), {"identity", "light", "medium", "severe", "robust"})
        self.assertEqual(self.summary["scenario_id"].nunique(), 5)
        self.assertEqual(set(self.summary["q2_scenario_id"]), {"primary"})
        self.assertTrue(self.contract["checks"]["deterministic_repeat_solve_all"])

    def test_robust_equals_severe_in_current_monotone_data(self) -> None:
        robust = self._scenario("robust").set_index("enterprise_id")
        severe = self._scenario("severe").set_index("enterprise_id")
        self.assertTrue(np.allclose(robust["risk_used"], severe["risk_used"], atol=1e-12, rtol=0))
        self.assertTrue(np.allclose(robust["offered_loan_amount_10k"], severe["offered_loan_amount_10k"], atol=1e-8, rtol=0))
        self.assertTrue(robust["decision"].astype(str).equals(severe["decision"].astype(str)))
        self.assertTrue(self.contract["checks"]["robust_severe_strategy_equal"])

    def test_fixed_q2_under_robust_uses_candidate_economics(self) -> None:
        fixed = pd.read_csv(ROOT / "outputs/q3/tables/q3_fixed_q2_strategy_under_robust_risk.csv", encoding="utf-8-sig")
        details = self.contract["fixed_q2_under_robust"]
        self.assertEqual(len(fixed), 302)
        self.assertTrue(details["candidate_economics_match"])
        self.assertAlmostEqual(float(fixed["offered_loan_amount_10k"].sum()), 10000.0, places=6)
        self.assertAlmostEqual(float(details["expected_net_return_10k"]), float(fixed["expected_net_return_10k"].sum()), places=8)
        self.assertNotEqual(float(details["reoptimization_net_return_delta_10k"]), 0.0)

    def test_industry_share_and_hhi_identities(self) -> None:
        industry = pd.read_csv(ROOT / "outputs/q3/tables/q3_industry_exposure_comparison.csv", encoding="utf-8-sig")
        for scenario, group in industry.groupby("scenario_id"):
            self.assertAlmostEqual(float(group["nominal_share"].sum()), 1.0, places=9, msg=scenario)
            hhi = float(group["industry_hhi"].iloc[0])
            self.assertAlmostEqual(hhi, float((group["nominal_share"] ** 2).sum()), places=9, msg=scenario)
            self.assertIn("unknown", set(group["industry_code"]))

    def test_cli_optimize_is_available(self) -> None:
        command = [sys.executable, str(ROOT / "src/q3.py"), "--stage", "optimize"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("stage=optimize", completed.stdout)
        self.assertEqual(json.loads((ROOT / "outputs/q3/reports/q3_robust_optimization_validation.json").read_text(encoding="utf-8"))["status"], "PASS")

    def test_repeat_run_keeps_key_artifact_hashes(self) -> None:
        paths = [
            ROOT / "data/processed/q3_robust_credit_strategy.csv",
            ROOT / "outputs/q3/tables/q3_scenario_credit_strategies.csv",
        ]
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        repeated = run_q3_robust_optimization()
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        self.assertEqual(before, after)
        self.assertTrue(repeated["checks"]["deterministic_repeat_solve_all"])


if __name__ == "__main__":
    unittest.main()
