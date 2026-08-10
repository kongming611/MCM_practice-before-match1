from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path

from src._internal.q3_stress_scenarios import (
    NBS_GROWTH_YOY_PCT,
    NBS_SOURCE_URL,
    apply_risk_stress,
    calculate_industry_stress,
    load_stress_settings,
    run_q3_stress_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]


class Q3StressScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.settings = load_stress_settings()

    def test_accommodation_catering_formula_and_clipping(self) -> None:
        stress = calculate_industry_stress(-35.3, **{key: self.settings[key] for key in ("c_pp", "rho", "eps", "lambdas")})
        self.assertAlmostEqual(stress["S_h"], 0.8825, places=10)
        self.assertAlmostEqual(stress["exposures"]["light"], 0.44125, places=10)
        self.assertAlmostEqual(stress["multipliers"]["light"], 1.1103125, places=10)
        self.assertAlmostEqual(stress["multipliers"]["medium"], 1.220625, places=10)
        self.assertEqual(stress["exposures"]["severe"], 1.0)
        self.assertAlmostEqual(stress["multipliers"]["severe"], 1.25, places=10)

    def test_positive_growth_has_no_risk_reward(self) -> None:
        stress = calculate_industry_stress(13.2, **{key: self.settings[key] for key in ("c_pp", "rho", "eps", "lambdas")})
        self.assertEqual(stress["S_h"], 0.0)
        self.assertTrue(all(value == 1.0 for value in stress["multipliers"].values()))

    def test_identity_is_exact_and_upper_bound_is_clipped(self) -> None:
        stress = calculate_industry_stress(-35.3, **{key: self.settings[key] for key in ("c_pp", "rho", "eps", "lambdas")})
        risks = apply_risk_stress(0.9142122803, stress, eps=self.settings["eps"])
        self.assertEqual(risks["identity"], 0.9142122803)
        self.assertLessEqual(risks["severe"], 1.0 - self.settings["eps"])
        self.assertEqual(risks["severe"], 1.0 - self.settings["eps"])

    def test_full_stage_passes_unknown_max_policy_and_is_deterministic(self) -> None:
        first = run_q3_stress_scenarios()
        risk_output = ROOT / "data/processed/q3_scenario_risk_scores.csv"
        first_bytes = risk_output.read_bytes()
        first_hash = hashlib.sha256(first_bytes).hexdigest()
        second = run_q3_stress_scenarios()
        second_bytes = risk_output.read_bytes()
        self.assertEqual(first["status"], "PASS")
        self.assertEqual(second["status"], "PASS")
        self.assertEqual(first["counts"]["output_rows"], 302)
        self.assertEqual(first["counts"]["unknown_rows"], 145)
        self.assertEqual(first["max_known_S"], 0.8825)
        self.assertEqual(first_hash, hashlib.sha256(second_bytes).hexdigest())
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first["source"]["url"], NBS_SOURCE_URL)
        self.assertEqual(first["source"]["period"], "2020Q1")
        self.assertEqual(first["assumptions"]["unknown_policy"], "max_known_S")
        params = ROOT / "outputs/q3/tables/q3_industry_shock_params.csv"
        self.assertTrue(params.is_file())
        self.assertEqual(json.loads((ROOT / "outputs/q3/reports/q3_stress_validation.json").read_text(encoding="utf-8"))["status"], "PASS")

    def test_cli_stress_is_available(self) -> None:
        command = [sys.executable, str(ROOT / "src/q3.py"), "--stage", "stress"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("stage=stress", completed.stdout)

    def test_source_growth_contract_uses_manufacturing_value(self) -> None:
        self.assertEqual(NBS_GROWTH_YOY_PCT["manufacturing_industry"], -10.2)
        self.assertNotEqual(NBS_GROWTH_YOY_PCT["manufacturing_industry"], -8.5)


if __name__ == "__main__":
    unittest.main()

