from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from src._internal.q3_baseline_preflight import run_q3_preflight


def _sha256_lf(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def _semantic_hash(path: Path) -> str:
    config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write(path: Path, text: str, *, crlf: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = text.encode("utf-8")
    if crlf:
        encoded = encoded.replace(b"\n", b"\r\n")
    path.write_bytes(encoded)


class Q3BaselinePreflightTests(unittest.TestCase):
    def test_success_accepts_crlf_artifacts_and_records_normalized_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            risk_path = root / "data/processed/q2_risk_rating_scores.csv"
            strategy_path = root / "data/processed/q2_credit_strategy.csv"
            summary_path = root / "outputs/q2/tables/q2_portfolio_summary.csv"
            validation_path = root / "outputs/q2/tables/q2_strategy_validation.csv"
            opt_config_path = root / "src/_internal/q2_optimization_config.yaml"
            model_config_path = root / "src/_internal/q2_config.yaml"
            opt_manifest_path = root / "outputs/q2/reports/q2_credit_optimization_manifest.json"
            risk_manifest_path = root / "outputs/q2/reports/q2_risk_rating_run_manifest.json"
            q3_config_path = root / "src/_internal/q3_config.yaml"

            risk_text = (
                "enterprise_id,main_risk_score,rating_prob_A,rating_prob_B,rating_prob_C,rating_prob_D\n"
                "E1,0.1,0.7,0.2,0.09,0.01\n"
                "E2,0.2,0.6,0.3,0.08,0.02\n"
            )
            strategy_text = "enterprise_id,scenario_id\nE1,primary\nE2,primary\n"
            summary_text = (
                "scenario_id,solve_status,is_optimal,validation_status,fallback_used,budget_definition,budget_10k,nominal_offered_amount_10k,budget_difference_10k\n"
                "primary,optimal,True,PASS,False,nominal_equality,10000,10000,0\n"
            )
            bool_columns = [
                "D_rule_not_violated",
                "acceptance_values_legal",
                "at_most_one_rate_per_enterprise",
                "budget_constraint",
                "enterprise_count_2",
                "enterprise_id_unique",
                "no_D_churn_curve",
                "numeric_decomposition",
                "rating_probability_nonnegative",
                "rating_probability_sum_one",
                "risk_values_legal",
                "selected_amounts_10_to_100",
                "selected_rate_is_observed",
                "unselected_amounts_zero",
            ]
            validation_header = (
                "scenario_id,validation_status,budget_10k,budget_difference_10k," + ",".join(bool_columns) + "\n"
            )
            validation_row = "primary,PASS,10000,0," + ",".join("True" for _ in bool_columns) + "\n"
            _write(risk_path, risk_text, crlf=True)
            _write(strategy_path, strategy_text, crlf=True)
            _write(summary_path, summary_text, crlf=True)
            _write(validation_path, validation_header + validation_row, crlf=True)
            _write(opt_config_path, "placeholder: true\n", crlf=True)
            _write(model_config_path, "modeling:\n  probability_check_tolerance: 0.00000001\n", crlf=True)
            _write(
                q3_config_path,
                "paths:\n"
                "  q2_optimization_manifest: outputs/q2/reports/q2_credit_optimization_manifest.json\n"
                "  q2_risk_manifest: outputs/q2/reports/q2_risk_rating_run_manifest.json\n"
                "  q2_model_config: src/_internal/q2_config.yaml\n"
                "  baseline_contract_output: outputs/q3/reports/q3_q2_baseline_contract.json\n"
                "checks:\n  numeric_tolerance: 0.000001\n",
            )

            opt_manifest = {
                "config_path": "src/_internal/q2_optimization_config.yaml",
                "config_sha256": _semantic_hash(opt_config_path),
                "fallback_used": False,
                "inputs": {
                    "q2_risk_rating_scores": {
                        "path": "data/processed/q2_risk_rating_scores.csv",
                        "sha256": _sha256_lf(risk_path.read_bytes()),
                    }
                },
                "outputs": {
                    "data/processed/q2_credit_strategy.csv": _sha256_lf(strategy_path.read_bytes()),
                    "outputs/q2/tables/q2_portfolio_summary.csv": _sha256_lf(summary_path.read_bytes()),
                    "outputs/q2/tables/q2_strategy_validation.csv": _sha256_lf(validation_path.read_bytes()),
                },
                "target_enterprise_count": 2,
                "primary_scenario": {
                    "scenario_id": "primary",
                    "solve_status": "optimal",
                    "is_optimal": True,
                    "validation_status": "PASS",
                    "fallback_used": False,
                    "budget_definition": "nominal_equality",
                    "budget_equality_required": True,
                    "budget_10k": 10000.0,
                    "nominal_offered_amount_10k": 10000.0,
                    "budget_difference_10k": 0.0,
                    "enterprise_count_total": 2,
                },
                "primary_validation": {
                    key: True for key in bool_columns
                } | {"scenario_id": "primary", "budget_10k": 10000.0, "budget_difference_10k": 0.0},
            }
            risk_manifest = {
                "counts": {"target_enterprises": 2},
                "q2_config_sha256": _semantic_hash(model_config_path),
                "input_hashes": {"src/_internal/q2_config.yaml": _sha256_lf(model_config_path.read_bytes())},
                "output_hashes": {
                    "data/processed/q2_risk_rating_scores.csv": _sha256_lf(risk_path.read_bytes())
                },
                "target_accuracy_reported": False,
            }
            _write(opt_manifest_path, json.dumps(opt_manifest, ensure_ascii=False, indent=2) + "\n")
            _write(risk_manifest_path, json.dumps(risk_manifest, ensure_ascii=False, indent=2) + "\n")

            contract = run_q3_preflight(root, q3_config_path)

            self.assertEqual(contract["status"], "PASS")
            self.assertEqual(
                contract["artifact_hashes"]["q2_risk_rating_scores"]["hash_status"],
                "PASS_LINE_ENDING_NORMALIZED",
            )
            self.assertEqual(
                contract["artifact_hashes"]["q2_strategy_validation"]["hash_status"],
                "PASS_LINE_ENDING_NORMALIZED",
            )
            self.assertEqual(
                contract["artifact_hashes"]["q2_optimization_config"]["hash_status"],
                "PASS_SEMANTIC_CONFIG",
            )
            self.assertEqual(
                contract["artifact_hashes"]["q2_model_config"]["semantic_hash_status"],
                "PASS_SEMANTIC_CONFIG",
            )
            output = root / "outputs/q3/reports/q3_q2_baseline_contract.json"
            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["status"], "PASS")


if __name__ == "__main__":
    unittest.main()
