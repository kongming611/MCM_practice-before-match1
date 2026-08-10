"""Q3 stage-one gate for freezing the accepted Q2 outputs.

The gate is intentionally read-only with respect to Q1/Q2.  It verifies the
artifacts referenced by the Q2 optimization manifest, records the evidence in
a contract JSON, and returns a non-zero outcome when any required check fails.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import yaml


CONTRACT_VERSION = "q3_q2_baseline_contract_v1"
DEFAULT_CONFIG_RELATIVE = Path("src/_internal/q3_config.yaml")
DEFAULT_CONTRACT_RELATIVE = Path("outputs/q3/reports/q3_q2_baseline_contract.json")


class PreflightError(RuntimeError):
    """Raised when the Q2 baseline contract cannot be accepted."""

    def __init__(self, message: str, contract: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.contract = contract


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _relative_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _semantic_config_hash(config: dict[str, Any]) -> str:
    """Match data_pipeline.config_hash (semantic YAML, not file bytes)."""

    clean = {key: value for key, value in config.items() if key != "_config_path"}
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
    return _sha256_bytes(payload.encode("utf-8"))


def _hash_record(
    repo_root: Path,
    path: Path,
    expected: str | None,
    *,
    hash_kind: str = "file_bytes",
    expected_semantic: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative_path(repo_root, path),
        "exists": path.is_file(),
        "hash_kind": hash_kind,
        "expected_sha256": expected,
        "actual_sha256": None,
        "line_ending_normalized_sha256": None,
        "semantic_config_sha256": None,
        "expected_semantic_config_sha256": expected_semantic,
        "semantic_hash_status": None,
        "hash_status": "FAIL",
    }
    if not path.is_file():
        return record

    data = path.read_bytes()
    actual = _sha256_bytes(data)
    normalized = _sha256_bytes(data.replace(b"\r\n", b"\n"))
    record["actual_sha256"] = actual
    record["line_ending_normalized_sha256"] = normalized
    if hash_kind == "semantic_config":
        try:
            semantic = _semantic_config_hash(_read_yaml(path))
        except (OSError, ValueError, yaml.YAMLError):
            semantic = None
        record["semantic_config_sha256"] = semantic
        if expected and semantic == expected:
            record["hash_status"] = "PASS_SEMANTIC_CONFIG"
    elif expected and actual == expected:
        record["hash_status"] = "PASS_BYTE_EXACT"
    elif expected and normalized == expected:
        record["hash_status"] = "PASS_LINE_ENDING_NORMALIZED"

    if expected_semantic is not None:
        try:
            semantic = record["semantic_config_sha256"] or _semantic_config_hash(_read_yaml(path))
        except (OSError, ValueError, yaml.YAMLError):
            semantic = None
        record["semantic_config_sha256"] = semantic
        record["semantic_hash_status"] = (
            "PASS_SEMANTIC_CONFIG" if semantic == expected_semantic else "FAIL"
        )
    return record


def _manifest_record(repo_root: Path, path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": _relative_path(repo_root, path),
        "exists": path.is_file(),
        "sha256": None,
        "line_ending_normalized_sha256": None,
        "parse_status": "FAIL",
    }
    if not path.is_file():
        return record
    data = path.read_bytes()
    record["sha256"] = _sha256_bytes(data)
    record["line_ending_normalized_sha256"] = _sha256_bytes(data.replace(b"\r\n", b"\n"))
    try:
        json.loads(data.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return record
    record["parse_status"] = "PASS"
    return record


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "pass"}:
        return True
    if normalized in {"false", "0", "no", "fail"}:
        return False
    return None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numbers_close(actual: Any, expected: Any, tolerance: float) -> bool:
    actual_number = _as_float(actual)
    expected_number = _as_float(expected)
    return (
        actual_number is not None
        and expected_number is not None
        and math.isclose(actual_number, expected_number, rel_tol=0.0, abs_tol=tolerance)
    )


def _all_true(values: Iterable[bool]) -> bool:
    return all(values)


def _find_output_hash(manifest: dict[str, Any], suffix: str) -> tuple[str, str]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Q2 optimization manifest has no outputs mapping")
    candidates = [
        (str(path), str(expected))
        for path, expected in outputs.items()
        if str(path).replace("\\", "/").endswith(suffix)
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one manifest output ending in {suffix!r}, found {len(candidates)}")
    return candidates[0]


def _required_artifact_specs(
    repo_root: Path,
    config: dict[str, Any],
    optimization_manifest: dict[str, Any],
    risk_manifest: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    inputs = optimization_manifest.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("q2_risk_rating_scores"), dict):
        raise ValueError("Q2 optimization manifest is missing inputs.q2_risk_rating_scores")

    risk_input = inputs["q2_risk_rating_scores"]
    risk_path_value = risk_input.get("path")
    risk_expected = risk_input.get("sha256")
    if not risk_path_value or not risk_expected:
        raise ValueError("q2_risk_rating_scores input lacks path or sha256")

    config_path_value = optimization_manifest.get("config_path")
    config_expected = optimization_manifest.get("config_sha256")
    if not config_path_value or not config_expected:
        raise ValueError("Q2 optimization manifest lacks config_path/config_sha256")

    strategy_path, strategy_expected = _find_output_hash(
        optimization_manifest, "data/processed/q2_credit_strategy.csv"
    )
    portfolio_path, portfolio_expected = _find_output_hash(
        optimization_manifest, "outputs/q2/tables/q2_portfolio_summary.csv"
    )
    validation_path, validation_expected = _find_output_hash(
        optimization_manifest, "outputs/q2/tables/q2_strategy_validation.csv"
    )

    paths = config.get("paths", {})
    configured_model_path = paths.get("q2_model_config")
    risk_input_hashes = risk_manifest.get("input_hashes", {})
    model_expected = risk_input_hashes.get(str(configured_model_path).replace("\\", "/"))
    model_semantic_expected = risk_manifest.get("q2_config_sha256")
    if not configured_model_path or not model_expected or not model_semantic_expected:
        raise ValueError("q3_config.paths.q2_model_config or risk manifest q2 config hashes are missing")

    return {
        "q2_risk_rating_scores": {
            "path": _resolve_path(repo_root, str(risk_path_value)),
            "expected": str(risk_expected),
            "hash_kind": "file_bytes",
        },
        "q2_credit_strategy": {
            "path": _resolve_path(repo_root, strategy_path),
            "expected": strategy_expected,
            "hash_kind": "file_bytes",
        },
        "q2_portfolio_summary": {
            "path": _resolve_path(repo_root, portfolio_path),
            "expected": portfolio_expected,
            "hash_kind": "file_bytes",
        },
        "q2_strategy_validation": {
            "path": _resolve_path(repo_root, validation_path),
            "expected": validation_expected,
            "hash_kind": "file_bytes",
        },
        "q2_optimization_config": {
            "path": _resolve_path(repo_root, str(config_path_value)),
            "expected": str(config_expected),
            "hash_kind": "semantic_config",
        },
        "q2_model_config": {
            "path": _resolve_path(repo_root, str(configured_model_path)),
            "expected": str(model_expected),
            "hash_kind": "file_bytes",
            "expected_semantic": str(model_semantic_expected),
        },
    }


def _check_risk_table(
    rows: list[dict[str, str]], target_count: int, probability_tolerance: float
) -> dict[str, Any]:
    required_columns = {
        "enterprise_id",
        "main_risk_score",
        "rating_prob_A",
        "rating_prob_B",
        "rating_prob_C",
        "rating_prob_D",
    }
    columns = set(rows[0]) if rows else set()
    columns_present = required_columns.issubset(columns)
    ids = [row.get("enterprise_id", "") for row in rows]
    ids_unique = len(ids) == len(set(ids)) and all(ids)
    risk_values_legal = True
    rating_values_legal = True
    rating_sums_one = True

    if columns_present:
        for row in rows:
            risk = _as_float(row.get("main_risk_score"))
            risk_values_legal &= risk is not None and -probability_tolerance <= risk <= 1.0 + probability_tolerance
            probabilities = [_as_float(row.get(f"rating_prob_{rating}")) for rating in "ABCD"]
            rating_values_legal &= all(
                p is not None and -probability_tolerance <= p <= 1.0 + probability_tolerance
                for p in probabilities
            )
            rating_sums_one &= (
                all(p is not None for p in probabilities)
                and _numbers_close(sum(p for p in probabilities if p is not None), 1.0, probability_tolerance)
            )
    else:
        risk_values_legal = rating_values_legal = rating_sums_one = False

    return {
        "required_columns_present": columns_present,
        "row_count": len(rows),
        "target_row_count": target_count,
        "row_count_target": len(rows) == target_count,
        "enterprise_id_unique": ids_unique,
        "risk_values_legal": risk_values_legal,
        "rating_probability_nonnegative_and_bounded": rating_values_legal,
        "rating_probability_sum_one": rating_sums_one,
    }


def _check_strategy_table(rows: list[dict[str, str]], target_count: int) -> dict[str, Any]:
    ids = [row.get("enterprise_id", "") for row in rows]
    return {
        "row_count": len(rows),
        "target_row_count": target_count,
        "row_count_target": len(rows) == target_count,
        "enterprise_id_unique": len(ids) == len(set(ids)) and all(ids),
        "required_columns_present": bool(rows) and {"enterprise_id", "scenario_id"}.issubset(rows[0]),
    }


def _check_validation_table(
    rows: list[dict[str, str]],
    primary_scenario: dict[str, Any],
    primary_validation: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    scenario_id = primary_scenario.get("scenario_id")
    matching = [row for row in rows if row.get("scenario_id") == str(scenario_id)]
    expected_boolean_fields = [
        key for key, value in primary_validation.items() if isinstance(value, bool)
    ]
    manifest_boolean_checks = {
        key: bool(primary_validation.get(key)) for key in expected_boolean_fields
    }
    row_boolean_checks = {
        key: _as_bool(matching[0].get(key)) if len(matching) == 1 else None
        for key in expected_boolean_fields
    }
    all_manifest_booleans_true = _all_true(manifest_boolean_checks.values())
    all_row_booleans_true = _all_true(value is True for value in row_boolean_checks.values())
    status_ok = (
        len(matching) == 1
        and matching[0].get("validation_status") == str(primary_scenario.get("validation_status"))
        and matching[0].get("validation_status") == "PASS"
    )
    numeric_consistency = False
    if len(matching) == 1:
        row = matching[0]
        numeric_consistency = all(
            _numbers_close(row.get(field), primary_validation.get(field), tolerance)
            for field in ("budget_10k", "budget_difference_10k")
            if field in primary_validation
        )
    return {
        "row_count": len(rows),
        "primary_row_count": len(matching),
        "primary_row_present_once": len(matching) == 1,
        "expected_boolean_fields": expected_boolean_fields,
        "manifest_boolean_checks": manifest_boolean_checks,
        "row_boolean_checks": row_boolean_checks,
        "manifest_required_booleans_true": all_manifest_booleans_true,
        "row_required_booleans_true": all_row_booleans_true,
        "primary_status_pass": status_ok,
        "numeric_fields_match_manifest": numeric_consistency,
    }


def _check_primary_summary(
    rows: list[dict[str, str]], primary_scenario: dict[str, Any], tolerance: float
) -> dict[str, Any]:
    scenario_id = str(primary_scenario.get("scenario_id"))
    matching = [row for row in rows if row.get("scenario_id") == scenario_id]
    if len(matching) != 1:
        return {"primary_row_present_once": False, "primary_row_count": len(matching)}
    row = matching[0]
    expected_budget = primary_scenario.get("budget_10k")
    expected_offered = primary_scenario.get("nominal_offered_amount_10k")
    expected_difference = primary_scenario.get("budget_difference_10k")
    return {
        "primary_row_present_once": True,
        "primary_row_count": len(matching),
        "solve_status_optimal": row.get("solve_status") == str(primary_scenario.get("solve_status"))
        and row.get("solve_status") == "optimal",
        "is_optimal": _as_bool(row.get("is_optimal")) is True,
        "validation_status_pass": row.get("validation_status") == "PASS",
        "fallback_unused": _as_bool(row.get("fallback_used")) is False,
        "budget_definition_matches": row.get("budget_definition") == str(primary_scenario.get("budget_definition")),
        "budget_matches_manifest": _numbers_close(row.get("budget_10k"), expected_budget, tolerance),
        "offered_amount_matches_manifest": _numbers_close(
            row.get("nominal_offered_amount_10k"), expected_offered, tolerance
        ),
        "budget_difference_matches_manifest": _numbers_close(
            row.get("budget_difference_10k"), expected_difference, tolerance
        ),
    }


def _check_risk_manifest(
    risk_manifest: dict[str, Any],
    optimization_manifest: dict[str, Any],
    target_count: int,
) -> dict[str, Any]:
    input_record = optimization_manifest.get("inputs", {}).get("q2_risk_rating_scores", {})
    expected_optimization_hash = input_record.get("sha256")
    output_hashes = risk_manifest.get("output_hashes", {})
    risk_hash_candidates = [
        value
        for path, value in output_hashes.items()
        if str(path).replace("\\", "/").endswith("data/processed/q2_risk_rating_scores.csv")
    ]
    target_from_risk = risk_manifest.get("counts", {}).get("target_enterprises")
    return {
        "target_enterprise_count_present": target_from_risk is not None,
        "target_enterprise_count_matches": target_from_risk == target_count,
        "risk_output_hash_declared_once": len(risk_hash_candidates) == 1,
        "risk_output_hash_matches_optimization_manifest": (
            len(risk_hash_candidates) == 1 and risk_hash_candidates[0] == expected_optimization_hash
        ),
        "target_accuracy_reported_false": risk_manifest.get("target_accuracy_reported") is False,
    }


def _load_config_and_paths(
    repo_root: Path, config_path: Path | None
) -> tuple[dict[str, Any], Path, Path, Path, Path]:
    resolved_config = config_path or (repo_root / DEFAULT_CONFIG_RELATIVE)
    config = _read_yaml(resolved_config)
    paths = config.get("paths")
    if not isinstance(paths, dict):
        raise ValueError("q3_config.paths must be a mapping")
    optimization_manifest_path = _resolve_path(repo_root, str(paths["q2_optimization_manifest"]))
    risk_manifest_path = _resolve_path(repo_root, str(paths["q2_risk_manifest"]))
    contract_path = _resolve_path(repo_root, str(paths.get("baseline_contract_output", DEFAULT_CONTRACT_RELATIVE)))
    return config, resolved_config, optimization_manifest_path, risk_manifest_path, contract_path


def _failure_contract(repo_root: Path, config_path: Path, message: str) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "status": "FAIL",
        "config_path": _relative_path(repo_root, config_path),
        "error": message,
        "semantic_checks": {},
        "artifact_hashes": {},
    }


def _write_contract(path: Path, contract: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run_q3_preflight(
    repo_root: Path | None = None, config_path: Path | None = None
) -> dict[str, Any]:
    """Run the Q2 baseline gate and write the contract JSON.

    A contract with ``status == "FAIL"`` is written for diagnostics before a
    :class:`PreflightError` is raised.  No failing run can produce a PASS
    contract.
    """

    root = (repo_root or _repo_root_from_file()).resolve()
    requested_config = (config_path or (root / DEFAULT_CONFIG_RELATIVE)).resolve()
    output_path = root / DEFAULT_CONTRACT_RELATIVE
    try:
        config, resolved_config, optimization_manifest_path, risk_manifest_path, output_path = _load_config_and_paths(
            root, requested_config
        )
        optimization_manifest = _read_json(optimization_manifest_path)
        risk_manifest = _read_json(risk_manifest_path)
        artifact_specs = _required_artifact_specs(root, config, optimization_manifest, risk_manifest)
        artifact_hashes = {
            name: _hash_record(
                root,
                spec["path"],
                spec.get("expected"),
                hash_kind=str(spec.get("hash_kind", "file_bytes")),
                expected_semantic=spec.get("expected_semantic"),
            )
            for name, spec in artifact_specs.items()
        }

        primary_scenario = optimization_manifest.get("primary_scenario")
        primary_validation = optimization_manifest.get("primary_validation")
        if not isinstance(primary_scenario, dict) or not isinstance(primary_validation, dict):
            raise ValueError("Q2 optimization manifest is missing primary_scenario/primary_validation")

        target_count = int(optimization_manifest["target_enterprise_count"])
        probability_tolerance = 1.0e-8
        model_config_path = artifact_specs["q2_model_config"]["path"]
        model_config = _read_yaml(model_config_path)
        model_tolerance = model_config.get("modeling", {}).get("probability_check_tolerance")
        if model_tolerance is not None:
            probability_tolerance = float(model_tolerance)
        numeric_tolerance = float(config.get("checks", {}).get("numeric_tolerance", 1.0e-6))

        risk_rows = _read_csv(artifact_specs["q2_risk_rating_scores"]["path"])
        strategy_rows = _read_csv(artifact_specs["q2_credit_strategy"]["path"])
        portfolio_rows = _read_csv(artifact_specs["q2_portfolio_summary"]["path"])
        validation_rows = _read_csv(artifact_specs["q2_strategy_validation"]["path"])
        risk_checks = _check_risk_table(risk_rows, target_count, probability_tolerance)
        strategy_checks = _check_strategy_table(strategy_rows, target_count)
        validation_checks = _check_validation_table(
            validation_rows, primary_scenario, primary_validation, numeric_tolerance
        )
        summary_checks = _check_primary_summary(portfolio_rows, primary_scenario, numeric_tolerance)
        risk_manifest_checks = _check_risk_manifest(risk_manifest, optimization_manifest, target_count)

        primary_semantics = {
            "scenario_id": primary_scenario.get("scenario_id"),
            "solve_status_optimal": primary_scenario.get("solve_status") == "optimal"
            and primary_scenario.get("is_optimal") is True,
            "validation_status_pass": primary_scenario.get("validation_status") == "PASS",
            "fallback_unused": primary_scenario.get("fallback_used") is False
            and optimization_manifest.get("fallback_used") is False,
            "budget_definition_nominal_equality": primary_scenario.get("budget_definition")
            == "nominal_equality",
            "budget_equality_required": primary_scenario.get("budget_equality_required") is True,
            "budget_matches_nominal_offered": _numbers_close(
                primary_scenario.get("budget_10k"),
                primary_scenario.get("nominal_offered_amount_10k"),
                numeric_tolerance,
            ),
            "budget_difference_zero": _numbers_close(
                primary_scenario.get("budget_difference_10k"), 0.0, numeric_tolerance
            ),
        }
        for key, value in primary_validation.items():
            if isinstance(value, bool):
                primary_semantics[f"manifest_validation_{key}"] = value

        artifact_checks_pass = _all_true(
            record["hash_status"] in {
                "PASS_BYTE_EXACT",
                "PASS_LINE_ENDING_NORMALIZED",
                "PASS_SEMANTIC_CONFIG",
            }
            and record.get("semantic_hash_status") in {None, "PASS_SEMANTIC_CONFIG"}
            for record in artifact_hashes.values()
        )
        table_checks_pass = _all_true(
            [
                risk_checks["row_count_target"],
                risk_checks["enterprise_id_unique"],
                risk_checks["risk_values_legal"],
                risk_checks["rating_probability_nonnegative_and_bounded"],
                risk_checks["rating_probability_sum_one"],
                strategy_checks["row_count_target"],
                strategy_checks["enterprise_id_unique"],
                strategy_checks["required_columns_present"],
                validation_checks["primary_row_present_once"],
                validation_checks["manifest_required_booleans_true"],
                validation_checks["row_required_booleans_true"],
                validation_checks["primary_status_pass"],
                validation_checks["numeric_fields_match_manifest"],
                summary_checks["primary_row_present_once"],
            ]
        )
        manifest_checks_pass = _all_true(
            [
                risk_manifest_checks["target_enterprise_count_matches"],
                risk_manifest_checks["risk_output_hash_declared_once"],
                risk_manifest_checks["risk_output_hash_matches_optimization_manifest"],
                risk_manifest_checks["target_accuracy_reported_false"],
            ]
        )
        semantic_checks_pass = _all_true(primary_semantics.values()) and _all_true(summary_checks.values())
        all_checks_pass = artifact_checks_pass and table_checks_pass and manifest_checks_pass and semantic_checks_pass

        contract = {
            "contract_version": CONTRACT_VERSION,
            "status": "PASS" if all_checks_pass else "FAIL",
            "config": {
                "path": _relative_path(root, resolved_config),
                "sha256": _sha256_bytes(resolved_config.read_bytes()),
            },
            "authoritative_manifest": _manifest_record(root, optimization_manifest_path),
            "risk_manifest": {
                **_manifest_record(root, risk_manifest_path),
                "semantic_checks": risk_manifest_checks,
            },
            "artifact_hashes": artifact_hashes,
            "baseline_parameters": {
                "target_enterprise_count": target_count,
                "enterprise_count_total": primary_scenario.get("enterprise_count_total"),
                "budget_10k": primary_scenario.get("budget_10k"),
                "nominal_offered_amount_10k": primary_scenario.get("nominal_offered_amount_10k"),
                "budget_difference_10k": primary_scenario.get("budget_difference_10k"),
                "budget_definition": primary_scenario.get("budget_definition"),
                "risk_variant": primary_scenario.get("risk_variant"),
                "lgd": primary_scenario.get("lgd"),
                "funding_cost_rate": primary_scenario.get("funding_cost_rate"),
                "d_probability_threshold": primary_scenario.get("d_probability_threshold"),
                "uncertainty_cap_10k": primary_scenario.get("uncertainty_cap_10k"),
                "selected_enterprise_count": primary_scenario.get("selected_enterprise_count"),
            },
            "table_checks": {
                "risk_scores": risk_checks,
                "credit_strategy": strategy_checks,
                "portfolio_summary": summary_checks,
                "strategy_validation": validation_checks,
            },
            "semantic_checks": {
                "primary_scenario": primary_semantics,
                "artifact_hashes_pass": artifact_checks_pass,
                "tables_pass": table_checks_pass,
                "risk_manifest_pass": manifest_checks_pass,
                "primary_summary_pass": _all_true(summary_checks.values()),
            },
            "evidence": {
                "q2_optimization_manifest_path": _relative_path(root, optimization_manifest_path),
                "q2_risk_manifest_path": _relative_path(root, risk_manifest_path),
                "values_are_read_from_q2_manifest": True,
                "probability_tolerance_source": _relative_path(root, model_config_path),
            },
        }
    except Exception as exc:  # noqa: BLE001 - convert all gate failures to a contract and non-zero exit
        contract = _failure_contract(root, requested_config, f"{type(exc).__name__}: {exc}")

    _write_contract(output_path, contract)
    if contract.get("status") != "PASS":
        raise PreflightError(
            f"Q3 Q2 baseline preflight failed; see {_relative_path(root, output_path)}",
            contract,
        )
    return contract
