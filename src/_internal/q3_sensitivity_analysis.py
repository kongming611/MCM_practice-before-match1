"""Q3 stage-five sensitivity, stability, and release-gate analysis.

This module deliberately keeps the stress transformation small and explicit,
then delegates every credit decision to the accepted Q2 ``_run_scenario``
MILP.  It is a deterministic parameter sweep, not a Monte-Carlo simulation;
there is no supported evidence for an industry recovery path or duration yet.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd
import yaml

try:  # script entry points expose ``src`` on sys.path
    from _internal import q2_credit_optimization as q2
    from _internal.data_pipeline import sha256_file
    from _internal.q3_baseline_preflight import PreflightError, run_q3_preflight
    from _internal.q3_industry_mapping import IndustryMappingError, run_q3_industry_mapping
    from _internal.q3_robust_optimization import RobustOptimizationError, run_q3_robust_optimization
    from _internal.q3_stress_scenarios import (
        INDUSTRY_CODES,
        NBS_GROWTH_YOY_PCT,
        NBS_INTERPRETATION,
        NBS_PERIOD,
        NBS_RELEASE_DATE,
        NBS_SOURCE_URL,
        NBS_SOURCE_STATUS,
        NBS_STANDARD,
        SCENARIOS as STRESS_SCENARIOS,
        StressScenarioError,
        apply_risk_stress,
        calculate_industry_stress,
        clip_risk,
        run_q3_stress_scenarios,
    )
except ModuleNotFoundError:  # pragma: no cover - package import compatibility
    _SRC = Path(__file__).resolve().parents[1]
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from _internal import q2_credit_optimization as q2
    from _internal.data_pipeline import sha256_file
    from _internal.q3_baseline_preflight import PreflightError, run_q3_preflight
    from _internal.q3_industry_mapping import IndustryMappingError, run_q3_industry_mapping
    from _internal.q3_robust_optimization import RobustOptimizationError, run_q3_robust_optimization
    from _internal.q3_stress_scenarios import (
        INDUSTRY_CODES,
        NBS_GROWTH_YOY_PCT,
        NBS_INTERPRETATION,
        NBS_PERIOD,
        NBS_RELEASE_DATE,
        NBS_SOURCE_URL,
        NBS_SOURCE_STATUS,
        NBS_STANDARD,
        SCENARIOS as STRESS_SCENARIOS,
        StressScenarioError,
        apply_risk_stress,
        calculate_industry_stress,
        clip_risk,
        run_q3_stress_scenarios,
    )


TOLERANCE = 1.0e-6
FORMAL_SCENARIO_ID = "formal_robust"
DURATION_ASSUMPTION_TEXT = "unconfirmed; Q1 retrospective stress marker only"
DURATION_LIMITATION_REASON = (
    "Q0 supplied only a retrospective 2020Q1 pressure marker; "
    "industry duration/recovery path requires team confirmation."
)
STABILITY_RULE = (
    "robust_selected_core if loan_frequency>=0.80; "
    "selection_sensitive if 0<loan_frequency<0.80; "
    "consistently_not_selected if loan_frequency=0; "
    "selection-frequency analysis only, not enterprise robustness or default conclusion"
)
RISK_SCENARIO_NAMES = ("identity", "light", "medium", "severe", "robust")
CORE_SCORE_COLUMNS = (
    "rating_prob_A",
    "rating_prob_B",
    "rating_prob_C",
    "rating_prob_D",
    "pD",
    "d_probability_reject_flag",
    "max_loan_limit_10k",
)


class SensitivityAnalysisError(RuntimeError):
    """Raised when the Q3 stage-five acceptance contract fails."""


def _root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise SensitivityAnalysisError(f"configuration must be a mapping: {path}")
    return value


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SensitivityAnalysisError(f"required table does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise SensitivityAnalysisError(f"required table is empty: {path}")
    return frame


def _duration_noncalibration_gate(
    scenario_specs: Sequence[Mapping[str, Any]],
    long_table: pd.DataFrame,
    limitation_reason: str,
) -> bool:
    """Check that duration is disclosed as an uncalibrated marker everywhere.

    This is deliberately a content check rather than a constant acceptance flag:
    removing the ``unconfirmed``/retrospective marker from either the scenario
    metadata or the long table must fail the sensitivity contract.
    """

    required_assumption_terms = ("unconfirmed", "q1", "retrospective", "stress marker")
    required_reason_terms = ("q0", "retrospective", "pressure marker", "duration", "team confirmation")

    def _has_terms(value: Any, terms: Sequence[str]) -> bool:
        text = " ".join(str(value).strip().lower().split())
        return bool(text) and all(term in text for term in terms)

    spec_values = [item.get("duration_assumption", "") for item in scenario_specs]
    if not spec_values or not all(_has_terms(value, required_assumption_terms) for value in spec_values):
        return False
    if "duration_assumption" not in long_table.columns or long_table.empty:
        return False
    long_values = long_table["duration_assumption"].tolist()
    if not long_values or not all(_has_terms(value, required_assumption_terms) for value in long_values):
        return False
    reason = " ".join(str(limitation_reason).strip().lower().split())
    return reason == " ".join(DURATION_LIMITATION_REASON.lower().split()) and all(term in reason for term in required_reason_terms)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _frame_hash(frame: pd.DataFrame, columns: Sequence[str] | None = None) -> str:
    selected = frame.loc[:, list(columns)] if columns is not None else frame
    selected = selected.copy()
    selected = selected.sort_values(list(selected.columns), kind="stable").reset_index(drop=True)
    payload = selected.to_csv(index=False, lineterminator="\n", float_format="%.12g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("sensitivity_analysis")
    if not isinstance(raw, Mapping):
        raise SensitivityAnalysisError("q3_config.sensitivity_analysis is missing")
    centre = raw.get("center")
    if not isinstance(centre, Mapping):
        raise SensitivityAnalysisError("sensitivity_analysis.center is missing")
    values: dict[str, Any] = {
        "risk_input": str(raw.get("risk_input", "data/processed/q3_scenario_risk_scores.csv")),
        "mapping_input": str(raw.get("mapping_input", "data/processed/q3_enterprise_industry_mapping.csv")),
        "q2_strategy_input": str(raw.get("q2_strategy_input", "data/processed/q2_credit_strategy.csv")),
        "q3_robust_strategy_input": str(raw.get("q3_robust_strategy_input", "data/processed/q3_robust_credit_strategy.csv")),
        "q2_optimization_config": str(raw.get("q2_optimization_config", "src/_internal/q2_optimization_config.yaml")),
        "summary_output": str(raw.get("summary_output", "outputs/q3/tables/q3_sensitivity_summary.csv")),
        "stability_output": str(raw.get("stability_output", "outputs/q3/tables/q3_enterprise_strategy_stability.csv")),
        "long_output": str(raw.get("long_output", "outputs/q3/tables/q3_sensitivity_enterprise_strategies.csv")),
        "industry_output": str(raw.get("industry_output", "outputs/q3/tables/q3_sensitivity_industry_shocks.csv")),
        "validation_output": str(raw.get("validation_output", "outputs/q3/reports/q3_sensitivity_validation.json")),
        "report_output": str(raw.get("report_output", "outputs/q3/reports/q3_sensitivity_validation.md")),
        "full_validation_output": str(raw.get("full_validation_output", "outputs/q3/reports/q3_full_validation.json")),
        "full_report_output": str(raw.get("full_report_output", "outputs/q3/reports/q3_full_validation.md")),
        "run_manifest_output": str(raw.get("run_manifest_output", "outputs/q3/reports/q3_run_manifest.json")),
        "center": {
            "c_pp": float(centre.get("c_pp", 40.0)),
            "rho": float(centre.get("rho", 0.25)),
            "lambda_severe": float(centre.get("lambda_severe", 1.5)),
            "unknown_policy": str(centre.get("unknown_policy", "max_known_S")),
            "lgd": float(centre.get("lgd", 0.5)),
            "funding_cost_rate": float(centre.get("funding_cost_rate", 0.03)),
        },
        "rho_values": [float(item) for item in raw.get("rho_values", [0.1, 0.25, 0.5])],
        "c_pp_values": [float(item) for item in raw.get("c_pp_values", [20, 40, 60])],
        "lambda_severe_values": [float(item) for item in raw.get("lambda_severe_values", [1.125, 1.5, 1.875])],
        "unknown_policies": [str(item) for item in raw.get("unknown_policies", ["p50_known_S", "max_known_S"])],
        "lgd_values": [float(item) for item in raw.get("lgd_values", [0.3, 0.5, 0.7])],
        "funding_cost_rates": [float(item) for item in raw.get("funding_cost_rates", [0.02, 0.03, 0.04])],
        "stability_frequency_threshold": float(raw.get("stability_frequency_threshold", 0.80)),
        "risk_eps": float(raw.get("risk_eps", 1.0e-6)),
    }
    centre_values = values["center"]
    if centre_values != {
        "c_pp": 40.0,
        "rho": 0.25,
        "lambda_severe": 1.5,
        "unknown_policy": "max_known_S",
        "lgd": 0.5,
        "funding_cost_rate": 0.03,
    }:
        raise SensitivityAnalysisError("stage-five centre must equal the accepted Q3 values")
    if not 0 < values["risk_eps"] < 1 or not 0 < values["stability_frequency_threshold"] <= 1:
        raise SensitivityAnalysisError("risk_eps and stability threshold are invalid")
    if any(policy not in {"p50_known_S", "max_known_S"} for policy in values["unknown_policies"]):
        raise SensitivityAnalysisError("unknown policy must be p50_known_S or max_known_S")
    return values


def load_sensitivity_settings(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or (_root_from_file() / "src" / "_internal" / "q3_config.yaml")
    return _load_settings(_read_yaml(path))


def _scenario_token(value: Any) -> str:
    return str(value).replace(".", "p").replace("-", "m").replace(" ", "_")


def _scenario_specs(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    centre = dict(settings["center"])
    specs: list[dict[str, Any]] = [
        {
            "scenario_id": FORMAL_SCENARIO_ID,
            "scenario_kind": "formal_robust",
            "sensitivity_axis": "formal_robust",
            "sensitivity_level": "centre",
            **centre,
            "parameter_source": "modeling_assumption_centre",
        }
    ]
    axes = (
        ("rho", settings["rho_values"]),
        ("c_pp", settings["c_pp_values"]),
        ("lambda_severe", settings["lambda_severe_values"]),
        ("unknown_policy", settings["unknown_policies"]),
        ("lgd", settings["lgd_values"]),
        ("funding_cost_rate", settings["funding_cost_rates"]),
    )
    for axis, values in axes:
        for value in values:
            candidate = dict(centre)
            candidate[axis] = value
            if all(candidate[key] == centre[key] for key in centre):
                continue  # one canonical centre row only
            specs.append(
                {
                    "scenario_id": f"sensitivity_{axis}_{_scenario_token(value)}",
                    "scenario_kind": "single_factor",
                    "sensitivity_axis": axis,
                    "sensitivity_level": str(value),
                    **candidate,
                    "parameter_source": "modeling_assumption_single_factor",
                }
            )
    adverse = dict(centre)
    adverse.update({"rho": 0.50, "c_pp": 20.0, "lambda_severe": 1.875, "unknown_policy": "max_known_S", "lgd": 0.70, "funding_cost_rate": 0.04})
    specs.append(
        {
            "scenario_id": "exploratory_combined_adverse",
            "scenario_kind": "exploratory_combined_adverse",
            "sensitivity_axis": "combined_adverse",
            "sensitivity_level": "boundary",
            **adverse,
            "parameter_source": "exploratory_combined_adverse_modeling_assumption",
        }
    )
    ids = [str(item["scenario_id"]) for item in specs]
    if len(ids) != len(set(ids)) or len(specs) != 13:
        raise SensitivityAnalysisError(f"unexpected or duplicate sensitivity scenario set: {ids}")
    for spec in specs:
        changed = [key for key in centre if spec[key] != centre[key]]
        expected = 1 if spec["scenario_kind"] == "single_factor" else (0 if spec["scenario_kind"] == "formal_robust" else 5)
        if len(changed) != expected:
            raise SensitivityAnalysisError(f"scenario {spec['scenario_id']} changes {changed}, expected {expected}")
        spec["changed_parameter_count"] = len(changed)
        spec["changed_parameters"] = ",".join(changed) if changed else "none"
        spec["analysis_rule"] = STABILITY_RULE
        spec["duration_assumption"] = DURATION_ASSUMPTION_TEXT
    return specs


def _industry_stress_table(spec: Mapping[str, Any], eps: float) -> tuple[pd.DataFrame, dict[str, float], float]:
    lambdas = {"identity": 0.0, "light": 0.5, "medium": 1.0, "severe": float(spec["lambda_severe"])}
    known: dict[str, dict[str, Any]] = {}
    for code in INDUSTRY_CODES:
        known[code] = calculate_industry_stress(
            NBS_GROWTH_YOY_PCT[code], c_pp=float(spec["c_pp"]), rho=float(spec["rho"]), eps=eps, lambdas=lambdas
        )
    known_s = [float(known[code]["S_h"]) for code in INDUSTRY_CODES]
    if spec["unknown_policy"] == "max_known_S":
        unknown_s = max(known_s)
    else:
        unknown_s = float(np.percentile(np.asarray(known_s, dtype=float), 50))
    rows: list[dict[str, Any]] = []
    for code in (*INDUSTRY_CODES, "unknown"):
        if code == "unknown":
            growth = None
            d_value = unknown_s * float(spec["c_pp"])
            s_value = unknown_s
            exposures = {name: min(max(float(lambdas[name]) * s_value, 0.0), 1.0) for name in STRESS_SCENARIOS}
            multipliers = {name: 1.0 + float(spec["rho"]) * exposures[name] for name in STRESS_SCENARIOS}
            source_status = "derived_policy_p50_known_S" if spec["unknown_policy"] == "p50_known_S" else "derived_policy_max_known_S"
        else:
            item = known[code]
            growth = float(item["growth_yoy_pct"])
            d_value = float(item["d_h_pp"])
            s_value = float(item["S_h"])
            exposures = {name: float(item["exposures"][name]) for name in STRESS_SCENARIOS}
            multipliers = {name: float(item["multipliers"][name]) for name in STRESS_SCENARIOS}
            source_status = NBS_SOURCE_STATUS
        rows.append(
            {
                "scenario_id": spec["scenario_id"],
                "scenario_kind": spec["scenario_kind"],
                "sensitivity_axis": spec["sensitivity_axis"],
                "industry_code": code,
                "growth_yoy_pct": growth,
                "d_h_pp": d_value,
                "S_h": s_value,
                "D_identity": exposures["identity"],
                "D_light": exposures["light"],
                "D_medium": exposures["medium"],
                "D_severe": exposures["severe"],
                "identity_multiplier": multipliers["identity"],
                "light_multiplier": multipliers["light"],
                "medium_multiplier": multipliers["medium"],
                "severe_multiplier": multipliers["severe"],
                "c_pp": float(spec["c_pp"]),
                "rho": float(spec["rho"]),
                "lambda_severe": float(spec["lambda_severe"]),
                "unknown_policy": spec["unknown_policy"],
                "unknown_S_value": unknown_s,
                "known_S_p50_value": float(np.percentile(np.asarray(known_s, dtype=float), 50)),
                "growth_source_url": NBS_SOURCE_URL,
                "source_release_date": NBS_RELEASE_DATE,
                "source_period": NBS_PERIOD,
                "source_standard": NBS_STANDARD,
                "source_status": source_status,
                "source_interpretation": NBS_INTERPRETATION,
                "parameter_source": "official_growth_plus_modeling_assumption" if code != "unknown" else "modeling_assumption_derived_policy",
                "duration_assumption": spec["duration_assumption"],
            }
        )
    table = pd.DataFrame(rows)
    return table, {code: float(known[code]["S_h"]) for code in INDUSTRY_CODES}, unknown_s


def _risk_frame(
    q2_scores: pd.DataFrame,
    mapping: pd.DataFrame,
    spec: Mapping[str, Any],
    eps: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int, float]:
    industry_table, known_s, unknown_s = _industry_stress_table(spec, eps)
    params = industry_table.set_index("industry_code")
    mapping_view = mapping[["enterprise_id", "enterprise_name", "industry_code", "confidence"]].copy()
    mapping_view["unknown_industry_flag"] = (mapping_view["industry_code"].astype(str) == "unknown").astype(int)
    mapping_view["industry_high_uncertainty_flag"] = ((mapping_view["industry_code"].astype(str) == "unknown") | (pd.to_numeric(mapping_view["confidence"], errors="coerce").fillna(0.0) < 1.0)).astype(int)
    mapping_view = mapping_view.drop(columns=["confidence"])
    mapping_view["enterprise_id"] = mapping_view["enterprise_id"].astype(str)
    scores = q2_scores.copy()
    scores["enterprise_id"] = scores["enterprise_id"].astype(str)
    if set(scores["enterprise_id"]) != set(mapping_view["enterprise_id"]):
        raise SensitivityAnalysisError("Q2 score and industry mapping ids do not match")
    scores = scores.sort_values("enterprise_id", kind="stable").reset_index(drop=True)
    mapping_view = mapping_view.sort_values("enterprise_id", kind="stable").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    multipliers = {code: {name: float(params.loc[code, f"{name}_multiplier"]) for name in STRESS_SCENARIOS} for code in params.index}
    for left, mapped in zip(scores.to_dict("records"), mapping_view.to_dict("records")):
        code = str(mapped["industry_code"])
        q2_risk = float(left["main_risk_score"])
        stress = {"multipliers": multipliers[code]}
        risks = apply_risk_stress(q2_risk, stress, eps=eps)
        row = {
            "enterprise_id": str(left["enterprise_id"]),
            "enterprise_name": str(left["enterprise_name"]),
            "industry_code": code,
            "unknown_industry_flag": int(mapped["unknown_industry_flag"]),
            "industry_high_uncertainty_flag": int(mapped["industry_high_uncertainty_flag"]),
            "q2_main_risk": q2_risk,
            "identity_risk": risks["identity"],
            "light_risk": risks["light"],
            "medium_risk": risks["medium"],
            "severe_risk": risks["severe"],
            "robust_risk": risks["robust"],
            "risk_upper_clip_count": int(any(q2_risk * multipliers[code][name] > 1.0 - eps for name in STRESS_SCENARIOS)),
            "risk_override_only": True,
        }
        rows.append(row)
    risk = pd.DataFrame(rows).sort_values("enterprise_id", kind="stable").reset_index(drop=True)
    return risk, industry_table, int(risk["risk_upper_clip_count"].sum()), unknown_s


def _override_scores(q2_scores: pd.DataFrame, risk: pd.DataFrame) -> pd.DataFrame:
    result = q2_scores.copy()
    result["enterprise_id"] = result["enterprise_id"].astype(str)
    values = risk.set_index("enterprise_id")["robust_risk"]
    result["main_risk_score"] = result["enterprise_id"].map(values).astype(float)
    if result["main_risk_score"].isna().any():
        raise SensitivityAnalysisError("risk override did not join all Q2 enterprises")
    return result.sort_values("enterprise_id", kind="stable").reset_index(drop=True)


def _strategy_with_metadata(strategy: pd.DataFrame, mapping: pd.DataFrame, spec: Mapping[str, Any]) -> pd.DataFrame:
    result = strategy.copy()
    result["enterprise_id"] = result["enterprise_id"].astype(str)
    mapping_view = mapping[["enterprise_id", "enterprise_name", "industry_code", "confidence"]].copy()
    mapping_view["unknown_industry_flag"] = (mapping_view["industry_code"].astype(str) == "unknown").astype(int)
    mapping_view["industry_high_uncertainty_flag"] = ((mapping_view["industry_code"].astype(str) == "unknown") | (pd.to_numeric(mapping_view["confidence"], errors="coerce").fillna(0.0) < 1.0)).astype(int)
    mapping_view = mapping_view.drop(columns=["confidence"])
    mapping_view["enterprise_id"] = mapping_view["enterprise_id"].astype(str)
    result = result.drop(columns=[c for c in ("industry_code", "unknown_industry_flag", "industry_high_uncertainty_flag") if c in result.columns])
    result = result.merge(mapping_view, on=["enterprise_id", "enterprise_name"], how="left", validate="one_to_one")
    if result["industry_code"].isna().any():
        raise SensitivityAnalysisError(f"strategy lost industry mapping for {spec['scenario_id']}")
    result["q3_scenario_id"] = spec["scenario_id"]
    result["scenario_id"] = spec["scenario_id"]
    result["scenario_kind"] = spec["scenario_kind"]
    result["sensitivity_axis"] = spec["sensitivity_axis"]
    result["sensitivity_level"] = spec["sensitivity_level"]
    result["unknown_policy"] = spec["unknown_policy"]
    result["q3_c_pp"] = float(spec["c_pp"])
    result["q3_rho"] = float(spec["rho"])
    result["q3_lambda_severe"] = float(spec["lambda_severe"])
    result["risk_override_column"] = "robust_risk"
    result["risk_override_only"] = True
    result["parameter_source"] = spec["parameter_source"]
    result["duration_assumption"] = spec["duration_assumption"]
    return result.sort_values("enterprise_id", kind="stable").reset_index(drop=True)


def _selected_ids(frame: pd.DataFrame) -> set[str]:
    amounts = pd.to_numeric(frame["offered_loan_amount_10k"], errors="coerce")
    return set(frame.loc[amounts > TOLERANCE, "enterprise_id"].astype(str))


def _hhi_and_unknown_share(strategy: pd.DataFrame, mapping: pd.DataFrame) -> tuple[float, float]:
    amount = pd.to_numeric(strategy["offered_loan_amount_10k"], errors="coerce").fillna(0.0)
    total = float(amount.sum())
    if abs(total - 10000.0) > 1e-5:
        raise SensitivityAnalysisError(f"sensitivity budget is not exactly 10000: {total}")
    shares = strategy.assign(_amount=amount).groupby("industry_code", sort=True)["_amount"].sum() / total
    hhi = float((shares * shares).sum())
    unknown_share = float(shares.get("unknown", 0.0))
    return hhi, unknown_share


def _summary_row(
    result: Mapping[str, Any],
    strategy: pd.DataFrame,
    risk: pd.DataFrame,
    spec: Mapping[str, Any],
    formal_strategy: pd.DataFrame,
    unknown_s: float,
    clip_count: int,
) -> dict[str, Any]:
    summary = result["summary"]
    formal_selected = _selected_ids(formal_strategy)
    selected = _selected_ids(strategy)
    union = formal_selected | selected
    jaccard = float(len(formal_selected & selected) / len(union)) if union else 1.0
    amounts = strategy.set_index("enterprise_id")["offered_loan_amount_10k"].astype(float)
    formal_amounts = formal_strategy.set_index("enterprise_id")["offered_loan_amount_10k"].astype(float)
    amount_l1 = float((amounts - formal_amounts).abs().sum())
    hhi, unknown_share = _hhi_and_unknown_share(strategy, strategy)
    return {
        "scenario_id": spec["scenario_id"],
        "scenario_kind": spec["scenario_kind"],
        "sensitivity_axis": spec["sensitivity_axis"],
        "sensitivity_level": spec["sensitivity_level"],
        "changed_parameters": spec["changed_parameters"],
        "changed_parameter_count": spec["changed_parameter_count"],
        "c_pp": float(spec["c_pp"]),
        "rho": float(spec["rho"]),
        "lambda_severe": float(spec["lambda_severe"]),
        "unknown_policy": spec["unknown_policy"],
        "unknown_S_value": float(unknown_s),
        "unknown_S_source": "actual_P50_of_11_known_S" if spec["unknown_policy"] == "p50_known_S" else "actual_max_of_11_known_S",
        "lgd": float(spec["lgd"]),
        "funding_cost_rate": float(spec["funding_cost_rate"]),
        "parameter_source": spec["parameter_source"],
        "duration_assumption": spec["duration_assumption"],
        "solve_status": str(summary.get("solve_status")),
        "solver_status_code": int(summary.get("solver_status_code", 0)),
        "is_optimal": bool(summary.get("is_optimal")),
        "validation_status": str(summary.get("validation_status")),
        "solver_called": bool(result["diagnostics"].get("solver_called")),
        "fallback_used": bool(summary.get("fallback_used", False)),
        "nominal_offered_amount_10k": float(summary.get("nominal_offered_amount_10k", 0.0)),
        "budget_difference_10k": float(summary.get("budget_difference_10k", 0.0)),
        "expected_credit_loss_10k": float(summary.get("expected_credit_loss_10k", 0.0)),
        "expected_funding_cost_10k": float(summary.get("expected_funding_cost_10k", 0.0)),
        "expected_interest_income_10k": float(summary.get("expected_interest_income_10k", 0.0)),
        "expected_net_return_10k": float(summary.get("expected_net_return_10k", 0.0)),
        "weighted_risk_used": float(summary.get("weighted_risk_used", 0.0)),
        "selected_enterprise_count": int(summary.get("selected_enterprise_count", 0)),
        "jaccard_vs_formal_robust": jaccard,
        "amount_l1_vs_formal_robust_10k": amount_l1,
        "industry_hhi": hhi,
        "unknown_amount_share": unknown_share,
        "risk_upper_clip_count": int(clip_count),
        "risk_upper_clip_applied": bool(clip_count > 0),
        "risk_override_column": "robust_risk",
        "risk_override_only": bool(strategy["risk_override_only"].all()),
        "analysis_rule": STABILITY_RULE,
    }


def _long_strategy(strategy: pd.DataFrame, spec: Mapping[str, Any], risk: pd.DataFrame, unknown_s: float) -> pd.DataFrame:
    columns = [
        "enterprise_id", "enterprise_name", "industry_code", "unknown_industry_flag",
        "industry_high_uncertainty_flag", "main_risk_score", "risk_used", "decision",
        "selected_rate", "offered_loan_amount_10k", "expected_credit_loss_10k",
        "expected_funding_cost_10k", "expected_net_return_10k", "selected_rate_is_observed_attachment3_point",
        "rating_prob_A", "rating_prob_B", "rating_prob_C", "rating_prob_D", "pD",
        "d_probability_reject_flag", "max_loan_limit_10k", "risk_override_only",
    ]
    selected = strategy[[c for c in columns if c in strategy.columns]].copy()
    selected = selected.rename(columns={"main_risk_score": "risk_used_main_score"})
    if "selected_rate" in selected.columns:
        selected["selected_rate_missing"] = selected["selected_rate"].isna().astype(int)
        selected["selected_rate"] = pd.to_numeric(selected["selected_rate"], errors="coerce").fillna(0.0)
    selected["q2_main_risk"] = risk.set_index("enterprise_id").loc[selected["enterprise_id"], "q2_main_risk"].to_numpy()
    selected["q3_robust_risk"] = risk.set_index("enterprise_id").loc[selected["enterprise_id"], "robust_risk"].to_numpy()
    selected["scenario_id"] = spec["scenario_id"]
    selected["scenario_kind"] = spec["scenario_kind"]
    selected["sensitivity_axis"] = spec["sensitivity_axis"]
    selected["sensitivity_level"] = spec["sensitivity_level"]
    selected["c_pp"] = float(spec["c_pp"])
    selected["rho"] = float(spec["rho"])
    selected["lambda_severe"] = float(spec["lambda_severe"])
    selected["unknown_policy"] = spec["unknown_policy"]
    selected["unknown_S_value"] = float(unknown_s)
    selected["lgd"] = float(spec["lgd"])
    selected["funding_cost_rate"] = float(spec["funding_cost_rate"])
    selected["parameter_source"] = spec["parameter_source"]
    selected["duration_assumption"] = spec["duration_assumption"]
    selected["risk_override_column"] = "robust_risk"
    return selected.sort_values("enterprise_id", kind="stable").reset_index(drop=True)


def _compare_deterministic(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    key = [c for c in ("enterprise_id", "decision", "selected_rate", "offered_loan_amount_10k", "expected_net_return_10k", "risk_used") if c in left.columns and c in right.columns]
    return _frame_hash(left, key) == _frame_hash(right, key)


def _validate_score_override(q2_scores: pd.DataFrame, scenario_scores: pd.DataFrame) -> bool:
    left = q2_scores.set_index("enterprise_id").sort_index()
    right = scenario_scores.set_index("enterprise_id").sort_index()
    if set(left.index) != set(right.index):
        return False
    present = [column for column in CORE_SCORE_COLUMNS if column in left and column in right]
    required = [column for column in ("rating_prob_A", "rating_prob_B", "rating_prob_C", "rating_prob_D") if column in left]
    if not set(required).issubset(present):
        return False
    for column in present:
        if not np.allclose(pd.to_numeric(left[column], errors="coerce"), pd.to_numeric(right[column], errors="coerce"), atol=1e-12, rtol=0):
            return False
    return True


def _validate_strategy_components(q2_strategy: pd.DataFrame, scenario_strategy: pd.DataFrame) -> bool:
    left = q2_strategy.set_index("enterprise_id").sort_index()
    right = scenario_strategy.set_index("enterprise_id").sort_index()
    if set(left.index) != set(right.index):
        return False
    for column in CORE_SCORE_COLUMNS:
        if column not in left or column not in right:
            continue
        if not np.allclose(pd.to_numeric(left[column], errors="coerce"), pd.to_numeric(right[column], errors="coerce"), atol=1e-12, rtol=0):
            return False
    return True


def _matches_reference_strategy(left: pd.DataFrame, right: pd.DataFrame, atol: float = 1.0e-8) -> bool:
    a = left.set_index("enterprise_id").sort_index()
    b = right.set_index("enterprise_id").sort_index()
    if set(a.index) != set(b.index):
        return False
    if not a["decision"].astype(str).equals(b["decision"].astype(str)):
        return False
    for column in ("selected_rate", "offered_loan_amount_10k", "expected_net_return_10k", "risk_used"):
        if column not in a or column not in b:
            return False
        if not np.allclose(pd.to_numeric(a[column], errors="coerce"), pd.to_numeric(b[column], errors="coerce"), atol=atol, rtol=0, equal_nan=True):
            return False
    return True


def _validate_results(
    summary: pd.DataFrame,
    long_table: pd.DataFrame,
    industry_table: pd.DataFrame,
    stability: pd.DataFrame,
    q2_scores: pd.DataFrame,
    q2_strategy: pd.DataFrame,
    formal_strategy: pd.DataFrame,
    scenario_specs: Sequence[Mapping[str, Any]],
    deterministic_matches: Mapping[str, bool],
    centre_reproduces: bool,
    q2_hash_before: Mapping[str, str],
    q2_hash_after: Mapping[str, str],
    score_override_checks: Mapping[str, bool],
    duration_limitation_reason: str,
) -> dict[str, Any]:
    scenario_ids = [str(item["scenario_id"]) for item in scenario_specs]
    checks: dict[str, Any] = {
        "scenario_count_13": len(scenario_ids) == 13,
        "scenario_ids_unique": len(scenario_ids) == len(set(scenario_ids)),
        "single_factor_only": all(item["changed_parameter_count"] == 1 for item in scenario_specs if item["scenario_kind"] == "single_factor"),
        "combined_adverse_labeled": any(item["scenario_kind"] == "exploratory_combined_adverse" for item in scenario_specs),
        "centre_parameters_exact": all(
            item["c_pp"] == 40.0 and item["rho"] == 0.25 and item["lambda_severe"] == 1.5 and item["unknown_policy"] == "max_known_S" and item["lgd"] == 0.5 and item["funding_cost_rate"] == 0.03
            for item in scenario_specs if item["scenario_id"] == FORMAL_SCENARIO_ID
        ),
        "summary_rows_13": len(summary) == 13,
        "long_rows_13x302": len(long_table) == 13 * 302,
        "stability_rows_302": len(stability) == 302,
        "enterprise_ids_exact": set(long_table["enterprise_id"].astype(str)) == set(q2_scores["enterprise_id"].astype(str)) and long_table.groupby("scenario_id")["enterprise_id"].nunique().eq(302).all(),
        "all_numeric_finite": bool(np.isfinite(summary.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()) and bool(np.isfinite(long_table.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()) and bool(np.isfinite(stability.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()),
        "all_optimal": bool(summary["is_optimal"].astype(bool).all() and (summary["solve_status"].astype(str) == "optimal").all() and summary["solver_called"].astype(bool).all()),
        "all_validation_pass": bool((summary["validation_status"].astype(str) == "PASS").all()),
        "no_fallback": bool((~summary["fallback_used"].astype(bool)).all()),
        "budget_exact_all": bool(np.allclose(summary["nominal_offered_amount_10k"], 10000.0, atol=1e-5, rtol=0)),
        "risk_override_only_all": bool(summary["risk_override_only"].astype(bool).all() and long_table["risk_override_only"].astype(bool).all()),
        "risk_upper_clip_recorded": bool("risk_upper_clip_applied" in summary),
        "unknown_p50_actual": False,
        "centre_reproduces_stage4_robust": bool(centre_reproduces),
        "deterministic_repeat_all": bool(all(deterministic_matches.values()) and len(deterministic_matches) == len(scenario_specs)),
        "q2_score_components_unchanged": bool(all(score_override_checks.values())),
        "q2_artifacts_unchanged": dict(q2_hash_before) == dict(q2_hash_after),
        "formal_selected_count_matches": bool(_selected_ids(formal_strategy) == _selected_ids(long_table.loc[long_table["scenario_id"] == FORMAL_SCENARIO_ID])),
        "stability_rule_recorded": bool((stability["analysis_rule"].astype(str) == STABILITY_RULE).all()),
        "stability_uses_12_formal_scenarios": bool((stability["scenario_count"].astype(int) == 12).all()),
        "exploratory_adverse_excluded_from_stability": bool((stability["excluded_scenario"].astype(str) == "exploratory_combined_adverse").all()),
        "no_duration_calibration_claim": bool((long_table["duration_assumption"].astype(str).str.contains("unconfirmed")).all()),
        "duration_noncalibration_disclosed": False,
    }
    p50_rows = summary.loc[summary["unknown_policy"].astype(str) == "p50_known_S"]
    p50_ok = True
    for row in p50_rows.itertuples(index=False):
        known = industry_table.loc[
            (industry_table["scenario_id"].astype(str) == str(row.scenario_id))
            & (industry_table["industry_code"].astype(str) != "unknown"),
            "S_h",
        ].astype(float)
        expected = float(np.percentile(known.to_numpy(dtype=float), 50)) if len(known) == 11 else float("nan")
        p50_ok = p50_ok and len(known) == 11 and abs(float(row.unknown_S_value) - expected) <= 1e-12 and str(row.unknown_S_source) == "actual_P50_of_11_known_S"
    checks["unknown_p50_actual"] = bool(p50_rows.empty is False and p50_ok)
    checks["duration_noncalibration_disclosed"] = _duration_noncalibration_gate(scenario_specs, long_table, duration_limitation_reason)
    return {"status": "PASS" if all(bool(value) for value in checks.values()) else "FAIL", "checks": checks}


def _write_sensitivity_report(path: Path, contract: Mapping[str, Any], summary: pd.DataFrame) -> None:
    lines = [
        "# Q3 stage-five sensitivity validation",
        "",
        "本阶段是确定性参数敏感性分析，不是随机模拟；所有策略仍由 Q2 `_run_scenario` 求解。风险仍是历史发票行为相对违约倾向，不是真实 PD。",
        "",
        "行业压力数据为包含 2020-03 的 2020Q1 回顾性外部压力标尺；行业持续期/全年恢复路径尚待团队确认，本阶段没有进行持续期校准。",
        "",
        f"状态：**{contract['status']}**；场景数：{len(summary)}；确定性重复：{contract['checks']['deterministic_repeat_all']}。",
        "",
        f"选择频率稳定性仅使用正式中心+11个单因素场景（12个）；排除 exploratory_combined_adverse。类别计数：{contract.get('stability_counts', {})}。这些类别是选择频率分析标签，不是企业真实稳健性或违约结论。",
        "",
        "持续期校准不可用：Q0只有回顾性2020Q1压力标尺，行业持续期/恢复路径尚需团队确认。",
        "",
        "| scenario_id | axis | c_pp | rho | severe lambda | unknown policy | LGD | funding | net return (10k) | credit loss (10k) | Jaccard | HHI | unknown share |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.scenario_id} | {row.sensitivity_axis} | {float(row.c_pp):.3f} | {float(row.rho):.3f} | {float(row.lambda_severe):.3f} | {row.unknown_policy} | {float(row.lgd):.3f} | {float(row.funding_cost_rate):.3f} | {float(row.expected_net_return_10k):.6f} | {float(row.expected_credit_loss_10k):.6f} | {float(row.jaccard_vs_formal_robust):.4f} | {float(row.industry_hhi):.4f} | {float(row.unknown_amount_share):.4f} |"
        )
    lines.extend(["", "## Acceptance checks", "", "```json", json.dumps(_json_safe(contract["checks"]), ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    _write_text("\n".join(lines), path)


def run_q3_sensitivity_analysis(
    repo_root: Path | None = None,
    config_path: Path | None = None,
    *,
    preflight_contract: Mapping[str, Any] | None = None,
    mapping_contract: Mapping[str, Any] | None = None,
    stress_contract: Mapping[str, Any] | None = None,
    optimization_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run stage five after the four prior Q3 contracts have passed."""

    root = (repo_root or _root_from_file()).resolve()
    q3_config_path = (config_path or (root / "src" / "_internal" / "q3_config.yaml")).resolve()
    settings = _load_settings(_read_yaml(q3_config_path))
    if preflight_contract is None:
        preflight_contract = run_q3_preflight(root, q3_config_path)
    if mapping_contract is None:
        mapping_contract = run_q3_industry_mapping(root, q3_config_path, preflight_contract=preflight_contract)
    if stress_contract is None:
        stress_contract = run_q3_stress_scenarios(root, q3_config_path, preflight_contract=preflight_contract, mapping_contract=mapping_contract)
    if optimization_contract is None:
        optimization_contract = run_q3_robust_optimization(root, q3_config_path, preflight_contract=preflight_contract, mapping_contract=mapping_contract, stress_contract=stress_contract)
    for name, contract in (("preflight", preflight_contract), ("mapping", mapping_contract), ("stress", stress_contract), ("optimization", optimization_contract)):
        if contract.get("status") != "PASS":
            raise SensitivityAnalysisError(f"stage-five requires PASS {name} contract")

    q2_config = q2.load_config(_resolve(root, settings["q2_optimization_config"]))
    q2_scores = q2._load_risk_rating_scores(q2.resolve_path(q2_config["inputs"]["risk_rating_scores"]))
    mapping = _read_csv(_resolve(root, settings["mapping_input"]))
    q2_strategy = _read_csv(_resolve(root, settings["q2_strategy_input"]))
    formal_strategy = _read_csv(_resolve(root, settings["q3_robust_strategy_input"]))
    if len(q2_scores) != 302 or len(mapping) != 302 or len(q2_strategy) != 302 or len(formal_strategy) != 302:
        raise SensitivityAnalysisError("stage-five input rows must all be 302")
    q2_scores["enterprise_id"] = q2_scores["enterprise_id"].astype(str)
    mapping["enterprise_id"] = mapping["enterprise_id"].astype(str)
    q2_strategy["enterprise_id"] = q2_strategy["enterprise_id"].astype(str)
    formal_strategy["enterprise_id"] = formal_strategy["enterprise_id"].astype(str)
    if set(q2_scores["enterprise_id"]) != set(mapping["enterprise_id"]) or set(q2_scores["enterprise_id"]) != set(q2_strategy["enterprise_id"]) or set(q2_scores["enterprise_id"]) != set(formal_strategy["enterprise_id"]):
        raise SensitivityAnalysisError("stage-five inputs do not have exact enterprise id joins")

    q2_hash_before = {
        "q2_scores": sha256_file(q2.resolve_path(q2_config["inputs"]["risk_rating_scores"])),
        "q2_strategy": sha256_file(_resolve(root, settings["q2_strategy_input"])),
        "q2_manifest": sha256_file(q2.resolve_path(q2_config["outputs"]["manifest"])),
    }
    churn, churn_evidence = q2._load_q1_accepted_churn(q2_config)
    primary = q2._scenario_from_config(q2_config)
    specs = _scenario_specs(settings)
    formal_by_id = formal_strategy.sort_values("enterprise_id", kind="stable").reset_index(drop=True)
    result_by_id: dict[str, dict[str, Any]] = {}
    strategy_by_id: dict[str, pd.DataFrame] = {}
    risk_by_id: dict[str, pd.DataFrame] = {}
    industry_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    long_frames: list[pd.DataFrame] = []
    deterministic_matches: dict[str, bool] = {}
    score_override_checks: dict[str, bool] = {}
    clip_counts: dict[str, int] = {}
    unknown_values: dict[str, float] = {}

    for spec in specs:
        risk, industry, clip_count, unknown_s = _risk_frame(q2_scores, mapping, spec, float(settings["risk_eps"]))
        scenario_scores = _override_scores(q2_scores, risk)
        score_override_checks[spec["scenario_id"]] = _validate_score_override(q2_scores, scenario_scores)
        scenario = replace(
            primary,
            scenario_id=str(spec["scenario_id"]),
            scenario_kind="primary" if spec["scenario_kind"] == "formal_robust" else "sensitivity",
            sensitivity_axis=str(spec["sensitivity_axis"]),
            sensitivity_level=str(spec["sensitivity_level"]),
            lgd=float(spec["lgd"]),
            funding_cost_rate=float(spec["funding_cost_rate"]),
        )
        result = q2._run_scenario(scenario_scores, churn, scenario, q2_config)
        strategy = _strategy_with_metadata(result["strategy"], mapping, spec)
        score_override_checks[spec["scenario_id"]] = score_override_checks[spec["scenario_id"]] and _validate_strategy_components(q2_strategy, strategy)
        result["strategy"] = strategy
        result_by_id[str(spec["scenario_id"])] = result
        strategy_by_id[str(spec["scenario_id"])] = strategy
        risk_by_id[str(spec["scenario_id"])] = risk
        industry["lgd"] = float(spec["lgd"])
        industry["funding_cost_rate"] = float(spec["funding_cost_rate"])
        industry_frames.append(industry)
        clip_counts[str(spec["scenario_id"])] = clip_count
        unknown_values[str(spec["scenario_id"])] = unknown_s
        summary_rows.append(_summary_row(result, strategy, risk, spec, formal_by_id, unknown_s, clip_count))
        long_frames.append(_long_strategy(strategy, spec, risk, unknown_s))

    # Repeat the complete deterministic sweep and compare the decision tables.
    for spec in specs:
        risk, _, _, _ = _risk_frame(q2_scores, mapping, spec, float(settings["risk_eps"]))
        scenario = replace(
            primary,
            scenario_id=str(spec["scenario_id"]),
            scenario_kind="primary" if spec["scenario_kind"] == "formal_robust" else "sensitivity",
            sensitivity_axis=str(spec["sensitivity_axis"]),
            sensitivity_level=str(spec["sensitivity_level"]),
            lgd=float(spec["lgd"]),
            funding_cost_rate=float(spec["funding_cost_rate"]),
        )
        repeat_result = q2._run_scenario(_override_scores(q2_scores, risk), churn, scenario, q2_config)
        repeat_strategy = _strategy_with_metadata(repeat_result["strategy"], mapping, spec)
        deterministic_matches[str(spec["scenario_id"])] = _compare_deterministic(strategy_by_id[str(spec["scenario_id"])], repeat_strategy)

    summary = pd.DataFrame(summary_rows).sort_values("scenario_id", kind="stable").reset_index(drop=True)
    long_table = pd.concat(long_frames, ignore_index=True).sort_values(["scenario_id", "enterprise_id"], kind="stable").reset_index(drop=True)
    industry_table = pd.concat(industry_frames, ignore_index=True).sort_values(["scenario_id", "industry_code"], kind="stable").reset_index(drop=True)

    formal_result = result_by_id[FORMAL_SCENARIO_ID]
    centre_key = ["enterprise_id", "decision", "selected_rate", "offered_loan_amount_10k", "expected_net_return_10k", "risk_used"]
    centre_reproduces = _matches_reference_strategy(formal_result["strategy"], formal_by_id)
    robust_summary = pd.read_csv(_resolve(root, "outputs/q3/tables/q3_portfolio_scenario_summary.csv"), encoding="utf-8-sig")
    robust_summary = robust_summary.loc[robust_summary["scenario_id"].astype(str) == "robust"].iloc[0]
    centre_reproduces = centre_reproduces and abs(float(formal_result["summary"]["expected_net_return_10k"]) - float(robust_summary["expected_net_return_10k"])) <= 1e-8
    centre_reproduces = centre_reproduces and abs(float(formal_result["summary"]["expected_credit_loss_10k"]) - float(robust_summary["expected_credit_loss_10k"])) <= 1e-8

    stability_rows: list[dict[str, Any]] = []
    formal_index = formal_result["strategy"].set_index("enterprise_id")
    stability_scenario_ids = [
        str(spec["scenario_id"])
        for spec in specs
        if spec["scenario_kind"] in {"formal_robust", "single_factor"}
    ]
    stability_long = long_table.loc[long_table["scenario_id"].astype(str).isin(stability_scenario_ids)].copy()
    for enterprise_id in sorted(q2_scores["enterprise_id"].astype(str)):
        entries = stability_long.loc[stability_long["enterprise_id"].astype(str) == enterprise_id].copy()
        amounts = pd.to_numeric(entries["offered_loan_amount_10k"], errors="coerce")
        rates = pd.to_numeric(entries["selected_rate"], errors="coerce")
        loan_mask = amounts > TOLERANCE
        formal_amount = float(formal_index.loc[enterprise_id, "offered_loan_amount_10k"])
        formal_rate = pd.to_numeric(formal_index.loc[enterprise_id, "selected_rate"], errors="coerce")
        rate_values = rates.loc[entries["selected_rate_missing"].astype(int) == 0]
        has_rate = not rate_values.empty
        formal_rate_missing = bool(pd.isna(formal_rate))
        rate_min = float(rate_values.min()) if has_rate else 0.0
        rate_max = float(rate_values.max()) if has_rate else 0.0
        rate_mean = float(rate_values.mean()) if has_rate else 0.0
        rate_sd = float(rate_values.std(ddof=0)) if has_rate else 0.0
        formal_rate_value = float(formal_rate) if not formal_rate_missing else 0.0
        rate_delta = float((rate_mean - formal_rate_value) * 100.0) if has_rate and not formal_rate_missing else 0.0
        stability_rows.append(
            {
                "enterprise_id": enterprise_id,
                "enterprise_name": str(entries.iloc[0]["enterprise_name"]),
                "scenario_count": int(len(entries)),
                "loan_frequency": float(loan_mask.mean()),
                "loan_count": int(loan_mask.sum()),
                "offered_amount_min_10k": float(amounts.min()),
                "offered_amount_max_10k": float(amounts.max()),
                "offered_amount_mean_10k": float(amounts.mean()),
                "offered_amount_sd_10k": float(amounts.std(ddof=0)),
                "selected_rate_observation_count": int(len(rate_values)),
                "selected_rate_missing_all": int(not has_rate),
                "selected_rate_min": rate_min,
                "selected_rate_max": rate_max,
                "selected_rate_mean": rate_mean,
                "selected_rate_sd": rate_sd,
                "formal_robust_amount_10k": formal_amount,
                "formal_robust_rate_missing": int(formal_rate_missing),
                "formal_robust_selected_rate": formal_rate_value,
                "amount_delta_mean_vs_formal_robust_10k": float(amounts.mean() - formal_amount),
                "rate_delta_mean_vs_formal_robust_pp": rate_delta,
                "stability_label": (
                    "robust_selected_core"
                    if float(loan_mask.mean()) >= float(settings["stability_frequency_threshold"])
                    else ("selection_sensitive" if float(loan_mask.mean()) > 0.0 else "consistently_not_selected")
                ),
                "analysis_rule": STABILITY_RULE,
                "excluded_scenario": "exploratory_combined_adverse",
                "exclusion_reason": "exploratory combined boundary is not a formal or single-factor scenario; excluded from selection frequency",
                "spread_interpretation": "deterministic scenario spread; not a statistical confidence interval",
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values("enterprise_id", kind="stable").reset_index(drop=True)
    q2_hash_after = {
        "q2_scores": sha256_file(q2.resolve_path(q2_config["inputs"]["risk_rating_scores"])),
        "q2_strategy": sha256_file(_resolve(root, settings["q2_strategy_input"])),
        "q2_manifest": sha256_file(q2.resolve_path(q2_config["outputs"]["manifest"])),
    }
    contract = _validate_results(
        summary,
        long_table,
        industry_table,
        stability,
        q2_scores,
        q2_strategy,
        formal_by_id,
        specs,
        deterministic_matches,
        centre_reproduces,
        q2_hash_before,
        q2_hash_after,
        score_override_checks,
        DURATION_LIMITATION_REASON,
    )
    contract.update(
        {
            "stage": "q3_sensitivity_analysis",
            "config_path": _relative(root, q3_config_path),
            "centre": settings["center"],
            "scenario_count": len(specs),
            "total_solve_time_seconds": float(summary.get("solve_status", pd.Series(dtype=str)).size * 0.0 + sum(float(result_by_id[sid]["diagnostics"].get("solve_time_seconds", 0.0)) for sid in result_by_id)),
            "scenario_ids": [str(spec["scenario_id"]) for spec in specs],
            "limitations": {
                "duration_calibration": {
                    "available": False,
                    "reason": DURATION_LIMITATION_REASON,
                },
                "stability_frequency": {
                    "scenario_count": int(len(stability_scenario_ids)),
                    "excluded_scenario": "exploratory_combined_adverse",
                    "exclusion_reason": "exploratory boundary is not a formal or single-factor scenario",
                },
            },
            "q2_reuse_evidence": {"solver_entry": "q2._run_scenario", "churn_source": churn_evidence, "risk_override": "only main_risk_score replaced; ratings/D/churn configuration retained"},
            "q2_artifact_hashes_before": q2_hash_before,
            "q2_artifact_hashes_after": q2_hash_after,
            "outputs": {},
        }
    )
    contract["stability_counts"] = {str(key): int(value) for key, value in stability["stability_label"].value_counts().to_dict().items()}
    summary_path = _resolve(root, settings["summary_output"])
    stability_path = _resolve(root, settings["stability_output"])
    long_path = _resolve(root, settings["long_output"])
    industry_path = _resolve(root, settings["industry_output"])
    validation_path = _resolve(root, settings["validation_output"])
    report_path = _resolve(root, settings["report_output"])
    _write_csv(summary, summary_path)
    _write_csv(stability, stability_path)
    _write_csv(long_table, long_path)
    _write_csv(industry_table, industry_path)
    contract["outputs"] = {
        "summary": {"path": _relative(root, summary_path), "row_count": int(len(summary)), "sha256": sha256_file(summary_path)},
        "stability": {"path": _relative(root, stability_path), "row_count": int(len(stability)), "sha256": sha256_file(stability_path)},
        "long": {"path": _relative(root, long_path), "row_count": int(len(long_table)), "sha256": sha256_file(long_path)},
        "industry": {"path": _relative(root, industry_path), "row_count": int(len(industry_table)), "sha256": sha256_file(industry_path)},
    }
    _write_json(_json_safe(contract), validation_path)
    _write_sensitivity_report(report_path, contract, summary)
    if contract["status"] != "PASS":
        raise SensitivityAnalysisError(f"Q3 sensitivity acceptance failed; see {_relative(root, validation_path)}")
    return contract


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _hash_paths(root: Path, paths: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative_path in paths:
        path = _resolve(root, relative_path)
        if path.is_file():
            result[relative_path] = sha256_file(path)
    return result


def build_q3_full_validation(
    repo_root: Path | None,
    config_path: Path | None,
    *,
    preflight_contract: Mapping[str, Any],
    mapping_contract: Mapping[str, Any],
    stress_contract: Mapping[str, Any],
    optimization_contract: Mapping[str, Any],
    sensitivity_contract: Mapping[str, Any],
    figure_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the fail-closed stage-one-to-five validation and run manifest."""

    root = (repo_root or _root_from_file()).resolve()
    q3_config_path = (config_path or (root / "src" / "_internal" / "q3_config.yaml")).resolve()
    settings = _load_settings(_read_yaml(q3_config_path))
    statuses = {"preflight": preflight_contract.get("status"), "mapping": mapping_contract.get("status"), "stress": stress_contract.get("status"), "optimization": optimization_contract.get("status"), "sensitivity": sensitivity_contract.get("status"), "figures": figure_contract.get("status")}
    q3_outputs = [
        "outputs/q3/reports/q3_q2_baseline_contract.json",
        "outputs/q3/reports/q3_industry_mapping_validation.json",
        "outputs/q3/tables/q3_industry_mapping_review_evidence.csv",
        "outputs/q3/reports/q3_stress_validation.json",
        "outputs/q3/reports/q3_robust_optimization_validation.json",
        settings["validation_output"],
        settings["summary_output"], settings["stability_output"], settings["long_output"], settings["industry_output"],
        "outputs/q3/tables/q3_portfolio_scenario_summary.csv",
        "outputs/q3/tables/q3_industry_exposure_comparison.csv",
        "outputs/q3/tables/q3_strategy_adjustments.csv",
        "outputs/q3/tables/q3_fixed_q2_strategy_under_robust_risk.csv",
        "data/processed/q3_scenario_risk_scores.csv",
        "data/processed/q3_robust_credit_strategy.csv",
        "outputs/q3/reports/q3_figure_qa.json",
        "outputs/q3/reports/q3_sensitivity_validation.md",
        "outputs/q3/reports/q3_figure_qa.md",
        "outputs/q3/tables/q3_figure_industry_shock_heatmap_source.csv",
        "outputs/q3/tables/q3_figure_industry_allocation_shift_source.csv",
        "outputs/q3/tables/q3_figure_sensitivity_results_source.csv",
        "outputs/q3/figures/q3_industry_shock_heatmap.png",
        "outputs/q3/figures/q3_industry_shock_heatmap.svg",
        "outputs/q3/figures/q3_industry_shock_heatmap.pdf",
        "outputs/q3/figures/q3_industry_shock_heatmap.tiff",
        "outputs/q3/figures/q3_industry_allocation_shift.png",
        "outputs/q3/figures/q3_industry_allocation_shift.svg",
        "outputs/q3/figures/q3_industry_allocation_shift.pdf",
        "outputs/q3/figures/q3_industry_allocation_shift.tiff",
        "outputs/q3/figures/q3_sensitivity_results.png",
        "outputs/q3/figures/q3_sensitivity_results.svg",
        "outputs/q3/figures/q3_sensitivity_results.pdf",
        "outputs/q3/figures/q3_sensitivity_results.tiff",
    ]
    q3_paths_hashes = _hash_paths(root, q3_outputs)
    q2_before = dict(optimization_contract.get("q2_artifact_hashes_before", {}))
    q2_after = dict(optimization_contract.get("q2_artifact_hashes_after", {}))
    q2_actual = {
        "q2_scores": sha256_file(_resolve(root, "data/processed/q2_risk_rating_scores.csv")),
        "q2_strategy": sha256_file(_resolve(root, "data/processed/q2_credit_strategy.csv")),
    }
    summary = _read_csv(_resolve(root, settings["summary_output"]))
    stability = _read_csv(_resolve(root, settings["stability_output"]))
    q3_score = _read_csv(_resolve(root, "data/processed/q3_scenario_risk_scores.csv"))
    checks = {
        "all_prior_contracts_pass": all(value == "PASS" for value in statuses.values()),
        "q3_output_hashes_present": len(q3_paths_hashes) == len(q3_outputs),
        "core_rows_302": len(q3_score) == 302 and len(stability) == 302,
        "sensitivity_summary_rows_13": len(summary) == 13,
        "all_solver_optimal": bool((summary["solve_status"].astype(str) == "optimal").all() and summary["is_optimal"].astype(bool).all()),
        "all_validation_pass": bool((summary["validation_status"].astype(str) == "PASS").all()),
        "all_fallback_false": bool((~summary["fallback_used"].astype(bool)).all()),
        "all_budget_exact": bool(np.allclose(summary["nominal_offered_amount_10k"], 10000.0, atol=1e-5, rtol=0)),
        "sensitivity_contract_pass": sensitivity_contract.get("status") == "PASS",
        "figure_qa_pass": figure_contract.get("status") == "PASS",
        "q2_hash_before_after_equal": q2_before == q2_after,
        "q2_actual_matches_after": all(q2_actual.get(key) == value for key, value in q2_after.items() if key in q2_actual),
        "formula_contract_recorded": bool(sensitivity_contract.get("centre")) and sensitivity_contract.get("limitations", {}).get("duration_calibration", {}).get("available") is False,
        "duration_noncalibration_disclosed": bool(sensitivity_contract.get("checks", {}).get("duration_noncalibration_disclosed")) and bool(sensitivity_contract.get("limitations", {}).get("duration_calibration", {}).get("reason")),
        "mapping_review_evidence_complete": bool(
            mapping_contract.get("checks", {}).get("review_evidence_rows_equal_unknown")
            and mapping_contract.get("checks", {}).get("review_evidence_ids_exact")
            and mapping_contract.get("review_evidence", {}).get("row_count") == 145
        ),
        "deterministic_repeat_pass": bool(sensitivity_contract.get("checks", {}).get("deterministic_repeat_all")),
    }
    full = {
        "status": "PASS" if all(bool(value) for value in checks.values()) else "FAIL",
        "stage": "q3_full_validation",
        "contracts": statuses,
        "checks": checks,
        "row_counts": {"q3_scenario_risk_scores": int(len(q3_score)), "sensitivity_summary": int(len(summary)), "enterprise_strategy_stability": int(len(stability))},
        "q2_artifact_hashes_before": q2_before,
        "q2_artifact_hashes_after": q2_after,
        "q3_artifact_hashes": q3_paths_hashes,
        "budget_definition": "nominal_equality; 10000 ten-thousand-CNY units",
        "solver_gate": "every sensitivity scenario optimal, validation PASS, solver called, fallback false",
        "formula": "risk_i,s=clip(q2_main_risk_i * (1+rho*clip(lambda_s*S_h,0,1)),0,1-eps); unknown main=max_known_S; P50 sensitivity is actual median of 11 known S values",
        "duration_note": "duration calibration unavailable; Q1 2020Q1 is retrospective external stress marker only",
        "figure_qa": _json_safe(figure_contract),
        "run_manifest_path": _relative(root, _resolve(root, settings["run_manifest_output"])),
    }
    full_path = _resolve(root, settings["full_validation_output"])
    full_report_path = _resolve(root, settings["full_report_output"])
    _write_json(_json_safe(full), full_path)
    md_lines = [
        "# Q3 full validation",
        "",
        f"状态：**{full['status']}**。这是阶段1到阶段5的fail-closed汇总；任何前序contract、求解、预算或图表QA不通过都不能发布。",
        "",
        "| Gate | Status |", "|---|---|",
    ]
    md_lines.extend(f"| {name} | {value} |" for name, value in statuses.items())
    md_lines.extend(["", "```json", json.dumps(_json_safe(checks), ensure_ascii=False, indent=2, sort_keys=True), "```", ""])
    _write_text("\n".join(md_lines), full_report_path)

    code_paths = [
        "src/q3.py", "src/_internal/q3_config.yaml", "src/_internal/q3_baseline_preflight.py", "src/_internal/q3_industry_mapping.py", "src/_internal/q3_stress_scenarios.py", "src/_internal/q3_robust_optimization.py", "src/_internal/q3_sensitivity_analysis.py", "src/_internal/q3_figures.py", "src/_internal/q2_credit_optimization.py",
    ]
    input_paths = [
        "data/processed/q2_risk_rating_scores.csv", "data/processed/q2_credit_strategy.csv", "data/processed/q3_enterprise_industry_mapping.csv", "data/processed/q3_scenario_risk_scores.csv", "data/processed/q3_robust_credit_strategy.csv", "src/_internal/q2_optimization_config.yaml",
    ]
    manifest = {
        "status": "PASS" if full["status"] == "PASS" else "FAIL",
        "stage": "q3_run_manifest",
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": _package_version("numpy"), "pandas": _package_version("pandas"), "scipy": _package_version("scipy"), "pyyaml": _package_version("PyYAML"), "python_executable": sys.executable},
        "contracts": statuses,
        "q2_artifact_hashes_before": q2_before,
        "q2_artifact_hashes_after": q2_after,
        "q2_artifact_hashes_unchanged": q2_before == q2_after,
        "code_hashes": _hash_paths(root, code_paths),
        "config_hashes": _hash_paths(root, ["src/_internal/q3_config.yaml", "src/_internal/q2_config.yaml", "src/_internal/q2_optimization_config.yaml"]),
        "input_hashes": _hash_paths(root, input_paths),
        "output_hashes": {**q3_paths_hashes, _relative(root, full_path): sha256_file(full_path)},
        "determinism": {"sensitivity_summary_sha256": sha256_file(_resolve(root, settings["summary_output"])), "sensitivity_stability_sha256": sha256_file(_resolve(root, settings["stability_output"])), "contract_repeat": sensitivity_contract.get("checks", {}).get("deterministic_repeat_all")},
        "formula_and_scope": {"risk_override_only": True, "q2_optimizer": "q2._run_scenario", "duration_calibration": "unconfirmed", "official_source": NBS_SOURCE_URL, "source_period": NBS_PERIOD},
    }
    manifest_path = _resolve(root, settings["run_manifest_output"])
    _write_json(_json_safe(manifest), manifest_path)
    return full


run_q3_analyze = run_q3_sensitivity_analysis
