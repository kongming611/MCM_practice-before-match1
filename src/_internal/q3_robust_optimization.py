"""Q3 scenario credit optimization built on the accepted Q2 MILP.

This module is intentionally an adapter.  It replaces only Q2's
``main_risk_score`` with one column from the stage-three stress table and
delegates policy construction, candidate economics, MILP solving, strategy
construction, and constraint validation to ``q2_credit_optimization``.
The resulting ``robust`` case uses the per-enterprise maximum over the four
stage-three risk scenarios.  Under the monotone multiplier construction this
is the same risk as ``severe``; that equivalence is reported explicitly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

try:  # script execution has ``src`` on sys.path
    from _internal import q2_credit_optimization as q2
    from _internal.data_pipeline import config_hash, sha256_file, write_csv, write_json, write_text
    from _internal.q3_baseline_preflight import PreflightError, run_q3_preflight
    from _internal.q3_industry_mapping import IndustryMappingError, run_q3_industry_mapping
    from _internal.q3_stress_scenarios import StressScenarioError, run_q3_stress_scenarios
except ModuleNotFoundError:  # tests import ``src._internal``
    _SRC = Path(__file__).resolve().parents[1]
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from _internal import q2_credit_optimization as q2
    from _internal.data_pipeline import config_hash, sha256_file, write_csv, write_json, write_text
    from _internal.q3_baseline_preflight import PreflightError, run_q3_preflight
    from _internal.q3_industry_mapping import IndustryMappingError, run_q3_industry_mapping
    from _internal.q3_stress_scenarios import StressScenarioError, run_q3_stress_scenarios


SCENARIOS = ("identity", "light", "medium", "severe", "robust")
STRESS_RISK_COLUMNS = {
    "identity": "identity_risk",
    "light": "light_risk",
    "medium": "medium_risk",
    "severe": "severe_risk",
    "robust": "robust_risk",
}
TOLERANCE = 1.0e-6
DECISION_NUMERIC_COLUMNS = (
    "risk_used",
    "selected_rate",
    "offered_loan_amount_10k",
    "selected_acceptance_probability",
    "selected_customer_churn_mass",
    "expected_disbursed_amount_10k",
    "expected_interest_income_10k",
    "expected_credit_loss_10k",
    "expected_funding_cost_10k",
    "expected_net_return_10k",
    "unit_expected_net_return_per_10k",
)
DECISION_TEXT_COLUMNS = ("decision",)


class RobustOptimizationError(RuntimeError):
    """Raised when the Q3 robust optimization acceptance gate fails."""


def _repo_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise RobustOptimizationError(f"Q3 config must be a mapping: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("robust_optimization")
    if not isinstance(raw, Mapping):
        raise RobustOptimizationError("q3_config.robust_optimization is missing")
    scenarios = tuple(str(value) for value in raw.get("scenarios", SCENARIOS))
    if scenarios != SCENARIOS:
        raise RobustOptimizationError("robust_optimization.scenarios must be identity/light/medium/severe/robust")
    settings = {
        "risk_input": str(raw.get("risk_input", "data/processed/q3_scenario_risk_scores.csv")),
        "q2_strategy_input": str(raw.get("q2_strategy_input", "data/processed/q2_credit_strategy.csv")),
        "q2_optimization_config": str(raw.get("q2_optimization_config", "src/_internal/q2_optimization_config.yaml")),
        "robust_strategy_output": str(raw.get("robust_strategy_output", "data/processed/q3_robust_credit_strategy.csv")),
        "scenario_strategy_output": str(raw.get("scenario_strategy_output", "outputs/q3/tables/q3_scenario_credit_strategies.csv")),
        "portfolio_summary_output": str(raw.get("portfolio_summary_output", "outputs/q3/tables/q3_portfolio_scenario_summary.csv")),
        "adjustments_output": str(raw.get("adjustments_output", "outputs/q3/tables/q3_strategy_adjustments.csv")),
        "industry_output": str(raw.get("industry_output", "outputs/q3/tables/q3_industry_exposure_comparison.csv")),
        "fixed_q2_output": str(raw.get("fixed_q2_output", "outputs/q3/tables/q3_fixed_q2_strategy_under_robust_risk.csv")),
        "validation_output": str(raw.get("validation_output", "outputs/q3/reports/q3_robust_optimization_validation.json")),
        "report_output": str(raw.get("report_output", "outputs/q3/reports/q3_robust_optimization_validation.md")),
        "numeric_tolerance": float(raw.get("numeric_tolerance", TOLERANCE)),
    }
    if settings["numeric_tolerance"] <= 0:
        raise RobustOptimizationError("robust_optimization.numeric_tolerance must be positive")
    return settings


def load_robust_settings(config_path: Path | None = None) -> dict[str, Any]:
    """Load Q3 robust settings for unit tests and collaborators."""

    path = config_path or (_repo_root_from_file() / "src" / "_internal" / "q3_config.yaml")
    return _load_settings(_read_yaml(path))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise RobustOptimizationError(f"required Q3 optimization table does not exist: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return {(str(row.enterprise_id), str(row.enterprise_name)) for row in frame.itertuples(index=False)}


def _index_by_id(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    if "enterprise_id" not in frame or "enterprise_name" not in frame:
        raise RobustOptimizationError(f"{label} must contain enterprise_id and enterprise_name")
    result = frame.copy()
    result["enterprise_id"] = result["enterprise_id"].astype(str).str.strip()
    result["enterprise_name"] = result["enterprise_name"].astype(str)
    if len(result) != 302 or result["enterprise_id"].duplicated().any():
        raise RobustOptimizationError(f"{label} must contain 302 unique enterprise ids")
    return result.sort_values("enterprise_id", kind="stable").reset_index(drop=True)


def _load_stress_and_q2(
    root: Path,
    settings: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = _index_by_id(
        q2._load_risk_rating_scores(q2.resolve_path(q2_config["inputs"]["risk_rating_scores"])),
        "Q2 risk/rating scores",
    )
    stress = _index_by_id(_read_csv(_resolve(root, settings["risk_input"])), "Q3 scenario risk scores")
    q2_strategy = _index_by_id(_read_csv(_resolve(root, settings["q2_strategy_input"])), "Q2 credit strategy")
    required_stress = {"industry_code", "q2_main_risk", "robust_risk", *STRESS_RISK_COLUMNS.values()}
    missing = sorted(required_stress.difference(stress.columns))
    if missing:
        raise RobustOptimizationError(f"Q3 scenario risk table is missing columns: {missing}")
    if _pairs(scores) != _pairs(stress) or _pairs(scores) != _pairs(q2_strategy):
        raise RobustOptimizationError("Q2 scores, Q3 stress rows, and Q2 strategy do not have an exact id/name join")
    for column in STRESS_RISK_COLUMNS.values():
        values = pd.to_numeric(stress[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all() or ((values < 0) | (values > 1)).any():
            raise RobustOptimizationError(f"Q3 risk column {column} is not finite and legal")
    robust = pd.to_numeric(stress["robust_risk"], errors="raise")
    expected_robust = stress[[STRESS_RISK_COLUMNS[name] for name in ("identity", "light", "medium", "severe")]].max(axis=1)
    if not np.allclose(robust.to_numpy(), expected_robust.to_numpy(), atol=1e-12, rtol=0):
        raise RobustOptimizationError("Q3 robust_risk is not the exact per-enterprise maximum")
    return scores, stress, q2_strategy


def _decorate_strategy(
    strategy: pd.DataFrame,
    stress: pd.DataFrame,
    scenario: str,
) -> pd.DataFrame:
    result = strategy.copy()
    stress_view = stress[
        [
            "enterprise_id",
            "enterprise_name",
            "industry_code",
            "q2_main_risk",
            "robust_risk",
            "unknown_industry_flag",
            "industry_high_uncertainty_flag",
        ]
    ].copy()
    result["enterprise_id"] = result["enterprise_id"].astype(str)
    result = result.merge(stress_view, on=["enterprise_id", "enterprise_name"], how="left", validate="one_to_one")
    if result["industry_code"].isna().any():
        raise RobustOptimizationError(f"Q3 strategy {scenario} lost stress rows during join")
    result["q2_baseline_risk"] = result["q2_main_risk"]
    result["q3_scenario_id"] = scenario
    result["scenario"] = scenario
    result["scenario_id"] = scenario
    result["q2_scenario_id"] = "primary"
    result["risk_override_column"] = STRESS_RISK_COLUMNS[scenario]
    result["risk_override_only"] = True
    result["risk_interpretation"] = "historical invoice behavior relative default tendency; not a verified PD"
    return result.sort_values("enterprise_id", kind="stable").reset_index(drop=True)


def _same_decision(left: pd.DataFrame, right: pd.DataFrame, *, atol: float) -> bool:
    a = _index_by_id(left, "left strategy").set_index("enterprise_id")
    b = _index_by_id(right, "right strategy").set_index("enterprise_id")
    if set(a.index) != set(b.index):
        return False
    for column in DECISION_TEXT_COLUMNS:
        if column in a and column in b and not a[column].astype(str).equals(b[column].astype(str)):
            return False
    for column in DECISION_NUMERIC_COLUMNS:
        if column not in a or column not in b:
            return False
        av = pd.to_numeric(a[column], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
        bv = pd.to_numeric(b[column], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
        if not np.allclose(av, bv, atol=atol, rtol=0):
            return False
    return True


def _scenario_scores(q2_scores: pd.DataFrame, stress: pd.DataFrame, scenario: str) -> pd.DataFrame:
    """Return Q2 scores with only ``main_risk_score`` overridden."""

    scores = q2_scores.copy()
    risk_by_id = stress.set_index("enterprise_id")[STRESS_RISK_COLUMNS[scenario]]
    scores["main_risk_score"] = scores["enterprise_id"].map(risk_by_id).astype(float)
    if scores["main_risk_score"].isna().any():
        raise RobustOptimizationError(f"risk override join failed for {scenario}")
    return scores


def _same_scenario_result(left: Mapping[str, Any], right: Mapping[str, Any], *, atol: float) -> bool:
    """Compare the repeated Q2 solve's key decisions, economics and summary."""

    if not _same_decision(left["strategy"], right["strategy"], atol=atol):
        return False
    for column in ("expected_interest_income_10k", "expected_credit_loss_10k", "expected_funding_cost_10k"):
        left_values = pd.to_numeric(left["strategy"][column], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
        right_values = pd.to_numeric(right["strategy"][column], errors="coerce").fillna(-999.0).to_numpy(dtype=float)
        if not np.allclose(left_values, right_values, atol=atol, rtol=0):
            return False
    for field in (
        "solve_status",
        "validation_status",
        "selected_enterprise_count",
        "nominal_offered_amount_10k",
        "expected_interest_income_10k",
        "expected_credit_loss_10k",
        "expected_funding_cost_10k",
        "expected_net_return_10k",
        "objective_value_solver_10k",
        "objective_value_recomputed_10k",
    ):
        left_value = left["summary"].get(field)
        right_value = right["summary"].get(field)
        if isinstance(left_value, str) or isinstance(right_value, str):
            if str(left_value) != str(right_value):
                return False
        elif not np.allclose(float(left_value), float(right_value), atol=atol, rtol=0):
            return False
    return True


def _numeric_delta(left: pd.Series, right: pd.Series) -> pd.Series:
    return pd.to_numeric(right, errors="coerce").fillna(0.0) - pd.to_numeric(left, errors="coerce").fillna(0.0)


def _build_adjustments(q2_strategy: pd.DataFrame, robust_strategy: pd.DataFrame) -> pd.DataFrame:
    left = _index_by_id(q2_strategy, "Q2 strategy")
    right = _index_by_id(robust_strategy, "Q3 robust strategy")
    merged = left.merge(right, on=["enterprise_id", "enterprise_name"], how="inner", suffixes=("_q2", "_q3"), validate="one_to_one")
    result = merged[["enterprise_id", "enterprise_name", "industry_code", "q2_main_risk", "robust_risk", "q3_scenario_id"]].copy()
    result = result.rename(columns={"q2_main_risk": "q2_baseline_risk"})
    result["q2_decision"] = merged["decision_q2"]
    result["robust_decision"] = merged["decision_q3"]
    result["q2_selected_rate"] = merged["selected_rate_q2"]
    result["robust_selected_rate"] = merged["selected_rate_q3"]
    result["delta_selected_rate"] = _numeric_delta(merged["selected_rate_q2"], merged["selected_rate_q3"])
    result["q2_offered_loan_amount_10k"] = pd.to_numeric(merged["offered_loan_amount_10k_q2"], errors="raise")
    result["robust_offered_loan_amount_10k"] = pd.to_numeric(merged["offered_loan_amount_10k_q3"], errors="raise")
    result["delta_offered_loan_amount_10k"] = _numeric_delta(merged["offered_loan_amount_10k_q2"], merged["offered_loan_amount_10k_q3"])
    result["q2_expected_net_return_10k"] = merged["expected_net_return_10k_q2"]
    result["robust_expected_net_return_10k"] = merged["expected_net_return_10k_q3"]
    result["delta_expected_net_return_10k"] = _numeric_delta(merged["expected_net_return_10k_q2"], merged["expected_net_return_10k_q3"])
    result["decision_changed"] = result["q2_decision"].astype(str) != result["robust_decision"].astype(str)
    result["rate_changed"] = result["delta_selected_rate"].abs() > TOLERANCE
    result["amount_changed"] = result["delta_offered_loan_amount_10k"].abs() > TOLERANCE
    return result.sort_values("enterprise_id", kind="stable").reset_index(drop=True)


def _build_fixed_q2_under_robust(
    q2_strategy: pd.DataFrame,
    robust_result: Mapping[str, Any],
    robust_strategy: pd.DataFrame,
    stress: pd.DataFrame,
    *,
    atol: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    q2_frame = _index_by_id(q2_strategy, "Q2 strategy")
    robust = _index_by_id(robust_strategy, "Q3 robust strategy")
    candidates = robust_result["candidates"].copy()
    candidates["enterprise_id"] = candidates["enterprise_id"].astype(str)
    lookup: dict[tuple[str, int], Any] = {}
    for row in candidates.itertuples(index=False):
        lookup[(str(row.enterprise_id), int(row.rate_index))] = row
    rates_by_enterprise: dict[str, list[tuple[float, Any]]] = {}
    for row in candidates.itertuples(index=False):
        rates_by_enterprise.setdefault(str(row.enterprise_id), []).append((float(row.interest_rate), row))
    stress_index = stress.set_index("enterprise_id")
    robust_index = robust.set_index("enterprise_id")
    rows: list[dict[str, Any]] = []
    candidate_match = True
    for q2_row in q2_frame.itertuples(index=False):
        enterprise_id = str(q2_row.enterprise_id)
        amount = float(pd.to_numeric(q2_row.offered_loan_amount_10k, errors="coerce") or 0.0)
        rate_value = pd.to_numeric(q2_row.selected_rate, errors="coerce")
        selected = amount > TOLERANCE and pd.notna(rate_value)
        candidate = None
        if selected:
            possible = rates_by_enterprise.get(enterprise_id, [])
            matches = [row for rate, row in possible if abs(rate - float(rate_value)) <= atol]
            if len(matches) != 1:
                candidate_match = False
                raise RobustOptimizationError(f"Q2 selected rate has no unique robust candidate for {enterprise_id}")
            candidate = matches[0]
        def value(field: str, default: float = 0.0) -> float:
            return float(getattr(candidate, field)) if candidate is not None else default
        fixed_amount = amount if selected else 0.0
        rows.append(
            {
                "enterprise_id": enterprise_id,
                "enterprise_name": str(q2_row.enterprise_name),
                "industry_code": str(stress_index.loc[enterprise_id, "industry_code"]),
                "q2_main_risk": float(stress_index.loc[enterprise_id, "q2_main_risk"]),
                "robust_risk": float(stress_index.loc[enterprise_id, "robust_risk"]),
                "scenario_id": "fixed_q2_under_robust",
                "scenario": "fixed_q2_under_robust",
                "risk_used": value("risk_used", float(robust_index.loc[enterprise_id, "risk_used"])),
                "q2_decision": str(q2_row.decision),
                "q2_selected_rate": float(rate_value) if pd.notna(rate_value) else np.nan,
                "q2_offered_loan_amount_10k": amount,
                "offered_loan_amount_10k": fixed_amount,
                "selected_rate": float(rate_value) if selected else np.nan,
                "acceptance_probability": value("acceptance_probability_A_i_r"),
                "customer_churn_mass": value("rating_mixture_customer_churn_mass"),
                "unit_expected_interest_per_10k": value("unit_expected_interest_per_10k"),
                "unit_expected_credit_loss_per_10k": value("unit_expected_credit_loss_per_10k"),
                "unit_expected_funding_cost_per_10k": value("unit_expected_funding_cost_per_10k"),
                "unit_expected_net_return_per_10k": value("unit_expected_net_return_per_10k"),
                "expected_disbursed_amount_10k": value("acceptance_probability_A_i_r") * fixed_amount,
                "expected_interest_income_10k": value("unit_expected_interest_per_10k") * fixed_amount,
                "expected_credit_loss_10k": value("unit_expected_credit_loss_per_10k") * fixed_amount,
                "expected_funding_cost_10k": value("unit_expected_funding_cost_per_10k") * fixed_amount,
                "expected_net_return_10k": value("unit_expected_net_return_per_10k") * fixed_amount,
                "source": "robust_result.candidates from q2._run_scenario; Q2 decision/rate/amount fixed",
            }
        )
    fixed = pd.DataFrame(rows).sort_values("enterprise_id", kind="stable").reset_index(drop=True)
    summary = {
        "scenario_id": "fixed_q2_under_robust",
        "scenario": "fixed_q2_under_robust",
        "enterprise_count": int(len(fixed)),
        "selected_count": int((fixed["offered_loan_amount_10k"] > TOLERANCE).sum()),
        "nominal_offered_amount_10k": float(fixed["offered_loan_amount_10k"].sum()),
        "expected_disbursed_amount_10k": float(fixed["expected_disbursed_amount_10k"].sum()),
        "expected_interest_income_10k": float(fixed["expected_interest_income_10k"].sum()),
        "expected_credit_loss_10k": float(fixed["expected_credit_loss_10k"].sum()),
        "expected_funding_cost_10k": float(fixed["expected_funding_cost_10k"].sum()),
        "expected_net_return_10k": float(fixed["expected_net_return_10k"].sum()),
        "candidate_economics_match": candidate_match,
        "risk_interpretation": "model-implied stress loss/net return under historical relative risk tendency; not real observed loss",
        "source": "Q2 selected decision/rate/amount + robust candidate economics",
    }
    summary["budget_difference_10k"] = summary["nominal_offered_amount_10k"] - 10000.0
    summary["decomposition_difference_10k"] = summary["expected_net_return_10k"] - (
        summary["expected_interest_income_10k"] - summary["expected_credit_loss_10k"] - summary["expected_funding_cost_10k"]
    )
    return fixed, summary


def _build_industry_exposure(strategies: pd.DataFrame, *, tolerance: float) -> pd.DataFrame:
    if strategies.empty:
        return pd.DataFrame()
    frame = strategies.copy()
    frame["offered_loan_amount_10k"] = pd.to_numeric(frame["offered_loan_amount_10k"], errors="raise")
    frame["expected_disbursed_amount_10k"] = pd.to_numeric(frame["expected_disbursed_amount_10k"], errors="raise")
    frame["risk_used"] = pd.to_numeric(frame["risk_used"], errors="raise")
    all_industries = sorted(frame["industry_code"].astype(str).unique().tolist())
    rows: list[dict[str, Any]] = []
    for scenario, group_scenario in frame.groupby("q3_scenario_id", sort=True):
        total = float(group_scenario["offered_loan_amount_10k"].sum())
        if total <= tolerance:
            raise RobustOptimizationError(f"industry exposure cannot compute zero-budget scenario: {scenario}")
        shares = {code: float(group_scenario.loc[group_scenario["industry_code"].astype(str) == code, "offered_loan_amount_10k"].sum()) / total for code in all_industries}
        hhi = float(sum(value * value for value in shares.values()))
        for code in all_industries:
            group = group_scenario.loc[group_scenario["industry_code"].astype(str) == code]
            offered = float(group["offered_loan_amount_10k"].sum())
            disbursed = float(group["expected_disbursed_amount_10k"].sum())
            weighted_risk = float((group["expected_disbursed_amount_10k"] * group["risk_used"]).sum() / disbursed) if disbursed > tolerance else 0.0
            rows.append(
                {
                    "scenario_id": str(scenario),
                    "industry_code": code,
                    "nominal_offered_amount_10k": offered,
                    "nominal_share": shares[code],
                    "selected_enterprise_count": int((group["offered_loan_amount_10k"] > tolerance).sum()),
                    "expected_disbursed_amount_10k": disbursed,
                    "weighted_risk_used": weighted_risk,
                    "industry_hhi": hhi,
                    "risk_interpretation": "historical invoice behavior relative default tendency; not a verified PD",
                }
            )
    return pd.DataFrame(rows).sort_values(["scenario_id", "industry_code"], kind="stable").reset_index(drop=True)


def _validate_results(
    root: Path,
    settings: Mapping[str, Any],
    q2_config: Mapping[str, Any],
    q2_scores: pd.DataFrame,
    stress: pd.DataFrame,
    q2_strategy: pd.DataFrame,
    results: Mapping[str, Mapping[str, Any]],
    scenario_strategies: Mapping[str, pd.DataFrame],
    fixed_summary: Mapping[str, Any],
    industry: pd.DataFrame,
    preflight: Mapping[str, Any],
    mapping: Mapping[str, Any],
    stress_contract: Mapping[str, Any],
    q2_hashes_before: Mapping[str, str],
    q2_hashes_after: Mapping[str, str],
    repeat_solve_matches: Mapping[str, bool],
) -> dict[str, Any]:
    atol = float(settings["numeric_tolerance"])
    checks: dict[str, bool] = {
        "five_scenarios_present": tuple(results) == SCENARIOS,
        "scenario_rows_302": all(len(scenario_strategies[name]) == 302 for name in SCENARIOS),
        "scenario_ids_unique": all(scenario_strategies[name]["enterprise_id"].is_unique for name in SCENARIOS),
        "exact_id_name_join": all(_pairs(q2_scores) == _pairs(scenario_strategies[name]) for name in SCENARIOS),
        "solver_optimal_all": all(bool(results[name]["diagnostics"].get("is_optimal")) and results[name]["diagnostics"].get("solver_status") == "optimal" for name in SCENARIOS),
        "solver_called_all": all(bool(results[name]["diagnostics"].get("solver_called")) for name in SCENARIOS),
        "fallback_false_all": all(not bool(results[name]["summary"].get("fallback_used")) for name in SCENARIOS),
        "validation_pass_all": all(results[name]["validation"].get("validation_status") == "PASS" for name in SCENARIOS),
        "budget_equality_all": all(abs(float(results[name]["summary"]["nominal_offered_amount_10k"]) - 10000.0) <= 10 * atol for name in SCENARIOS),
        "identity_reproduces_q2": _same_decision(scenario_strategies["identity"], q2_strategy, atol=atol),
        "robust_risk_is_max": np.allclose(stress["robust_risk"].to_numpy(dtype=float), stress[[STRESS_RISK_COLUMNS[name] for name in ("identity", "light", "medium", "severe")]].max(axis=1).to_numpy(dtype=float), atol=1e-12, rtol=0),
        "risk_used_matches_override": all(np.allclose(scenario_strategies[name]["risk_used"].to_numpy(dtype=float), stress.set_index("enterprise_id").loc[scenario_strategies[name]["enterprise_id"], STRESS_RISK_COLUMNS[name]].to_numpy(dtype=float), atol=atol, rtol=0) for name in SCENARIOS),
        "robust_severe_risk_equal": np.allclose(stress["robust_risk"].to_numpy(dtype=float), stress["severe_risk"].to_numpy(dtype=float), atol=1e-12, rtol=0),
        "robust_severe_strategy_equal": _same_decision(scenario_strategies["robust"], scenario_strategies["severe"], atol=atol),
        "risk_only_override": all(bool(scenario_strategies[name]["risk_override_only"].all()) for name in SCENARIOS),
        "rating_probabilities_and_pD_unchanged": all(
            all(
                np.allclose(
                    scenario_strategies[name].set_index("enterprise_id")[column].to_numpy(dtype=float),
                    q2_scores.set_index("enterprise_id").loc[scenario_strategies[name]["enterprise_id"], "rating_prob_D" if column == "pD" else column].to_numpy(dtype=float),
                    atol=atol,
                    rtol=0,
                )
                for column in (*q2.RATING_COLUMNS, "pD")
            )
            for name in SCENARIOS
        ),
        "D_rule_and_uncertainty_policy_unchanged": all(
            all(
                np.allclose(
                    scenario_strategies[name].set_index("enterprise_id")[column].to_numpy(dtype=float),
                    q2_strategy.set_index("enterprise_id").loc[scenario_strategies[name]["enterprise_id"], column].to_numpy(dtype=float),
                    atol=atol,
                    rtol=0,
                )
                for column in ("pD", "d_probability_reject_flag", "optimization_uncertainty_flag", "max_loan_limit_10k")
            )
            for name in SCENARIOS
        ),
        "fixed_q2_candidate_match": bool(fixed_summary.get("candidate_economics_match")),
        "fixed_q2_budget_identity": abs(float(fixed_summary["nominal_offered_amount_10k"]) - 10000.0) <= 10 * atol,
        "fixed_q2_net_decomposition": abs(float(fixed_summary["decomposition_difference_10k"])) <= 1e-8,
        "industry_share_identity": all(abs(float(group["nominal_share"].sum()) - 1.0) <= 1e-9 for _, group in industry.groupby("scenario_id")),
        "industry_hhi_identity": all(abs(float(group["industry_hhi"].iloc[0]) - float((group["nominal_share"] ** 2).sum())) <= 1e-9 for _, group in industry.groupby("scenario_id")),
        "q2_artifacts_unchanged": dict(q2_hashes_before) == dict(q2_hashes_after),
        "preflight_pass": preflight.get("status") == "PASS",
        "mapping_pass": mapping.get("status") == "PASS",
        "stress_pass": stress_contract.get("status") == "PASS",
        "deterministic_repeat_solve_all": tuple(repeat_solve_matches) == SCENARIOS and all(repeat_solve_matches.values()),
    }
    return {
        "contract_version": "q3_robust_optimization_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "q2_rows": int(len(q2_scores)),
            "stress_rows": int(len(stress)),
            "scenario_rows": {name: int(len(scenario_strategies[name])) for name in SCENARIOS},
            "unknown_industry_rows": int((stress["industry_code"] == "unknown").sum()),
        },
        "scenario_summary": {name: dict(results[name]["summary"]) for name in SCENARIOS},
        "fixed_q2_under_robust": dict(fixed_summary),
        "robust_interpretation": "robust_risk=max(identity/light/medium/severe) per enterprise; under the monotone interval this equals severe, so robust and severe may have the same optimal strategy; risk is not verified PD and loss/net return are model-implied stress quantities",
        "contracts": {
            "q2_preflight": preflight.get("status"),
            "q3_mapping": mapping.get("status"),
            "q3_stress": stress_contract.get("status"),
        },
        "hashes": {
            "q2_risk_rating_scores": sha256_file(q2.resolve_path(q2_config["inputs"]["risk_rating_scores"])),
            "q2_strategy_before": q2_hashes_before.get("q2_strategy"),
            "q2_strategy_after": q2_hashes_after.get("q2_strategy"),
            "q2_optimization_config_sha256": sha256_file(_resolve(root, settings["q2_optimization_config"])),
            "q3_config_sha256": sha256_file(root / "src" / "_internal" / "q3_config.yaml"),
            "q2_optimizer_code_sha256": sha256_file(root / "src" / "_internal" / "q2_credit_optimization.py"),
            "q3_robust_code_sha256": sha256_file(Path(__file__)),
        },
        "q2_artifact_hashes_before": dict(q2_hashes_before),
        "q2_artifact_hashes_after": dict(q2_hashes_after),
        "deterministic_repeat_solve": {"scenarios": dict(repeat_solve_matches), "all": all(repeat_solve_matches.values())},
        "inputs": {
            "q2_scores": _rel(root, q2.resolve_path(q2_config["inputs"]["risk_rating_scores"])),
            "q2_strategy": _rel(root, _resolve(root, settings["q2_strategy_input"])),
            "q3_stress": _rel(root, _resolve(root, settings["risk_input"])),
            "q2_config": _rel(root, _resolve(root, settings["q2_optimization_config"])),
        },
    }


def _q2_artifact_hashes(root: Path, q2_config: Mapping[str, Any]) -> dict[str, str]:
    manifest_path = q2.resolve_path(q2_config["outputs"]["manifest"])
    paths: dict[str, Path] = {
        "q2_manifest": manifest_path,
        "q2_risk_scores": q2.resolve_path(q2_config["inputs"]["risk_rating_scores"]),
        "q2_strategy": q2.resolve_path(q2_config["outputs"]["strategy_file"]),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in manifest.get("outputs", {}):
            paths[key] = _resolve(root, key)
    return {name: sha256_file(path) for name, path in sorted(paths.items()) if path.is_file()}


def _write_report(path: Path, contract: Mapping[str, Any], industry: pd.DataFrame) -> None:
    lines = [
        "# Q3 稳健信贷策略验证",
        "",
        f"状态：**{contract['status']}**",
        "",
        "本阶段仅通过 Q2 `_run_scenario` 重用既有 MILP、评级概率、D 规则、不确定性上限、附件3 A/B/C 流失曲线、LGD、资金成本、利率点与预算约束；Q3 只覆盖 `main_risk_score`。风险是历史发票行为相对违约倾向，不是真实 PD。",
        "",
        "稳健定义：对每户在阶段3的离散风险区间取 `max(identity, light, medium, severe)`；在当前单调压力变换下它等于 severe，因此两者相同不代表额外独立信息。",
        "",
        "## 五场景组合",
        "",
        "| scenario | solve | validation | offered(10k CNY) | expected loss(10k CNY) | expected net(10k CNY) | selected | solve seconds |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in contract["scenario_summary"].items():
        diagnostics_seconds = f"{float(summary.get('solve_time_seconds', 0.0)):.6f}"
        lines.append(
            f"| {name} | {summary.get('solve_status')} | {summary.get('validation_status')} | {float(summary.get('nominal_offered_amount_10k', 0.0)):.6f} | {float(summary.get('expected_credit_loss_10k', 0.0)):.6f} | {float(summary.get('expected_net_return_10k', 0.0)):.6f} | {int(summary.get('selected_enterprise_count', 0))} | {diagnostics_seconds} |"
        )
    fixed = contract["fixed_q2_under_robust"]
    lines.extend(
        [
            "",
            "## 固定Q2策略与稳健重优化",
            "",
            f"固定 Q2 决策在 robust 候选经济学下的模型隐含 expected loss={float(fixed['expected_credit_loss_10k']):.6f}、expected net={float(fixed['expected_net_return_10k']):.6f}（不是实际观测损失）；候选经济学来源为 Q2 `_run_scenario` 的 robust candidate 表。稳健重优化 net={float(fixed['robust_reoptimized_expected_net_return_10k']):.6f}，相对固定策略净收益变化={float(fixed['reoptimization_net_return_delta_10k']):.6f}，信用损失变化={float(fixed['reoptimization_credit_loss_delta_10k']):.6f}。",
            "",
            "## 行业资金迁移",
            "",
            "行业表的 nominal_share 对每个场景严格求和为1，HHI 为行业名义额度占比平方和；未知行业单列，不增加没有依据的行业上限。",
            "",
            "```json",
            json.dumps({"checks": contract["checks"], "hashes": contract["hashes"]}, ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            "输出：`data/processed/q3_robust_credit_strategy.csv`、`q3_scenario_credit_strategies.csv`、`q3_portfolio_scenario_summary.csv`、`q3_strategy_adjustments.csv`、`q3_industry_exposure_comparison.csv`、`q3_fixed_q2_strategy_under_robust_risk.csv`。",
        ]
    )
    write_text("\n".join(lines) + "\n", path)


def run_q3_robust_optimization(
    repo_root: Path | None = None,
    config_path: Path | None = None,
    *,
    preflight_contract: Mapping[str, Any] | None = None,
    mapping_contract: Mapping[str, Any] | None = None,
    stress_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the Q3 five-scenario adapter after stages 0--3 have passed."""

    root = (repo_root or _repo_root_from_file()).resolve()
    q3_config_path = (config_path or (root / "src" / "_internal" / "q3_config.yaml")).resolve()
    config = _read_yaml(q3_config_path)
    settings = _load_settings(config)
    if preflight_contract is None:
        try:
            preflight_contract = run_q3_preflight(root, q3_config_path)
        except PreflightError as exc:
            raise RobustOptimizationError(str(exc)) from exc
    if preflight_contract.get("status") != "PASS":
        raise RobustOptimizationError("Q3 optimize requires a PASS Q2 baseline preflight")
    if mapping_contract is None:
        try:
            mapping_contract = run_q3_industry_mapping(root, q3_config_path, preflight_contract=preflight_contract)
        except (IndustryMappingError, PreflightError) as exc:
            raise RobustOptimizationError(str(exc)) from exc
    if mapping_contract.get("status") != "PASS":
        raise RobustOptimizationError("Q3 optimize requires a PASS industry mapping")
    if stress_contract is None:
        try:
            stress_contract = run_q3_stress_scenarios(root, q3_config_path, preflight_contract=preflight_contract, mapping_contract=mapping_contract)
        except (StressScenarioError, IndustryMappingError, PreflightError) as exc:
            raise RobustOptimizationError(str(exc)) from exc
    if stress_contract.get("status") != "PASS":
        raise RobustOptimizationError("Q3 optimize requires a PASS stress scenario contract")

    q2_config_path = _resolve(root, settings["q2_optimization_config"])
    q2_config = q2.load_config(q2_config_path)
    q2_scores, stress, q2_strategy = _load_stress_and_q2(root, settings, q2_config)
    q2_hashes_before = _q2_artifact_hashes(root, q2_config)
    churn, evidence = q2._load_q1_accepted_churn(q2_config)
    evidence.update(q2._load_q2_threshold_evidence(q2_config))
    primary = q2._scenario_from_config(q2_config)
    if primary.risk_variant != "mean":
        raise RobustOptimizationError("Q3 v1 must reuse the Q2 primary mean-risk scenario")

    results: dict[str, dict[str, Any]] = {}
    scenario_strategies: dict[str, pd.DataFrame] = {}
    for name in SCENARIOS:
        stress_column = STRESS_RISK_COLUMNS[name]
        scores = _scenario_scores(q2_scores, stress, name)
        result = q2._run_scenario(scores, churn, primary, q2_config)
        decorated = _decorate_strategy(result["strategy"], stress, name)
        result["strategy"] = decorated
        q2_scenario_id = str(result["summary"].get("scenario_id", "primary"))
        result["summary"] = {
            **result["summary"],
            "scenario_id": name,
            "q3_scenario_id": name,
            "scenario": name,
            "q2_scenario_id": q2_scenario_id,
            "risk_override_column": stress_column,
            "solver_called": bool(result["diagnostics"].get("solver_called")),
            "solve_time_seconds": float(result["diagnostics"].get("solve_time_seconds", 0.0)),
            "fallback_used": bool(result["summary"].get("fallback_used", False)),
        }
        results[name] = result
        scenario_strategies[name] = decorated

    # A second complete solve is an acceptance gate for deterministic output;
    # it is deliberately not replaced by a constant or a hash-only shortcut.
    repeat_matches: dict[str, bool] = {}
    for name in SCENARIOS:
        repeat = q2._run_scenario(_scenario_scores(q2_scores, stress, name), churn, primary, q2_config)
        repeat["strategy"] = _decorate_strategy(repeat["strategy"], stress, name)
        repeat_matches[name] = _same_scenario_result(results[name], repeat, atol=float(settings["numeric_tolerance"]))

    scenario_table = pd.concat([scenario_strategies[name] for name in SCENARIOS], ignore_index=True)
    robust_strategy = scenario_strategies["robust"]
    adjustments = _build_adjustments(q2_strategy, robust_strategy)
    fixed_q2, fixed_summary = _build_fixed_q2_under_robust(q2_strategy, results["robust"], robust_strategy, stress, atol=float(settings["numeric_tolerance"]))
    fixed_summary["robust_reoptimized_expected_net_return_10k"] = float(results["robust"]["summary"]["expected_net_return_10k"])
    fixed_summary["robust_reoptimized_expected_credit_loss_10k"] = float(results["robust"]["summary"]["expected_credit_loss_10k"])
    fixed_summary["reoptimization_net_return_delta_10k"] = fixed_summary["robust_reoptimized_expected_net_return_10k"] - fixed_summary["expected_net_return_10k"]
    fixed_summary["reoptimization_credit_loss_delta_10k"] = fixed_summary["robust_reoptimized_expected_credit_loss_10k"] - fixed_summary["expected_credit_loss_10k"]
    industry = _build_industry_exposure(scenario_table, tolerance=float(settings["numeric_tolerance"]))

    robust_path = _resolve(root, settings["robust_strategy_output"])
    scenario_path = _resolve(root, settings["scenario_strategy_output"])
    summary_path = _resolve(root, settings["portfolio_summary_output"])
    adjustments_path = _resolve(root, settings["adjustments_output"])
    industry_path = _resolve(root, settings["industry_output"])
    fixed_path = _resolve(root, settings["fixed_q2_output"])
    validation_path = _resolve(root, settings["validation_output"])
    report_path = _resolve(root, settings["report_output"])
    write_csv(robust_strategy, robust_path)
    write_csv(scenario_table, scenario_path)
    write_csv(pd.DataFrame([results[name]["summary"] for name in SCENARIOS]), summary_path)
    write_csv(adjustments, adjustments_path)
    write_csv(industry, industry_path)
    write_csv(fixed_q2, fixed_path)
    q2_hashes_after = _q2_artifact_hashes(root, q2_config)
    contract = _validate_results(root, settings, q2_config, q2_scores, stress, q2_strategy, results, scenario_strategies, fixed_summary, industry, preflight_contract, mapping_contract, stress_contract, q2_hashes_before, q2_hashes_after, repeat_matches)
    contract.update(
        {
            "config_path": _rel(root, q3_config_path),
            "config_semantic_hash": config_hash(config),
            "outputs": {
                "robust_strategy": {"path": _rel(root, robust_path), "sha256": sha256_file(robust_path), "row_count": int(len(robust_strategy))},
                "scenario_strategies": {"path": _rel(root, scenario_path), "sha256": sha256_file(scenario_path), "row_count": int(len(scenario_table))},
                "portfolio_summary": {"path": _rel(root, summary_path), "sha256": sha256_file(summary_path), "row_count": int(len(SCENARIOS))},
                "adjustments": {"path": _rel(root, adjustments_path), "sha256": sha256_file(adjustments_path), "row_count": int(len(adjustments))},
                "industry_exposure": {"path": _rel(root, industry_path), "sha256": sha256_file(industry_path), "row_count": int(len(industry))},
                "fixed_q2_under_robust": {"path": _rel(root, fixed_path), "sha256": sha256_file(fixed_path), "row_count": int(len(fixed_q2))},
            },
            "q2_reuse_evidence": {
                "risk_loader": "q2._load_risk_rating_scores",
                "churn_loader": "q2._load_q1_accepted_churn",
                "threshold_loader": "q2._load_q2_threshold_evidence",
                "scenario_loader": "q2._scenario_from_config",
                "solver_entry": "q2._run_scenario",
                "q1_evidence": evidence,
            },
        }
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(contract, validation_path)
    _write_report(report_path, contract, industry)
    if contract["status"] != "PASS":
        raise RobustOptimizationError(f"Q3 robust optimization acceptance failed; see {_rel(root, validation_path)}")
    return contract


run_q3_optimize = run_q3_robust_optimization
