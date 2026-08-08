"""Probability-aware credit portfolio optimization for question two.

This module is an adapter layer for the 302 unlabeled enterprises.  It does
not call the deterministic-rating optimizer in ``credit_strategy.py`` and it
does not modify that file.  The only question-one credit artifact reused here
is the accepted attachment-3 isotonic churn table.

The decision model uses one binary variable ``x[i,k]`` and one continuous
nominal-loan variable ``s[i,k]`` for every eligible enterprise and every
observed attachment-3 interest-rate point.  Rating probabilities enter only
through the acceptance probability

    A_i(r) = sum_{g in {A,B,C}} pi[i,g] * (1 - L_g(r)).

The D probability is never assigned a churn curve.  A registered D-probability
rule rejects an enterprise before optimization, while registered uncertainty
rules cap its maximum nominal loan.  All sensitivity scenarios are solved
independently; an infeasible nominal equality is reported as infeasible and is
never silently changed to a budget cap.
"""

from __future__ import annotations

import json
import math
import platform
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from _internal.data_pipeline import (
    ROOT,
    config_hash,
    load_config,
    relative,
    resolve_path,
    sha256_file,
    write_csv,
    write_json,
    write_text,
)


SCORE_REQUIRED_COLUMNS = [
    "enterprise_id",
    "enterprise_name",
    "main_risk_score",
    "risk_instability_p90",
    "risk_instability_interval_width",
    "risk_disagreement_abs",
    "rating_entropy_normalized",
    "rating_prob_A",
    "rating_prob_B",
    "rating_prob_C",
    "rating_prob_D",
    "ood_flag",
    "novelty_score_max_tail_distance",
]
RATING_COLUMNS = ["rating_prob_A", "rating_prob_B", "rating_prob_C", "rating_prob_D"]
ABC_RATING_COLUMNS = ["rating_prob_A", "rating_prob_B", "rating_prob_C"]
VALID_BUDGET_DEFINITIONS = {
    "nominal_equality",
    "nominal_cap",
    "expected_disbursement_cap",
}
TOLERANCE = 1e-6


@dataclass(frozen=True)
class Q2Scenario:
    """Complete immutable parameter set for one independently solved case."""

    scenario_id: str
    scenario_kind: str
    sensitivity_axis: str
    sensitivity_level: str
    risk_variant: str
    lgd: float
    funding_cost_rate: float
    budget_definition: str
    d_probability_threshold: float
    uncertainty_cap_10k: float
    ood_novelty_threshold: float
    interval_width_threshold: float
    risk_disagreement_threshold: float
    rating_entropy_threshold: float

    def as_record(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "scenario_kind": self.scenario_kind,
            "sensitivity_axis": self.sensitivity_axis,
            "sensitivity_level": self.sensitivity_level,
            "risk_variant": self.risk_variant,
            "lgd": float(self.lgd),
            "funding_cost_rate": float(self.funding_cost_rate),
            "budget_definition": self.budget_definition,
            "d_probability_threshold": float(self.d_probability_threshold),
            "uncertainty_cap_10k": float(self.uncertainty_cap_10k),
            "ood_novelty_threshold": float(self.ood_novelty_threshold),
            "interval_width_threshold": float(self.interval_width_threshold),
            "risk_disagreement_threshold": float(self.risk_disagreement_threshold),
            "rating_entropy_threshold": float(self.rating_entropy_threshold),
        }


def _default_config_path() -> Path:
    return Path(__file__).resolve().with_name("q2_optimization_config.yaml")


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y", "pass"})


def _finite_numeric(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{label} column {column} contains non-finite values")


def _load_risk_rating_scores(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"question-two risk/rating score file not found: {path}")
    scores = pd.read_csv(path, encoding="utf-8-sig")
    missing = [column for column in SCORE_REQUIRED_COLUMNS if column not in scores.columns]
    if missing:
        raise ValueError(f"question-two score table is missing columns: {missing}")
    if len(scores) != 302:
        raise ValueError(f"question-two score table must contain 302 enterprises, got {len(scores)}")
    if scores["enterprise_id"].duplicated().any():
        raise ValueError("question-two enterprise ids are not unique")

    numeric_columns = [
        "main_risk_score",
        "risk_instability_p90",
        "risk_instability_interval_width",
        "risk_disagreement_abs",
        "rating_entropy_normalized",
        "novelty_score_max_tail_distance",
        *RATING_COLUMNS,
    ]
    _finite_numeric(scores, numeric_columns, "question-two score")
    for column in numeric_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(float)
    scores["ood_flag"] = _as_bool_series(scores["ood_flag"])

    if ((scores[RATING_COLUMNS] < -TOLERANCE) | (scores[RATING_COLUMNS] > 1 + TOLERANCE)).any().any():
        raise ValueError("rating probabilities must all lie in [0, 1]")
    probability_sum = scores[RATING_COLUMNS].sum(axis=1)
    if not np.allclose(probability_sum.to_numpy(), 1.0, atol=1e-8, rtol=0):
        raise ValueError("rating probabilities do not sum to one")
    if ((scores[["main_risk_score", "risk_instability_p90"]] < -TOLERANCE) | (scores[["main_risk_score", "risk_instability_p90"]] > 1 + TOLERANCE)).any().any():
        raise ValueError("risk values must lie in [0, 1]")
    if (scores["risk_instability_interval_width"] < -TOLERANCE).any():
        raise ValueError("risk interval widths must be non-negative")
    scores = scores.sort_values("enterprise_id", kind="stable").reset_index(drop=True)
    scores["rating_probability_sum_recomputed"] = scores[RATING_COLUMNS].sum(axis=1)
    return scores


def _load_q1_accepted_churn(opt_config: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load, rather than refit, the final accepted q1 attachment-3 result."""

    inputs = opt_config["inputs"]
    manifest_path = resolve_path(inputs["q1_final_manifest"])
    validation_path = resolve_path(inputs["q1_final_validation_report"])
    churn_path = resolve_path(inputs["q1_accepted_churn"])
    if not manifest_path.exists() or not validation_path.exists() or not churn_path.exists():
        raise FileNotFoundError("q1 final manifest, validation report, or accepted churn table is missing")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation_text = validation_path.read_text(encoding="utf-8")
    if manifest.get("final_validation_status") != "PASS" or "PASS" not in validation_text:
        raise ValueError("q1 attachment-3 churn source is not marked PASS in the final acceptance artifacts")
    expected_hash = manifest.get("input_hashes", {}).get("churn_fitted")
    actual_hash = sha256_file(churn_path)
    byte_hash_matches_manifest = not expected_hash or expected_hash == actual_hash

    churn = pd.read_csv(churn_path, encoding="utf-8-sig")
    semantic_match_final_delivery = False
    delivery_workbook_path = manifest_path.parent / "q1_final_delivery.xlsx"
    # The local CSV can have a harmless serialization-only difference (for
    # example a regenerated UTF-8/line-ending representation) while the final
    # accepted workbook and the fitted values remain identical.  Verify that
    # case against the final delivery workbook; never accept a changed curve
    # merely because its shape looks plausible.
    if not byte_hash_matches_manifest and delivery_workbook_path.exists():
        workbook_churn = pd.read_excel(delivery_workbook_path, sheet_name="ChurnFitted")
        semantic_match_final_delivery = bool(churn.equals(workbook_churn))
        if not semantic_match_final_delivery:
            raise ValueError("q1 churn CSV hash differs from the final manifest and its values differ from the accepted workbook")
    elif byte_hash_matches_manifest:
        semantic_match_final_delivery = True
    if not byte_hash_matches_manifest and not semantic_match_final_delivery:
        raise ValueError("q1 accepted churn file cannot be tied to q1 final acceptance artifacts")
    required = {
        "credit_rating",
        "interest_rate",
        "fitted_churn_rate",
        "acceptance_probability",
        "is_observed_rate",
        "fit_method",
    }
    missing = sorted(required.difference(churn.columns))
    if missing:
        raise ValueError(f"q1 accepted churn table is missing columns: {missing}")
    churn["credit_rating"] = churn["credit_rating"].astype(str).str.strip()
    churn["interest_rate"] = pd.to_numeric(churn["interest_rate"], errors="raise").astype(float)
    churn["fitted_churn_rate"] = pd.to_numeric(churn["fitted_churn_rate"], errors="raise").astype(float)
    churn["acceptance_probability"] = pd.to_numeric(churn["acceptance_probability"], errors="raise").astype(float)
    churn["is_observed_rate"] = _as_bool_series(churn["is_observed_rate"])
    if set(churn["credit_rating"]) != {"A", "B", "C"}:
        raise ValueError("q1 accepted churn source must contain exactly A/B/C curves; D is forbidden")
    if not churn["is_observed_rate"].all():
        raise ValueError("q2 must use only observed attachment-3 rate points")
    if ((churn[["fitted_churn_rate", "acceptance_probability"]] < -TOLERANCE) | (churn[["fitted_churn_rate", "acceptance_probability"]] > 1 + TOLERANCE)).any().any():
        raise ValueError("q1 churn/acceptance values are outside [0, 1]")
    if not np.allclose(churn["acceptance_probability"], 1.0 - churn["fitted_churn_rate"], atol=1e-8, rtol=0):
        raise ValueError("q1 acceptance probability is not 1 - fitted customer churn")
    rates_by_rating = {
        rating: sorted(churn.loc[churn["credit_rating"] == rating, "interest_rate"].unique().tolist())
        for rating in ["A", "B", "C"]
    }
    if len(set(map(tuple, rates_by_rating.values()))) != 1:
        raise ValueError("q1 accepted A/B/C curves do not share the same observed rate points")
    for rating, group in churn.groupby("credit_rating", sort=True):
        group = group.sort_values("interest_rate")
        if (np.diff(group["fitted_churn_rate"].to_numpy()) < -TOLERANCE).any():
            raise ValueError(f"q1 fitted customer churn curve is not nondecreasing for {rating}")

    evidence = {
        "source_file": relative(churn_path),
        "source_sha256": actual_hash,
        "source_byte_hash_matches_q1_manifest": byte_hash_matches_manifest,
        "semantic_match_q1_final_delivery_workbook": semantic_match_final_delivery,
        "q1_final_delivery_workbook": relative(delivery_workbook_path) if delivery_workbook_path.exists() else None,
        "q1_manifest": relative(manifest_path),
        "q1_validation_report": relative(validation_path),
        "q1_final_validation_status": manifest.get("final_validation_status"),
        "q1_expected_churn_sha256": expected_hash,
        "q1_credit_strategy_code_sha256": sha256_file(ROOT / "src" / "_internal" / "credit_strategy.py"),
        "curve_count": int(len(churn)),
        "rate_point_count": int(len(rates_by_rating["A"])),
        "ratings": ["A", "B", "C"],
        "D_curve_used": False,
        "fit_method_values": sorted(churn["fit_method"].astype(str).unique().tolist()),
    }
    return churn.sort_values(["interest_rate", "credit_rating"], kind="stable").reset_index(drop=True), evidence


def _load_q2_threshold_evidence(opt_config: Mapping[str, Any]) -> dict[str, Any]:
    """Tie optimization uncertainty thresholds to the registered q2 model config."""

    path = resolve_path(opt_config["inputs"]["q2_model_config"])
    q2_config = load_config(path)
    instability = q2_config.get("modeling", {}).get("instability", {})
    primary_uncertainty = opt_config["primary"]["uncertainty"]
    for key in ["risk_disagreement_threshold", "rating_entropy_threshold"]:
        if key not in instability or abs(float(instability[key]) - float(primary_uncertainty[key])) > TOLERANCE:
            raise ValueError(f"registered q2 threshold mismatch for {key}")
    return {
        "q2_model_config": relative(path),
        "q2_model_config_sha256": sha256_file(path),
        "ood_reference_lower_quantile": float(q2_config["eda"]["ood_quantile_lower"]),
        "ood_reference_upper_quantile": float(q2_config["eda"]["ood_quantile_upper"]),
        "ood_reference_source": str(q2_config["eda"].get("ood_threshold_source", "attachment1_123_only")),
        "ood_reference_rule": str(q2_config["eda"].get("ood_flag_rule", "any_primary_feature_outside_reference_quantile_interval")),
        "risk_disagreement_threshold_source": "q2_config.modeling.instability.risk_disagreement_threshold",
        "rating_entropy_threshold_source": "q2_config.modeling.instability.rating_entropy_threshold",
        "interval_width_threshold_source": "q2_optimization_config.primary.uncertainty.interval_width_threshold; one-factor sensitivity registered",
        "novelty_threshold_source": "q2_optimization_config.primary.uncertainty.ood_novelty_threshold; one-factor sensitivity registered",
    }


def _scenario_from_config(opt_config: Mapping[str, Any]) -> Q2Scenario:
    primary = opt_config["primary"]
    uncertainty = primary["uncertainty"]
    scenario = Q2Scenario(
        scenario_id="primary",
        scenario_kind="primary",
        sensitivity_axis="primary",
        sensitivity_level="baseline",
        risk_variant=str(primary["risk_variant"]),
        lgd=float(primary["lgd"]),
        funding_cost_rate=float(primary["funding_cost_rate"]),
        budget_definition=str(primary["budget_definition"]),
        d_probability_threshold=float(primary["d_probability_threshold"]),
        uncertainty_cap_10k=float(uncertainty["cap_10k"]),
        ood_novelty_threshold=float(uncertainty["ood_novelty_threshold"]),
        interval_width_threshold=float(uncertainty["interval_width_threshold"]),
        risk_disagreement_threshold=float(uncertainty["risk_disagreement_threshold"]),
        rating_entropy_threshold=float(uncertainty["rating_entropy_threshold"]),
    )
    if scenario.risk_variant not in {"mean", "p90"}:
        raise ValueError("risk_variant must be mean or p90")
    if scenario.budget_definition not in VALID_BUDGET_DEFINITIONS:
        raise ValueError(f"unsupported budget definition: {scenario.budget_definition}")
    if not 0 <= scenario.d_probability_threshold <= 1:
        raise ValueError("D probability threshold must be in [0, 1]")
    if not 0 <= scenario.lgd <= 1 or scenario.funding_cost_rate < 0:
        raise ValueError("LGD and funding cost rate are not legal")
    return scenario


def _scenario_id(axis: str, value: Any) -> str:
    text = str(value).replace(".", "p").replace("-", "m").replace(" ", "_")
    return f"sensitivity_{axis}_{text}"


def _sensitivity_scenarios(opt_config: Mapping[str, Any], primary: Q2Scenario) -> list[Q2Scenario]:
    sensitivity = opt_config["sensitivity"]
    scenarios: list[Q2Scenario] = [primary]
    axis_values = [
        ("risk_variant", sensitivity["risk_variants"]),
        ("lgd", sensitivity["lgd_values"]),
        ("funding_cost_rate", sensitivity["funding_cost_rates"]),
        ("d_probability_threshold", sensitivity["d_probability_thresholds"]),
        ("uncertainty_cap_10k", sensitivity["uncertainty_caps_10k"]),
        ("ood_novelty_threshold", sensitivity["ood_novelty_thresholds"]),
        ("interval_width_threshold", sensitivity["interval_width_thresholds"]),
        ("budget_definition", sensitivity["budget_definitions"]),
    ]
    for axis, values in axis_values:
        for value in values:
            kwargs: dict[str, Any] = {
                "scenario_id": _scenario_id(axis, value),
                "scenario_kind": "sensitivity",
                "sensitivity_axis": axis,
                "sensitivity_level": str(value),
            }
            if axis == "risk_variant":
                kwargs["risk_variant"] = str(value)
            elif axis == "lgd":
                kwargs["lgd"] = float(value)
            elif axis == "funding_cost_rate":
                kwargs["funding_cost_rate"] = float(value)
            elif axis == "d_probability_threshold":
                kwargs["d_probability_threshold"] = float(value)
            elif axis == "uncertainty_cap_10k":
                kwargs["uncertainty_cap_10k"] = float(value)
            elif axis == "ood_novelty_threshold":
                kwargs["ood_novelty_threshold"] = float(value)
            elif axis == "interval_width_threshold":
                kwargs["interval_width_threshold"] = float(value)
            elif axis == "budget_definition":
                kwargs["budget_definition"] = str(value)
            scenarios.append(replace(primary, **kwargs))
    return scenarios


def _build_policy(scores: pd.DataFrame, scenario: Q2Scenario, opt_config: Mapping[str, Any]) -> pd.DataFrame:
    policy = scores.copy()
    policy["risk_mean"] = policy["main_risk_score"].astype(float)
    policy["risk_p90"] = policy["risk_instability_p90"].astype(float)
    risk_column = "risk_mean" if scenario.risk_variant == "mean" else "risk_p90"
    policy["risk_used"] = policy[risk_column].astype(float).clip(0.0, 1.0)
    policy["risk_used_column"] = risk_column

    policy["pD"] = policy["rating_prob_D"].astype(float)
    policy["d_probability_reject_flag"] = policy["pD"] >= scenario.d_probability_threshold - TOLERANCE
    policy["uncertainty_reason_ood_optimization"] = policy["ood_flag"] | (
        policy["novelty_score_max_tail_distance"] >= scenario.ood_novelty_threshold - TOLERANCE
    )
    policy["uncertainty_reason_interval_optimization"] = policy["risk_instability_interval_width"] >= scenario.interval_width_threshold - TOLERANCE
    policy["uncertainty_reason_disagreement_optimization"] = policy["risk_disagreement_abs"] >= scenario.risk_disagreement_threshold - TOLERANCE
    policy["uncertainty_reason_entropy_optimization"] = policy["rating_entropy_normalized"] >= scenario.rating_entropy_threshold - TOLERANCE
    policy["optimization_uncertainty_flag"] = policy[
        [
            "uncertainty_reason_ood_optimization",
            "uncertainty_reason_interval_optimization",
            "uncertainty_reason_disagreement_optimization",
            "uncertainty_reason_entropy_optimization",
        ]
    ].any(axis=1)
    policy["uncertainty_cap_applied"] = policy["optimization_uncertainty_flag"] & ~policy["d_probability_reject_flag"]
    max_loan = float(opt_config["portfolio"]["max_loan_10k"])
    policy["max_loan_limit_10k"] = np.where(
        policy["d_probability_reject_flag"],
        0.0,
        np.where(policy["optimization_uncertainty_flag"], scenario.uncertainty_cap_10k, max_loan),
    )
    policy["eligible_for_optimization"] = (~policy["d_probability_reject_flag"]) & (
        policy["max_loan_limit_10k"] >= float(opt_config["portfolio"]["min_loan_10k"]) - TOLERANCE
    )
    policy["optimization_threshold_source"] = "q2_optimization_config_pre_registered"
    policy["d_rule_description"] = "pD >= tau_D => reject; D mass has no churn curve"
    policy["uncertainty_rule_description"] = "OOD/interval/disagreement/entropy flag => max-loan cap"
    return policy


def _build_candidate_economics(
    policy: pd.DataFrame,
    churn: pd.DataFrame,
    scenario: Q2Scenario,
    opt_config: Mapping[str, Any],
) -> pd.DataFrame:
    rates = sorted(pd.to_numeric(churn["interest_rate"], errors="raise").unique().tolist())
    churn_pivot = churn.pivot(index="interest_rate", columns="credit_rating", values="fitted_churn_rate").reindex(rates)
    if churn_pivot[["A", "B", "C"]].isna().any().any():
        raise ValueError("all A/B/C curves must be available at every observed rate point")
    churn_matrix = churn_pivot[["A", "B", "C"]].to_numpy(dtype=float)
    rating_probabilities = policy[ABC_RATING_COLUMNS].to_numpy(dtype=float)
    risk = policy["risk_used"].to_numpy(dtype=float)
    pD = policy["pD"].to_numpy(dtype=float)
    records: list[pd.DataFrame] = []
    for rate_index, rate in enumerate(rates):
        fitted_churn = churn_matrix[rate_index]
        acceptance = rating_probabilities @ (1.0 - fitted_churn)
        churn_mass = rating_probabilities @ fitted_churn
        if ((acceptance < -TOLERANCE) | (acceptance > 1.0 + TOLERANCE)).any():
            raise ValueError("A_i(r) is outside [0, 1]")
        if (acceptance > 1.0 - pD + 1e-8).any():
            raise ValueError("A_i(r) exceeds the non-D probability mass")
        block = policy[
            [
                "enterprise_id",
                "enterprise_name",
                "risk_mean",
                "risk_p90",
                "risk_used",
                "risk_used_column",
                "pD",
                "rating_prob_A",
                "rating_prob_B",
                "rating_prob_C",
                "rating_prob_D",
                "max_loan_limit_10k",
                "eligible_for_optimization",
                "optimization_uncertainty_flag",
                "uncertainty_cap_applied",
                "d_probability_reject_flag",
            ]
        ].copy()
        block["rate_index"] = int(rate_index)
        block["interest_rate"] = float(rate)
        block["interest_rate_percent"] = 100.0 * float(rate)
        block["fitted_churn_A"] = float(fitted_churn[0])
        block["fitted_churn_B"] = float(fitted_churn[1])
        block["fitted_churn_C"] = float(fitted_churn[2])
        block["rating_mixture_customer_churn_mass"] = churn_mass
        block["acceptance_probability_A_i_r"] = acceptance
        block["nonacceptance_probability_including_D"] = 1.0 - acceptance
        block["risk_variant"] = scenario.risk_variant
        block["lgd"] = scenario.lgd
        block["funding_cost_rate"] = scenario.funding_cost_rate
        block["unit_expected_interest_per_10k"] = acceptance * (1.0 - risk) * float(rate)
        block["unit_expected_credit_loss_per_10k"] = acceptance * risk * scenario.lgd
        block["unit_expected_funding_cost_per_10k"] = acceptance * scenario.funding_cost_rate
        block["unit_expected_net_return_per_10k"] = (
            block["unit_expected_interest_per_10k"]
            - block["unit_expected_credit_loss_per_10k"]
            - block["unit_expected_funding_cost_per_10k"]
        )
        block["candidate_rate_is_observed_attachment3_point"] = True
        records.append(block)
    candidates = pd.concat(records, ignore_index=True)
    candidates["eligible_candidate"] = candidates["eligible_for_optimization"] & (
        candidates["max_loan_limit_10k"] >= float(opt_config["portfolio"]["min_loan_10k"]) - TOLERANCE
    )
    return candidates


def _add_sparse_row(
    row_index: int,
    columns: Sequence[int],
    values: Sequence[float],
    row_indices: list[int],
    col_indices: list[int],
    data: list[float],
) -> None:
    for column, value in zip(columns, values):
        if value:
            row_indices.append(row_index)
            col_indices.append(int(column))
            data.append(float(value))


def _solve_scenario(
    policy: pd.DataFrame,
    candidates: pd.DataFrame,
    scenario: Q2Scenario,
    opt_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    budget = float(opt_config["portfolio"]["budget_10k"])
    min_loan = float(opt_config["portfolio"]["min_loan_10k"])
    max_loan = float(opt_config["portfolio"]["max_loan_10k"])
    eligible_policy = policy.loc[policy["eligible_for_optimization"]].sort_values("enterprise_id", kind="stable")
    enterprise_ids = eligible_policy["enterprise_id"].astype(str).tolist()
    rates = sorted(candidates["interest_rate"].astype(float).unique().tolist())
    candidate_eligible = candidates.loc[candidates["eligible_candidate"]].copy()
    candidate_eligible["enterprise_id"] = candidate_eligible["enterprise_id"].astype(str)
    candidate_lookup = {
        (str(row.enterprise_id), int(row.rate_index)): row
        for row in candidate_eligible.itertuples(index=False)
    }
    n_enterprises = len(enterprise_ids)
    n_rates = len(rates)
    variable_count = 2 * n_enterprises * n_rates
    x_index = {(i, k): i * n_rates + k for i in range(n_enterprises) for k in range(n_rates)}
    s_offset = n_enterprises * n_rates
    s_index = {(i, k): s_offset + i * n_rates + k for i in range(n_enterprises) for k in range(n_rates)}
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.full(variable_count, np.inf, dtype=float)
    integrality = np.zeros(variable_count, dtype=int)
    for index in x_index.values():
        upper_bounds[index] = 1.0
        integrality[index] = 1
    objective = np.zeros(variable_count, dtype=float)
    for i, enterprise_id in enumerate(enterprise_ids):
        cap = float(eligible_policy.iloc[i]["max_loan_limit_10k"])
        for k in range(n_rates):
            row = candidate_lookup[(enterprise_id, k)]
            objective[s_index[(i, k)]] = -float(row.unit_expected_net_return_per_10k)
            upper_bounds[s_index[(i, k)]] = cap

    capacity_max = float(eligible_policy["max_loan_limit_10k"].sum())
    diagnostics: dict[str, Any] = {
        **scenario.as_record(),
        "solver": "scipy.optimize.milp",
        "scipy_version": scipy.__version__,
        "solver_status_code": None,
        "solver_status": None,
        "solver_message": None,
        "is_optimal": False,
        "solver_called": False,
        "objective_value_minimization": np.nan,
        "objective_value_maximization": np.nan,
        "solve_time_seconds": 0.0,
        "mip_rel_gap": float(opt_config["solver"]["mip_rel_gap"]),
        "feasibility_tolerance": float(opt_config["solver"]["feasibility_tolerance"]),
        "variable_count": int(variable_count),
        "integer_variable_count": int(n_enterprises * n_rates),
        "constraint_count": None,
        "enterprise_count_total": int(len(policy)),
        "eligible_enterprise_count": int(n_enterprises),
        "d_reject_count": int(policy["d_probability_reject_flag"].sum()),
        "uncertainty_cap_count": int(policy["uncertainty_cap_applied"].sum()),
        "rate_point_count": int(n_rates),
        "budget_10k": budget,
        "capacity_max_10k": capacity_max,
        "fallback_used": False,
        "precheck_pass": True,
        "budget_projection_adjustment_10k": 0.0,
    }
    if scenario.budget_definition == "nominal_equality":
        if budget > TOLERANCE and budget < min_loan - TOLERANCE:
            diagnostics.update(
                solver_status_code=2,
                solver_status="infeasible",
                solver_message="Precheck: strict nominal equality is below one minimum loan",
                precheck_pass=False,
            )
            return pd.DataFrame(columns=["enterprise_id", "rate_index", "interest_rate", "offered_loan_amount_10k"]), diagnostics
        if budget > capacity_max + TOLERANCE:
            diagnostics.update(
                solver_status_code=2,
                solver_status="infeasible",
                solver_message="Precheck: strict nominal equality exceeds total legal loan capacity",
                precheck_pass=False,
            )
            return pd.DataFrame(columns=["enterprise_id", "rate_index", "interest_rate", "offered_loan_amount_10k"]), diagnostics

    row_indices: list[int] = []
    col_indices: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row_number = 0
    for i in range(n_enterprises):
        _add_sparse_row(
            row_number,
            [x_index[(i, k)] for k in range(n_rates)],
            [1.0] * n_rates,
            row_indices,
            col_indices,
            data,
        )
        lower.append(-np.inf)
        upper.append(1.0)
        row_number += 1
    for i, enterprise_id in enumerate(enterprise_ids):
        cap = float(eligible_policy.iloc[i]["max_loan_limit_10k"])
        for k in range(n_rates):
            _add_sparse_row(
                row_number,
                [s_index[(i, k)], x_index[(i, k)]],
                [1.0, -cap],
                row_indices,
                col_indices,
                data,
            )
            lower.append(-np.inf)
            upper.append(0.0)
            row_number += 1
            _add_sparse_row(
                row_number,
                [s_index[(i, k)], x_index[(i, k)]],
                [-1.0, min_loan],
                row_indices,
                col_indices,
                data,
            )
            lower.append(-np.inf)
            upper.append(0.0)
            row_number += 1
    budget_columns: list[int] = []
    budget_values: list[float] = []
    for i, enterprise_id in enumerate(enterprise_ids):
        for k in range(n_rates):
            budget_columns.append(s_index[(i, k)])
            if scenario.budget_definition == "expected_disbursement_cap":
                budget_values.append(float(candidate_lookup[(enterprise_id, k)].acceptance_probability_A_i_r))
            else:
                budget_values.append(1.0)
    _add_sparse_row(row_number, budget_columns, budget_values, row_indices, col_indices, data)
    if scenario.budget_definition == "nominal_equality":
        lower.append(budget)
        upper.append(budget)
    else:
        lower.append(-np.inf)
        upper.append(budget)
    row_number += 1
    matrix = coo_matrix((data, (row_indices, col_indices)), shape=(row_number, variable_count)).tocsr()
    diagnostics["constraint_count"] = int(row_number)
    start = time.perf_counter()
    diagnostics["solver_called"] = True
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, np.asarray(lower), np.asarray(upper)),
        options={
            "time_limit": float(opt_config["solver"]["time_limit_seconds"]),
            "mip_rel_gap": float(opt_config["solver"]["mip_rel_gap"]),
        },
    )
    diagnostics["solve_time_seconds"] = float(time.perf_counter() - start)
    diagnostics["solver_status_code"] = int(result.status)
    diagnostics["solver_message"] = str(result.message)
    diagnostics["is_optimal"] = int(result.status) == 0
    diagnostics["solver_status"] = "optimal" if result.status == 0 else ("infeasible" if result.status == 2 else "not_optimal")
    if result.fun is not None and np.isfinite(result.fun):
        diagnostics["objective_value_minimization"] = float(result.fun)
        diagnostics["objective_value_maximization"] = float(-result.fun)
    if result.x is None or result.status != 0:
        return pd.DataFrame(columns=["enterprise_id", "rate_index", "interest_rate", "offered_loan_amount_10k"]), diagnostics

    selected_records: list[dict[str, Any]] = []
    for i, enterprise_id in enumerate(enterprise_ids):
        for k, rate in enumerate(rates):
            x_value = float(result.x[x_index[(i, k)]])
            amount = max(0.0, float(result.x[s_index[(i, k)]]))
            if x_value > 0.5 or amount > TOLERANCE:
                row = candidate_lookup[(enterprise_id, k)]
                selected_records.append(
                    {
                        "enterprise_id": enterprise_id,
                        "rate_index": int(k),
                        "interest_rate": float(rate),
                        "offered_loan_amount_10k": amount,
                        "acceptance_probability_A_i_r": float(row.acceptance_probability_A_i_r),
                        "rating_mixture_customer_churn_mass": float(row.rating_mixture_customer_churn_mass),
                        "unit_expected_interest_per_10k": float(row.unit_expected_interest_per_10k),
                        "unit_expected_credit_loss_per_10k": float(row.unit_expected_credit_loss_per_10k),
                        "unit_expected_funding_cost_per_10k": float(row.unit_expected_funding_cost_per_10k),
                        "unit_expected_net_return_per_10k": float(row.unit_expected_net_return_per_10k),
                        "solver_x_value": x_value,
                    }
                )
    selected = pd.DataFrame(selected_records)
    if selected.empty:
        selected = pd.DataFrame(columns=["enterprise_id", "rate_index", "interest_rate", "offered_loan_amount_10k"])
    if scenario.budget_definition == "nominal_equality" and not selected.empty:
        residual = budget - float(selected["offered_loan_amount_10k"].sum())
        if abs(residual) <= 10 * TOLERANCE:
            for index in reversed(selected.index.tolist()):
                proposed = float(selected.loc[index, "offered_loan_amount_10k"]) + residual
                cap = float(eligible_policy.loc[eligible_policy["enterprise_id"].astype(str) == str(selected.loc[index, "enterprise_id"]), "max_loan_limit_10k"].iloc[0])
                if min_loan - TOLERANCE <= proposed <= cap + TOLERANCE:
                    selected.loc[index, "offered_loan_amount_10k"] = proposed
                    diagnostics["budget_projection_adjustment_10k"] = float(residual)
                    break
    diagnostics["selected_enterprise_count"] = int(len(selected))
    diagnostics["selected_nominal_amount_10k"] = float(selected["offered_loan_amount_10k"].sum()) if not selected.empty else 0.0
    return selected, diagnostics


def _build_strategy(
    policy: pd.DataFrame,
    selected: pd.DataFrame,
    scenario: Q2Scenario,
    opt_config: Mapping[str, Any],
) -> pd.DataFrame:
    strategy = policy.copy()
    strategy["scenario_id"] = scenario.scenario_id
    strategy["risk_variant"] = scenario.risk_variant
    strategy["lgd"] = scenario.lgd
    strategy["funding_cost_rate"] = scenario.funding_cost_rate
    strategy["budget_definition"] = scenario.budget_definition
    strategy["nominal_budget_10k"] = float(opt_config["portfolio"]["budget_10k"])
    strategy["decision"] = "不贷"
    strategy["decision_reason"] = np.where(
        strategy["d_probability_reject_flag"],
        "D概率规则拒贷",
        np.where(strategy["optimization_uncertainty_flag"], "高不确定性规则降额后未被优化选择", "优化未选择"),
    )
    strategy["selected_rate"] = np.nan
    strategy["selected_rate_percent"] = np.nan
    strategy["offered_loan_amount_10k"] = 0.0
    strategy["offered_loan_amount_yuan"] = 0.0
    strategy["selected_acceptance_probability"] = 0.0
    strategy["selected_customer_churn_mass"] = 0.0
    strategy["expected_disbursed_amount_10k"] = 0.0
    strategy["expected_interest_income_10k"] = 0.0
    strategy["expected_credit_loss_10k"] = 0.0
    strategy["expected_funding_cost_10k"] = 0.0
    strategy["expected_net_return_10k"] = 0.0
    strategy["unit_expected_net_return_per_10k"] = 0.0
    strategy["budget_projection_adjustment_10k"] = 0.0

    if not selected.empty:
        selected = selected.copy()
        selected["enterprise_id"] = selected["enterprise_id"].astype(str)
        strategy["enterprise_id"] = strategy["enterprise_id"].astype(str)
        selection = selected.set_index("enterprise_id")
        for index, row in strategy.iterrows():
            enterprise_id = str(row["enterprise_id"])
            if enterprise_id not in selection.index:
                continue
            chosen = selection.loc[enterprise_id]
            amount = float(chosen["offered_loan_amount_10k"])
            acceptance = float(chosen["acceptance_probability_A_i_r"])
            rate = float(chosen["interest_rate"])
            risk = float(row["risk_used"])
            expected_disbursed = acceptance * amount
            expected_interest = expected_disbursed * (1.0 - risk) * rate
            expected_loss = expected_disbursed * risk * scenario.lgd
            expected_funding = expected_disbursed * scenario.funding_cost_rate
            expected_net = expected_interest - expected_loss - expected_funding
            strategy.at[index, "decision"] = "贷"
            strategy.at[index, "decision_reason"] = (
                "优化选择；高不确定性额度上限"
                if bool(row["optimization_uncertainty_flag"])
                else "优化选择"
            )
            strategy.at[index, "selected_rate"] = rate
            strategy.at[index, "selected_rate_percent"] = 100.0 * rate
            strategy.at[index, "offered_loan_amount_10k"] = amount
            strategy.at[index, "offered_loan_amount_yuan"] = amount * 10000.0
            strategy.at[index, "selected_acceptance_probability"] = acceptance
            strategy.at[index, "selected_customer_churn_mass"] = float(chosen["rating_mixture_customer_churn_mass"])
            strategy.at[index, "expected_disbursed_amount_10k"] = expected_disbursed
            strategy.at[index, "expected_interest_income_10k"] = expected_interest
            strategy.at[index, "expected_credit_loss_10k"] = expected_loss
            strategy.at[index, "expected_funding_cost_10k"] = expected_funding
            strategy.at[index, "expected_net_return_10k"] = expected_net
            strategy.at[index, "unit_expected_net_return_per_10k"] = float(chosen["unit_expected_net_return_per_10k"])
    strategy["selected_rate_is_observed_attachment3_point"] = strategy["selected_rate"].notna()
    strategy["no_D_churn_extrapolation_used"] = True
    strategy["risk_is_historical_relative_default_tendency"] = True
    strategy["customer_churn_is_not_default_rate"] = True
    return strategy.sort_values("enterprise_id", kind="stable").reset_index(drop=True)


def _validate_strategy(
    strategy: pd.DataFrame,
    candidates: pd.DataFrame,
    selected: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    scenario: Q2Scenario,
    opt_config: Mapping[str, Any],
) -> dict[str, Any]:
    budget = float(opt_config["portfolio"]["budget_10k"])
    min_loan = float(opt_config["portfolio"]["min_loan_10k"])
    max_loan = float(opt_config["portfolio"]["max_loan_10k"])
    selected_rows = strategy.loc[strategy["offered_loan_amount_10k"] > TOLERANCE]
    checks: dict[str, bool] = {
        "enterprise_count_302": len(strategy) == 302,
        "enterprise_id_unique": strategy["enterprise_id"].is_unique,
        "rating_probability_nonnegative": bool((strategy[RATING_COLUMNS] >= -TOLERANCE).all().all()),
        "rating_probability_sum_one": bool(np.allclose(strategy[RATING_COLUMNS].sum(axis=1), 1.0, atol=1e-8, rtol=0)),
        "risk_values_legal": bool(((strategy[["risk_mean", "risk_p90", "risk_used"]] >= -TOLERANCE) & (strategy[["risk_mean", "risk_p90", "risk_used"]] <= 1 + TOLERANCE)).all().all()),
        "acceptance_values_legal": bool(((strategy["selected_acceptance_probability"] >= -TOLERANCE) & (strategy["selected_acceptance_probability"] <= 1 + TOLERANCE)).all()),
        "at_most_one_rate_per_enterprise": bool(selected_rows["enterprise_id"].value_counts().max() <= 1 if not selected_rows.empty else True),
        "selected_rate_is_observed": bool(strategy.loc[selected_rows.index, "selected_rate_is_observed_attachment3_point"].all() if not selected_rows.empty else True),
        "D_rule_not_violated": bool((selected_rows["pD"] < scenario.d_probability_threshold - TOLERANCE).all() if not selected_rows.empty else True),
        "selected_amounts_10_to_100": bool(((selected_rows["offered_loan_amount_10k"] >= min_loan - TOLERANCE) & (selected_rows["offered_loan_amount_10k"] <= max_loan + TOLERANCE)).all() if not selected_rows.empty else True),
        "unselected_amounts_zero": bool((strategy.loc[~strategy.index.isin(selected_rows.index), "offered_loan_amount_10k"].abs() <= TOLERANCE).all()),
        "numeric_decomposition": bool(
            np.allclose(
                strategy["expected_net_return_10k"],
                strategy["expected_interest_income_10k"] - strategy["expected_credit_loss_10k"] - strategy["expected_funding_cost_10k"],
                atol=1e-8,
                rtol=0,
            )
        ),
        "no_D_churn_curve": bool(strategy["no_D_churn_extrapolation_used"].all()),
    }
    offered = float(strategy["offered_loan_amount_10k"].sum())
    expected_disbursed = float(strategy["expected_disbursed_amount_10k"].sum())
    if scenario.budget_definition == "nominal_equality":
        checks["budget_constraint"] = bool(abs(offered - budget) <= 10 * TOLERANCE) if diagnostics.get("is_optimal") else False
    elif scenario.budget_definition == "nominal_cap":
        checks["budget_constraint"] = bool(offered <= budget + 10 * TOLERANCE)
    else:
        checks["budget_constraint"] = bool(expected_disbursed <= budget + 10 * TOLERANCE)
    status = "PASS" if all(checks.values()) else "FAIL"
    if diagnostics.get("solver_status") != "optimal":
        status = "NOT_EVALUATED_INFEASIBLE_OR_NOT_OPTIMAL"
    return {
        "scenario_id": scenario.scenario_id,
        "validation_status": status,
        **checks,
        "offered_sum_10k": offered,
        "expected_disbursed_sum_10k": expected_disbursed,
        "budget_10k": budget,
        "budget_difference_10k": offered - budget,
        "selected_count": int(len(selected_rows)),
    }


def _portfolio_summary(
    strategy: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    validation: Mapping[str, Any],
    scenario: Q2Scenario,
    opt_config: Mapping[str, Any],
) -> dict[str, Any]:
    budget = float(opt_config["portfolio"]["budget_10k"])
    selected = strategy.loc[strategy["offered_loan_amount_10k"] > TOLERANCE]
    offered = float(strategy["offered_loan_amount_10k"].sum())
    expected_disbursed = float(strategy["expected_disbursed_amount_10k"].sum())
    expected_interest = float(strategy["expected_interest_income_10k"].sum())
    expected_loss = float(strategy["expected_credit_loss_10k"].sum())
    expected_funding = float(strategy["expected_funding_cost_10k"].sum())
    expected_net = float(strategy["expected_net_return_10k"].sum())
    if expected_disbursed > TOLERANCE and not selected.empty:
        weighted_risk = float((selected["expected_disbursed_amount_10k"] * selected["risk_used"]).sum() / expected_disbursed)
        weighted_pD = float((selected["expected_disbursed_amount_10k"] * selected["pD"]).sum() / expected_disbursed)
        weighted_acceptance = float((selected["offered_loan_amount_10k"] * selected["selected_acceptance_probability"]).sum() / offered)
        weighted_churn_mass = float((selected["offered_loan_amount_10k"] * selected["selected_customer_churn_mass"]).sum() / offered)
    else:
        weighted_risk = weighted_pD = weighted_acceptance = weighted_churn_mass = np.nan
    summary = {
        **scenario.as_record(),
        "solve_status": diagnostics.get("solver_status"),
        "solver_status_code": diagnostics.get("solver_status_code"),
        "is_optimal": bool(diagnostics.get("is_optimal", False)),
        "validation_status": validation.get("validation_status"),
        "enterprise_count_total": int(len(strategy)),
        "eligible_enterprise_count": int(strategy["eligible_for_optimization"].sum()),
        "D_reject_count": int(strategy["d_probability_reject_flag"].sum()),
        "uncertainty_cap_count": int(strategy["uncertainty_cap_applied"].sum()),
        "selected_enterprise_count": int(len(selected)),
        "nominal_offered_amount_10k": offered,
        "nominal_offered_amount_yuan": offered * 10000.0,
        "expected_disbursed_amount_10k": expected_disbursed,
        "expected_interest_income_10k": expected_interest,
        "expected_credit_loss_10k": expected_loss,
        "expected_funding_cost_10k": expected_funding,
        "expected_net_return_10k": expected_net,
        "decomposition_difference_10k": expected_net - (expected_interest - expected_loss - expected_funding),
        "objective_value_solver_10k": diagnostics.get("objective_value_maximization", np.nan),
        "objective_value_recomputed_10k": expected_net,
        "objective_recomputation_difference_10k": expected_net - float(diagnostics.get("objective_value_maximization", np.nan)) if pd.notna(diagnostics.get("objective_value_maximization", np.nan)) else np.nan,
        "budget_10k": budget,
        "budget_difference_10k": offered - budget,
        "expected_disbursement_difference_10k": expected_disbursed - budget,
        "budget_constraint_pass": bool(validation.get("budget_constraint", False)),
        "budget_equality_required": scenario.budget_definition == "nominal_equality",
        "fallback_used": False,
        "weighted_risk_used": weighted_risk,
        "weighted_D_probability": weighted_pD,
        "weighted_acceptance_probability": weighted_acceptance,
        "weighted_customer_churn_mass_ABC": weighted_churn_mass,
        "risk_interpretation": "historical invoice behavior relative default tendency; not a verified PD",
        "customer_churn_interpretation": "attachment-3 customer churn/acceptance relationship; not default rate",
        "risk_interval_is_external_confidence_interval": False,
    }
    return summary


def _run_scenario(
    scores: pd.DataFrame,
    churn: pd.DataFrame,
    scenario: Q2Scenario,
    opt_config: Mapping[str, Any],
) -> dict[str, Any]:
    policy = _build_policy(scores, scenario, opt_config)
    candidates = _build_candidate_economics(policy, churn, scenario, opt_config)
    selected, diagnostics = _solve_scenario(policy, candidates, scenario, opt_config)
    strategy = _build_strategy(policy, selected, scenario, opt_config)
    validation = _validate_strategy(strategy, candidates, selected, diagnostics, scenario, opt_config)
    summary = _portfolio_summary(strategy, diagnostics, validation, scenario, opt_config)
    return {
        "policy": policy,
        "candidates": candidates,
        "selected": selected,
        "diagnostics": diagnostics,
        "strategy": strategy,
        "validation": validation,
        "summary": summary,
    }


def _markdown_table(df: pd.DataFrame, columns: Sequence[str], limit: int | None = None) -> str:
    table = df.loc[:, list(columns)].copy()
    if limit is not None:
        table = table.head(limit)
    headers = [str(column) for column in table.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in table.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                values.append("" if not np.isfinite(float(value)) else f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_figures(strategy: pd.DataFrame, summaries: pd.DataFrame, figures_dir: Path) -> list[Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(8.5, 5.2))
    no_loan = strategy["offered_loan_amount_10k"] <= TOLERANCE
    plt.scatter(
        strategy.loc[no_loan, "risk_used"],
        strategy.loc[no_loan, "offered_loan_amount_10k"],
        c=strategy.loc[no_loan, "pD"],
        cmap="viridis",
        alpha=0.45,
        label="不贷款",
    )
    loaned = ~no_loan
    if loaned.any():
        plt.scatter(
            strategy.loc[loaned, "risk_used"],
            strategy.loc[loaned, "offered_loan_amount_10k"],
            c=strategy.loc[loaned, "pD"],
            cmap="viridis",
            edgecolor="black",
            linewidth=0.4,
            label="贷款",
        )
    plt.xlabel("使用风险值（相对违约倾向）")
    plt.ylabel("名义贷款额度（万元）")
    plt.title("问题二主场景配置：风险、D级概率与贷款额度")
    plt.colorbar(label="评级概率 pD")
    plt.legend()
    plt.tight_layout()
    allocation_path = figures_dir / "q2_credit_allocation_risk.png"
    plt.savefig(allocation_path, dpi=300)
    plt.close()
    paths.append(allocation_path)

    axes = [
        "risk_variant",
        "lgd",
        "funding_cost_rate",
        "d_probability_threshold",
        "uncertainty_cap_10k",
        "ood_novelty_threshold",
        "interval_width_threshold",
        "budget_definition",
    ]
    axis_labels = {
        "risk_variant": "风险口径",
        "lgd": "LGD",
        "funding_cost_rate": "资金成本率",
        "d_probability_threshold": "D级概率阈值",
        "uncertainty_cap_10k": "不确定性额度上限（万元）",
        "ood_novelty_threshold": "OOD新颖度阈值",
        "interval_width_threshold": "预测区间宽度阈值",
        "budget_definition": "预算口径",
    }
    value_labels = {
        "mean": "均值",
        "p90": "P90",
        "expected_disbursement_cap": "预期放款上限",
        "nominal_cap": "名义额度上限",
        "nominal_equality": "名义额度等式",
    }
    fig, plot_axes = plt.subplots(4, 2, figsize=(13, 14), squeeze=False)
    for axis_name, axis in zip(axes, plot_axes.flat):
        group = summaries.loc[summaries["sensitivity_axis"] == axis_name].copy()
        if group.empty:
            axis.axis("off")
            continue
        group["plot_x"] = pd.factorize(group["sensitivity_level"], sort=True)[0]
        feasible = group["is_optimal"].astype(bool)
        axis.plot(group.loc[feasible, "plot_x"], group.loc[feasible, "expected_net_return_10k"], "o-")
        if (~feasible).any():
            axis.scatter(group.loc[~feasible, "plot_x"], np.zeros((~feasible).sum()), marker="x", color="red")
        axis.set_xticks(group["plot_x"])
        axis.set_xticklabels(
            [value_labels.get(str(value), str(value)) for value in group["sensitivity_level"]],
            rotation=35,
            ha="right",
            fontsize=8,
        )
        axis.set_title(axis_labels[axis_name])
        axis.set_ylabel("净收益（万元）")
        axis.grid(alpha=0.25)
    fig.suptitle("问题二单因素敏感性：预期净收益")
    fig.tight_layout()
    sensitivity_path = figures_dir / "q2_credit_sensitivity_net_return.png"
    fig.savefig(sensitivity_path, dpi=300)
    plt.close(fig)
    paths.append(sensitivity_path)
    return paths


def _write_report(
    report_path: Path,
    opt_config: Mapping[str, Any],
    evidence: Mapping[str, Any],
    primary: Mapping[str, Any],
    summaries: pd.DataFrame,
    diagnostics: pd.DataFrame,
    validation: Mapping[str, Any],
    output_index: pd.DataFrame,
) -> None:
    def f(value: Any, digits: int = 6) -> str:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return "NA"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return str(value)

    primary_status = str(primary.get("solve_status"))
    primary_ok = bool(primary.get("is_optimal")) and str(primary.get("validation_status")) == "PASS"
    summary_rows = summaries.loc[
        summaries["scenario_kind"].eq("sensitivity"),
        [
            "sensitivity_axis",
            "sensitivity_level",
            "risk_variant",
            "lgd",
            "funding_cost_rate",
            "budget_definition",
            "solve_status",
            "is_optimal",
            "selected_enterprise_count",
            "nominal_offered_amount_10k",
            "expected_disbursed_amount_10k",
            "expected_net_return_10k",
            "D_reject_count",
            "uncertainty_cap_count",
        ],
    ]
    lines = [
        "# 问题二：概率评级信贷组合优化报告",
        "",
        f"主场景状态：**{('PASS / optimal' if primary_ok else primary_status.upper())}**。",
        "",
        "## 1. 适配层边界与复用证据",
        "",
        "本轮新增 `src/_internal/q2_credit_optimization.py`，不调用或修改问题一确定评级的 `credit_strategy.py` 求解逻辑。问题二只使用评级概率；问题一仅提供已经验收的附件3 A/B/C 保序客户流失率曲线。",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"item": "q1 churn source", "value": evidence["source_file"]},
                    {"item": "q1 churn SHA256", "value": evidence["source_sha256"]},
                    {"item": "q1 churn byte hash matches manifest", "value": evidence["source_byte_hash_matches_q1_manifest"]},
                    {"item": "q1 churn semantic match final workbook", "value": evidence["semantic_match_q1_final_delivery_workbook"]},
                    {"item": "q1 final validation", "value": evidence["q1_final_validation_status"]},
                    {"item": "curve method", "value": ", ".join(evidence["fit_method_values"])},
                    {"item": "observed rate points", "value": evidence["rate_point_count"]},
                    {"item": "D churn curve used", "value": evidence["D_curve_used"]},
                    {"item": "q2 threshold config", "value": evidence["q2_model_config"]},
                    {"item": "OOD reference source", "value": evidence["ood_reference_source"]},
                    {"item": "OOD reference quantiles", "value": f"{evidence['ood_reference_lower_quantile']} / {evidence['ood_reference_upper_quantile']}"},
                    {"item": "q1 credit_strategy code hash", "value": evidence["q1_credit_strategy_code_sha256"]},
                ]
            ),
            ["item", "value"],
        ),
        "",
        "## 2. 概率评级经济学与目标",
        "",
        "对每个企业和附件3实际利率点，使用 `A_i(r)=sum_g pi_ig[1-L_g(r)]`，其中 `g` 只取 A、B、C。`pi_iD` 不外推、不插值客户流失率曲线，因此 `A_i(r) <= 1-pi_iD`。目标是最大化 `sum A_i(r_k) s_ik [(1-p_i)r_k-c_f-p_i*LGD]`。金额单位为万元；`x_ik` 为二元选择变量，`s_ik` 为连续名义额度。",
        "",
        "客户流失率/接受概率是附件3的客户行为关系，不是违约率。预期信用损失只由锁定风险分数 `p_i` 与 LGD 计算；本报告不把客户流失率解释成违约概率。",
        "",
        "## 3. 主场景规则（预先锁定）",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"rule": "D probability", "threshold": f"pD >= {opt_config['primary']['d_probability_threshold']}", "action": "reject; no D churn curve"},
                    {"rule": "OOD", "threshold": f"q2 ood_flag OR novelty >= {opt_config['primary']['uncertainty']['ood_novelty_threshold']}", "action": f"cap max loan at {opt_config['primary']['uncertainty']['cap_10k']} 10k CNY"},
                    {"rule": "risk interval width", "threshold": f"width >= {opt_config['primary']['uncertainty']['interval_width_threshold']}", "action": f"cap max loan at {opt_config['primary']['uncertainty']['cap_10k']} 10k CNY"},
                    {"rule": "risk disagreement", "threshold": f"abs difference >= {opt_config['primary']['uncertainty']['risk_disagreement_threshold']}", "action": f"cap max loan at {opt_config['primary']['uncertainty']['cap_10k']} 10k CNY"},
                    {"rule": "rating entropy", "threshold": f"normalized entropy >= {opt_config['primary']['uncertainty']['rating_entropy_threshold']}", "action": f"cap max loan at {opt_config['primary']['uncertainty']['cap_10k']} 10k CNY"},
                    {"rule": "budget", "threshold": "10000 10k CNY", "action": "strict nominal equality in primary"},
                ]
            ),
            ["rule", "threshold", "action"],
        ),
        "",
        "阈值来源：`q2_optimization_config.yaml` 的预注册政策；`ood_flag` 来自问题二仅用123家参考分布的1%/99%特征区间规则，风险区间、风险分歧和评级熵阈值沿用问题二模型配置中已登记的稳定性诊断。OOD和不确定性只触发额度上限，不改变风险分数；D概率规则才触发拒贷。敏感性表覆盖这些阈值和规则。",
        "",
        "## 4. 主场景组合摘要",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"metric": "solve status", "value": primary.get("solve_status")},
                    {"metric": "validation status", "value": primary.get("validation_status")},
                    {"metric": "total enterprises", "value": primary.get("enterprise_count_total")},
                    {"metric": "eligible enterprises", "value": primary.get("eligible_enterprise_count")},
                    {"metric": "D rejects", "value": primary.get("D_reject_count")},
                    {"metric": "uncertainty caps", "value": primary.get("uncertainty_cap_count")},
                    {"metric": "selected enterprises", "value": primary.get("selected_enterprise_count")},
                    {"metric": "nominal offered (10k CNY)", "value": f(primary.get("nominal_offered_amount_10k"))},
                    {"metric": "expected disbursed (10k CNY)", "value": f(primary.get("expected_disbursed_amount_10k"))},
                    {"metric": "expected interest (10k CNY)", "value": f(primary.get("expected_interest_income_10k"))},
                    {"metric": "expected credit loss (10k CNY)", "value": f(primary.get("expected_credit_loss_10k"))},
                    {"metric": "expected funding cost (10k CNY)", "value": f(primary.get("expected_funding_cost_10k"))},
                    {"metric": "expected net return (10k CNY)", "value": f(primary.get("expected_net_return_10k"))},
                    {"metric": "weighted risk used", "value": f(primary.get("weighted_risk_used"))},
                    {"metric": "weighted pD", "value": f(primary.get("weighted_D_probability"))},
                    {"metric": "weighted acceptance A", "value": f(primary.get("weighted_acceptance_probability"))},
                    {"metric": "strict budget difference (10k CNY)", "value": f(primary.get("budget_difference_10k"), 10)},
                ]
            ),
            ["metric", "value"],
        ),
        "",
        f"预算恒等式：`sum_i offered_i = {f(primary.get('nominal_offered_amount_10k'), 10)} 10k CNY`, 目标为 `10000.0000000000 10k CNY`，差额 `{f(primary.get('budget_difference_10k'), 10)}`。`fallback_used=False`；若主场景不可行，以上状态会明确为 `infeasible`，不会用 nominal cap 替代。",
        "",
        "## 5. 约束与求解器诊断",
        "",
        "主场景约束包括：每家至多一个实际利率点；获贷额度为0或10--100万元并受不确定性上限约束；D概率拒贷；评级概率、风险、接受概率均在合法区间；主场景名义额度严格等于10000万元。模型为连续额度+二元利率选择的MILP，求解器为已有 `scipy.optimize.milp`/HiGHS，未新增有序Logistic或优化依赖。",
        "",
        _markdown_table(
            pd.DataFrame(
                [
                    {"diagnostic": key, "value": diagnostics.iloc[0][key] if key in diagnostics.columns else "NA"}
                    for key in [
                        "solver_status",
                        "solver_status_code",
                        "solver_message",
                        "is_optimal",
                        "solver_called",
                        "variable_count",
                        "integer_variable_count",
                        "constraint_count",
                        "solve_time_seconds",
                        "capacity_max_10k",
                        "budget_projection_adjustment_10k",
                    ]
                ]
            ),
            ["diagnostic", "value"],
        ),
        "",
        "## 6. 敏感性分析",
        "",
        "以下为单因素敏感性：每次只改变一项，其他参数保持主场景值。`nominal_equality` 的任何不可行结果保留为不可行，不转成上限；`nominal_cap` 和 `expected_disbursement_cap` 仅作为口径敏感性，不替代主场景。",
        "",
        _markdown_table(
            summary_rows,
            [
                "sensitivity_axis",
                "sensitivity_level",
                "risk_variant",
                "lgd",
                "funding_cost_rate",
                "budget_definition",
                "solve_status",
                "is_optimal",
                "selected_enterprise_count",
                "nominal_offered_amount_10k",
                "expected_disbursed_amount_10k",
                "expected_net_return_10k",
                "D_reject_count",
                "uncertainty_cap_count",
            ],
        ),
        "",
        "## 7. 不确定性与使用边界",
        "",
        "问题二的302家没有真实标签，因此没有报告、也不能计算302家预测准确率。风险值只解释为历史发票行为对应的相对违约倾向；重复折模型得到的区间是不稳定性区间，不是真实外部置信区间。评级概率是模型概率，需结合D概率拒贷和不确定性额度上限使用。",
        "",
        "完整302家策略表、候选利率经济学、预算恒等式、组合分解、敏感性和诊断均以CSV落盘。",
        "",
        "## 8. 输出文件",
        "",
        _markdown_table(output_index, ["role", "path"]),
        "",
    ]
    write_text("\n".join(lines) + "\n", report_path)


def run_q2_optimization(config_path: Path | None = None) -> int:
    """Run primary q2 optimization plus one-factor sensitivity scenarios."""

    opt_path = config_path or _default_config_path()
    opt_config = load_config(opt_path)
    score_path = resolve_path(opt_config["inputs"]["risk_rating_scores"])
    scores = _load_risk_rating_scores(score_path)
    churn, evidence = _load_q1_accepted_churn(opt_config)
    evidence.update(_load_q2_threshold_evidence(opt_config))
    primary_scenario = _scenario_from_config(opt_config)
    scenarios = _sensitivity_scenarios(opt_config, primary_scenario)

    primary_result = _run_scenario(scores, churn, primary_scenario, opt_config)
    all_results = [primary_result]
    for scenario in scenarios[1:]:
        all_results.append(_run_scenario(scores, churn, scenario, opt_config))

    primary_strategy = primary_result["strategy"]
    primary_candidates = primary_result["candidates"]
    primary_validation = primary_result["validation"]
    primary_summary = primary_result["summary"]
    primary_diagnostics = primary_result["diagnostics"]

    output_config = opt_config["outputs"]
    strategy_path = resolve_path(output_config["strategy_file"])
    tables_dir = resolve_path(output_config["tables_dir"])
    figures_dir = resolve_path(output_config["figures_dir"])
    report_path = resolve_path(output_config["report"])
    manifest_path = resolve_path(output_config["manifest"])
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    strategy_table_path = tables_dir / "q2_credit_strategy.csv"
    candidate_path = tables_dir / "q2_primary_candidate_economics.csv"
    summary_df = pd.DataFrame([result["summary"] for result in all_results])
    diagnostics_df = pd.DataFrame([result["diagnostics"] for result in all_results])
    validation_df = pd.DataFrame([result["validation"] for result in all_results])
    budget_identity_df = summary_df[
        [
            "scenario_id",
            "scenario_kind",
            "sensitivity_axis",
            "sensitivity_level",
            "budget_definition",
            "solve_status",
            "is_optimal",
            "budget_10k",
            "nominal_offered_amount_10k",
            "budget_difference_10k",
            "expected_disbursed_amount_10k",
            "expected_disbursement_difference_10k",
            "budget_constraint_pass",
            "budget_equality_required",
            "fallback_used",
        ]
    ].copy()

    write_csv(primary_strategy, strategy_path)
    write_csv(primary_strategy, strategy_table_path)
    write_csv(primary_candidates, candidate_path)
    write_csv(summary_df, tables_dir / "q2_portfolio_summary.csv")
    write_csv(summary_df, tables_dir / "q2_optimization_sensitivity.csv")
    write_csv(budget_identity_df, tables_dir / "q2_budget_identity.csv")
    write_csv(diagnostics_df, tables_dir / "q2_solver_diagnostics.csv")
    write_csv(validation_df, tables_dir / "q2_strategy_validation.csv")

    output_index = pd.DataFrame(
        [
            {"role": "302-enterprise primary strategy", "path": relative(strategy_path)},
            {"role": "302-enterprise strategy table copy", "path": relative(strategy_table_path)},
            {"role": "primary candidate economics at observed rates", "path": relative(candidate_path)},
            {"role": "portfolio summary and one-factor sensitivity", "path": relative(tables_dir / "q2_portfolio_summary.csv")},
            {"role": "budget identity and budget-口径 diagnostics", "path": relative(tables_dir / "q2_budget_identity.csv")},
            {"role": "solver diagnostics", "path": relative(tables_dir / "q2_solver_diagnostics.csv")},
            {"role": "constraint validation", "path": relative(tables_dir / "q2_strategy_validation.csv")},
            {"role": "risk-allocation figure", "path": relative(figures_dir / "q2_credit_allocation_risk.png")},
            {"role": "sensitivity figure", "path": relative(figures_dir / "q2_credit_sensitivity_net_return.png")},
            {"role": "model report", "path": relative(report_path)},
        ]
    )
    write_csv(output_index, tables_dir / "q2_optimization_output_index.csv")
    figure_paths = _write_figures(primary_strategy, summary_df, figures_dir)
    _write_report(
        report_path,
        opt_config,
        evidence,
        primary_summary,
        summary_df,
        diagnostics_df,
        primary_validation,
        output_index,
    )

    output_files = [
        strategy_path,
        strategy_table_path,
        candidate_path,
        tables_dir / "q2_portfolio_summary.csv",
        tables_dir / "q2_optimization_sensitivity.csv",
        tables_dir / "q2_budget_identity.csv",
        tables_dir / "q2_solver_diagnostics.csv",
        tables_dir / "q2_strategy_validation.csv",
        tables_dir / "q2_optimization_output_index.csv",
        *figure_paths,
        report_path,
    ]
    output_hashes = {relative(path): sha256_file(path) for path in output_files}
    manifest = {
        "workflow": "question_two_probability_aware_credit_optimization",
        "config_path": relative(opt_path),
        "config_sha256": config_hash(opt_config),
        "random_seed": int(opt_config["random_seed"]),
        "model_type": "MILP with binary rate choice and continuous loan amount",
        "risk_model_inherited": "question_one locked elastic_net_logistic risk score; no random forest or composite score",
        "risk_interpretation": "historical invoice behavior relative default tendency; not a verified probability of default",
        "target_enterprise_count": int(len(scores)),
        "target_has_real_labels": False,
        "target_accuracy_reported": False,
        "q1_reuse_evidence": evidence,
        "threshold_evidence": {
            key: value
            for key, value in evidence.items()
            if key.startswith("q2_") or key.startswith("ood_") or key.endswith("_source")
        },
        "inputs": {
            "q2_risk_rating_scores": {"path": relative(score_path), "sha256": sha256_file(score_path)},
            "q1_final_manifest": {"path": evidence["q1_manifest"], "sha256": sha256_file(resolve_path(opt_config["inputs"]["q1_final_manifest"]))},
            "q1_final_validation_report": {"path": evidence["q1_validation_report"], "sha256": sha256_file(resolve_path(opt_config["inputs"]["q1_final_validation_report"]))},
            "q1_accepted_churn": {"path": evidence["source_file"], "sha256": evidence["source_sha256"]},
        },
        "primary_scenario": primary_summary,
        "primary_validation": primary_validation,
        "sensitivity_scenario_count": int(len(summary_df)),
        "sensitivity_axes": sorted(summary_df["sensitivity_axis"].astype(str).unique().tolist()),
        "fallback_used": False,
        "outputs": output_hashes,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "scipy": scipy.__version__,
        },
    }
    write_json(manifest, manifest_path)

    print(
        "q2 optimization completed: "
        f"status={primary_summary['solve_status']}, "
        f"validation={primary_summary['validation_status']}, "
        f"selected={primary_summary['selected_enterprise_count']}, "
        f"offered_10k={primary_summary['nominal_offered_amount_10k']:.10f}, "
        f"net_10k={primary_summary['expected_net_return_10k']:.6f}"
    )
    return 0
