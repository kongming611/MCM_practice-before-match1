"""Q3 deterministic industry-pressure risk overlay.

The overlay is deliberately a transparent stress transformation of Q2's
``main_risk_score``.  That Q2 value is a relative historical-invoice risk
tendency, not a calibrated probability of default; this module does not
rename it or claim a new target.  The only Q3 override is the risk column.

The official 2020Q1 National Bureau of Statistics industry growth data is a
retrospective external stress marker.  It contains March 2020 and therefore
is not used as a Q2 training input or as an ex-ante validation result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

try:  # Script entry points expose ``src`` on sys.path; tests import ``src``.
    from _internal.q3_baseline_preflight import PreflightError, run_q3_preflight
    from _internal.q3_industry_mapping import IndustryMappingError, run_q3_industry_mapping
except ModuleNotFoundError:  # pragma: no cover - import-path compatibility
    from src._internal.q3_baseline_preflight import PreflightError, run_q3_preflight
    from src._internal.q3_industry_mapping import IndustryMappingError, run_q3_industry_mapping


INDUSTRY_CODES = (
    "agriculture",
    "manufacturing_industry",
    "construction",
    "wholesale_retail",
    "transport_storage_post",
    "accommodation_catering",
    "finance",
    "real_estate",
    "information_software_it",
    "leasing_business_services",
    "other_services",
)
SCENARIOS = ("identity", "light", "medium", "severe")

NBS_SOURCE_URL = "https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900688.html"
NBS_RELEASE_DATE = "2020-04-18"
NBS_PERIOD = "2020Q1"
NBS_STANDARD = "GB/T4754-2017"
NBS_SOURCE_STATUS = "official_primary_preliminary"
NBS_INTERPRETATION = (
    "retrospective_external_stress_marker; includes 2020-03; after Q2 invoice "
    "window; not Q2 training input or ex_ante validation"
)

# These are the exact NBS Q1 industry year-on-year percentage-point values
# accepted at Q0.  Manufacturing deliberately uses manufacturing (-10.2),
# not the broader industry total (-8.5).
NBS_GROWTH_YOY_PCT = {
    "agriculture": -2.8,
    "manufacturing_industry": -10.2,
    "construction": -17.5,
    "wholesale_retail": -17.8,
    "transport_storage_post": -14.0,
    "accommodation_catering": -35.3,
    "finance": 6.0,
    "real_estate": -6.1,
    "information_software_it": 13.2,
    "leasing_business_services": -9.4,
    "other_services": -1.8,
}

INDUSTRY_PARAMETER_COLUMNS = (
    "industry_code",
    "growth_yoy_pct",
    "d_h_pp",
    "S_h",
    "c_pp",
    "rho",
    "lambda_identity",
    "lambda_light",
    "lambda_medium",
    "lambda_severe",
    "identity_multiplier",
    "light_multiplier",
    "medium_multiplier",
    "severe_multiplier",
    "growth_source_url",
    "source_release_date",
    "source_period",
    "source_standard",
    "source_status",
    "source_interpretation",
    "c_pp_source",
    "rho_source",
    "lambda_source",
    "unknown_policy",
    "parameter_status",
)

SCENARIO_OUTPUT_COLUMNS = (
    "enterprise_id",
    "enterprise_name",
    "industry_code",
    "mapping_status",
    "confidence",
    "mapping_confidence",
    "q2_main_risk",
    "q2_risk_interpretation",
    "growth_yoy_pct",
    "growth_source",
    "d_h_pp",
    "S_h",
    "identity_multiplier",
    "identity_risk",
    "light_multiplier",
    "light_risk",
    "medium_multiplier",
    "medium_risk",
    "severe_multiplier",
    "severe_risk",
    "robust_risk",
    "unknown_industry_flag",
    "industry_high_uncertainty_flag",
    "high_uncertainty_flag",
    "unknown_policy",
    "growth_source_url",
    "source_release_date",
    "source_period",
    "source_standard",
    "source_status",
    "source_interpretation",
    "risk_override_only",
)


class StressScenarioError(RuntimeError):
    """Raised when Q3 stress scenario construction fails its acceptance gate."""


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


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise StressScenarioError(f"Q3 config must be a mapping: {path}")
    return value


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise StressScenarioError(f"required table does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise StressScenarioError(f"required table is empty: {path}")
    return rows


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _as_float(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise StressScenarioError(f"{field} must be numeric, got {value!r}") from exc
    if not math.isfinite(number):
        raise StressScenarioError(f"{field} must be finite, got {value!r}")
    return number


def _as_int_flag(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def clip_risk(value: float, eps: float = 1.0e-6) -> float:
    """Clip a risk tendency to the Q3 legal interval ``[0, 1-eps]``."""

    return min(max(float(value), 0.0), 1.0 - float(eps))


def _load_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = config.get("stress_scenarios")
    if not isinstance(settings, Mapping):
        raise StressScenarioError("q3_config.stress_scenarios is missing")
    paths = config.get("industry_mapping")
    if not isinstance(paths, Mapping):
        raise StressScenarioError("q3_config.industry_mapping is missing")

    values = {
        "input": settings.get("input", "data/processed/q2_risk_rating_scores.csv"),
        "industry_mapping_input": settings.get(
            "industry_mapping_input", paths.get("output", "data/processed/q3_enterprise_industry_mapping.csv")
        ),
        "risk_output": settings.get("risk_output", "data/processed/q3_scenario_risk_scores.csv"),
        "industry_output": settings.get(
            "industry_output", "outputs/q3/tables/q3_industry_shock_params.csv"
        ),
        "validation_output": settings.get(
            "validation_output", "outputs/q3/reports/q3_stress_validation.json"
        ),
        "report_output": settings.get("report_output", "outputs/q3/reports/q3_stress_validation.md"),
        "c_pp": float(settings.get("c_pp", 40.0)),
        "rho": float(settings.get("rho", 0.25)),
        "eps": float(settings.get("eps", 1.0e-6)),
        "lambdas": settings.get("lambdas", {"identity": 0.0, "light": 0.5, "medium": 1.0, "severe": 1.5}),
        "unknown_policy": str(settings.get("unknown_policy", "max_known_S")),
    }
    lambdas = values["lambdas"]
    if not isinstance(lambdas, Mapping) or set(lambdas) != set(SCENARIOS):
        raise StressScenarioError("stress_scenarios.lambdas must cover identity/light/medium/severe")
    values["lambdas"] = {name: float(lambdas[name]) for name in SCENARIOS}
    if values["c_pp"] <= 0 or not (0 <= values["rho"] <= 1) or not (0 < values["eps"] < 1):
        raise StressScenarioError("invalid c_pp/rho/eps in stress_scenarios")
    if values["unknown_policy"] != "max_known_S":
        raise StressScenarioError("Q3 v1 unknown_policy must be max_known_S")
    return values


def load_stress_settings(config_path: Path | None = None) -> dict[str, Any]:
    """Load stress settings for unit-level tests and collaborators."""

    path = config_path or (_repo_root_from_file() / "src" / "_internal" / "q3_config.yaml")
    config = _read_yaml(path)
    return _load_settings(config)


def calculate_industry_stress(
    growth_yoy_pct: float,
    *,
    c_pp: float = 40.0,
    rho: float = 0.25,
    eps: float = 1.0e-6,
    lambdas: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Calculate ``d_h_pp``, ``S_h`` and scenario multipliers for one industry."""

    growth = _as_float(growth_yoy_pct, "growth_yoy_pct")
    c_value = _as_float(c_pp, "c_pp")
    rho_value = _as_float(rho, "rho")
    eps_value = _as_float(eps, "eps")
    if c_value <= 0 or not 0 <= rho_value <= 1 or not 0 < eps_value < 1:
        raise StressScenarioError("invalid stress calculation parameters")
    lambda_values = {"identity": 0.0, "light": 0.5, "medium": 1.0, "severe": 1.5}
    if lambdas is not None:
        lambda_values = {name: _as_float(lambdas[name], f"lambda_{name}") for name in SCENARIOS}
    d_value = max(0.0, -growth)
    s_value = min(max(d_value / c_value, 0.0), 1.0)
    multipliers: dict[str, float] = {}
    exposures: dict[str, float] = {}
    for scenario in SCENARIOS:
        exposure = min(max(lambda_values[scenario] * s_value, 0.0), 1.0)
        exposures[scenario] = exposure
        multipliers[scenario] = 1.0 + rho_value * exposure
    return {
        "growth_yoy_pct": growth,
        "d_h_pp": d_value,
        "S_h": s_value,
        "exposures": exposures,
        "multipliers": multipliers,
        "c_pp": c_value,
        "rho": rho_value,
        "eps": eps_value,
    }


def apply_risk_stress(
    q2_main_risk: float,
    stress: Mapping[str, Any],
    *,
    eps: float = 1.0e-6,
) -> dict[str, float]:
    """Apply scenario multipliers to one Q2 relative risk tendency."""

    baseline = clip_risk(_as_float(q2_main_risk, "q2_main_risk"), eps)
    multipliers = stress["multipliers"]
    risks = {scenario: clip_risk(baseline * float(multipliers[scenario]), eps) for scenario in SCENARIOS}
    # Keep the identity value exactly tied to the Q2 float and not to an
    # independently rounded multiplication.
    risks["identity"] = baseline
    risks["robust"] = max(risks[scenario] for scenario in ("identity", "light", "medium", "severe"))
    return risks


def _industry_rows(settings: Mapping[str, Any]) -> tuple[list[dict[str, Any]], float]:
    known: dict[str, dict[str, Any]] = {}
    for code in INDUSTRY_CODES:
        stress = calculate_industry_stress(
            NBS_GROWTH_YOY_PCT[code],
            c_pp=float(settings["c_pp"]),
            rho=float(settings["rho"]),
            eps=float(settings["eps"]),
            lambdas=settings["lambdas"],
        )
        known[code] = stress
    max_known_s = max(float(item["S_h"]) for item in known.values())
    rows: list[dict[str, Any]] = []
    for code in INDUSTRY_CODES:
        stress = known[code]
        rows.append(_parameter_row(code, stress, settings, source_status=NBS_SOURCE_STATUS))
    unknown_stress = calculate_industry_stress(
        0.0,
        c_pp=float(settings["c_pp"]),
        rho=float(settings["rho"]),
        eps=float(settings["eps"]),
        lambdas=settings["lambdas"],
    )
    unknown_stress = dict(unknown_stress)
    unknown_stress["d_h_pp"] = max_known_s * float(settings["c_pp"])
    unknown_stress["S_h"] = max_known_s
    unknown_stress["growth_yoy_pct"] = None
    unknown_stress["exposures"] = {
        scenario: min(max(float(settings["lambdas"][scenario]) * max_known_s, 0.0), 1.0)
        for scenario in SCENARIOS
    }
    unknown_stress["multipliers"] = {
        scenario: 1.0 + float(settings["rho"]) * unknown_stress["exposures"][scenario]
        for scenario in SCENARIOS
    }
    rows.append(_parameter_row("unknown", unknown_stress, settings, source_status="derived_policy_max_known_S"))
    return rows, max_known_s


def _parameter_row(
    code: str,
    stress: Mapping[str, Any],
    settings: Mapping[str, Any],
    *,
    source_status: str,
) -> dict[str, Any]:
    multipliers = stress["multipliers"]
    return {
        "industry_code": code,
        "growth_yoy_pct": stress.get("growth_yoy_pct"),
        "d_h_pp": stress["d_h_pp"],
        "S_h": stress["S_h"],
        "c_pp": settings["c_pp"],
        "rho": settings["rho"],
        "lambda_identity": settings["lambdas"]["identity"],
        "lambda_light": settings["lambdas"]["light"],
        "lambda_medium": settings["lambdas"]["medium"],
        "lambda_severe": settings["lambdas"]["severe"],
        "identity_multiplier": multipliers["identity"],
        "light_multiplier": multipliers["light"],
        "medium_multiplier": multipliers["medium"],
        "severe_multiplier": multipliers["severe"],
        "growth_source_url": NBS_SOURCE_URL,
        "source_release_date": NBS_RELEASE_DATE,
        "source_period": NBS_PERIOD,
        "source_standard": NBS_STANDARD,
        "source_status": source_status,
        "source_interpretation": NBS_INTERPRETATION,
        "c_pp_source": "modeling_assumption",
        "rho_source": "modeling_assumption",
        "lambda_source": "modeling_assumption",
        "unknown_policy": settings["unknown_policy"] if code == "unknown" else "not_applicable",
        "parameter_status": "modeling_assumption" if code == "unknown" else "official_growth_plus_modeling_assumption",
    }


def _build_enterprise_rows(
    q2_rows: list[dict[str, str]],
    mapping_rows: list[dict[str, str]],
    parameter_rows: list[dict[str, Any]],
    settings: Mapping[str, Any],
) -> list[dict[str, Any]]:
    params = {row["industry_code"]: row for row in parameter_rows}
    q2_by_id = {str(row.get("enterprise_id", "")).strip(): row for row in q2_rows}
    mapping_by_id = {str(row.get("enterprise_id", "")).strip(): row for row in mapping_rows}
    output: list[dict[str, Any]] = []
    for enterprise_id in sorted(q2_by_id):
        q2_row = q2_by_id[enterprise_id]
        mapping = mapping_by_id[enterprise_id]
        code = str(mapping.get("industry_code", "unknown"))
        parameter = params[code if code in params else "unknown"]
        q2_risk = _as_float(q2_row.get("main_risk_score"), "main_risk_score")
        stress = {
            "multipliers": {
                scenario: _as_float(parameter[f"{scenario}_multiplier"], f"{scenario}_multiplier")
                for scenario in SCENARIOS
            }
        }
        risks = apply_risk_stress(q2_risk, stress, eps=float(settings["eps"]))
        output.append(
            {
                "enterprise_id": enterprise_id,
                "enterprise_name": str(q2_row.get("enterprise_name", "")),
                "industry_code": code,
                "mapping_status": str(mapping.get("mapping_status", "")),
                "confidence": _as_float(mapping.get("confidence", 0.0), "mapping confidence"),
                "mapping_confidence": _as_float(mapping.get("confidence", 0.0), "mapping confidence"),
                "q2_main_risk": q2_risk,
                "q2_risk_interpretation": str(
                    q2_row.get(
                        "risk_interpretation",
                        "historical invoice behavior relative default tendency; no observed default labels",
                    )
                ),
                "growth_yoy_pct": parameter["growth_yoy_pct"],
                "growth_source": NBS_SOURCE_URL,
                "d_h_pp": parameter["d_h_pp"],
                "S_h": parameter["S_h"],
                "identity_multiplier": parameter["identity_multiplier"],
                "identity_risk": risks["identity"],
                "light_multiplier": parameter["light_multiplier"],
                "light_risk": risks["light"],
                "medium_multiplier": parameter["medium_multiplier"],
                "medium_risk": risks["medium"],
                "severe_multiplier": parameter["severe_multiplier"],
                "severe_risk": risks["severe"],
                "robust_risk": risks["robust"],
                "unknown_industry_flag": int(code == "unknown"),
                "industry_high_uncertainty_flag": int(code == "unknown" or float(mapping.get("confidence", 0.0)) < 1.0),
                "high_uncertainty_flag": _as_int_flag(q2_row.get("high_uncertainty_flag", 0)),
                "unknown_policy": settings["unknown_policy"] if code == "unknown" else "not_applicable",
                "growth_source_url": NBS_SOURCE_URL,
                "source_release_date": NBS_RELEASE_DATE,
                "source_period": NBS_PERIOD,
                "source_standard": NBS_STANDARD,
                "source_status": parameter["source_status"],
                "source_interpretation": NBS_INTERPRETATION,
                "risk_override_only": True,
            }
        )
    return output


def _risk_summary(rows: list[dict[str, Any]], parameter_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenario_summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        values = [float(row[f"{scenario}_risk"]) for row in rows]
        baseline = [float(row["q2_main_risk"]) for row in rows]
        changes = [value - base for value, base in zip(values, baseline)]
        scenario_summary[scenario] = {
            "mean_risk": sum(values) / len(values),
            "min_risk": min(values),
            "max_risk": max(values),
            "mean_change_from_q2": sum(changes) / len(changes),
            "max_change_from_q2": max(changes),
            "clipped_at_upper_bound_count": sum(value >= 1.0 - 1.0e-6 for value in values),
        }
    by_industry: dict[str, Any] = {}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["industry_code"])].append(row)
    for code in sorted(groups):
        group = groups[code]
        by_industry[code] = {
            "count": len(group),
            "q2_mean_risk": sum(float(row["q2_main_risk"]) for row in group) / len(group),
            "severe_mean_risk": sum(float(row["severe_risk"]) for row in group) / len(group),
            "robust_mean_risk": sum(float(row["robust_risk"]) for row in group) / len(group),
            "max_robust_change_from_q2": max(
                float(row["robust_risk"]) - float(row["q2_main_risk"]) for row in group
            ),
        }
    return {
        "scenario": scenario_summary,
        "by_industry": by_industry,
        "parameter_rows": len(parameter_rows),
    }


def _validate(
    q2_rows: list[dict[str, str]],
    mapping_rows: list[dict[str, str]],
    output_rows: list[dict[str, Any]],
    parameter_rows: list[dict[str, Any]],
    settings: Mapping[str, Any],
    repeat_rows: list[dict[str, Any]],
    max_known_s: float,
    repo_root: Path,
) -> dict[str, Any]:
    q2_pairs = {(str(row.get("enterprise_id", "")).strip(), str(row.get("enterprise_name", ""))) for row in q2_rows}
    mapping_pairs = {(str(row.get("enterprise_id", "")).strip(), str(row.get("enterprise_name", ""))) for row in mapping_rows}
    output_pairs = {(str(row.get("enterprise_id", "")), str(row.get("enterprise_name", ""))) for row in output_rows}
    q2_ids = [str(row.get("enterprise_id", "")).strip() for row in q2_rows]
    output_ids = [str(row.get("enterprise_id", "")) for row in output_rows]
    eps = float(settings["eps"])
    numeric_fields = [
        "q2_main_risk", "growth_yoy_pct", "d_h_pp", "S_h",
        *(f"{scenario}_multiplier" for scenario in SCENARIOS),
        *(f"{scenario}_risk" for scenario in SCENARIOS), "robust_risk",
    ]
    finite_legal = all(
        (row[field] is None or (isinstance(row[field], (int, float)) and math.isfinite(float(row[field]))))
        for row in output_rows
        for field in numeric_fields
    )
    risk_legal = all(
        0.0 <= float(row[f"{scenario}_risk"]) <= 1.0 - eps
        for row in output_rows
        for scenario in SCENARIOS
    )
    multiplier_legal = all(
        float(row[f"{scenario}_multiplier"]) >= 1.0 for row in output_rows for scenario in SCENARIOS
    )
    identity_exact = all(float(row["identity_risk"]) == float(row["q2_main_risk"]) for row in output_rows)
    scenario_order = all(
        float(row["light_risk"]) <= float(row["medium_risk"]) <= float(row["severe_risk"]) <= float(row["robust_risk"])
        for row in output_rows
    )
    robust_exact = all(
        float(row["robust_risk"]) == max(float(row[f"{scenario}_risk"]) for scenario in SCENARIOS)
        for row in output_rows
    )
    positive_growth_neutral = all(
        row["growth_yoy_pct"] is None
        or float(row["growth_yoy_pct"]) <= 0
        or all(float(row[f"{scenario}_multiplier"]) == 1.0 for scenario in SCENARIOS)
        for row in output_rows
    )
    unknown_rows = [row for row in output_rows if row["industry_code"] == "unknown"]
    unknown_max_policy = (
        len(unknown_rows) == 145
        and bool(unknown_rows)
        and all(
            float(row["S_h"]) == max_known_s
            and row["unknown_policy"] == "max_known_S"
            for row in unknown_rows
        )
    )
    source_fields = (
        "growth_source_url", "source_release_date", "source_period", "source_standard",
        "source_status", "source_interpretation",
    )
    source_metadata_complete = all(all(str(row[field]).strip() for field in source_fields) for row in output_rows)
    growth_values_match = all(
        code == "unknown" or float(row["growth_yoy_pct"]) == NBS_GROWTH_YOY_PCT[code]
        for row in parameter_rows
        for code in [str(row["industry_code"])]
    )
    q2_name_join = q2_pairs == mapping_pairs == output_pairs
    no_rating_or_churn = not any(
        field.startswith("rating_prob") or "churn" in field.lower() or field.startswith("predicted_rating")
        for field in SCENARIO_OUTPUT_COLUMNS
    ) and all(row["risk_override_only"] is True for row in output_rows)
    checks = {
        "row_count_302": len(output_rows) == 302,
        "q2_ids_unique": len(q2_ids) == len(set(q2_ids)) and all(q2_ids),
        "output_ids_unique": len(output_ids) == len(set(output_ids)) and all(output_ids),
        "exact_id_name_join": q2_name_join,
        "finite_values": finite_legal,
        "risk_values_legal": risk_legal,
        "multipliers_ge_one": multiplier_legal,
        "identity_exact_q2": identity_exact,
        "scenario_order_monotone": scenario_order,
        "robust_is_exact_max": robust_exact,
        "positive_growth_neutral": positive_growth_neutral,
        "unknown_count_145_max_known_policy": unknown_max_policy,
        "source_metadata_complete": source_metadata_complete,
        "official_growth_values_match": growth_values_match,
        "risk_override_only": no_rating_or_churn,
        "deterministic_rows": output_rows == repeat_rows,
        "parameter_columns_complete": bool(parameter_rows) and set(INDUSTRY_PARAMETER_COLUMNS).issubset(parameter_rows[0]),
        "output_columns_complete": bool(output_rows) and set(SCENARIO_OUTPUT_COLUMNS).issubset(output_rows[0]),
    }
    return {
        "contract_version": "q3_stress_scenarios_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "counts": {
            "q2_rows": len(q2_rows),
            "mapping_rows": len(mapping_rows),
            "output_rows": len(output_rows),
            "unknown_rows": len(unknown_rows),
            "industry_code": dict(sorted(Counter(str(row["industry_code"]) for row in output_rows).items())),
        },
        "source": {
            "url": NBS_SOURCE_URL,
            "release_date": NBS_RELEASE_DATE,
            "period": NBS_PERIOD,
            "standard": NBS_STANDARD,
            "status": NBS_SOURCE_STATUS,
            "interpretation": NBS_INTERPRETATION,
        },
        "assumptions": {
            "c_pp": settings["c_pp"],
            "rho": settings["rho"],
            "eps": settings["eps"],
            "lambdas": settings["lambdas"],
            "unknown_policy": settings["unknown_policy"],
            "all_assumption_fields_marked_modeling_assumption": True,
        },
        "inputs": {
            "q2_risk_table": _relative_path(repo_root, _resolve_path(repo_root, settings["input"])),
            "industry_mapping": _relative_path(repo_root, _resolve_path(repo_root, settings["industry_mapping_input"])),
        },
        "risk_change_summary": _risk_summary(output_rows, parameter_rows),
        "max_known_S": max_known_s,
    }


def _write_report(path: Path, contract: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Q3 疫情行业压力场景验证",
        "",
        f"状态：**{contract['status']}**",
        "",
        "本阶段仅覆盖 Q2 `main_risk_score` 的相对风险倾向；不改变评级概率、流失曲线或其他 Q2 参数。",
        "官方分行业 GDP 数据是包含 2020 年 3 月的回顾性外部压力标尺，不是 Q2 训练输入或事前预测验证。",
        "",
        "## 门禁",
        "",
        "| 检查 | 结果 |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {'PASS' if value else 'FAIL'} |" for name, value in contract["checks"].items())
    lines.extend(
        [
            "",
            "## 场景风险变化摘要",
            "",
            "```json",
            json.dumps(contract["risk_change_summary"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_q3_stress_scenarios(
    repo_root: Path | None = None,
    config_path: Path | None = None,
    *,
    preflight_contract: Mapping[str, Any] | None = None,
    mapping_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run stage three after the Q2 preflight and industry mapping gates."""

    root = (repo_root or _repo_root_from_file()).resolve()
    resolved_config = (config_path or (root / "src" / "_internal" / "q3_config.yaml")).resolve()
    config = _read_yaml(resolved_config)
    settings = _load_settings(config)
    if preflight_contract is None:
        try:
            preflight_contract = run_q3_preflight(root, resolved_config)
        except PreflightError as exc:
            raise StressScenarioError(str(exc)) from exc
    if preflight_contract.get("status") != "PASS":
        raise StressScenarioError("Q3 stress scenarios require a PASS Q2 baseline preflight")
    if mapping_contract is None:
        try:
            mapping_contract = run_q3_industry_mapping(
                root, resolved_config, preflight_contract=preflight_contract
            )
        except (IndustryMappingError, PreflightError) as exc:
            raise StressScenarioError(str(exc)) from exc
    if mapping_contract.get("status") != "PASS":
        raise StressScenarioError("Q3 stress scenarios require a PASS industry mapping")

    q2_path = _resolve_path(root, settings["input"])
    mapping_path = _resolve_path(root, settings["industry_mapping_input"])
    q2_rows = _read_rows(q2_path)
    mapping_rows = _read_rows(mapping_path)
    if not {"enterprise_id", "enterprise_name", "main_risk_score"}.issubset(q2_rows[0]):
        raise StressScenarioError("Q2 risk table must contain enterprise_id, enterprise_name, main_risk_score")
    if not {"enterprise_id", "enterprise_name", "industry_code", "mapping_status", "confidence"}.issubset(mapping_rows[0]):
        raise StressScenarioError("industry mapping must contain enterprise identity and mapping fields")
    parameter_rows, max_known_s = _industry_rows(settings)
    output_rows = _build_enterprise_rows(q2_rows, mapping_rows, parameter_rows, settings)
    repeat_rows = _build_enterprise_rows(q2_rows, mapping_rows, parameter_rows, settings)
    risk_output = _resolve_path(root, settings["risk_output"])
    industry_output = _resolve_path(root, settings["industry_output"])
    validation_output = _resolve_path(root, settings["validation_output"])
    report_output = _resolve_path(root, settings["report_output"])
    _write_csv(risk_output, SCENARIO_OUTPUT_COLUMNS, output_rows)
    _write_csv(industry_output, INDUSTRY_PARAMETER_COLUMNS, parameter_rows)
    contract = _validate(
        q2_rows,
        mapping_rows,
        output_rows,
        parameter_rows,
        settings,
        repeat_rows,
        max_known_s,
        root,
    )
    contract.update(
        {
            "config_path": _relative_path(root, resolved_config),
            "config_sha256": _sha256_bytes(resolved_config.read_bytes()),
            "preflight_status": preflight_contract.get("status"),
            "mapping_status": mapping_contract.get("status"),
            "outputs": {
                "risk_scores": {
                    "path": _relative_path(root, risk_output),
                    "sha256": _sha256_bytes(risk_output.read_bytes()),
                    "row_count": len(output_rows),
                },
                "industry_parameters": {
                    "path": _relative_path(root, industry_output),
                    "sha256": _sha256_bytes(industry_output.read_bytes()),
                    "row_count": len(parameter_rows),
                },
            },
        }
    )
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_report(report_output, contract)
    if contract["status"] != "PASS":
        raise StressScenarioError(
            f"Q3 stress scenario acceptance failed; see {_relative_path(root, validation_output)}"
        )
    return contract


# Short aliases make the stage callable without coupling tests to a long name.
run_q3_stress = run_q3_stress_scenarios
