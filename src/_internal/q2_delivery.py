"""Final paper-handoff validation for question two.

This stage does not fit a model or alter a strategy.  It verifies that the
current Q2 outputs and the paper-writer package agree, then writes a compact
final index, manifest and PASS/FAIL report.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _internal.data_pipeline import ROOT, relative, sha256_file, write_csv, write_json, write_text


TOLERANCE = 1e-6
FINAL_DIR = ROOT / "outputs" / "q2" / "final"
PAPER_FILES = [
    ROOT / "paper" / "q2" / "README.md",
    ROOT / "paper" / "q2" / "q2_v1_problem_analysis.md",
    ROOT / "paper" / "q2" / "q2_v2_data_and_model_report.md",
    ROOT / "paper" / "q2" / "q2_v3_results_and_validation_report.md",
    ROOT / "paper" / "q2" / "q2_final_paper_materials.md",
    ROOT / "paper" / "q2" / "q2_assumptions_and_limitations.md",
    ROOT / "paper" / "q2" / "q2_submission_checklist.md",
]
FIGURE_SOURCES = {
    "outputs/q2/figures/eda/standardized_mean_difference.png": "outputs/q2/reports/q2_feature_distribution_comparison.csv",
    "outputs/q2/figures/eda/training_quantile_coverage.png": "outputs/q2/reports/q2_feature_distribution_comparison.csv",
    "outputs/q2/figures/eda/ood_novelty_score_distribution.png": "outputs/q2/reports/q2_ood_scores.csv",
    "outputs/q2/figures/eda/standardized_feature_distributions.png": "data/processed/q1_enterprise_features.csv + data/processed/q2_enterprise_features.csv",
    "outputs/q2/figures/model/q2_risk_probability_calibration_oof.png": "outputs/q2/tables/q2_risk_oof_enterprise.csv + outputs/q2/tables/q2_label_spreading_oof_enterprise.csv",
    "outputs/q2/figures/model/q2_risk_ranking_disagreement.png": "data/processed/q2_risk_rating_scores.csv",
    "outputs/q2/figures/model/q2_risk_model_instability_interval.png": "data/processed/q2_risk_rating_scores.csv",
    "outputs/q2/figures/model/q2_rating_confusion_matrix_oof.png": "outputs/q2/tables/q2_rating_confusion_matrix.csv",
    "outputs/q2/figures/model/q2_rating_probability_heatmap_302.png": "data/processed/q2_risk_rating_scores.csv",
    "outputs/q2/figures/credit/q2_credit_allocation_risk.png": "outputs/q2/tables/q2_credit_strategy.csv",
    "outputs/q2/figures/credit/q2_credit_sensitivity_net_return.png": "outputs/q2/tables/q2_optimization_sensitivity.csv",
}


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, encoding="utf-8-sig")


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _fixed(value: Any, digits: int) -> str:
    quantum = Decimal(1).scaleb(-digits)
    return format(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP), f".{digits}f")


def _record(checks: list[dict[str, Any]], name: str, passed: bool, detail: Any) -> None:
    checks.append(
        {
            "check_name": name,
            "status": "PASS" if bool(passed) else "FAIL",
            "detail": json.dumps(detail, ensure_ascii=False, default=str) if not isinstance(detail, str) else detail,
        }
    )


def _png_ok(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    with path.open("rb") as handle:
        return handle.read(8) == b"\x89PNG\r\n\x1a\n"


def _index_rows(paths: list[tuple[str, Path]]) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    for role, path in paths:
        key = relative(path)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "role": role,
                "path": key,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "path"], kind="stable").reset_index(drop=True)


def run_q2_delivery() -> int:
    """Validate and index the final Q2 paper handoff."""

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    required_data = {
        "features": "data/processed/q2_enterprise_features.csv",
        "scores": "data/processed/q2_risk_rating_scores.csv",
        "strategy": "data/processed/q2_credit_strategy.csv",
        "model_metrics": "outputs/q2/tables/q2_model_metrics.csv",
        "rating_metrics": "outputs/q2/tables/q2_rating_metrics.csv",
        "risk_oof": "outputs/q2/tables/q2_risk_oof_enterprise.csv",
        "rating_oof": "outputs/q2/tables/q2_rating_oof_enterprise.csv",
        "sensitivity": "outputs/q2/tables/q2_optimization_sensitivity.csv",
        "strategy_validation": "outputs/q2/tables/q2_strategy_validation.csv",
        "bootstrap": "outputs/q2/tables/q2_model_comparison_bootstrap.csv",
    }
    missing = [path for path in required_data.values() if not (ROOT / path).is_file()]
    _record(checks, "required_result_files_present", not missing, missing or list(required_data.values()))
    if missing:
        return _write_failed_early(checks)

    frames = {name: _read_csv(path) for name, path in required_data.items()}
    features = frames["features"]
    scores = frames["scores"]
    strategy = frames["strategy"]
    sensitivity = frames["sensitivity"]
    strategy_validation = frames["strategy_validation"]

    _record(
        checks,
        "q2_feature_table_302_unique",
        len(features) == 302 and features["enterprise_id"].nunique() == 302,
        {"rows": len(features), "unique_ids": int(features["enterprise_id"].nunique())},
    )
    _record(
        checks,
        "q2_score_table_302_unique",
        len(scores) == 302 and scores["enterprise_id"].nunique() == 302,
        {"rows": len(scores), "unique_ids": int(scores["enterprise_id"].nunique())},
    )
    probability_columns = ["rating_prob_A", "rating_prob_B", "rating_prob_C", "rating_prob_D"]
    probabilities = scores[probability_columns].to_numpy(dtype=float)
    _record(
        checks,
        "rating_probabilities_legal",
        bool(np.isfinite(probabilities).all())
        and bool((probabilities >= -1e-8).all())
        and bool(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-8)),
        {
            "minimum": float(probabilities.min()),
            "maximum_row_sum_error": float(np.max(np.abs(probabilities.sum(axis=1) - 1.0))),
        },
    )
    risk_columns = ["main_risk_score", "risk_instability_mean", "risk_instability_p90", "label_spreading_risk_score"]
    risk_values = scores[risk_columns].to_numpy(dtype=float)
    _record(
        checks,
        "risk_diagnostics_in_unit_interval",
        bool(np.isfinite(risk_values).all()) and bool(((risk_values >= 0) & (risk_values <= 1)).all()),
        {"minimum": float(risk_values.min()), "maximum": float(risk_values.max())},
    )

    amount = pd.to_numeric(strategy["offered_loan_amount_10k"], errors="raise")
    rate = pd.to_numeric(strategy["selected_rate"], errors="coerce")
    loaned = amount > TOLERANCE
    _record(
        checks,
        "q2_strategy_302_unique",
        len(strategy) == 302 and strategy["enterprise_id"].nunique() == 302,
        {"rows": len(strategy), "unique_ids": int(strategy["enterprise_id"].nunique())},
    )
    _record(
        checks,
        "strict_nominal_budget_10000_10k",
        abs(float(amount.sum()) - 10000.0) <= TOLERANCE,
        {"offered_sum_10k": float(amount.sum()), "difference_10k": float(amount.sum() - 10000.0)},
    )
    legal_amount = (~loaned) | ((amount >= 10.0 - TOLERANCE) & (amount <= 100.0 + TOLERANCE))
    _record(checks, "loan_amount_bounds", bool(legal_amount.all()), {"loaned_count": int(loaned.sum())})
    observed_rate_ok = _truthy(strategy.loc[loaned, "selected_rate_is_observed_attachment3_point"]).all()
    legal_rate = rate.loc[loaned].between(0.04 - TOLERANCE, 0.15 + TOLERANCE).all()
    _record(
        checks,
        "selected_rates_legal_and_observed",
        bool(observed_rate_ok and legal_rate),
        {"selected_rate_min": float(rate.loc[loaned].min()), "selected_rate_max": float(rate.loc[loaned].max())},
    )
    d_reject = _truthy(strategy["d_probability_reject_flag"])
    _record(
        checks,
        "D_probability_rejects_have_zero_loan",
        bool((amount.loc[d_reject] <= TOLERANCE).all()),
        {"D_reject_count": int(d_reject.sum())},
    )
    _record(
        checks,
        "all_24_optimization_scenarios_pass",
        len(sensitivity) == 24
        and sensitivity["solve_status"].eq("optimal").all()
        and sensitivity["validation_status"].eq("PASS").all(),
        {
            "scenario_count": len(sensitivity),
            "optimal_count": int(sensitivity["solve_status"].eq("optimal").sum()),
            "pass_count": int(sensitivity["validation_status"].eq("PASS").sum()),
        },
    )
    _record(
        checks,
        "strategy_validation_all_pass",
        len(strategy_validation) == 24 and strategy_validation["validation_status"].eq("PASS").all(),
        {"rows": len(strategy_validation), "pass_count": int(strategy_validation["validation_status"].eq("PASS").sum())},
    )
    _record(
        checks,
        "enterprise_OOF_contract",
        len(frames["risk_oof"]) == 123
        and len(frames["rating_oof"]) == 123
        and pd.to_numeric(frames["risk_oof"]["risk_oof_test_count"]).eq(10).all(),
        {"risk_oof_rows": len(frames["risk_oof"]), "rating_oof_rows": len(frames["rating_oof"])},
    )
    _record(
        checks,
        "paired_bootstrap_contract",
        len(frames["bootstrap"]) >= 4
        and pd.to_numeric(frames["bootstrap"]["bootstrap_replicates_valid"]).eq(1000).all()
        and pd.to_numeric(frames["bootstrap"]["enterprise_sample_size"]).eq(123).all(),
        {"metric_rows": len(frames["bootstrap"]), "replicates": sorted(frames["bootstrap"]["bootstrap_replicates_valid"].unique().tolist())},
    )

    data_validation = (ROOT / "outputs/q2/reports/q2_validation_report.md").read_text(encoding="utf-8")
    _record(checks, "data_validation_report_pass", "最终状态：**PASS**" in data_validation, "outputs/q2/reports/q2_validation_report.md")
    _record(checks, "paper_package_files_present", all(path.is_file() for path in PAPER_FILES), [relative(path) for path in PAPER_FILES if not path.is_file()] or [relative(path) for path in PAPER_FILES])
    figure_paths = [ROOT / path for path in FIGURE_SOURCES]
    _record(
        checks,
        "eleven_nonempty_png_figures",
        len(figure_paths) == 11 and all(_png_ok(path) for path in figure_paths),
        {"figure_count": len(figure_paths), "invalid": [relative(path) for path in figure_paths if not _png_ok(path)]},
    )
    source_paths: list[Path] = []
    for source_text in FIGURE_SOURCES.values():
        source_paths.extend(ROOT / item.strip() for item in source_text.split("+") if item.strip())
    _record(
        checks,
        "figure_source_data_present",
        all(path.is_file() for path in source_paths),
        [relative(path) for path in source_paths if not path.is_file()] or sorted({relative(path) for path in source_paths}),
    )

    primary = sensitivity.loc[sensitivity["scenario_id"].eq("primary")].iloc[0]
    final_text = (ROOT / "paper/q2/q2_final_paper_materials.md").read_text(encoding="utf-8") if (ROOT / "paper/q2/q2_final_paper_materials.md").is_file() else ""
    required_fragments = [
        str(int(primary["selected_enterprise_count"])),
        _fixed(primary["nominal_offered_amount_10k"], 0),
        _fixed(primary["expected_net_return_10k"], 6),
        _fixed(primary["expected_disbursed_amount_10k"], 6),
        "302家没有真实违约标签",
    ]
    _record(
        checks,
        "final_paper_materials_match_primary_numbers",
        all(fragment in final_text for fragment in required_fragments),
        {"required_fragments": required_fragments},
    )

    indexed_paths: list[tuple[str, Path]] = []
    indexed_paths.extend(("paper_handoff", path) for path in PAPER_FILES)
    indexed_paths.extend(("figure", path) for path in figure_paths)
    indexed_paths.extend(("figure_source_data", path) for path in source_paths)
    indexed_paths.extend(("key_result", ROOT / path) for path in required_data.values())
    indexed_paths.extend(
        [
            ("configuration", ROOT / "src/_internal/q2_config.yaml"),
            ("configuration", ROOT / "src/_internal/q2_optimization_config.yaml"),
            ("entrypoint", ROOT / "src/q2.py"),
            ("model_implementation", ROOT / "src/_internal/q2_models.py"),
            ("optimization_implementation", ROOT / "src/_internal/q2_credit_optimization.py"),
        ]
    )
    index = _index_rows(indexed_paths)
    index_path = FINAL_DIR / "q2_delivery_output_index.csv"
    write_csv(index, index_path)

    status = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    check_frame = pd.DataFrame(checks)
    check_path = FINAL_DIR / "q2_delivery_checks.csv"
    write_csv(check_frame, check_path)
    report_path = FINAL_DIR / "q2_delivery_validation_report.md"
    report_lines = [
        "# 问题二论文材料包最终验收",
        "",
        f"最终状态：**{status}**",
        "",
        f"- 验收时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- 检查项：{len(checks)}，通过：{sum(row['status'] == 'PASS' for row in checks)}，失败：{sum(row['status'] == 'FAIL' for row in checks)}。",
        f"- 正式索引文件数：{len(index)}。",
        "- 本验收只确认仓库内模型、结果、图表和论文材料一致；另一位建模同学的人工复核与签字仍须单独完成。",
        "",
        "| check_name | status | detail |",
        "|---|---|---|",
    ]
    for row in checks:
        detail = str(row["detail"]).replace("|", "\\|").replace("\n", " ")
        report_lines.append(f"| {row['check_name']} | {row['status']} | {detail} |")
    report_lines.extend(["", "论文手总入口：`paper/q2/README.md`。", ""])
    write_text("\n".join(report_lines), report_path)

    manifest = {
        "workflow": "question_two_final_paper_handoff_validation",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "check_count": len(checks),
        "pass_count": sum(row["status"] == "PASS" for row in checks),
        "fail_count": sum(row["status"] == "FAIL" for row in checks),
        "primary_summary": {
            "selected_enterprise_count": int(primary["selected_enterprise_count"]),
            "nominal_offered_amount_10k": float(primary["nominal_offered_amount_10k"]),
            "expected_disbursed_amount_10k": float(primary["expected_disbursed_amount_10k"]),
            "expected_net_return_10k": float(primary["expected_net_return_10k"]),
            "validation_status": str(primary["validation_status"]),
        },
        "runtime": {"python": sys.version, "platform": platform.platform()},
        "files": {
            relative(index_path): sha256_file(index_path),
            relative(check_path): sha256_file(check_path),
            relative(report_path): sha256_file(report_path),
        },
        "paper_files": {relative(path): sha256_file(path) for path in PAPER_FILES if path.is_file()},
    }
    write_json(manifest, FINAL_DIR / "q2_delivery_manifest.json")
    print(f"q2_delivery_stage={status} report={relative(report_path)} indexed_files={len(index)}")
    return 0 if status == "PASS" else 1


def _write_failed_early(checks: list[dict[str, Any]]) -> int:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(checks), FINAL_DIR / "q2_delivery_checks.csv")
    write_text(
        "# 问题二论文材料包最终验收\n\n最终状态：**FAIL**\n\n缺少必要结果文件，未继续验收。\n",
        FINAL_DIR / "q2_delivery_validation_report.md",
    )
    print("q2_delivery_stage=FAIL missing required result files")
    return 1


if __name__ == "__main__":
    raise SystemExit(run_q2_delivery())
