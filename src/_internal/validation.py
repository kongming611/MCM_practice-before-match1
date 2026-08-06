"""Final question-one acceptance checks for the stable project layout."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from _internal.data_pipeline import (
    OUTPUT_ROOT,
    PAPER_ROOT,
    PROCESSED_ROOT,
    RUNTIME_ROOT,
    ROOT,
    config_hash,
    load_config,
    relative,
    sha256_file,
    write_csv,
    write_json,
    write_text,
)


REQUIRED_FIGURES = [
    "churn_curve_raw_vs_fitted.png",
    "acceptance_probability_curves.png",
    "baseline_risk_amount_scatter.png",
    "baseline_amount_by_rating.png",
    "baseline_rate_by_rating.png",
    "budget_expected_return_curve.png",
    "budget_risk_curve.png",
    "budget_enterprise_count_curve.png",
    "lgd_sensitivity.png",
    "funding_cost_sensitivity.png",
    "risk_mean_vs_p90.png",
    "budget_definition_comparison.png",
    "lgd_funding_expected_return_heatmap.png",
    "enterprise_selection_stability.png",
]


def check(condition: bool, name: str, detail: str, failures: list[str], rows: list[dict[str, object]]) -> None:
    status = "PASS" if condition else "FAIL"
    rows.append({"check_name": name, "status": status, "detail": detail})
    if not condition:
        failures.append(f"{name}: {detail}")


def main() -> int:
    config = load_config()
    failures: list[str] = []
    rows: list[dict[str, object]] = []

    required_paths = [
        RUNTIME_ROOT / "audit" / "invoice_status_summary.csv",
        OUTPUT_ROOT / "reports" / "q1_data_quality_report.md",
        RUNTIME_ROOT / "feature_validation" / "feature_validation_summary.json",
        OUTPUT_ROOT / "reports" / "model_training_report.md",
        PROCESSED_ROOT / "q1_enterprise_features.csv",
        PROCESSED_ROOT / "q1_risk_scores.csv",
        RUNTIME_ROOT / "model_training" / "model_training_manifest.json",
        RUNTIME_ROOT / "model_training" / "model_training_config_snapshot.yaml",
        OUTPUT_ROOT / "tables" / "attachment3_audit.csv",
        OUTPUT_ROOT / "tables" / "attachment3_normalized.csv",
        OUTPUT_ROOT / "tables" / "churn_curve_fitted.csv",
        OUTPUT_ROOT / "tables" / "churn_curve_metrics.csv",
        OUTPUT_ROOT / "tables" / "churn_curve_cross_rating_check.csv",
        OUTPUT_ROOT / "tables" / "baseline_enterprise_strategy.csv",
        OUTPUT_ROOT / "tables" / "baseline_portfolio_summary.csv",
        OUTPUT_ROOT / "reports" / "baseline_solver_diagnostics.json",
        OUTPUT_ROOT / "tables" / "budget_sensitivity_summary.csv",
        OUTPUT_ROOT / "tables" / "parameter_sensitivity_summary.csv",
        OUTPUT_ROOT / "tables" / "enterprise_strategy_stability.csv",
        RUNTIME_ROOT / "sensitivity" / "sensitivity_solver_diagnostics.csv",
        OUTPUT_ROOT / "final" / "q1_final_delivery.xlsx",
        OUTPUT_ROOT / "final" / "q1_final_manifest.json",
        OUTPUT_ROOT / "final" / "q1_final_output_index.csv",
        PAPER_ROOT / "q1_final_results_for_paper.md",
        PAPER_ROOT / "q1_final_assumptions_and_limitations.md",
        PAPER_ROOT / "q1_submission_checklist.md",
    ]
    check(all(path.exists() for path in required_paths), "required_files_exist", "all first-to-final delivery inputs exist", failures, rows)

    phase_a_path = RUNTIME_ROOT / "feature_validation" / "feature_validation_summary.json"
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8")) if phase_a_path.exists() else {}
    check(phase_a.get("status") == "PASS" and phase_a.get("blocking_failure_count") == 0, "phase_a_feature_validation_pass", str(phase_a.get("status")), failures, rows)
    config_features = config["features"]
    check("zero_amount_invoice_rate" not in config_features["primary_model_features"] and "zero_amount_invoice_rate" not in config_features["sensitivity_model_features"], "zero_amount_excluded_from_models", "zero_amount_invoice_rate absent from formal feature lists", failures, rows)

    risk_path = PROCESSED_ROOT / "q1_risk_scores.csv"
    risk = pd.read_csv(risk_path) if risk_path.exists() else pd.DataFrame()
    check(len(risk) == 123 and risk.get("enterprise_id", pd.Series(dtype=str)).nunique() == 123, "risk_table_123_unique", str(risk.shape), failures, rows)
    if not risk.empty:
        check(risk["selected_model"].nunique() == 1 and risk["selected_model"].iloc[0] == "elastic_net_logistic", "selected_model_elastic_net_logistic", str(risk["selected_model"].value_counts().to_dict()), failures, rows)
        check(risk["selected_model_risk_score"].notna().all() and risk["selected_model_risk_score"].between(0, 1).all(), "risk_values_legal", "selected_model_risk_score in [0,1]", failures, rows)

    manifest_path = RUNTIME_ROOT / "model_training" / "cv_split_manifest.csv"
    if manifest_path.exists():
        cv = pd.read_csv(manifest_path)
        test = cv[cv["split_role"] == "test"]
        count_ok = len(test) == 1230 and test.groupby("enterprise_id").size().eq(10).all()
    else:
        count_ok = False
    check(count_ok, "risk_test_count_10_per_enterprise", "cv_split_manifest test rows", failures, rows)

    audit_path = OUTPUT_ROOT / "tables" / "attachment3_audit.csv"
    audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    check(not audit.empty and audit["status"].eq("PASS").all(), "attachment3_audit_pass", "all attachment-3 checks PASS", failures, rows)
    fitted_path = OUTPUT_ROOT / "tables" / "churn_curve_fitted.csv"
    fitted = pd.read_csv(fitted_path) if fitted_path.exists() else pd.DataFrame()
    check(len(fitted) == 87 and set(fitted.get("credit_rating", pd.Series(dtype=str)).unique()) == {"A", "B", "C"}, "churn_fitted_complete", str(fitted.shape), failures, rows)
    if not fitted.empty:
        check(fitted["fitted_churn_rate"].between(0, 1).all() and fitted["acceptance_probability"].between(0, 1).all(), "churn_and_acceptance_legal", "fitted churn and acceptance in [0,1]", failures, rows)
        monotone = fitted.sort_values(["credit_rating", "interest_rate"]).groupby("credit_rating")["fitted_churn_rate"].diff().dropna().ge(-1e-8).all()
        acceptance_monotone = fitted.sort_values(["credit_rating", "interest_rate"]).groupby("credit_rating")["acceptance_probability"].diff().dropna().le(1e-8).all()
        check(bool(monotone and acceptance_monotone), "churn_curves_monotone", "churn nondecreasing and acceptance nonincreasing", failures, rows)

    baseline_diag_path = OUTPUT_ROOT / "reports" / "baseline_solver_diagnostics.json"
    baseline_diag = json.loads(baseline_diag_path.read_text(encoding="utf-8")) if baseline_diag_path.exists() else {}
    check(bool(baseline_diag.get("is_optimal")), "baseline_milp_optimal", str(baseline_diag.get("solver_message")), failures, rows)
    baseline_path = OUTPUT_ROOT / "tables" / "baseline_enterprise_strategy.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    check(len(baseline) == 123 and baseline.get("enterprise_id", pd.Series(dtype=str)).nunique() == 123, "baseline_strategy_123_unique", str(baseline.shape), failures, rows)
    if not baseline.empty:
        d_zero = (baseline.loc[baseline["credit_rating"] == "D", "offered_loan_amount_10k"] == 0).all()
        check(bool(d_zero), "D_enterprises_zero_loan", "D-level business rule", failures, rows)
        selected = baseline[baseline["loan_decision"] == "放贷"]
        amount_ok = selected["offered_loan_amount_10k"].between(10 - 1e-6, 100 + 1e-6).all()
        rate_ok = selected["interest_rate"].between(0.04 - 1e-6, 0.15 + 1e-6).all()
        check(bool(amount_ok and rate_ok), "baseline_amount_rate_constraints", "selected amount 10-100万元 and rate 4%-15%", failures, rows)
        recomputed = baseline["expected_interest_income_10k"] - baseline["expected_credit_loss_10k"] - baseline["expected_funding_cost_10k"]
        check(bool(np.allclose(recomputed, baseline["expected_net_return_10k"], atol=1e-6, rtol=0)), "baseline_return_decomposition", "interest - loss - funding = net return", failures, rows)
        check(bool(np.isfinite(baseline[["offered_loan_amount_10k", "expected_disbursed_amount_10k", "expected_interest_income_10k", "expected_credit_loss_10k", "expected_funding_cost_10k", "expected_net_return_10k"]].to_numpy(dtype=float)).all()), "baseline_numeric_finite", "monetary strategy fields finite", failures, rows)
    summary_path = OUTPUT_ROOT / "tables" / "baseline_portfolio_summary.csv"
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not summary.empty:
        check(abs(float(summary.loc[0, "budget_used_10k"]) - float(summary.loc[0, "budget_10k"])) <= 1e-6, "baseline_nominal_budget_equality", str(summary.loc[0, "budget_used_10k"]), failures, rows)

    diag_path = RUNTIME_ROOT / "sensitivity" / "sensitivity_solver_diagnostics.csv"
    diagnostics = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    all_optimal = not diagnostics.empty and diagnostics["is_optimal"].astype(bool).all()
    failures_empty = not diagnostics.empty and diagnostics["validation_failures"].astype(str).eq("[]").all()
    check(bool(all_optimal), "all_official_scenarios_optimal", f"{len(diagnostics)} scenarios", failures, rows)
    check(bool(failures_empty), "all_scenario_validation_failures_empty", "scenario constraint checks", failures, rows)
    if not diagnostics.empty:
        check(len(diagnostics) == 30, "sensitivity_scenario_count", str(len(diagnostics)), failures, rows)

    stability_path = OUTPUT_ROOT / "tables" / "enterprise_strategy_stability.csv"
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    check(len(stability) == 123 and stability.get("enterprise_id", pd.Series(dtype=str)).nunique() == 123, "stability_table_123_unique", str(stability.shape), failures, rows)

    figures_dir = OUTPUT_ROOT / "figures" / "credit"
    figure_ok = all((figures_dir / name).exists() and (figures_dir / name).stat().st_size > 0 for name in REQUIRED_FIGURES)
    check(figure_ok, "required_figures_nonempty", "all requested q1_credit figures", failures, rows)
    excel_path = OUTPUT_ROOT / "final" / "q1_final_delivery.xlsx"
    required_sheets = {"README", "RiskScores", "ChurnRaw", "ChurnFitted", "BaselineStrategy", "BaselinePortfolio", "BudgetSensitivity", "ParameterSensitivity", "StrategyStability", "ModelMetrics", "Assumptions", "Validation"}
    if excel_path.exists():
        sheets = set(pd.ExcelFile(excel_path).sheet_names)
    else:
        sheets = set()
    check(required_sheets.issubset(sheets), "final_excel_re_readable", str(sorted(sheets)), failures, rows)

    docs = [PAPER_ROOT / "q1_final_results_for_paper.md", PAPER_ROOT / "q1_final_assumptions_and_limitations.md", PAPER_ROOT / "q1_submission_checklist.md"]
    placeholder_pattern = re.compile(r"TODO|TBD|PLACEHOLDER|待填|占位", re.IGNORECASE)
    docs_ok = all(not placeholder_pattern.search(path.read_text(encoding="utf-8")) for path in docs if path.exists())
    check(docs_ok, "final_docs_no_placeholders", "final paper/assumptions/checklist documents", failures, rows)

    final_manifest_path = OUTPUT_ROOT / "final" / "q1_final_manifest.json"
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8")) if final_manifest_path.exists() else {}
    check(final_manifest.get("selected_model") == "elastic_net_logistic", "final_manifest_selected_model", str(final_manifest.get("selected_model")), failures, rows)
    check(final_manifest.get("risk_column") == "selected_model_risk_score", "final_manifest_risk_column", str(final_manifest.get("risk_column")), failures, rows)
    check(final_manifest.get("config_sha256") == config_hash(config), "final_config_hash_recorded", "configuration hash matches current config", failures, rows)
    if final_manifest.get("code_hashes"):
        code_ok = True
        for path_text, expected in final_manifest["code_hashes"].items():
            path = ROOT / path_text
            code_ok &= path.exists() and sha256_file(path) == expected
        check(bool(code_ok), "final_code_hashes_match", "core fourth-batch code hashes", failures, rows)
    else:
        check(False, "final_code_hashes_match", "code_hashes missing", failures, rows)
    output_index_path = OUTPUT_ROOT / "final" / "q1_final_output_index.csv"
    if output_index_path.exists():
        output_index = pd.read_csv(output_index_path)
        index_ok = True
        for row in output_index.itertuples(index=False):
            path = ROOT / str(row.path)
            index_ok &= path.exists() and sha256_file(path) == str(row.sha256)
        check(bool(index_ok), "output_index_hashes_match", "final output index hashes", failures, rows)
    else:
        check(False, "output_index_hashes_match", "output index missing", failures, rows)

    report_path = OUTPUT_ROOT / "final" / "q1_final_validation_report.md"
    report_lines = [
        "# 问题一最终验收报告",
        "",
        f"最终状态：{'PASS' if not failures else 'FAIL'}",
        "",
        "| 检查 | 状态 | 证据 |",
        "|---|---|---|",
    ]
    report_lines.extend(f"| {row['check_name']} | {row['status']} | {row['detail']} |" for row in rows)
    report_lines.extend(["", "所有官方MILP场景必须达到最优状态；收益、损失和风险均按情景解释。", ""])
    write_text("\n".join(report_lines), report_path)

    # Add the final validation report to the output index and manifest after it
    # exists.  The manifest does not hash itself.
    if output_index_path.exists() and report_path.exists():
        output_index = pd.read_csv(output_index_path)
        if relative(report_path) not in set(output_index["path"]):
            output_index = pd.concat([output_index, pd.DataFrame([{"path": relative(report_path), "sha256": sha256_file(report_path), "exists": True, "size_bytes": report_path.stat().st_size}])], ignore_index=True)
        else:
            output_index.loc[output_index["path"] == relative(report_path), "sha256"] = sha256_file(report_path)
        write_csv(output_index.sort_values("path"), output_index_path)
        if final_manifest_path.exists():
            final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
            final_manifest["final_validation_status"] = "PASS" if not failures else "FAIL"
            final_manifest["final_validation_report_sha256"] = sha256_file(report_path)
            final_manifest["output_index_sha256"] = sha256_file(output_index_path)
            final_manifest["output_hashes"] = {row["path"]: row["sha256"] for row in output_index.to_dict("records")}
            write_json(final_manifest, final_manifest_path)

    if failures:
        print("FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
