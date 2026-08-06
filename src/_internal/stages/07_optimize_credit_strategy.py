"""Solve the representative parameterized credit strategy with scipy.milp."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from _internal.data_pipeline import (
    OUTPUT_ROOT,
    PROCESSED_ROOT,
    RUNTIME_ROOT,
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
from _internal.credit_strategy import (
    ALL_RATINGS,
    CHURN_RATINGS,
    load_fitted_churn,
    portfolio_summary,
    solve_credit_milp,
    strategy_from_solution,
    validate_strategy,
)


def _risk_table(config: dict) -> pd.DataFrame:
    path = PROCESSED_ROOT / "q1_risk_scores.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "enterprise_id", "enterprise_name", "credit_rating", "default_label",
        "selected_model", "selected_model_risk_score", "logistic_oof_p90",
        "logistic_risk_rank", "logistic_risk_percentile",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Risk table missing columns: {sorted(missing)}")
    if len(frame) != 123 or frame["enterprise_id"].nunique() != 123:
        raise ValueError("Risk table must contain 123 unique enterprises")
    if frame["selected_model"].nunique() != 1 or frame["selected_model"].iloc[0] != "elastic_net_logistic":
        raise ValueError("Fourth batch requires selected_model=elastic_net_logistic for every enterprise")
    if frame["selected_model_risk_score"].isna().any() or not frame["selected_model_risk_score"].between(0, 1).all():
        raise ValueError("selected_model_risk_score must be complete and in [0, 1]")
    if frame["logistic_oof_p90"].isna().any() or not frame["logistic_oof_p90"].between(0, 1).all():
        raise ValueError("logistic_oof_p90 must be complete and in [0, 1]")
    if not set(frame["credit_rating"].dropna().unique()).issubset(set(ALL_RATINGS)):
        raise ValueError("Unexpected credit rating in risk table")

    # This is a label/count integrity check only; default_label is never passed
    # to the objective, constraints, or risk calculation.
    enterprise_sheet = config["input"]["expected_sheets"]["enterprise"]
    original = pd.read_excel(resolve_path(config["input"]["attachment1"]), sheet_name=enterprise_sheet)
    rating_col = "信誉评级"
    if rating_col not in original.columns:
        raise ValueError("Original enterprise sheet has no credit-rating column")
    if int((original[rating_col].astype(str).str.strip() == "D").sum()) != int((frame["credit_rating"] == "D").sum()):
        raise ValueError("D-rating count differs between risk table and enterprise information")
    return frame.sort_values("enterprise_id").reset_index(drop=True)


def _write_baseline_figures(strategy: pd.DataFrame, summary: dict, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c", "D": "#888888"}
    paths: list[Path] = []
    eligible = strategy[strategy["credit_rating"].isin(CHURN_RATINGS)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for rating in ALL_RATINGS:
        group = strategy[strategy["credit_rating"] == rating]
        ax.scatter(group["risk_score"], group["offered_loan_amount_10k"], s=42, alpha=0.8, color=colors[rating], label=f"评级{rating}")
    ax.set_title("代表性基准策略：历史违约倾向与名义授信额度")
    ax.set_xlabel("selected_model_risk_score（历史发票行为相对违约倾向）")
    ax.set_ylabel("名义授信额度（万元）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.01, "代表性参数化情景：预算=0.50×B_max，LGD=0.50，资金成本=3%", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "baseline_risk_amount_scatter.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    amount_by_rating = strategy.groupby("credit_rating", sort=False)["offered_loan_amount_10k"].sum().reindex(ALL_RATINGS, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(amount_by_rating.index, amount_by_rating.values, color=[colors[r] for r in amount_by_rating.index])
    ax.set_title("代表性基准策略：各评级名义授信总额")
    ax.set_xlabel("信誉评级")
    ax.set_ylabel("授信总额（万元）")
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, "D级企业原则上不予放贷；结果为情景参数化决策", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "baseline_amount_by_rating.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    rate_by_rating = eligible[eligible["loan_decision"] == "放贷"].groupby("credit_rating").apply(
        lambda group: np.average(group["interest_rate_percent"], weights=group["offered_loan_amount_10k"]) if group["offered_loan_amount_10k"].sum() > 0 else np.nan,
        include_groups=False,
    ).reindex(CHURN_RATINGS)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rate_by_rating.index, rate_by_rating.values, color=[colors[r] for r in rate_by_rating.index])
    ax.set_title("代表性基准策略：各评级额度加权平均利率")
    ax.set_xlabel("信誉评级")
    ax.set_ylabel("利率（%）")
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, "利率取附件3实际观测点；风险值为历史发票行为相对违约倾向", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "baseline_rate_by_rating.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def _baseline_report(summary: dict, diagnostics: dict, validation_failures: list[str], eligible_count: int, b_max: float, baseline_budget: float) -> str:
    return "\n".join([
        "# 代表性基准信贷策略报告",
        "",
        "本报告是问题一没有给定年度预算时的代表性参数化情景，不把基准预算解释为题目给定的银行年度信贷总额。收益、损失和策略均为情景结果。",
        "",
        "## 参数",
        "",
        f"- 候选企业数：{eligible_count}；最大名义可投放额度B_max：{b_max:.10g}万元。",
        f"- 基准预算：0.50×B_max={baseline_budget:.10g}万元；预算口径：`nominal_equality`。",
        "- LGD=0.50、资金成本率=0.03是无真实参数时的示例中心情景，不是题目给定值，也不是由数据估计的真实参数。",
        "- 风险字段：`selected_model_risk_score`，含义是依据历史发票行为得到的相对违约倾向，不是严格一年期PD。",
        "- D级企业原则上不予放贷；A/B/C使用附件3实际利率点及独立评级保序拟合曲线。",
        "",
        "## 基准结果",
        "",
        f"- 获贷企业数：{summary['offered_enterprise_count']}；名义总额度：{summary['budget_used_10k']:.10g}万元；预算未使用：{summary['budget_unused_10k']:.10g}万元。",
        f"- 预期实际放款：{summary['expected_disbursed_amount_10k']:.10g}万元；额度加权平均利率：{summary['weighted_average_interest_rate_by_offer'] * 100:.6g}%。",
        f"- 额度加权组合风险：{summary['weighted_average_risk_by_offer']:.6g}。",
        f"- 情景预期利息收入：{summary['expected_interest_income_10k']:.10g}万元；情景预期信用损失：{summary['expected_credit_loss_10k']:.10g}万元；情景预期资金成本：{summary['expected_funding_cost_10k']:.10g}万元。",
        f"- 情景预期净收益：{summary['expected_net_return_10k']:.10g}万元；负单位情景收益贷款数：{summary['negative_unit_return_selected_count']}。",
        "",
        "## 求解与验收",
        "",
        f"- 求解器：scipy.optimize.milp；状态：{diagnostics['solver_status_code']}；是否最优：{diagnostics['is_optimal']}；耗时：{diagnostics['solve_time_seconds']:.6g}秒。",
        f"- 本策略分解恒等式及预算约束检查：{'PASS' if not validation_failures else 'FAIL'}。",
        "",
        "完整企业策略见`baseline_enterprise_strategy.csv`；未获贷企业仍保留在表中，D级企业与优化未选择企业的原因分开记录。",
        "",
    ])


def main() -> int:
    config = load_config()
    risk = _risk_table(config)
    churn = load_fitted_churn(config)
    eligible_count = int(risk["credit_rating"].isin(CHURN_RATINGS).sum())
    max_loan = float(config["credit_optimization"]["max_loan_10k"])
    min_loan = float(config["credit_optimization"]["min_loan_10k"])
    b_max = eligible_count * max_loan
    baseline_config = config["baseline"]
    baseline_budget = float(baseline_config["budget_fraction_of_max"]) * b_max
    if baseline_budget < min_loan or baseline_budget > b_max:
        raise ValueError(f"Baseline nominal equality budget is infeasible: {baseline_budget} not in [{min_loan}, {b_max}]")
    selected, diagnostics = solve_credit_milp(
        risk,
        churn,
        budget=baseline_budget,
        lgd=float(baseline_config["lgd"]),
        funding_cost_rate=float(baseline_config["funding_cost_rate"]),
        risk_column=str(config["credit_optimization"]["risk_column"]),
        budget_definition=str(baseline_config["budget_definition"]),
        config=config,
        scenario_id="baseline_representative_parameterized_scenario",
    )
    if not diagnostics["is_optimal"]:
        raise RuntimeError(f"Baseline MILP did not reach optimal status: {diagnostics}")
    strategy, summary = strategy_from_solution(
        risk,
        selected,
        diagnostics,
        risk_column=str(config["credit_optimization"]["risk_column"]),
        conservative_risk_column=str(config["credit_optimization"]["conservative_risk_column"]),
        lgd=float(baseline_config["lgd"]),
        funding_cost_rate=float(baseline_config["funding_cost_rate"]),
        scenario_id="baseline_representative_parameterized_scenario",
        budget_definition=str(baseline_config["budget_definition"]),
        representative=True,
    )
    validation_failures = validate_strategy(
        strategy,
        summary,
        churn,
        budget=baseline_budget,
        budget_definition=str(baseline_config["budget_definition"]),
        min_loan=min_loan,
        max_loan=max_loan,
        tolerance=float(config["solver"]["feasibility_tolerance"]),
    )
    if validation_failures:
        raise RuntimeError(f"Baseline strategy validation failed: {validation_failures}")

    out_dir = RUNTIME_ROOT / "credit_strategy"
    figure_dir = OUTPUT_ROOT / "figures" / "credit"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(strategy, out_dir / "baseline_enterprise_strategy.csv")
    write_csv(pd.DataFrame([summary]), out_dir / "baseline_portfolio_summary.csv")
    diagnostics_payload = {
        **diagnostics,
        "validation_failures": validation_failures,
        "eligible_count": eligible_count,
        "b_max_10k": b_max,
        "baseline_budget_10k": baseline_budget,
        "risk_input_sha256": sha256_file(PROCESSED_ROOT / "q1_risk_scores.csv"),
        "churn_input_sha256": sha256_file(RUNTIME_ROOT / "churn_model" / "churn_curve_fitted.csv"),
        "config_sha256": config_hash(config),
        "code_hashes": {
            "src/_internal/stages/07_optimize_credit_strategy.py": sha256_file(ROOT / "src" / "_internal" / "stages" / "07_optimize_credit_strategy.py"),
            "src/_internal/credit_strategy.py": sha256_file(ROOT / "src" / "_internal" / "credit_strategy.py"),
        },
    }
    write_json(diagnostics_payload, out_dir / "baseline_solver_diagnostics.json")
    write_text(_baseline_report(summary, diagnostics, validation_failures, eligible_count, b_max, baseline_budget), out_dir / "baseline_strategy_report.md")
    figures = _write_baseline_figures(strategy, summary, figure_dir)
    manifest = {
        "workflow": "fourth_batch_baseline_credit_strategy",
        "selected_model": str(risk["selected_model"].iloc[0]),
        "risk_column": str(config["credit_optimization"]["risk_column"]),
        "eligible_count": eligible_count,
        "b_max_10k": b_max,
        "baseline_budget_10k": baseline_budget,
        "baseline_parameters_are_representative": True,
        "official_solve_optimal": bool(diagnostics["is_optimal"]),
        "config_sha256": config_hash(config),
        "output_hashes": {
            relative(path): sha256_file(path)
            for path in [
                out_dir / "baseline_enterprise_strategy.csv",
                out_dir / "baseline_portfolio_summary.csv",
                out_dir / "baseline_solver_diagnostics.json",
                out_dir / "baseline_strategy_report.md",
                *figures,
            ]
        },
    }
    write_json(manifest, out_dir / "credit_strategy_manifest.json")
    print(
        "07_optimize_credit_strategy.py SUCCEEDED: "
        f"eligible={eligible_count} B_max_10k={b_max:.10g} "
        f"baseline_budget_10k={baseline_budget:.10g} selected={summary['offered_enterprise_count']} "
        f"optimal={diagnostics['is_optimal']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
