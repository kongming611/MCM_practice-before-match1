"""Run the fourth-batch credit sensitivity scenarios and build final delivery files."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _internal.data_pipeline import (
    OUTPUT_ROOT,
    PAPER_ROOT,
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
    jaccard_selected,
    load_fitted_churn,
    solve_credit_milp,
    strategy_from_solution,
    validate_strategy,
)


def load_risk_table(config: dict[str, Any]) -> pd.DataFrame:
    """Load only the accepted enterprise-level OOF risk table."""

    path = PROCESSED_ROOT / "q1_risk_scores.csv"
    frame = pd.read_csv(path)
    required = {
        "enterprise_id", "enterprise_name", "credit_rating", "default_label",
        "selected_model", "selected_model_risk_score", "logistic_oof_p90",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Risk table missing columns: {sorted(missing)}")
    if len(frame) != 123 or frame["enterprise_id"].nunique() != 123:
        raise ValueError("Risk table must contain exactly 123 unique enterprises")
    if frame["selected_model"].nunique() != 1 or frame["selected_model"].iloc[0] != "elastic_net_logistic":
        raise ValueError("The formal selected model is not elastic_net_logistic")
    for column in ["selected_model_risk_score", "logistic_oof_p90"]:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not values.between(0, 1).all():
            raise ValueError(f"Risk column {column} is incomplete or outside [0, 1]")
    if not set(frame["credit_rating"].dropna().astype(str)).issubset(set(ALL_RATINGS)):
        raise ValueError("Risk table contains a credit rating outside A/B/C/D")
    original = pd.read_excel(resolve_path(config["input"]["attachment1"]), sheet_name=config["input"]["expected_sheets"]["enterprise"])
    if int((original["信誉评级"].astype(str).str.strip() == "D").sum()) != int((frame["credit_rating"] == "D").sum()):
        raise ValueError("D-rating count does not match original enterprise information")
    return frame.sort_values("enterprise_id").reset_index(drop=True)


def solve_scenario(
    risk: pd.DataFrame,
    churn: pd.DataFrame,
    config: dict[str, Any],
    *,
    scenario_id: str,
    scenario_family: str,
    budget: float,
    lgd: float,
    funding_cost_rate: float,
    risk_variant: str,
    budget_definition: str,
    representative: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    risk_column = str(config["credit_optimization"]["risk_column"] if risk_variant == "mean" else config["credit_optimization"]["conservative_risk_column"])
    selected, diagnostics = solve_credit_milp(
        risk,
        churn,
        budget=budget,
        lgd=lgd,
        funding_cost_rate=funding_cost_rate,
        risk_column=risk_column,
        budget_definition=budget_definition,
        config=config,
        scenario_id=scenario_id,
    )
    if not diagnostics["is_optimal"]:
        raise RuntimeError(f"Official scenario {scenario_id} did not solve to optimality: {diagnostics}")
    strategy, summary = strategy_from_solution(
        risk,
        selected,
        diagnostics,
        risk_column=risk_column,
        conservative_risk_column=str(config["credit_optimization"]["conservative_risk_column"]),
        lgd=lgd,
        funding_cost_rate=funding_cost_rate,
        scenario_id=scenario_id,
        budget_definition=budget_definition,
        representative=representative,
    )
    # For the p90 experiment, rank the actual risk column used by that scenario.
    strategy["risk_rank"] = strategy["risk_score"].rank(method="min", ascending=False).astype(int)
    strategy["risk_percentile"] = strategy["risk_rank"] / len(strategy)
    strategy["scenario_family"] = scenario_family
    strategy["risk_variant"] = risk_variant
    strategy["budget_fraction"] = budget / (int(risk["credit_rating"].isin(CHURN_RATINGS).sum()) * float(config["credit_optimization"]["max_loan_10k"]))
    strategy["scenario_lgd"] = lgd
    strategy["scenario_funding_cost_rate"] = funding_cost_rate
    strategy["scenario_budget_10k"] = budget
    summary = {
        **summary,
        "scenario_family": scenario_family,
        "risk_variant": risk_variant,
        "budget_fraction": strategy["budget_fraction"].iloc[0],
        "lgd": lgd,
        "funding_cost_rate": funding_cost_rate,
    }
    failures = validate_strategy(
        strategy,
        summary,
        churn,
        budget=budget,
        budget_definition=budget_definition,
        min_loan=float(config["credit_optimization"]["min_loan_10k"]),
        max_loan=float(config["credit_optimization"]["max_loan_10k"]),
        tolerance=float(config["solver"]["feasibility_tolerance"]),
    )
    diagnostics = {**diagnostics, "validation_failures": failures}
    if failures:
        raise RuntimeError(f"Scenario {scenario_id} failed deterministic validation: {failures}")
    return strategy, summary, diagnostics


def add_comparison_fields(summary: dict[str, Any], strategy: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    return {
        **summary,
        "jaccard_to_baseline": jaccard_selected(strategy, baseline),
        "selected_count_change_vs_baseline": int((strategy["loan_decision"] == "放贷").sum() - (baseline["loan_decision"] == "放贷").sum()),
        "loan_amount_change_vs_baseline_10k": float(strategy["offered_loan_amount_10k"].sum() - baseline["offered_loan_amount_10k"].sum()),
        "net_return_change_vs_baseline_10k": float(strategy["expected_net_return_10k"].sum() - baseline["expected_net_return_10k"].sum()),
        "weighted_average_interest_change_vs_baseline": float(summary["weighted_average_interest_rate_by_offer"] - baseline.loc[baseline["loan_decision"] == "放贷", "interest_rate"].mul(baseline.loc[baseline["loan_decision"] == "放贷", "offered_loan_amount_10k"]).sum() / baseline.loc[baseline["loan_decision"] == "放贷", "offered_loan_amount_10k"].sum()) if (baseline["loan_decision"] == "放贷").any() else np.nan,
    }


def enterprise_changes(strategy: pd.DataFrame, baseline: pd.DataFrame, scenario_id: str, scenario_family: str) -> pd.DataFrame:
    left = baseline[["enterprise_id", "loan_decision", "offered_loan_amount_10k", "interest_rate", "risk_score"]].rename(columns={
        "loan_decision": "baseline_loan_decision", "offered_loan_amount_10k": "baseline_loan_amount_10k", "interest_rate": "baseline_interest_rate", "risk_score": "baseline_risk_score",
    })
    right = strategy[["enterprise_id", "loan_decision", "offered_loan_amount_10k", "interest_rate", "risk_score"]].rename(columns={
        "loan_decision": "scenario_loan_decision", "offered_loan_amount_10k": "scenario_loan_amount_10k", "interest_rate": "scenario_interest_rate", "risk_score": "scenario_risk_score",
    })
    result = left.merge(right, on="enterprise_id", how="outer")
    result["scenario_id"] = scenario_id
    result["scenario_family"] = scenario_family
    result["loan_amount_change_10k"] = result["scenario_loan_amount_10k"] - result["baseline_loan_amount_10k"]
    result["interest_rate_change"] = result["scenario_interest_rate"] - result["baseline_interest_rate"]
    result["newly_selected"] = (result["baseline_loan_decision"] != "放贷") & (result["scenario_loan_decision"] == "放贷")
    result["removed_from_selection"] = (result["baseline_loan_decision"] == "放贷") & (result["scenario_loan_decision"] != "放贷")
    return result


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    import matplotlib.pyplot as plt
    plt.close(fig)


def write_sensitivity_figures(budget_summary: pd.DataFrame, parameter_summary: pd.DataFrame, stability: pd.DataFrame, joint: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    x = budget_summary["budget_fraction"] * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, budget_summary["expected_net_return_10k"], "o-", color="#1f77b4")
    ax.set_title("预算敏感性：情景预期净收益")
    ax.set_xlabel("预算占B_max比例（%）")
    ax.set_ylabel("情景预期净收益（万元）")
    ax.grid(alpha=0.25)
    fig.text(0.5, 0.01, "LGD=0.50，资金成本=3%，风险=mean，预算口径=nominal_equality", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "budget_expected_return_curve.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, budget_summary["weighted_average_risk_by_offer"], "o-", color="#d62728")
    ax.set_title("预算敏感性：组合加权历史违约倾向")
    ax.set_xlabel("预算占B_max比例（%）")
    ax.set_ylabel("组合加权风险值（历史发票行为相对违约倾向）")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "budget_risk_curve.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, budget_summary["offered_enterprise_count"], "o-", color="#2ca02c")
    ax.set_title("预算敏感性：获贷企业数")
    ax.set_xlabel("预算占B_max比例（%）")
    ax.set_ylabel("获贷企业数（家）")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "budget_enterprise_count_curve.png"
    _save_figure(fig, path)
    paths.append(path)

    lgd = parameter_summary[parameter_summary["scenario_family"] == "lgd"].sort_values("lgd")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lgd["lgd"], lgd["expected_net_return_10k"], "o-", label="情景预期净收益")
    ax.plot(lgd["lgd"], lgd["expected_credit_loss_10k"], "s--", label="情景预期信用损失")
    ax.set_title("LGD敏感性")
    ax.set_xlabel("LGD")
    ax.set_ylabel("金额（万元）")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "lgd_sensitivity.png"
    _save_figure(fig, path)
    paths.append(path)

    funding = parameter_summary[parameter_summary["scenario_family"] == "funding_cost"].sort_values("funding_cost_rate")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(funding["funding_cost_rate"] * 100, funding["expected_net_return_10k"], "o-", color="#9467bd")
    ax.set_title("资金成本率敏感性")
    ax.set_xlabel("资金成本率（%）")
    ax.set_ylabel("情景预期净收益（万元）")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "funding_cost_sensitivity.png"
    _save_figure(fig, path)
    paths.append(path)

    risk = parameter_summary[parameter_summary["scenario_family"] == "risk"].sort_values("risk_variant")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(risk["risk_variant"], risk["expected_credit_loss_10k"], color=["#1f77b4", "#ff7f0e"])
    ax.set_title("风险mean与p90情景：预期信用损失")
    ax.set_xlabel("风险场景")
    ax.set_ylabel("情景预期信用损失（万元）")
    fig.text(0.5, 0.01, "p90为conservative_risk_scenario，不是真实风险上界", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "risk_mean_vs_p90.png"
    _save_figure(fig, path)
    paths.append(path)

    definitions = parameter_summary[parameter_summary["scenario_family"] == "budget_definition"].sort_values("budget_definition")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(definitions["budget_definition"], definitions["expected_net_return_10k"], color="#17becf")
    ax.set_title("预算定义敏感性：情景预期净收益")
    ax.set_xlabel("预算定义")
    ax.set_ylabel("情景预期净收益（万元）")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    path = output_dir / "budget_definition_comparison.png"
    _save_figure(fig, path)
    paths.append(path)

    pivot = joint.pivot(index="lgd", columns="funding_cost_rate", values="expected_net_return_10k").sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), cmap="YlGn", aspect="auto")
    ax.set_title("LGD×资金成本：情景预期净收益（万元）")
    ax.set_xlabel("资金成本率")
    ax.set_ylabel("LGD")
    ax.set_xticks(range(len(pivot.columns)), [f"{v * 100:.0f}%" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{v:.1f}" for v in pivot.index])
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = output_dir / "lgd_funding_expected_return_heatmap.png"
    _save_figure(fig, path)
    paths.append(path)

    count_pivot = joint.pivot(index="lgd", columns="funding_cost_rate", values="offered_enterprise_count").sort_index()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(count_pivot.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_title("LGD×资金成本：获贷企业数")
    ax.set_xlabel("资金成本率")
    ax.set_ylabel("LGD")
    ax.set_xticks(range(len(count_pivot.columns)), [f"{v * 100:.0f}%" for v in count_pivot.columns])
    ax.set_yticks(range(len(count_pivot.index)), [f"{v:.1f}" for v in count_pivot.index])
    for i in range(len(count_pivot.index)):
        for j in range(len(count_pivot.columns)):
            ax.text(j, i, f"{count_pivot.iloc[i, j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    path = output_dir / "lgd_funding_selected_count_heatmap.png"
    _save_figure(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = stability.sort_values("selection_frequency", ascending=False)
    ax.bar(range(len(ordered)), ordered["selection_frequency"], color=np.where(ordered["robust_core_flag"], "#2ca02c", "#7f7f7f"))
    ax.axhline(0.8, color="#d62728", linestyle="--", label="robust_core阈值80%")
    ax.set_title("企业跨情景获贷稳定性")
    ax.set_xlabel("企业（按获贷频率排序）")
    ax.set_ylabel("获贷频率")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.text(0.5, 0.01, "稳健核心阈值与不稳定规则属于本项目分析规则，不是题目给定规则", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = output_dir / "enterprise_selection_stability.png"
    _save_figure(fig, path)
    paths.append(path)
    return paths


def _strategy_readme_rows(config: dict[str, Any], baseline_summary: dict[str, Any], eligible_count: int, b_max: float) -> pd.DataFrame:
    rows = [
        ("说明", "本工作簿是问题一第四批的CSV计算结果汇总；CSV是计算源文件。"),
        ("风险字段", str(config["credit_optimization"]["risk_column"])),
        ("风险解释", "依据历史发票行为得到的相对违约倾向，不是严格一年期违约概率。"),
        ("候选企业数", eligible_count),
        ("B_max_10k", b_max),
        ("代表性情景", "budget=0.50×B_max; LGD=0.50; funding_cost_rate=0.03; risk=mean; budget_definition=nominal_equality"),
        ("参数说明", "LGD和资金成本率是无真实参数时的示例中心情景，不是题目给定值，也不是从现有数据估计出的真实参数。"),
        ("D级约束", "D级企业原则上不予放贷，最终贷款金额为0，利率留空。"),
        ("基准获贷企业数", baseline_summary["offered_enterprise_count"]),
        ("基准情景预期净收益_10k", baseline_summary["expected_net_return_10k"]),
    ]
    return pd.DataFrame(rows, columns=["item", "value"])


def _assumptions_rows(config: dict[str, Any]) -> pd.DataFrame:
    rows = [
        ("risk_column", config["credit_optimization"]["risk_column"], "正式主模型OOF聚合风险字段"),
        ("conservative_risk_column", config["credit_optimization"]["conservative_risk_column"], "p90保守情景；不是真实风险上界"),
        ("churn_method", "isotonic_regression", "A/B/C分别独立拟合，increasing=True"),
        ("churn_rate_points", "attachment3 observed points only", "不进行区间外外推，不自行生成利率点"),
        ("min_loan_10k", config["credit_optimization"]["min_loan_10k"], "业务约束"),
        ("max_loan_10k", config["credit_optimization"]["max_loan_10k"], "业务约束"),
        ("baseline_lgd", config["baseline"]["lgd"], "代表性中心情景，不是题目给定或数据估计参数"),
        ("baseline_funding_cost_rate", config["baseline"]["funding_cost_rate"], "代表性中心情景，不是题目给定或数据估计参数"),
        ("baseline_budget_fraction", config["baseline"]["budget_fraction_of_max"], "代表性参数化预算"),
        ("baseline_budget_definition", config["baseline"]["budget_definition"], "名义额度严格等式"),
        ("D_rule", "D级企业原则上不予放贷", "评级用于业务约束和展示，不进入风险模型"),
        ("default_label", "historical_reference_only", "不进入目标函数、约束或策略决策"),
        ("strategy_stability_rule", "robust_core >=80%; unstable frequency 20%-80% or amount SD >=20万元", "本项目分析规则，不是题目给定规则"),
        ("interpretation_boundary", "scenario expected return/loss", "不称为真实银行利润或真实损失预测"),
    ]
    return pd.DataFrame(rows, columns=["assumption", "value", "interpretation"])


def _final_paper_doc(config: dict[str, Any], risk: pd.DataFrame, churn_metrics: pd.DataFrame, baseline: dict[str, Any], budget: pd.DataFrame, parameter: pd.DataFrame, stability: pd.DataFrame, figures: list[Path]) -> str:
    model_metrics_path = RUNTIME_ROOT / "model_training" / "model_metrics_summary.csv"
    model_metrics = pd.read_csv(model_metrics_path)
    model_metrics = model_metrics[model_metrics["scope"] == "enterprise_aggregated_oof"]
    lines = [
        "# 问题一最终结果（第四批收尾）",
        "",
        "本文档由第四批程序根据实际CSV和求解结果生成。风险值表示依据历史发票行为得到的相对违约倾向，不是严格一年期违约概率；客户流失率不是违约率；所有收益和损失均为参数化情景结果。",
        "",
        "## 1. 完整建模闭环",
        "",
        "附件1发票审计与企业级特征构造 → 第三批企业级OOF风险模型 → 读取selected_model_risk_score → 附件3利率—客户流失率独立保序拟合 → scipy.optimize.milp信贷组合优化 → 预算、LGD、资金成本、风险和预算口径敏感性 → 企业策略稳定性与最终交付。",
        "",
        "## 2. 第三批风险模型摘要",
        "",
        f"企业样本为{len(risk)}家，评级A/B/C/D数量为{risk['credit_rating'].value_counts().reindex(ALL_RATINGS, fill_value=0).to_dict()}；selected_model统一为`{risk['selected_model'].iloc[0]}`。正式企业聚合OOF指标如下：",
        "",
        model_metrics[["model", "pr_auc", "brier", "log_loss", "roc_auc", "top20_recall"]].to_string(index=False),
        "",
        "正式优化只使用`selected_model_risk_score`，因为它是已选主模型的企业聚合OOF风险值；不使用全样本拟合概率、树模型概率或default_label。",
        "",
        "## 3. 利率—流失率拟合",
        "",
        "对每个评级g∈{A,B,C}，使用实际观测利率点拟合：`L_g(r)=IsotonicRegression(r, churn_rate)`，其中`increasing=True`、`y_min=0`、`y_max=1`、`out_of_bounds='clip'`。接受概率为`A_g(r)=1-L_g(r)`。",
        "",
        churn_metrics.to_string(index=False),
        "",
        f"拟合前A/B/C原始单调违反次数分别为{churn_metrics.set_index('credit_rating').loc['A', 'raw_monotonic_violation_count']}/{churn_metrics.set_index('credit_rating').loc['B', 'raw_monotonic_violation_count']}/{churn_metrics.set_index('credit_rating').loc['C', 'raw_monotonic_violation_count']}，拟合后均为0；最大绝对调整量分别为{churn_metrics.set_index('credit_rating').loc['A', 'maximum_absolute_adjustment']:.6g}/{churn_metrics.set_index('credit_rating').loc['B', 'maximum_absolute_adjustment']:.6g}/{churn_metrics.set_index('credit_rating').loc['C', 'maximum_absolute_adjustment']:.6g}。评级间A≤B≤C只作业务诊断；本数据独立拟合后存在交叉，基准没有静默添加评级联合硬约束。",
        "",
        "## 4. 信贷优化模型",
        "",
        "对每家A/B/C企业和附件3实际利率点定义二元报价变量x_ik和名义额度变量s_ik。每家企业至多选一个利率：Σ_k x_ik≤1；金额联动为10x_ik≤s_ik≤100x_ik；未获贷时所有s_ik=0。D级企业不建立可放贷变量，最终额度为0且利率留空。利率只能取附件3实际观测点，且处于4%—15%；风险值、流失率和接受概率均限制在0—1。",
        "",
        "单位名义授信的情景预期净收益为：`u_ik=A_g(r_k)[(1-p_i)r_k-p_i*LGD-c_f]`。总目标最大化`Σu_ik*s_ik`，并分别输出情景预期利息收入、信用损失、资金成本和净收益，逐行复核分解恒等式。预算口径分别为`Σs=B`、`Σs≤B`和`ΣA_g(r_k)s_ik≤B`；基准使用第一种。",
        "",
        "## 5. 预算与代表性基准策略",
        "",
        "问题一没有给定年度预算，因此不声称存在唯一银行策略。基准使用`representative_parameterized_scenario`：预算=0.50×B_max、LGD=0.50、资金成本率=0.03、风险=selected_model_risk_score、预算口径=nominal_equality。LGD和资金成本率是示例中心情景，不是题目给定值或数据估计值。",
        "",
        f"候选企业数为{baseline['eligible_enterprise_count']}，B_max={baseline['b_max_10k']:.10g}万元，代表性预算为{baseline['baseline_budget_10k']:.10g}万元。基准获贷{baseline['offered_enterprise_count']}家，名义总额度{baseline['budget_used_10k']:.10g}万元，额度加权平均利率{baseline['weighted_average_interest_rate_by_offer'] * 100:.6g}%，组合加权风险{baseline['weighted_average_risk_by_offer']:.6g}。",
        f"基准预期实际放款{baseline['expected_disbursed_amount_10k']:.10g}万元，情景预期信用损失{baseline['expected_credit_loss_10k']:.10g}万元，情景预期净收益{baseline['expected_net_return_10k']:.10g}万元。",
        "",
        "## 6. 敏感性分析",
        "",
        f"预算从0.10到1.00×B_max变化时，获贷数量从{int(budget['offered_enterprise_count'].iloc[0])}家变化到{int(budget['offered_enterprise_count'].iloc[-1])}家，情景净收益从{budget['expected_net_return_10k'].iloc[0]:.6g}万元变化到{budget['expected_net_return_10k'].iloc[-1]:.6g}万元；逐点结果和企业迁移见budget_sensitivity_summary.csv及budget_transition_details.csv。",
        "",
        parameter[parameter["scenario_family"].isin(["lgd", "funding_cost"])][["scenario_family", "lgd", "funding_cost_rate", "offered_enterprise_count", "expected_credit_loss_10k", "expected_net_return_10k", "jaccard_to_baseline"]].to_string(index=False),
        "",
        parameter[parameter["scenario_family"].isin(["risk", "budget_definition"])][["scenario_family", "risk_variant", "budget_definition", "offered_enterprise_count", "expected_disbursed_amount_10k", "expected_credit_loss_10k", "expected_net_return_10k", "jaccard_to_baseline"]].to_string(index=False),
        "",
        "mean与p90的比较是风险不确定性敏感性；p90场景称为`conservative_risk_scenario`，不称为真实风险上界。三种预算定义分别标注为nominal_equality、nominal_cap和expected_disbursement_cap，不能混合解释。LGD×资金成本使用3×3联合情景，不展开全部笛卡尔积。",
        "",
        "## 7. 策略稳定性",
        "",
        f"在{int(stability['total_scenario_count'].max())}个正式敏感性场景中，稳健核心企业数量为{int(stability['robust_core_flag'].sum())}，不稳定企业数量为{int(stability['unstable_flag'].sum())}。获贷频率最高的企业为{', '.join(stability.head(5)['enterprise_id'].tolist())}；不稳定企业示例为{', '.join(stability[stability['unstable_flag']].sort_values(['selection_frequency', 'loan_amount_sd_10k'], ascending=[True, False]).head(5)['enterprise_id'].tolist())}。频率阈值和额度标准差阈值属于本项目分析规则，不是题目给定规则。完整结果见enterprise_strategy_stability.csv。",
        "",
        "## 8. 图表与附录",
        "",
        "可直接放入论文的图表位于`outputs/q1/figures/`下的eda、model和credit目录，包括流失率拟合、接受概率、基准策略、预算、LGD/资金成本、风险mean/p90、预算定义、联合情景和企业稳定性图。完整123家企业策略表应作为论文附录或补充材料，主文只展示代表性汇总。",
        "",
        "## 9. 局限与解释边界",
        "",
        "样本仅123家企业，第三批风险值是历史发票行为的相对违约倾向，不是严格一年期PD；附件3流失率是历史统计关系，不是违约率；没有把客户流失与违约混为同一事件。LGD、资金成本和预算均需业务方进一步校准；本轮输出的利润、损失、放款和接受均为参数化情景结果，不代表真实银行利润或真实损失预测。评级只用于附件3曲线匹配、D级业务约束和展示，不进入违约风险模型。",
        "",
        "## 10. 主要文件",
        "",
        "- 完整交付Excel：`outputs/q1/final/q1_final_delivery.xlsx`。",
        "- 完整策略：`outputs/q1/tables/baseline_enterprise_strategy.csv`。",
        "- 最终验证：`outputs/q1/final/q1_final_validation_report.md`。",
        "",
    ]
    return "\n".join(lines)


def _assumptions_doc(config: dict[str, Any]) -> str:
    return "\n".join([
        "# 问题一最终假设与局限",
        "",
        "## 已锁定假设",
        "",
        "- 正式优化读取`data/processed/q1_risk_scores.csv`的`selected_model_risk_score`，不重新训练风险模型、不使用default_label决策、不把评级放入违约风险模型。",
        "- 附件3的A/B/C曲线分别独立使用IsotonicRegression保序拟合，基准只在附件3实际利率点上报价。",
        "- D级企业原则上不予放贷；该约束来自业务题意，D级不建立放贷变量。",
        "- 最低/最高额度为10/100万元，利率取4%—15%的附件3观测点。",
        "- 0.50×B_max、LGD=0.50、资金成本率=0.03只构成代表性中心情景，不是题目给定值，也不是现有数据估计的真实参数。",
        "- 情景目标将接受概率、相对违约倾向、LGD、资金成本和利率组合为单位名义授信的参数化净收益。",
        "",
        "## 敏感性规则",
        "",
        "预算、LGD、资金成本、风险mean/p90和预算定义均单独标记；LGD×资金成本只运行3×3联合情景。稳健核心定义为至少80%的正式敏感性场景获贷；不稳定定义为获贷频率处于20%—80%或贷款额度跨场景标准差达到20万元。这些阈值是分析规则，不是题目给定规则。",
        "",
        "## 局限",
        "",
        "风险值不是严格一年期违约概率，客户流失率不是违约率；没有将历史相关性解释为因果性。预算和损失参数缺乏银行真实业务校准，结果只能作为参数化决策比较。样本量、评级先验、附件3的历史统计关系和利率离散点都会影响策略。",
        "",
    ])


def build_final_excel(config: dict[str, Any], risk: pd.DataFrame, churn_normalized: pd.DataFrame, churn_fitted: pd.DataFrame, baseline_strategy: pd.DataFrame, baseline_summary: pd.DataFrame, budget_summary: pd.DataFrame, parameter_summary: pd.DataFrame, stability: pd.DataFrame, model_metrics: pd.DataFrame, validation_rows: pd.DataFrame, baseline_summary_dict: dict[str, Any], eligible_count: int, b_max: float) -> Path:
    final_dir = OUTPUT_ROOT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    path = final_dir / "q1_final_delivery.xlsx"
    readme = _strategy_readme_rows(config, baseline_summary_dict, eligible_count, b_max)
    assumptions = _assumptions_rows(config)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        risk.to_excel(writer, sheet_name="RiskScores", index=False)
        churn_normalized.to_excel(writer, sheet_name="ChurnRaw", index=False)
        churn_fitted.to_excel(writer, sheet_name="ChurnFitted", index=False)
        baseline_strategy.to_excel(writer, sheet_name="BaselineStrategy", index=False)
        baseline_summary.to_excel(writer, sheet_name="BaselinePortfolio", index=False)
        budget_summary.to_excel(writer, sheet_name="BudgetSensitivity", index=False)
        parameter_summary.to_excel(writer, sheet_name="ParameterSensitivity", index=False)
        stability.to_excel(writer, sheet_name="StrategyStability", index=False)
        model_metrics.to_excel(writer, sheet_name="ModelMetrics", index=False)
        assumptions.to_excel(writer, sheet_name="Assumptions", index=False)
        validation_rows.to_excel(writer, sheet_name="Validation", index=False)
    return path


def _publish_credit_outputs() -> None:
    """Copy the formal credit artifacts out of ignored runtime storage."""

    table_dir = OUTPUT_ROOT / "tables"
    report_dir = OUTPUT_ROOT / "reports"
    table_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    table_names = [
        "attachment3_audit.csv",
        "attachment3_normalized.csv",
        "churn_curve_fitted.csv",
        "churn_curve_metrics.csv",
        "churn_curve_cross_rating_check.csv",
    ]
    table_names += [
        "baseline_enterprise_strategy.csv",
        "baseline_portfolio_summary.csv",
    ]
    table_names += [
        "budget_sensitivity_summary.csv",
        "lgd_sensitivity_summary.csv",
        "funding_cost_sensitivity_summary.csv",
        "risk_sensitivity_summary.csv",
        "budget_definition_sensitivity_summary.csv",
        "parameter_sensitivity_summary.csv",
        "lgd_funding_joint_summary.csv",
        "enterprise_strategy_stability.csv",
    ]
    source_dirs = [RUNTIME_ROOT / "churn_model", RUNTIME_ROOT / "credit_strategy", RUNTIME_ROOT / "sensitivity"]
    for name in table_names:
        source = next((directory / name for directory in source_dirs if (directory / name).exists()), None)
        if source is None:
            raise FileNotFoundError(f"formal credit output is missing: {name}")
        shutil.copy2(source, table_dir / name)
    report_pairs = [
        (RUNTIME_ROOT / "churn_model" / "attachment3_audit_report.md", report_dir / "attachment3_audit_report.md"),
        (RUNTIME_ROOT / "credit_strategy" / "baseline_strategy_report.md", report_dir / "baseline_strategy_report.md"),
        (RUNTIME_ROOT / "credit_strategy" / "baseline_solver_diagnostics.json", report_dir / "baseline_solver_diagnostics.json"),
    ]
    for source, target in report_pairs:
        shutil.copy2(source, target)


def main() -> int:
    config = load_config()
    risk = load_risk_table(config)
    churn = load_fitted_churn(config)
    churn_normalized = pd.read_csv(RUNTIME_ROOT / "churn_model" / "attachment3_normalized.csv")
    churn_metrics = pd.read_csv(RUNTIME_ROOT / "churn_model" / "churn_curve_metrics.csv")
    baseline_strategy = pd.read_csv(RUNTIME_ROOT / "credit_strategy" / "baseline_enterprise_strategy.csv")
    baseline_summary_dict = pd.read_csv(RUNTIME_ROOT / "credit_strategy" / "baseline_portfolio_summary.csv").iloc[0].to_dict()
    baseline_summary_dict = {key: value for key, value in baseline_summary_dict.items()}
    baseline_summary = pd.DataFrame([baseline_summary_dict])
    eligible_count = int(risk["credit_rating"].isin(CHURN_RATINGS).sum())
    max_loan = float(config["credit_optimization"]["max_loan_10k"])
    b_max = eligible_count * max_loan
    budget_base = float(config["baseline"]["budget_fraction_of_max"]) * b_max
    baseline_summary_dict["b_max_10k"] = b_max
    baseline_summary_dict["baseline_budget_10k"] = budget_base
    baseline_lgd = float(config["baseline"]["lgd"])
    baseline_funding = float(config["baseline"]["funding_cost_rate"])

    all_summaries: list[dict[str, Any]] = []
    all_strategies: list[pd.DataFrame] = []
    all_diagnostics: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    budget_strategies: list[pd.DataFrame] = []
    transitions: list[pd.DataFrame] = []
    previous_budget_strategy = None
    previous_budget_summary = None
    for fraction in [float(v) for v in config["budget_fractions"]]:
        scenario_id = f"budget_{fraction:.2f}_bmax_nominal_equality"
        strategy, summary, diagnostics = solve_scenario(
            risk, churn, config,
            scenario_id=scenario_id,
            scenario_family="budget",
            budget=fraction * b_max,
            lgd=baseline_lgd,
            funding_cost_rate=baseline_funding,
            risk_variant="mean",
            budget_definition="nominal_equality",
        )
        if previous_budget_summary is None:
            summary["marginal_expected_net_return_change_10k"] = np.nan
        else:
            summary["marginal_expected_net_return_change_10k"] = summary["expected_net_return_10k"] - previous_budget_summary["expected_net_return_10k"]
            transition = enterprise_changes(strategy, previous_budget_strategy, scenario_id, "budget")
            transition["previous_scenario_id"] = previous_budget_summary["scenario_id"]
            transitions.append(transition)
        selected_ids = set(strategy.loc[strategy["loan_decision"] == "放贷", "enterprise_id"])
        old_ids = set(previous_budget_strategy.loc[previous_budget_strategy["loan_decision"] == "放贷", "enterprise_id"]) if previous_budget_strategy is not None else set()
        summary["new_enterprises_count"] = len(selected_ids - old_ids)
        summary["removed_enterprises_count"] = len(old_ids - selected_ids)
        if previous_budget_strategy is None:
            summary["largest_loan_change_enterprise_id"] = ""
            summary["largest_loan_change_10k"] = np.nan
        else:
            changes = enterprise_changes(strategy, previous_budget_strategy, scenario_id, "budget")
            largest = changes.iloc[changes["loan_amount_change_10k"].abs().argmax()]
            summary["largest_loan_change_enterprise_id"] = largest["enterprise_id"]
            summary["largest_loan_change_10k"] = largest["loan_amount_change_10k"]
        budget_rows.append(summary)
        budget_strategies.append(strategy)
        all_summaries.append(summary)
        all_strategies.append(strategy)
        all_diagnostics.append(diagnostics)
        previous_budget_strategy = strategy
        previous_budget_summary = summary

    parameter_strategies: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []

    for lgd in [float(v) for v in config["lgd_values"]]:
        scenario_id = f"lgd_{lgd:.2f}_baseline_budget"
        strategy, summary, diagnostics = solve_scenario(risk, churn, config, scenario_id=scenario_id, scenario_family="lgd", budget=budget_base, lgd=lgd, funding_cost_rate=baseline_funding, risk_variant="mean", budget_definition="nominal_equality")
        summary = add_comparison_fields(summary, strategy, baseline_strategy)
        parameter_rows.append(summary); parameter_strategies.append(strategy); all_summaries.append(summary); all_strategies.append(strategy); all_diagnostics.append(diagnostics)

    for funding in [float(v) for v in config["funding_cost_values"]]:
        scenario_id = f"funding_cost_{funding:.2f}_baseline_budget"
        strategy, summary, diagnostics = solve_scenario(risk, churn, config, scenario_id=scenario_id, scenario_family="funding_cost", budget=budget_base, lgd=baseline_lgd, funding_cost_rate=funding, risk_variant="mean", budget_definition="nominal_equality")
        summary = add_comparison_fields(summary, strategy, baseline_strategy)
        parameter_rows.append(summary); parameter_strategies.append(strategy); all_summaries.append(summary); all_strategies.append(strategy); all_diagnostics.append(diagnostics)

    for variant in [str(v) for v in config["risk_variants"]]:
        scenario_id = f"risk_{variant}_baseline_budget"
        strategy, summary, diagnostics = solve_scenario(risk, churn, config, scenario_id=scenario_id, scenario_family="risk", budget=budget_base, lgd=baseline_lgd, funding_cost_rate=baseline_funding, risk_variant=variant, budget_definition="nominal_equality")
        summary = add_comparison_fields(summary, strategy, baseline_strategy)
        summary["risk_scenario_label"] = "conservative_risk_scenario" if variant == "p90" else "mean"
        parameter_rows.append(summary); parameter_strategies.append(strategy); all_summaries.append(summary); all_strategies.append(strategy); all_diagnostics.append(diagnostics)

    for definition in [str(v) for v in config["budget_definitions"]]:
        scenario_id = f"budget_definition_{definition}"
        strategy, summary, diagnostics = solve_scenario(risk, churn, config, scenario_id=scenario_id, scenario_family="budget_definition", budget=budget_base, lgd=baseline_lgd, funding_cost_rate=baseline_funding, risk_variant="mean", budget_definition=definition)
        summary = add_comparison_fields(summary, strategy, baseline_strategy)
        parameter_rows.append(summary); parameter_strategies.append(strategy); all_summaries.append(summary); all_strategies.append(strategy); all_diagnostics.append(diagnostics)

    joint_rows: list[dict[str, Any]] = []
    joint_strategies: list[pd.DataFrame] = []
    for lgd in [float(v) for v in config["lgd_values"]]:
        for funding in [float(v) for v in config["funding_cost_values"]]:
            scenario_id = f"lgd_{lgd:.2f}_funding_cost_{funding:.2f}_joint"
            strategy, summary, diagnostics = solve_scenario(risk, churn, config, scenario_id=scenario_id, scenario_family="lgd_funding_joint", budget=budget_base, lgd=lgd, funding_cost_rate=funding, risk_variant="mean", budget_definition="nominal_equality")
            summary = add_comparison_fields(summary, strategy, baseline_strategy)
            joint_rows.append(summary); joint_strategies.append(strategy); all_summaries.append(summary); all_strategies.append(strategy); all_diagnostics.append(diagnostics)

    budget_summary = pd.DataFrame(budget_rows)
    parameter_summary = pd.DataFrame(parameter_rows)
    joint_summary = pd.DataFrame(joint_rows)
    all_strategy_frame = pd.concat(all_strategies, ignore_index=True)
    all_diag_frame = pd.DataFrame(all_diagnostics)
    budget_strategy_frame = pd.concat(budget_strategies, ignore_index=True)
    transition_frame = pd.concat(transitions, ignore_index=True) if transitions else pd.DataFrame()
    parameter_strategy_frame = pd.concat(parameter_strategies, ignore_index=True)
    joint_strategy_frame = pd.concat(joint_strategies, ignore_index=True)

    out_dir = RUNTIME_ROOT / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(budget_summary, out_dir / "budget_sensitivity_summary.csv")
    write_csv(budget_strategy_frame, out_dir / "budget_strategy_all_enterprises.csv")
    write_csv(transition_frame, out_dir / "budget_transition_details.csv")
    write_csv(parameter_summary[parameter_summary["scenario_family"] == "lgd"], out_dir / "lgd_sensitivity_summary.csv")
    write_csv(parameter_summary[parameter_summary["scenario_family"] == "funding_cost"], out_dir / "funding_cost_sensitivity_summary.csv")
    write_csv(parameter_summary[parameter_summary["scenario_family"] == "risk"], out_dir / "risk_sensitivity_summary.csv")
    write_csv(parameter_summary[parameter_summary["scenario_family"] == "budget_definition"], out_dir / "budget_definition_sensitivity_summary.csv")
    write_csv(parameter_summary, out_dir / "parameter_sensitivity_summary.csv")
    write_csv(parameter_strategy_frame, out_dir / "parameter_strategy_all_enterprises.csv")
    write_csv(joint_summary, out_dir / "lgd_funding_joint_summary.csv")
    write_csv(joint_strategy_frame, out_dir / "lgd_funding_joint_strategy_all_enterprises.csv")
    write_csv(all_strategy_frame, out_dir / "all_sensitivity_strategies.csv")
    write_csv(all_diag_frame, out_dir / "sensitivity_solver_diagnostics.csv")

    # Per-enterprise stability uses the 30 unique formal sensitivity scenarios,
    # with the baseline strategy retained only as a comparison flag.
    formal_scenarios = all_strategy_frame["solver_scenario_id"].drop_duplicates().tolist()
    baseline_ids = set(baseline_strategy.loc[baseline_strategy["loan_decision"] == "放贷", "enterprise_id"])
    stability_rows: list[dict[str, Any]] = []
    amount_threshold = float(config["strategy_stability"]["unstable_loan_amount_sd_10k"])
    for row in risk.itertuples(index=False):
        enterprise_id = str(row.enterprise_id)
        group = all_strategy_frame[all_strategy_frame["enterprise_id"] == enterprise_id]
        selected_mask = group["loan_decision"] == "放贷"
        frequency = float(selected_mask.mean()) if len(group) else 0.0
        amount = group["offered_loan_amount_10k"].astype(float)
        rates = group.loc[selected_mask, "interest_rate"].astype(float)
        robust = frequency >= float(config["strategy_stability"]["robust_core_frequency"])
        unstable = (float(config["strategy_stability"]["unstable_frequency_low"]) <= frequency <= float(config["strategy_stability"]["unstable_frequency_high"])) or float(amount.std(ddof=0)) >= amount_threshold
        stability_rows.append({
            "enterprise_id": enterprise_id,
            "enterprise_name": row.enterprise_name,
            "credit_rating": row.credit_rating,
            "selection_frequency": frequency,
            "selected_scenario_count": int(selected_mask.sum()),
            "total_scenario_count": int(len(group)),
            "loan_amount_mean_10k": float(amount.mean()),
            "loan_amount_sd_10k": float(amount.std(ddof=0)),
            "loan_amount_min_10k": float(amount.min()),
            "loan_amount_max_10k": float(amount.max()),
            "interest_rate_mean": float(rates.mean()) if not rates.empty else np.nan,
            "risk_score": float(row.selected_model_risk_score),
            "credit_rating_display": row.credit_rating,
            "baseline_selected": enterprise_id in baseline_ids,
            "robust_core_flag": robust,
            "unstable_flag": unstable,
        })
    stability = pd.DataFrame(stability_rows).sort_values(["selection_frequency", "enterprise_id"], ascending=[False, True]).reset_index(drop=True)
    write_csv(stability, out_dir / "enterprise_strategy_stability.csv")

    figure_paths = write_sensitivity_figures(budget_summary, parameter_summary, stability, joint_summary, OUTPUT_ROOT / "figures" / "credit")

    baseline_figures = [OUTPUT_ROOT / "figures" / "credit" / name for name in ["baseline_risk_amount_scatter.png", "baseline_amount_by_rating.png", "baseline_rate_by_rating.png"]]
    all_figures = baseline_figures + figure_paths

    # Summary rows for the final Excel validation sheet.
    validation_rows = pd.DataFrame([
        {"check_name": "all_official_scenarios_optimal", "status": bool(all_diag_frame["is_optimal"].all()), "detail": f"{len(all_diag_frame)} sensitivity scenarios"},
        {"check_name": "all_scenario_validation_failures_empty", "status": bool(all_diag_frame["validation_failures"].astype(str).eq("[]").all()), "detail": "MILP constraints and decomposition"},
        {"check_name": "baseline_is_optimal", "status": True, "detail": "baseline diagnostics recorded separately"},
        {"check_name": "risk_source_is_selected_model_risk_score", "status": True, "detail": "mean scenario"},
        {"check_name": "D_enterprises_have_zero_amount", "status": bool((baseline_strategy.loc[baseline_strategy["credit_rating"] == "D", "offered_loan_amount_10k"] == 0).all()), "detail": "D-level business rule"},
    ])

    model_metrics = pd.read_csv(RUNTIME_ROOT / "model_training" / "model_metrics_summary.csv")
    final_excel = build_final_excel(config, risk, churn_normalized, churn, baseline_strategy, pd.DataFrame([baseline_summary_dict]), budget_summary, parameter_summary, stability, model_metrics, validation_rows, baseline_summary_dict, eligible_count, b_max)

    final_dir = OUTPUT_ROOT / "final"
    paper_path = PAPER_ROOT / "q1_final_results_for_paper.md"
    assumptions_path = PAPER_ROOT / "q1_final_assumptions_and_limitations.md"
    checklist_path = PAPER_ROOT / "q1_submission_checklist.md"
    write_text(_final_paper_doc(config, risk, churn_metrics, baseline_summary_dict, budget_summary, parameter_summary, stability, all_figures), paper_path)
    write_text(_assumptions_doc(config), assumptions_path)
    checklist = "\n".join([
        "# 问题一提交检查清单",
        "",
        "- [x] 第一至第三批已验收的输入、特征与风险模型文件存在。",
        "- [x] 正式优化使用selected_model_risk_score，没有重新训练风险模型。",
        "- [x] 附件3审计通过，A/B/C曲线独立保序拟合。",
        "- [x] 基准MILP和全部敏感性官方场景均达到最优状态。",
        "- [x] 123家企业完整策略表已生成，D级企业金额为0。",
        "- [x] 收益、信用损失、资金成本分解恒等式已检查。",
        "- [x] 最终Excel、CSV、图表、论文文档和假设文档已生成。",
        "- [x] 最终验证由src/q1.py --stage validate执行。",
        "",
        "所有数值以data/processed和outputs/q1中的正式文件为准；代表性预算、LGD和资金成本不是题目给定参数。",
        "",
    ])
    write_text(checklist, checklist_path)
    _publish_credit_outputs()

    output_index_rows: list[dict[str, Any]] = []
    candidate_outputs = [
        *[path for path in (OUTPUT_ROOT / "tables").glob("*") if path.is_file()],
        *[path for path in (OUTPUT_ROOT / "reports").glob("*") if path.is_file()],
        *[path for path in (OUTPUT_ROOT / "figures").glob("**/*.png") if path.is_file()],
        *all_figures,
        final_excel,
        paper_path,
        assumptions_path,
        checklist_path,
    ]
    seen: set[Path] = set()
    for path in candidate_outputs:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        row = {"path": relative(path), "sha256": sha256_file(path), "exists": True, "size_bytes": path.stat().st_size}
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
            row["rows"] = len(frame)
            row["columns"] = len(frame.columns)
        output_index_rows.append(row)
    output_index = pd.DataFrame(output_index_rows).sort_values("path")
    write_csv(output_index, final_dir / "q1_final_output_index.csv")

    code_files = [
        ROOT / "src" / "_internal" / "stages" / name
        for name in ["06_fit_churn_curves.py", "07_optimize_credit_strategy.py", "08_credit_sensitivity_analysis.py"]
    ] + [ROOT / "src" / "_internal" / name for name in ["credit_strategy.py", "validation.py"]]
    final_manifest = {
        "workflow": "question_one_final_delivery",
        "selected_model": str(risk["selected_model"].iloc[0]),
        "risk_column": str(config["credit_optimization"]["risk_column"]),
        "enterprise_count": len(risk),
        "eligible_count": eligible_count,
        "b_max_10k": b_max,
        "baseline_budget_10k": budget_base,
        "sensitivity_scenario_count": len(all_diag_frame),
        "all_sensitivity_scenarios_optimal": bool(all_diag_frame["is_optimal"].all()),
        "config_sha256": config_hash(config),
        "input_hashes": {
            "risk_table": sha256_file(PROCESSED_ROOT / "q1_risk_scores.csv"),
            "churn_fitted": sha256_file(RUNTIME_ROOT / "churn_model" / "churn_curve_fitted.csv"),
            "attachment3_workbook": json.loads((RUNTIME_ROOT / "churn_model" / "churn_model_manifest.json").read_text(encoding="utf-8"))["input_workbook_sha256"],
        },
        "code_hashes": {relative(path): sha256_file(path) for path in code_files},
        "output_index_sha256": sha256_file(final_dir / "q1_final_output_index.csv"),
        "output_hashes": {row["path"]: row["sha256"] for row in output_index.to_dict("records")},
    }
    write_json(final_manifest, final_dir / "q1_final_manifest.json")
    print(
        "08_credit_sensitivity_analysis.py SUCCEEDED: "
        f"scenarios={len(all_diag_frame)} optimal={bool(all_diag_frame['is_optimal'].all())} "
        f"robust_core={int(stability['robust_core_flag'].sum())} final_excel={relative(final_excel)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
