"""Descriptive EDA for question-one enterprise features.

No classifier, probability model, churn curve or credit optimization is run in
this module.  The plotting backend uses deterministic primitives so the script
also runs in the minimal project environment without matplotlib.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from _internal.data_pipeline import (
    OUTPUT_ROOT,
    PROCESSED_ROOT,
    RUNTIME_ROOT,
    ROOT,
    load_config,
    output_paths,
    setup_logging,
    write_csv,
    write_text,
)
from _internal.plotting import line, rect, save_png, text


FEATURE_FILE = PROCESSED_ROOT / "q1_enterprise_features.csv"
VALIDATION_DIR = RUNTIME_ROOT / "feature_validation"
FIGURE_DIR = OUTPUT_ROOT / "figures" / "eda"
MAIN_FEATURES = [
    "business_scale_10k", "sales_growth_trend", "sales_monthly_cv", "invoice_activity_per_month",
    "sales_return_rate", "void_invoice_rate", "customer_hhi", "supplier_hhi",
    "customer_count", "supplier_count",
]
FEATURE_LABELS = {
    "business_scale_10k": "经营规模(万元)",
    "sales_scale_10k": "销售规模(万元)",
    "purchase_scale_10k": "采购规模(万元)",
    "net_sales_10k": "净销售额(万元)",
    "operating_net_inflow_proxy_10k": "经营差额代理(万元)",
    "sales_growth_trend": "销售增长趋势",
    "sales_monthly_cv": "销售月度波动CV",
    "invoice_activity_per_month": "月均交易活跃度",
    "sales_return_rate": "销售退货率",
    "purchase_return_rate": "采购退货率",
    "void_invoice_rate": "作废率",
    "zero_amount_invoice_rate": "零金额发票比例",
    "customer_count": "客户数量",
    "supplier_count": "供应商数量",
    "customer_hhi": "客户HHI",
    "supplier_hhi": "供应商HHI",
    "max_customer_share": "最大客户占比",
    "max_supplier_share": "最大供应商占比",
    "purchase_sales_ratio": "进销比",
    "active_month_ratio": "活跃月份比例",
    "longest_active_streak_ratio": "最长连续交易比例",
}
PALETTE = {"0": "#4472C4", "1": "#C00000", "A": "#70AD47", "B": "#FFC000", "C": "#ED7D31", "D": "#7030A0"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Question-one descriptive EDA.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--feature-file", type=Path, default=FEATURE_FILE)
    return parser.parse_args()


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _transform_values(values: pd.Series, feature: str) -> tuple[np.ndarray, str]:
    """Use a labelled robust plotting scale while retaining raw statistics."""

    array = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if feature.endswith("_10k") or feature in {"business_scale_10k", "customer_count", "supplier_count", "invoice_activity_per_month", "sales_monthly_cv"}:
        return np.sign(array) * np.log1p(np.abs(array)), "signed log1p"
    return array, "raw"


def _scale(value: float, lower: float, upper: float, start: float, end: float) -> float:
    if not np.isfinite(value):
        return (start + end) / 2
    if upper <= lower:
        return (start + end) / 2
    ratio = np.clip((value - lower) / (upper - lower), 0, 1)
    return start + ratio * (end - start)


def _axes(elements: list[dict[str, Any]], x: float, y: float, width: float, height: float,
          x_label: str = "", y_label: str = "", title: str = "") -> None:
    """Draw a lightweight chart frame."""

    line(elements, x, y + height, x + width, y + height, color="#666666", width=2)
    line(elements, x, y, x, y + height, color="#666666", width=2)
    if title:
        text(elements, x + width / 2, y - 48, title, size=22, align="center")
    if x_label:
        text(elements, x + width / 2, y + height + 58, x_label, size=16, align="center", color="#444444")
    if y_label:
        text(elements, x - 70, y + height / 2 - 10, y_label, size=16, align="center", color="#444444")


def _bar_chart(path: Path, categories: list[str], values: list[float], title: str, y_label: str,
               *, labels: list[str] | None = None, colors: list[str] | None = None) -> None:
    width, height = 1500, 900
    elements: list[dict[str, Any]] = []
    text(elements, width / 2, 25, title, size=32, align="center")
    x, y, w, h = 170, 130, 1200, 600
    _axes(elements, x, y, w, h, "类别", y_label)
    max_value = max(values) if values else 1
    max_value = max(max_value * 1.15, 1)
    bar_w = w / max(len(categories), 1) * .55
    colors = colors or ["#4472C4"] * len(values)
    for index, (category, value) in enumerate(zip(categories, values)):
        center = x + (index + .5) * w / len(categories)
        top = _scale(value, 0, max_value, y + h, y)
        rect(elements, center - bar_w / 2, top, bar_w, y + h - top, fill=colors[index], outline="#333333")
        text(elements, center, y + h + 20, category, size=21, align="center")
        label = labels[index] if labels else f"{value:g}"
        text(elements, center, top - 32, label, size=19, align="center")
    save_png(path, width, height, elements)


def _grouped_bar(path: Path, categories: list[str], series: dict[str, list[float]], title: str,
                 y_label: str, *, percentage: bool = False) -> None:
    width, height = 1600, 950
    elements: list[dict[str, Any]] = []
    text(elements, width / 2, 25, title, size=32, align="center")
    x, y, w, h = 170, 130, 1250, 620
    _axes(elements, x, y, w, h, "类别", y_label)
    max_value = max(max(vals) for vals in series.values()) if series else 1
    max_value = max(max_value * 1.18, 1)
    names = list(series)
    group_width = w / len(categories)
    bar_w = group_width / (len(names) + 1) * .72
    for j, name in enumerate(names):
        color = PALETTE.get(name, "#4472C4")
        for i, value in enumerate(series[name]):
            center = x + (i + .5) * group_width + (j - (len(names) - 1) / 2) * bar_w
            top = _scale(value, 0, max_value, y + h, y)
            rect(elements, center - bar_w / 2, top, bar_w, y + h - top, fill=color, outline="#333333")
            label = f"{value:.1%}" if percentage else f"{value:g}"
            text(elements, center, top - 27, label, size=14, align="center")
    for i, category in enumerate(categories):
        center = x + (i + .5) * group_width
        text(elements, center, y + h + 20, category, size=20, align="center")
    legend_x = x + w - 180 * len(names)
    for j, name in enumerate(names):
        rect(elements, legend_x + j * 180, 90, 22, 22, fill=PALETTE.get(name, "#4472C4"))
        text(elements, legend_x + 30 + j * 180, 88, f"{name}组", size=18)
    save_png(path, width, height, elements)


def _histogram_chart(path: Path, frame: pd.DataFrame) -> None:
    width, height = 2200, 1900
    elements: list[dict[str, Any]] = []
    text(elements, width / 2, 24, "主要经营特征总体分布（绘图尺度按特征标注）", size=32, align="center")
    for index, feature in enumerate(MAIN_FEATURES):
        row, col = divmod(index, 2)
        x, y, w, h = 150 + col * 1090, 160 + row * 345, 780, 190
        values, scale_name = _transform_values(frame[feature], feature)
        if len(values) == 0:
            continue
        low, high = float(np.nanmin(values)), float(np.nanmax(values))
        if low == high:
            low, high = low - .5, high + .5
        counts, edges = np.histogram(values, bins=12, range=(low, high))
        max_count = max(int(counts.max()), 1)
        _axes(elements, x, y, w, h, scale_name, "n", FEATURE_LABELS[feature])
        for j, count in enumerate(counts):
            left = _scale(edges[j], low, high, x, x + w)
            right = _scale(edges[j + 1], low, high, x, x + w)
            top = _scale(count, 0, max_count * 1.1, y + h, y)
            rect(elements, left + 1, top, max(1, right - left - 2), y + h - top, fill="#5B9BD5", outline="#FFFFFF")
        median = float(np.median(values))
        median_x = _scale(median, low, high, x, x + w)
        line(elements, median_x, y, median_x, y + h, color="#C00000", width=3)
        text(elements, x + w + 30, y + 40, f"中位数={median:.3g}", size=16, color="#C00000")
        text(elements, x + w + 30, y + 75, f"n={len(values)}", size=16)
    save_png(path, width, height, elements)


def _boxplot_chart(path: Path, frame: pd.DataFrame) -> None:
    width, height = 2200, 1900
    elements: list[dict[str, Any]] = []
    text(elements, width / 2, 24, "主要特征按违约标签的箱线图（仅描述性差异）", size=32, align="center")
    for index, feature in enumerate(MAIN_FEATURES):
        row, col = divmod(index, 2)
        x, y, w, h = 150 + col * 1090, 160 + row * 345, 780, 190
        values, scale_name = _transform_values(frame[feature], feature)
        # Group labels already identify the horizontal dimension; a second axis
        # label would compete for the same narrow band below every subplot.
        _axes(elements, x, y, w, h, "", scale_name, FEATURE_LABELS[feature])
        groups: list[tuple[str, np.ndarray]] = []
        for label in [0, 1]:
            raw = frame.loc[frame["default_label"].eq(label), feature]
            transformed, _ = _transform_values(raw, feature)
            if len(transformed):
                groups.append((str(label), transformed))
        for j, (label, group) in enumerate(groups):
            q1, med, q3 = np.quantile(group, [.25, .5, .75])
            iqr = q3 - q1
            lo = max(float(group.min()), float(q1 - 1.5 * iqr))
            hi = min(float(group.max()), float(q3 + 1.5 * iqr))
            center = x + (j + 1) * w / (len(groups) + 1)
            y_q1 = _scale(q1, values.min(), values.max(), y + h, y)
            y_q3 = _scale(q3, values.min(), values.max(), y + h, y)
            y_med = _scale(med, values.min(), values.max(), y + h, y)
            y_lo = _scale(lo, values.min(), values.max(), y + h, y)
            y_hi = _scale(hi, values.min(), values.max(), y + h, y)
            rect(elements, center - 28, y_q3, 56, max(2, y_q1 - y_q3), fill=PALETTE[label], outline="#333333")
            line(elements, center - 28, y_med, center + 28, y_med, color="#FFFFFF", width=3)
            line(elements, center, y_q3, center, y_hi, color="#333333", width=2)
            line(elements, center, y_q1, center, y_lo, color="#333333", width=2)
            line(elements, center - 15, y_hi, center + 15, y_hi, color="#333333", width=2)
            line(elements, center - 15, y_lo, center + 15, y_lo, color="#333333", width=2)
            text(elements, center, y + h + 22, f"{label}组 (n={len(group)})", size=16, align="center")
    save_png(path, width, height, elements)


def _correlation_heatmap(path: Path, frame: pd.DataFrame, feature_names: list[str]) -> None:
    width, height = 2200, 1900
    elements: list[dict[str, Any]] = []
    corr = frame[feature_names].corr(method="spearman")
    n = len(feature_names)
    left, top, cell = 400, 410, min(64, 1344 / max(n, 1))
    text(elements, width / 2, 24, "候选特征 Spearman 相关性热力图", size=32, align="center")
    for i, feature in enumerate(feature_names):
        label = FEATURE_LABELS.get(feature, feature)
        text(elements, left - 18, top + i * cell + cell / 2 - 10, label, size=15, align="right")
        text(elements, left + i * cell + cell / 2, top - 18, label, size=15, angle=-62)
        for j, other in enumerate(feature_names):
            rho = _safe_float(corr.loc[feature, other])
            if not np.isfinite(rho):
                fill = "#E7E6E6"
            elif rho >= 0:
                intensity = int(255 - min(1, rho) * 170)
                fill = f"#{intensity:02X}{intensity:02X}FF"
            else:
                intensity = int(255 - min(1, abs(rho)) * 170)
                fill = f"#FF{intensity:02X}{intensity:02X}"
            rect(elements, left + j * cell, top + i * cell, cell, cell, fill=fill, outline="#FFFFFF")
            if n <= 21:
                text(elements, left + j * cell + cell / 2, top + i * cell + cell / 2 - 9, f"{rho:.1f}" if np.isfinite(rho) else "—", size=12, align="center")
    text(elements, left, top + n * cell + 45, "蓝色 = 正相关，红色 = 负相关；|rho| > 0.85 的组合见 CSV，不自动删除", size=18)
    save_png(path, width, height, elements)


def _feature_group_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in MAIN_FEATURES:
        for label, group in frame.groupby("default_label", sort=True):
            values = pd.to_numeric(group[feature], errors="coerce").dropna()
            q1, q3 = values.quantile([.25, .75]) if len(values) else (np.nan, np.nan)
            iqr = q3 - q1 if len(values) else np.nan
            rows.append({
                "feature": feature, "feature_label": FEATURE_LABELS[feature], "default_label": int(label),
                "group_name": "违约" if int(label) == 1 else "未违约", "n": int(len(values)),
                "missing_count": int(group[feature].isna().sum()),
                "min": float(values.min()) if len(values) else np.nan,
                "q25": float(q1) if len(values) else np.nan,
                "median": float(values.median()) if len(values) else np.nan,
                "q75": float(q3) if len(values) else np.nan,
                "max": float(values.max()) if len(values) else np.nan,
                "iqr_outlier_count": int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum()) if len(values) else 0,
            })
    return pd.DataFrame(rows)


def _write_initial_findings(frame: pd.DataFrame, group_stats: pd.DataFrame, out_path: Path) -> None:
    """Write a paper-facing findings note with facts, interpretations and hypotheses separated."""

    audit = pd.read_csv(RUNTIME_ROOT / "audit" / "invoice_status_sign_summary.csv", encoding="utf-8-sig")
    status_lines = []
    for direction, name in [("input", "进项"), ("output", "销项")]:
        subset = audit[audit["direction"].eq(direction)]
        total = int(subset["record_count"].sum())
        valid = int(subset.loc[subset["status_canonical"].eq("有效"), "record_count"].sum())
        void = int(subset.loc[subset["status_canonical"].eq("作废"), "record_count"].sum())
        negative = int(subset["negative_amount_count"].sum())
        zero = int(subset["zero_amount_count"].sum())
        status_lines.append(f"- {name}发票共{total:,}条；有效{valid:,}条；作废{void:,}条；负数金额记录{negative:,}条；零金额记录{zero:,}条。")
    rating_counts = frame["credit_rating"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    default_counts = frame["default_label"].value_counts().reindex([0, 1], fill_value=0)
    contingency = pd.crosstab(frame["credit_rating"], frame["default_label"]).reindex(index=["A", "B", "C", "D"], columns=[0, 1], fill_value=0)
    diff_lines = []
    for feature in MAIN_FEATURES:
        med0 = group_stats.loc[(group_stats.feature == feature) & (group_stats.default_label == 0), "median"].iloc[0]
        med1 = group_stats.loc[(group_stats.feature == feature) & (group_stats.default_label == 1), "median"].iloc[0]
        relation = "高于" if med1 > med0 else "低于" if med1 < med0 else "接近"
        diff_lines.append(f"- {FEATURE_LABELS[feature]}：违约组中位数{med1:.6g}，未违约组{med0:.6g}，违约组{relation}未违约组。")
    validation_summary = json.loads((VALIDATION_DIR / "feature_validation_summary.json").read_text(encoding="utf-8"))
    high = pd.read_csv(VALIDATION_DIR / "high_correlation_pairs.csv", encoding="utf-8-sig")
    zero_summary_path = VALIDATION_DIR / "zero_amount_invoice_business_summary.csv"
    zero_summary = pd.read_csv(zero_summary_path, encoding="utf-8-sig") if zero_summary_path.exists() else pd.DataFrame()
    zero_raw = zero_summary.loc[
        zero_summary["stage"].eq("raw_detail")
        & zero_summary["direction"].eq("all")
        & zero_summary["status_scope"].eq("all")
    ].iloc[0] if not zero_summary.empty else None
    lines = [
        "# 问题一初步数据发现",
        "",
        "本文件只记录当前程序实际计算出的数据事实和谨慎解释，不包含风险模型指标、企业风险预测值或贷款策略。",
        "",
        "## 1. 直接由数据计算得到的事实",
        "",
        f"- 企业级特征表为{len(frame)}家企业、{frame.shape[1]}列；企业代号唯一，企业信息表的企业全部匹配。",
        f"- 标签：未违约{int(default_counts[0])}家，违约{int(default_counts[1])}家，违约占比{default_counts[1] / len(frame):.2%}。这构成少数类不平衡，后续不能只报告准确率。",
        *status_lines,
        "- 发票日期在进项和销项审计表中均无非法日期；审计报告记录了重复行、金额恒等式超差和分位数外金额，原始记录未被删除。",
        "",
        "### 评级与违约列联表（企业数）",
        "",
        "| 评级 | 未违约 | 违约 | 合计 | 违约率 |",
        "|---|---:|---:|---:|---:|",
    ]
    for rating in ["A", "B", "C", "D"]:
        row = contingency.loc[rating]
        total = int(row.sum())
        lines.append(f"| {rating} | {int(row[0])} | {int(row[1])} | {total} | {row[1] / total:.2%} |")
    lines.extend([
        "",
        "- 评级企业数：" + "、".join(f"{r}级{int(rating_counts[r])}家" for r in ["A", "B", "C", "D"]) + "。",
        "- 当前验收抽查企业：" + "、".join(validation_summary["sampled_enterprises"]) + "；每个抽查特征的绝对/相对误差均见 `feature_recalculation_check.csv`。",
        "",
        "## 2. 主要特征分布与描述性差异",
        "",
        "以下只表述分布差异或相关，不作因果解释。",
        *diff_lines,
        "",
        "- 客户和供应商集中度由HHI及最大对手占比共同描述；两类指标出现高相关组合时，不应在未审查的情况下同时解释系数。",
        f"- Spearman绝对相关系数大于0.85的特征组合共有{len(high)}组，详见 `high_correlation_pairs.csv`；本轮不自动删除。",
        "",
        "## 3. 合理解释（不是已验证因果）",
        "",
        "- 评级与违约的列联关系可以用于辅助校准和D级规则核对，但评级是银行人工先验，且附件2没有该字段，因此不能直接作为主违约模型行为输入。",
        "- 规模、交易活跃度和对手数量的差异可能同时反映企业规模、成立时间或行业结构；这只是混杂解释，不能写成规模导致或防止违约。",
        "- 退货率、作废率和波动率的组间差异只能作为风险假设的线索，仍需企业级交叉验证检验稳定性。",
        "",
        "## 4. 需要后续处理或核查",
        "",
        f"- 当前全常数构造特征：{', '.join(validation_summary['constant_features']) if validation_summary['constant_features'] else '无'}；主模型候选中的全常数特征：{', '.join(validation_summary.get('primary_constant_features', [])) if validation_summary.get('primary_constant_features') else '无'}。zero_amount_invoice_rate虽为全常数，但已按审计型排除，不阻断主模型验收。",
        f"- 零金额业务核验由程序实际生成；原始明细层精确零值数为{int(zero_raw['exact_zero_count'])}、容差零值数为{int(zero_raw['tolerance_zero_count'])}。完整的去重、原子发票、方向、状态和企业统计见zero_amount_invoice_business_check.csv。" if zero_raw is not None else "- 零金额业务核验报告尚未生成。",
        "- 金额、交易数量和月度波动可能右偏，后续按字典在训练折内采用log1p/有符号log1p、1%/99%缩尾或RobustScaler；不得覆盖本轮原始特征。",
        "- 规模分项、HHI与最大对手占比的高相关需要建模手决定变量组策略；当前保留并输出诊断。",
        "",
        "## 5. 尚不能形成的结论",
        "",
        "- 不能据此断言任何特征导致违约，也不能给出企业未来一年PD、最优利率、额度或收益。",
        "- 不能以评级列联表替代发票行为模型验证；也不能把当前样本的描述性中位数差异当作可迁移规律。",
        "- 尚未训练风险模型、未做重复分层交叉验证、未拟合利率—流失率曲线、未进行预算优化，因此尚无模型指标或信贷策略结果。",
        "",
        "## 6. 下一阶段需要验证的假设",
        "",
        "1. 发票行为特征在企业级重复分层交叉验证中能否稳定区分违约与未违约。",
        "2. 训练折内的缩尾、对数和标准化是否改善PR-AUC、Brier和LogLoss及校准。",
        "3. 去除评级、标签、身份列和全常数特征后，模型结论是否仍稳健。",
        "4. 边界月份、重复行和金额恒等式超差记录的处理口径是否改变描述性与模型结果。",
        "",
    ])
    write_text("\n".join(lines), out_path)


def run_eda(config: dict[str, Any], feature_path: Path) -> dict[str, Any]:
    """Generate descriptive tables, PNG figures and the initial findings note."""

    paths = output_paths(config)
    logger = setup_logging("04_exploratory_analysis", config, paths)
    feature_path = feature_path.resolve()
    frame = pd.read_csv(feature_path, encoding="utf-8-sig")
    missing = [feature for feature in MAIN_FEATURES + ["credit_rating", "default_label"] if feature not in frame.columns]
    if missing:
        raise ValueError(f"EDA输入缺少字段: {missing}")
    if len(frame) != 123 or frame["enterprise_id"].duplicated().any():
        raise ValueError("EDA输入不满足123家企业/主键唯一契约")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for file in FIGURE_DIR.glob("*.png"):
        # Only remove files created by this script's fixed names; no raw data is touched.
        if file.name in {"class_distribution.png", "rating_distribution.png", "rating_default_distribution.png", "rating_default_rate.png", "main_feature_distributions.png", "main_features_by_default.png", "feature_correlation_heatmap.png"}:
            file.unlink()

    default_counts = frame["default_label"].value_counts().reindex([0, 1], fill_value=0)
    rating_counts = frame["credit_rating"].value_counts().reindex(["A", "B", "C", "D"], fill_value=0)
    contingency = pd.crosstab(frame["credit_rating"], frame["default_label"]).reindex(index=["A", "B", "C", "D"], columns=[0, 1], fill_value=0)
    contingency_out = contingency.reset_index().rename(columns={0: "not_default_count", 1: "default_count", "credit_rating": "credit_rating"})
    contingency_out["total_count"] = contingency_out["not_default_count"] + contingency_out["default_count"]
    contingency_out["default_rate"] = contingency_out["default_count"] / contingency_out["total_count"]
    write_csv(contingency_out, VALIDATION_DIR / "eda_rating_default_contingency.csv")
    group_stats = _feature_group_stats(frame)
    write_csv(group_stats, VALIDATION_DIR / "eda_main_feature_group_stats.csv")

    chart_rows = [
        {"file": "outputs/q1/figures/eda/class_distribution.png", "title": "违约与未违约企业数量", "source": "data/processed/q1_enterprise_features.csv:default_label"},
        {"file": "outputs/q1/figures/eda/rating_distribution.png", "title": "A/B/C/D评级企业数量", "source": "data/processed/q1_enterprise_features.csv:credit_rating"},
        {"file": "outputs/q1/figures/eda/rating_default_distribution.png", "title": "评级—是否违约交叉分布（企业数）", "source": "data/processed/_runtime/feature_validation/eda_rating_default_contingency.csv"},
        {"file": "outputs/q1/figures/eda/rating_default_rate.png", "title": "各评级违约率（括号显示样本数）", "source": "data/processed/_runtime/feature_validation/eda_rating_default_contingency.csv"},
        {"file": "outputs/q1/figures/eda/main_feature_distributions.png", "title": "主要特征总体分布", "source": "data/processed/q1_enterprise_features.csv:main features"},
        {"file": "outputs/q1/figures/eda/main_features_by_default.png", "title": "主要特征按违约标签箱线图", "source": "data/processed/q1_enterprise_features.csv:main features/default_label"},
        {"file": "outputs/q1/figures/eda/feature_correlation_heatmap.png", "title": "候选特征Spearman相关性", "source": "data/processed/q1_enterprise_features.csv:main features"},
    ]
    _bar_chart(FIGURE_DIR / "class_distribution.png", ["未违约", "违约"], [int(default_counts[0]), int(default_counts[1])], "违约与未违约企业数量", "企业数", labels=[f"n={int(default_counts[0])}", f"n={int(default_counts[1])}"], colors=[PALETTE["0"], PALETTE["1"]])
    _bar_chart(FIGURE_DIR / "rating_distribution.png", ["A", "B", "C", "D"], [int(rating_counts[r]) for r in ["A", "B", "C", "D"]], "A/B/C/D评级企业数量", "企业数", labels=[f"n={int(rating_counts[r])}" for r in ["A", "B", "C", "D"]], colors=[PALETTE[r] for r in ["A", "B", "C", "D"]])
    _grouped_bar(FIGURE_DIR / "rating_default_distribution.png", ["A", "B", "C", "D"], {"0": contingency[0].astype(float).tolist(), "1": contingency[1].astype(float).tolist()}, "评级—是否违约交叉分布（显示企业数）", "企业数")
    rates = (contingency[1] / contingency.sum(axis=1)).fillna(0).tolist()
    _bar_chart(FIGURE_DIR / "rating_default_rate.png", ["A", "B", "C", "D"], rates, "各评级违约率（标签显示样本数）", "违约率", labels=[f"{r:.1%} (n={int(rating_counts.iloc[i])})" for i, r in enumerate(rates)], colors=[PALETTE[r] for r in ["A", "B", "C", "D"]])
    _histogram_chart(FIGURE_DIR / "main_feature_distributions.png", frame)
    _boxplot_chart(FIGURE_DIR / "main_features_by_default.png", frame)
    # Include all 21 dictionary features in the heatmap, not only the ten plotted distributions.
    feature_names = [column for column in frame.columns if column in config["features"]["names"]]
    _correlation_heatmap(FIGURE_DIR / "feature_correlation_heatmap.png", frame, feature_names)
    write_csv(pd.DataFrame(chart_rows), VALIDATION_DIR / "eda_chart_index.csv")
    write_text("# 问题一EDA图表索引\n\n" + "\n".join([f"- `{row['file']}`：{row['title']}；数据源：{row['source']}。" for row in chart_rows]) + "\n", OUTPUT_ROOT / "reports" / "q1_eda_chart_index.md")
    _write_initial_findings(frame, group_stats, OUTPUT_ROOT / "reports" / "q1_initial_findings.md")
    logger.info("EDA_STATUS=PASS figures=%d rows=%d", len(chart_rows), len(frame))
    return {"status": "PASS", "figure_count": len(chart_rows), "default_counts": default_counts.to_dict(), "rating_counts": rating_counts.to_dict()}


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    try:
        summary = run_eda(config, args.feature_file)
    except Exception as exc:
        print(f"04_exploratory_analysis.py FAILED: {exc}")
        return 1
    print(f"04_exploratory_analysis.py SUCCEEDED: figures={summary['figure_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
