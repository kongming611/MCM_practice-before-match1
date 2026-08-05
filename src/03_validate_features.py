"""Strict validation and raw-invoice spot checks for question-one features.

This script validates the already-built enterprise table.  It does not fit a
classifier, estimate a churn curve, or make a credit decision.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from q1_common import (
    ROOT,
    DataQualityError,
    collapse_invoice_lines,
    load_config,
    longest_consecutive_run,
    output_paths,
    read_role_sheet,
    sha256_file,
    setup_logging,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_FEATURE_FILE = ROOT / "results" / "features" / "enterprise_features_123.csv"
DEFAULT_DICTIONARY = ROOT / "docs" / "q1_feature_dictionary.md"
DEFAULT_REPORT = ROOT / "results" / "feature_validation" / "feature_validation_report.md"
COMPARISON_ATOL = 1e-6
COMPARISON_RTOL = 1e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate question-one enterprise features.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "q1.yaml")
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    return parser.parse_args()


def _md_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    """Render a compact deterministic Markdown table."""

    if frame.empty:
        return "_无记录_"
    view = frame.head(max_rows)
    columns = [str(c) for c in view.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in view.itertuples(index=False, name=None):
        values: list[str] = []
        for value in row:
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                values.append(json.dumps(value, ensure_ascii=False, default=str))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n> 仅展示前{max_rows}行，共{len(frame)}行。")
    return "\n".join(lines)


def _load_feature_table(path: Path) -> pd.DataFrame:
    """Read the output table without altering it."""

    if not path.exists():
        raise FileNotFoundError(f"企业级特征表不存在: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if frame.empty:
        raise DataQualityError("企业级特征表为空")
    return frame


def _load_enterprise_info(config: dict[str, Any], input_path: Path) -> pd.DataFrame:
    """Read the enterprise sheet and normalize only identifiers and labels."""

    frame, _ = read_role_sheet(input_path, "enterprise", config)
    info = frame.rename(
        columns={
            "企业代号": "enterprise_id",
            "企业名称": "enterprise_name",
            "信誉评级": "credit_rating",
            "是否违约": "default_label_raw",
        }
    ).copy()
    for column in ["enterprise_id", "enterprise_name", "credit_rating", "default_label_raw"]:
        info[column] = info[column].astype("string").str.strip()
    info["default_label"] = info["default_label_raw"].map({"是": 1, "否": 0})
    if info["default_label"].isna().any():
        raise DataQualityError("企业信息的是否违约存在缺失或未识别值")
    return info.drop(columns=["default_label_raw"])


def _load_ledgers(config: dict[str, Any], input_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read raw invoice sheets and apply the documented atomic-invoice collapse."""

    input_raw, input_sheet = read_role_sheet(input_path, "input", config)
    output_raw, output_sheet = read_role_sheet(input_path, "output", config)
    input_ledger, input_diag = collapse_invoice_lines(input_raw, "input", config)
    output_ledger, output_diag = collapse_invoice_lines(output_raw, "output", config)
    diagnostics = {
        "input_sheet": input_sheet,
        "output_sheet": output_sheet,
        "input": input_diag,
        "output": output_diag,
    }
    return input_ledger, output_ledger, diagnostics


def _months(input_ledger: pd.DataFrame, output_ledger: pd.DataFrame, config: dict[str, Any]) -> pd.PeriodIndex:
    """Return the common invoice month index used by the builder."""

    dates = pd.concat([input_ledger["invoice_date"], output_ledger["invoice_date"]], ignore_index=True)
    dates = dates.dropna()
    if dates.empty:
        raise DataQualityError("原始发票没有可用日期")
    start = dates.min().to_period("M")
    end = dates.max().to_period("M")
    configured_start = config["processing"].get("analysis_start")
    configured_end = config["processing"].get("analysis_end")
    if configured_start:
        start = max(start, pd.Period(str(configured_start), freq="M"))
    if configured_end:
        end = min(end, pd.Period(str(configured_end), freq="M"))
    if start > end:
        raise DataQualityError(f"analysis_start={start} is after analysis_end={end}")
    return pd.period_range(start, end, freq="M")


def _ratio(numerator: float, denominator: float) -> float:
    """Apply the dictionary zero-denominator rule."""

    return float(numerator / denominator) if denominator > 0 else float("nan")


def _counterparty(ledger: pd.DataFrame, direction: str) -> tuple[float, float, float]:
    """Return positive counterparty count, HHI and largest share."""

    positive = ledger[
        ledger["is_valid"]
        & ledger["is_positive"]
        & ledger["counterparty_known"]
        & ledger["direction"].eq(direction)
    ]
    if positive.empty:
        return 0.0, float("nan"), float("nan")
    amounts = positive.groupby("counterparty_id", sort=False)["total_10k"].sum()
    amounts = amounts[amounts.gt(0)]
    if amounts.empty:
        return 0.0, float("nan"), float("nan")
    shares = amounts / amounts.sum()
    return float(len(shares)), float((shares**2).sum()), float(shares.max())


def _recalculate_one(
    enterprise_id: str,
    input_ledger: pd.DataFrame,
    output_ledger: pd.DataFrame,
    months: pd.PeriodIndex,
) -> dict[str, float]:
    """Independently recalculate core features for one sampled enterprise."""

    ledger = pd.concat([input_ledger, output_ledger], ignore_index=True)
    ledger = ledger[ledger["enterprise_id"].eq(enterprise_id)].copy()
    valid = ledger[ledger["is_valid"]].copy()
    inp = valid[valid["direction"].eq("input")]
    out = valid[valid["direction"].eq("output")]
    sales_positive = float(out.loc[out["is_positive"], "total_10k"].sum())
    purchase_positive = float(inp.loc[inp["is_positive"], "total_10k"].sum())
    sales_net = float(out["total_10k"].sum())
    purchase_net = float(inp["total_10k"].sum())
    sales_negative_amount = float(-out.loc[out["is_negative"], "total_10k"].sum())
    purchase_negative_amount = float(-inp.loc[inp["is_negative"], "total_10k"].sum())
    all_count = float(len(ledger))
    valid_count = float(len(valid))
    void_count = float(ledger["is_void"].sum())
    negative_count = float(valid["is_negative"].sum())
    zero_count = float(valid["is_zero"].sum())
    nonzero_count = float(valid.loc[valid["total_10k"].ne(0)].shape[0])

    monthly_sales = (
        out.loc[out["is_positive"]].groupby("month")["total_10k"].sum()
        .reindex(months.astype("string"), fill_value=0.0)
        .astype(float)
    )
    t = np.arange(len(months), dtype=float)
    centered = t - (len(months) - 1) / 2.0
    denominator = float((centered**2).sum() / 123.0)
    growth = float((centered * np.log1p(monthly_sales.to_numpy())).sum() / denominator) if denominator > 0 else float("nan")
    if sales_positive <= 0:
        growth = float("nan")
    mean_sales = float(monthly_sales.mean())
    cv = float(monthly_sales.std() / mean_sales) if mean_sales > 0 and len(months) >= 2 else float("nan")

    active_month = (
        ledger.loc[ledger["is_valid"]]
        .assign(abs_amount=lambda d: d["total_10k"].abs())
        .groupby("month")["abs_amount"].sum()
        .reindex(months.astype("string"), fill_value=0.0)
        .gt(0)
    )
    active_ratio = float(active_month.mean()) if len(months) else float("nan")
    longest = float(longest_consecutive_run(active_month.tolist()) / len(months)) if len(months) else float("nan")
    customer_count, customer_hhi, max_customer = _counterparty(ledger, "output")
    supplier_count, supplier_hhi, max_supplier = _counterparty(ledger, "input")
    return {
        "sales_scale_10k": sales_positive,
        "purchase_scale_10k": purchase_positive,
        "net_sales_10k": sales_net,
        "operating_net_inflow_proxy_10k": sales_net - purchase_net,
        "sales_growth_trend": growth,
        "sales_monthly_cv": cv,
        "invoice_activity_per_month": nonzero_count / len(months),
        "sales_return_rate": _ratio(sales_negative_amount, sales_positive),
        "purchase_return_rate": _ratio(purchase_negative_amount, purchase_positive),
        "void_invoice_rate": _ratio(void_count, all_count),
        "zero_amount_invoice_rate": _ratio(zero_count, valid_count),
        "customer_count": customer_count,
        "supplier_count": supplier_count,
        "customer_hhi": customer_hhi,
        "supplier_hhi": supplier_hhi,
        "max_customer_share": max_customer,
        "max_supplier_share": max_supplier,
        "purchase_sales_ratio": _ratio(purchase_positive, sales_positive),
        "active_month_ratio": active_ratio,
        "longest_active_streak_ratio": longest,
        "audit_valid_invoice_count": valid_count,
        "audit_negative_invoice_count": negative_count,
        "audit_zero_invoice_count": zero_count,
        "audit_negative_amount_10k": sales_negative_amount + purchase_negative_amount,
    }


def _dictionary_names(path: Path) -> tuple[list[str], dict[str, str]]:
    """Extract English variable names from the two dictionary tables."""

    text = path.read_text(encoding="utf-8")
    names: list[str] = []
    sections: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or cells[1] in {"英文变量名", "English variable"}:
            continue
        candidate = cells[1].strip("`")
        if re.fullmatch(r"[a-z][a-z0-9_]+", candidate):
            names.append(candidate)
            sections[candidate] = cells[0]
    unique = list(dict.fromkeys(names))
    return unique, sections


def _feature_stats(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Compute distribution, constant and IQR diagnostics for all main features."""

    rows: list[dict[str, Any]] = []
    for feature in feature_names:
        values = pd.to_numeric(frame[feature], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        q01, q05, q25, q50, q75, q95, q99 = finite.quantile([.01, .05, .25, .50, .75, .95, .99])
        iqr = q75 - q25
        lower, upper = q25 - 1.5 * iqr, q75 + 1.5 * iqr
        top_share = float(finite.value_counts(normalize=True).iloc[0]) if len(finite) else float("nan")
        rows.append({
            "feature": feature,
            "count": int(len(finite)),
            "missing_count": int(values.isna().sum()),
            "missing_rate": float(values.isna().mean()),
            "unique_count": int(finite.nunique()),
            "constant_flag": bool(finite.nunique() <= 1),
            "near_constant_flag": bool(finite.nunique() <= 2 or top_share >= .95),
            "top_frequency_share": top_share,
            "min": float(finite.min()) if len(finite) else np.nan,
            "q01": float(q01), "q05": float(q05), "q25": float(q25), "median": float(q50),
            "q75": float(q75), "q95": float(q95), "q99": float(q99),
            "max": float(finite.max()) if len(finite) else np.nan,
            "mean": float(finite.mean()) if len(finite) else np.nan,
            "std": float(finite.std()) if len(finite) > 1 else np.nan,
            "iqr_outlier_count": int(((finite < lower) | (finite > upper)).sum()),
            "below_q01_count": int((finite < q01).sum()),
            "above_q99_count": int((finite > q99).sum()),
        })
    return pd.DataFrame(rows)


def _recalculation_check(
    features: pd.DataFrame,
    input_ledger: pd.DataFrame,
    output_ledger: pd.DataFrame,
    months: pd.PeriodIndex,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Recalculate at least five deterministic enterprises and compare with tolerance."""

    ids = sorted(features["enterprise_id"].astype(str).tolist())
    sample_size = min(max(5, int(round(len(ids) * .08))), len(ids))
    rng = np.random.default_rng(seed)
    sampled = sorted(rng.choice(np.array(ids), size=sample_size, replace=False).tolist())
    names = [
        "sales_scale_10k", "purchase_scale_10k", "net_sales_10k",
        "operating_net_inflow_proxy_10k", "audit_valid_invoice_count",
        "void_invoice_rate", "audit_negative_amount_10k", "sales_return_rate",
        "purchase_return_rate", "active_month_ratio", "customer_count", "supplier_count",
        "max_customer_share", "max_supplier_share", "customer_hhi", "supplier_hhi",
        "sales_monthly_cv",
    ]
    rows: list[dict[str, Any]] = []
    for enterprise_id in sampled:
        expected_row = features.loc[features["enterprise_id"].astype(str).eq(enterprise_id)].iloc[0]
        actual = _recalculate_one(enterprise_id, input_ledger, output_ledger, months)
        for feature in names:
            expected = float(pd.to_numeric(pd.Series([expected_row[feature]]), errors="coerce").iloc[0])
            recalculated = float(actual.get(feature, np.nan))
            both_nan = bool(np.isnan(expected) and np.isnan(recalculated))
            if both_nan:
                absolute_error, relative_error, passed = 0.0, 0.0, True
            elif np.isfinite(expected) and np.isfinite(recalculated):
                absolute_error = abs(expected - recalculated)
                relative_error = absolute_error / max(abs(expected), COMPARISON_ATOL)
                passed = bool(np.isclose(expected, recalculated, atol=COMPARISON_ATOL, rtol=COMPARISON_RTOL))
            else:
                absolute_error, relative_error, passed = np.nan, np.nan, False
            rows.append({
                "enterprise_id": enterprise_id,
                "feature": feature,
                "feature_table_value": expected,
                "recalculated_value": recalculated,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "absolute_tolerance": COMPARISON_ATOL,
                "relative_tolerance": COMPARISON_RTOL,
                "passed": passed,
            })
    return pd.DataFrame(rows), sampled


def _write_review_decisions(
    feature_names: list[str],
    stats: pd.DataFrame,
    high_pairs: pd.DataFrame,
    validation_path: Path,
) -> None:
    """Write the feature-by-feature economic review using actual validation outputs."""

    stat_map = stats.set_index("feature").to_dict(orient="index")
    constant = set(stats.loc[stats["constant_flag"], "feature"])
    high_features = set(high_pairs["feature_a"]).union(set(high_pairs["feature_b"])) if not high_pairs.empty else set()
    categories: dict[str, str] = {
        "sales_growth_trend": "保留为主模型候选特征",
        "void_invoice_rate": "保留为主模型候选特征",
        "customer_hhi": "保留为主模型候选特征",
        "supplier_hhi": "保留为主模型候选特征",
        "active_month_ratio": "保留为主模型候选特征",
        "longest_active_streak_ratio": "保留为主模型候选特征",
        "business_scale_10k": "仅用于描述性分析",
        "sales_scale_10k": "保留但需要转换",
        "purchase_scale_10k": "保留但需要转换",
        "net_sales_10k": "保留但需要转换",
        "operating_net_inflow_proxy_10k": "保留但需要转换",
        "sales_monthly_cv": "保留但需要转换",
        "invoice_activity_per_month": "保留但需要转换",
        "sales_return_rate": "保留但需要转换",
        "purchase_return_rate": "保留但需要转换",
        "customer_count": "保留但需要转换",
        "supplier_count": "保留但需要转换",
        "purchase_sales_ratio": "保留但需要转换",
        "max_customer_share": "需要建模手决定",
        "max_supplier_share": "需要建模手决定",
        "zero_amount_invoice_rate": "暂时删除",
    }
    lines = [
        "# 问题一特征评审与处理决定",
        "",
        f"本文件由 `{validation_path.relative_to(ROOT).as_posix()}` 的实际输出生成；本轮没有训练风险模型。",
        "分类只表示进入后续建模前的处理建议，不覆盖原始特征表。‘经营净流入代理’与‘经营差额’均不是利润。",
        "",
        "## 分类总表",
        "",
        "| 特征 | 决定 | 依据与原因 |",
        "|---|---|---|",
    ]
    for feature in feature_names:
        row = stat_map[feature]
        category = categories.get(feature, "需要建模手决定")
        reasons: list[str] = []
        if row["constant_flag"]:
            reasons.append("当前样本为全常数，不能提供主模型区分度")
        elif row["near_constant_flag"]:
            reasons.append(f"近似常数（最高频值占比={row['top_frequency_share']:.3f}），需核查")
        if feature.endswith("_10k") or feature in {"customer_count", "supplier_count", "invoice_activity_per_month", "sales_monthly_cv", "purchase_sales_ratio", "sales_return_rate", "purchase_return_rate"}:
            reasons.append("规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾")
        if feature in high_features:
            reasons.append("参与|Spearman|>0.85高相关对，不能与相关变量同时入模而不做审查")
        if feature == "business_scale_10k":
            reasons.append("是销售和采购规模之和，适合报告总体规模；与分项规模重叠")
        if feature == "operating_net_inflow_proxy_10k":
            reasons.append("只能解释为销售净额减采购净额的经营差额代理，不代表利润")
        if feature == "zero_amount_invoice_rate":
            reasons.append("需要确认零金额作废票的业务含义；当前特征全常数")
        if not reasons:
            reasons.append("边界在[0,1]或趋势/连续性定义清晰，可保留并在折内预处理")
        lines.append(f"| `{feature}` | {category} | {'；'.join(reasons)} |")
    lines.extend([
        "",
        "## 高相关变量处理",
        "",
        "高相关只作诊断，不在本轮自动删除。后续建模手需要在弹性网、变量组选择或保留一个代表变量之间做决定。",
        "",
        _md_table(high_pairs),
        "",
        "## 当前明确的口径缺口",
        "",
        "- 需确认审计报告中完全重复行是否确属重复导入；当前构建按 `keep_first`。",
        "- 需确认是否纳入2016-10与2020-02两个可能不完整的边界月份。",
        "- 3条金额+税额与价税合计超差记录目前仅标记并保留，是否修正/剔除由建模手决定。",
        "- 信誉评级和是否违约保留在表中做标签/辅助分析，但不进入主行为特征矩阵；`audit_`列同样只作审计辅助。",
        "",
    ])
    write_text("\n".join(lines), ROOT / "docs" / "q1_feature_review_decisions.md")


def run_validation(config: dict[str, Any], feature_path: Path) -> dict[str, Any]:
    """Run all feature assertions, recalc checks and diagnostic outputs."""

    paths = output_paths(config)
    logger = setup_logging("03_validate_features", config, paths)
    input_path = (ROOT / config["input"]["attachment1"]).resolve()
    feature_path = feature_path.resolve()
    out_dir = ROOT / "results" / "feature_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    input_hash_before = sha256_file(input_path)
    features = _load_feature_table(feature_path)
    info = _load_enterprise_info(config, input_path)
    input_ledger, output_ledger, ledger_diagnostics = _load_ledgers(config, input_path)
    months = _months(input_ledger, output_ledger, config)
    feature_names = [str(name) for name in config["features"]["names"]]
    bounded = set(config["features"]["bounded_01"])
    dictionary_names, dictionary_labels = _dictionary_names(DEFAULT_DICTIONARY)
    logger.info("feature_rows=%d feature_columns=%d month_count=%d", len(features), len(features.columns), len(months))

    checks: list[dict[str, Any]] = []

    def add_check(check_id: str, passed: bool, observed: Any, expected: Any,
                  *, severity: str = "blocking", detail: str = "") -> None:
        checks.append({
            "check_id": check_id, "passed": bool(passed), "severity": severity,
            "observed": observed, "expected": expected, "detail": detail,
        })

    feature_ids = features["enterprise_id"].astype("string") if "enterprise_id" in features else pd.Series(dtype="string")
    info_ids = info["enterprise_id"].astype("string")
    add_check("row_count_123", len(features) == 123, len(features), 123)
    add_check("one_row_per_enterprise", feature_ids.nunique(dropna=False) == len(features), int(feature_ids.nunique(dropna=False)), len(features))
    add_check("enterprise_id_unique", not feature_ids.duplicated().any(), int(feature_ids.duplicated().sum()), 0)
    missing_ids = sorted(set(info_ids) - set(feature_ids))
    extra_ids = sorted(set(feature_ids) - set(info_ids))
    add_check("all_info_enterprises_present", not missing_ids, len(missing_ids), 0, detail=json.dumps(missing_ids, ensure_ascii=False))
    add_check("no_extra_enterprises", not extra_ids, len(extra_ids), 0, detail=json.dumps(extra_ids, ensure_ascii=False))
    add_check("default_label_complete", "default_label" in features and features["default_label"].notna().all(), int(features.get("default_label", pd.Series(dtype=float)).isna().sum()) if "default_label" in features else 1, 0)
    add_check("credit_rating_complete", "credit_rating" in features and features["credit_rating"].notna().all(), int(features.get("credit_rating", pd.Series(dtype=float)).isna().sum()) if "credit_rating" in features else 1, 0)

    missing_feature_columns = [name for name in feature_names if name not in features.columns]
    add_check("dictionary_main_features_present", not missing_feature_columns, len(missing_feature_columns), 0, detail=json.dumps(missing_feature_columns, ensure_ascii=False))
    dictionary_missing_from_table = sorted(set(dictionary_names) - set(features.columns))
    add_check("dictionary_defined_features_present", not dictionary_missing_from_table, len(dictionary_missing_from_table), 0, detail=json.dumps(dictionary_missing_from_table, ensure_ascii=False))
    allowed_metadata = {"enterprise_id", "enterprise_name", "credit_rating", "default_label"}
    aux_columns = [column for column in features.columns if column.startswith("audit_")]
    unexplained_extra = sorted(set(features.columns) - set(dictionary_names) - allowed_metadata - set(aux_columns))
    add_check("unexplained_extra_columns", not unexplained_extra, len(unexplained_extra), 0, severity="warning", detail=json.dumps(unexplained_extra, ensure_ascii=False))

    numeric_frame = pd.DataFrame({name: pd.to_numeric(features[name], errors="coerce") for name in feature_names if name in features})
    all_numeric_columns = [
        column for column in features.columns
        if column not in {"enterprise_id", "enterprise_name", "credit_rating"}
    ]
    all_numeric_frame = pd.DataFrame({
        column: pd.to_numeric(features[column], errors="coerce") for column in all_numeric_columns
    })
    nan_columns = numeric_frame.columns[numeric_frame.isna().any()].tolist()
    add_check("numeric_features_no_nan", not nan_columns, len(nan_columns), 0, detail=json.dumps(nan_columns, ensure_ascii=False))
    inf_columns = [name for name in numeric_frame.columns if np.isinf(numeric_frame[name].to_numpy(dtype=float)).any()]
    add_check("numeric_features_no_infinity", not inf_columns, len(inf_columns), 0, detail=json.dumps(inf_columns, ensure_ascii=False))
    all_nan_columns = all_numeric_frame.columns[all_numeric_frame.isna().any()].tolist()
    all_inf_columns = [name for name in all_numeric_frame.columns if np.isinf(all_numeric_frame[name].to_numpy(dtype=float)).any()]
    add_check("all_numeric_output_columns_no_nan", not all_nan_columns, len(all_nan_columns), 0, detail=json.dumps(all_nan_columns, ensure_ascii=False))
    add_check("all_numeric_output_columns_no_infinity", not all_inf_columns, len(all_inf_columns), 0, detail=json.dumps(all_inf_columns, ensure_ascii=False))

    stats = _feature_stats(features, feature_names)
    constant_features = stats.loc[stats["constant_flag"], "feature"].tolist()
    near_constant_features = stats.loc[stats["near_constant_flag"], "feature"].tolist()
    add_check("no_constant_main_feature", not constant_features, constant_features, [], detail="全常数特征应在建模前移出候选集")
    add_check("near_constant_feature_reported", True, near_constant_features, "仅报告，不自动删除", severity="warning")
    duplicate_pairs: list[tuple[str, str]] = []
    for i, left in enumerate(feature_names):
        for right in feature_names[i + 1:]:
            if features[left].equals(features[right]):
                duplicate_pairs.append((left, right))
    add_check("no_duplicate_main_feature_columns", not duplicate_pairs, duplicate_pairs, [], detail=json.dumps(duplicate_pairs, ensure_ascii=False))

    range_violations: dict[str, int] = {}
    impossible: dict[str, int] = {}
    for feature in feature_names:
        values = numeric_frame[feature]
        if feature in bounded:
            count = int((values.lt(0) | values.gt(1)).sum())
            if count:
                range_violations[feature] = count
        if feature in {"business_scale_10k", "sales_scale_10k", "purchase_scale_10k", "customer_count", "supplier_count", "invoice_activity_per_month", "sales_monthly_cv", "sales_return_rate", "purchase_return_rate", "purchase_sales_ratio"}:
            count = int(values.lt(0).sum())
            if count:
                impossible[feature] = count
    add_check("ratio_features_in_0_1", not range_violations, range_violations, {}, detail=json.dumps(range_violations, ensure_ascii=False))
    add_check("amount_count_features_nonnegative", not impossible, impossible, {}, detail=json.dumps(impossible, ensure_ascii=False))
    count_features = {"customer_count", "supplier_count"}
    integer_violations = {feature: int((numeric_frame[feature] % 1 != 0).sum()) for feature in count_features if feature in numeric_frame and (numeric_frame[feature] % 1 != 0).any()}
    add_check("count_features_integer_valued", not integer_violations, integer_violations, {}, detail=json.dumps(integer_violations, ensure_ascii=False))
    amount_name_check = all(name.endswith("_10k") for name in ["business_scale_10k", "sales_scale_10k", "purchase_scale_10k", "net_sales_10k", "operating_net_inflow_proxy_10k"])
    add_check("amount_units_match_dictionary", amount_name_check, amount_name_check, True, detail="企业级金额字段统一为万元，变量名带_10k")
    add_check("main_names_match_config", feature_names == [name for name in dictionary_names if name in feature_names], feature_names, [name for name in dictionary_names if name in feature_names], detail="配置与字典主特征顺序")

    leakage_rows: list[dict[str, Any]] = []
    for column in features.columns:
        if column in {"enterprise_id", "enterprise_name"}:
            leakage_type, in_model, status = "identity", False, "excluded_metadata"
        elif column in {"credit_rating", "default_label"}:
            leakage_type, in_model, status = "label_or_rating", False, "excluded_label_or_auxiliary"
        elif column.startswith("audit_"):
            leakage_type, in_model, status = "audit_auxiliary", False, "excluded_audit_auxiliary"
        elif column in feature_names:
            leakage_type, in_model, status = "behavior_feature", False, "candidate_not_yet_modeled"
        else:
            leakage_type, in_model, status = "unrecognized", True, "review_required"
        leakage_rows.append({"column": column, "is_main_feature": column in feature_names, "leakage_type": leakage_type, "in_main_model": in_model, "status": status})
    leakage_frame = pd.DataFrame(leakage_rows)
    leakage_bad = leakage_frame.loc[leakage_frame["in_main_model"], "column"].tolist()
    add_check("no_identity_label_leakage_in_main_features", not leakage_bad, leakage_bad, [], detail="评级和标签保留在表中但不进入行为特征矩阵")

    # Spearman correlation is computed directly from the raw feature values.
    corr = numeric_frame.corr(method="spearman")
    corr_rows: list[dict[str, Any]] = []
    high_rows: list[dict[str, Any]] = []
    for i, left in enumerate(corr.columns):
        for right in corr.columns[i + 1:]:
            rho = float(corr.loc[left, right])
            pair = {"feature_a": left, "feature_b": right, "spearman_rho": rho, "abs_spearman_rho": abs(rho), "n_pair": int(numeric_frame[[left, right]].dropna().shape[0])}
            corr_rows.append(pair)
            if abs(rho) > .85:
                high_rows.append(pair)
    corr_frame = pd.DataFrame(corr_rows)
    high_frame = pd.DataFrame(high_rows).sort_values(["abs_spearman_rho", "feature_a", "feature_b"], ascending=[False, True, True]) if high_rows else pd.DataFrame(columns=["feature_a", "feature_b", "spearman_rho", "abs_spearman_rho", "n_pair"])

    recalculation, sampled_ids = _recalculation_check(features, input_ledger, output_ledger, months, int(config["random_seed"]))
    recalc_failed = recalculation.loc[~recalculation["passed"]]
    add_check("raw_invoice_recalculation", recalc_failed.empty, int(len(recalc_failed)), 0, detail=f"deterministic_sample={sampled_ids}")
    input_hash_after = sha256_file(input_path)
    add_check("raw_input_unchanged", input_hash_before == input_hash_after, input_hash_after, input_hash_before)
    try:
        reloaded = pd.read_csv(feature_path, encoding="utf-8-sig")
        output_readable = reloaded.shape == features.shape and reloaded.columns.tolist() == features.columns.tolist()
    except Exception as exc:
        output_readable = False
        logger.exception("feature table re-read failed")
        raise DataQualityError(f"特征表重新读取失败: {exc}") from exc
    add_check("feature_output_re_readable", output_readable, str(features.shape), str(reloaded.shape) if output_readable else "unreadable")
    add_check("fixed_random_seed_20260805", int(config["random_seed"]) == 20260805, int(config["random_seed"]), 20260805)
    manifest_path = ROOT / "results" / "reports" / "q1_run_manifest.json"
    add_check("upstream_reproducibility_manifest_present", manifest_path.exists(), str(manifest_path) if manifest_path.exists() else "missing", "present", severity="warning")

    # Persist machine-readable outputs.
    write_csv(pd.DataFrame(checks), out_dir / "feature_validation_summary.csv")
    write_csv(stats, out_dir / "feature_descriptive_statistics.csv")
    write_csv(recalculation, out_dir / "feature_recalculation_check.csv")
    write_csv(stats[["feature", "min", "q01", "q05", "median", "q95", "q99", "max", "iqr_outlier_count", "below_q01_count", "above_q99_count", "constant_flag", "near_constant_flag"]], out_dir / "feature_outlier_report.csv")
    write_csv(corr_frame, out_dir / "feature_correlation.csv")
    write_csv(high_frame, out_dir / "high_correlation_pairs.csv")
    write_csv(leakage_frame, out_dir / "leakage_check.csv")

    alignment_rows: list[dict[str, Any]] = []
    for column in features.columns:
        if column in dictionary_names:
            role, status = "main_dictionary_feature", "documented"
        elif column in allowed_metadata:
            role, status = "metadata_or_label", "explicitly_excluded_from_main_feature_matrix"
        elif column in aux_columns:
            role, status = "audit_auxiliary", "kept_for_audit_not_main_model"
        else:
            role, status = "unexplained", "review_required"
        alignment_rows.append({"column": column, "in_feature_table": True, "in_dictionary": column in dictionary_names, "role": role, "alignment_status": status, "dictionary_label": dictionary_labels.get(column, "")})
    for column in sorted(set(dictionary_names) - set(features.columns)):
        alignment_rows.append({"column": column, "in_feature_table": False, "in_dictionary": True, "role": "missing_dictionary_feature", "alignment_status": "FAIL", "dictionary_label": dictionary_labels.get(column, "")})
    alignment_frame = pd.DataFrame(alignment_rows)
    write_csv(alignment_frame, out_dir / "feature_dictionary_alignment.csv")

    overall_failures = [row for row in checks if not row["passed"] and row["severity"] == "blocking"]
    status = "FAIL" if overall_failures else "PASS_WITH_WARNINGS" if any(not row["passed"] for row in checks) else "PASS"
    summary = {
        "status": status,
        "blocking_failure_count": len(overall_failures),
        "warning_count": sum(1 for row in checks if not row["passed"] and row["severity"] != "blocking"),
        "sampled_enterprises": sampled_ids,
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "main_feature_count": int(len(feature_names)),
        "constant_features": constant_features,
        "near_constant_features": near_constant_features,
        "auxiliary_columns": aux_columns,
        "unexplained_extra_columns": unexplained_extra,
        "high_correlation_pair_count": int(len(high_frame)),
        "raw_input_sha256": input_hash_before,
        "comparison_atol": COMPARISON_ATOL,
        "comparison_rtol": COMPARISON_RTOL,
        "ledger_diagnostics": ledger_diagnostics,
    }
    write_json(summary, out_dir / "feature_validation_summary.json")
    review_path = out_dir / "feature_validation_report.md"
    failed_frame = pd.DataFrame(overall_failures)
    warning_frame = pd.DataFrame([row for row in checks if not row["passed"] and row["severity"] != "blocking"])
    report_lines = [
        "# 问题一企业级特征验收报告",
        "",
        f"- 状态：**{status}**",
        f"- 特征表：`{feature_path.relative_to(ROOT).as_posix()}`，形状={features.shape[0]}行×{features.shape[1]}列",
        f"- 主特征数：{len(feature_names)}；抽查企业：{', '.join(sampled_ids)}",
        "- 本轮仅做数据验收、重算抽查和描述性分析；未训练风险模型、未拟合流失率、未优化信贷。",
        "",
        "## 验收检查",
        "",
        _md_table(pd.DataFrame(checks)),
        "",
        "## 阻断项",
        "",
        _md_table(failed_frame) if not failed_frame.empty else "_无阻断项_",
        "",
        "## 警告项与口径说明",
        "",
        _md_table(warning_frame) if not warning_frame.empty else "_无警告项_",
        "",
        f"- 允许的 `audit_` 辅助列：{', '.join(aux_columns) if aux_columns else '无'}；这些列不属于字典主模型特征。",
        f"- 全常数主特征：{', '.join(constant_features) if constant_features else '无'}。",
        f"- 近似常数主特征：{', '.join(near_constant_features) if near_constant_features else '无'}。",
        f"- |Spearman|>0.85的组合数：{len(high_frame)}；本轮不自动删除。",
        "- VIF未计算：当前环境没有额外统计包，且小样本下VIF仅作辅助判断；Spearman结果已完整输出。",
        "",
        "## 抽查重算",
        "",
        f"使用固定种子20260805抽取至少5家企业，容差为absolute={COMPARISON_ATOL:g}、relative={COMPARISON_RTOL:g}；逐特征结果见 `feature_recalculation_check.csv`。",
        _md_table(recalculation.head(40)),
        "",
        "## 后续不能静默决定的事项",
        "",
        "- `zero_amount_invoice_rate`是否全为0反映真实业务，还是零额作废票应从分母排除，需要建模手确认。",
        "- 完全重复行、边界月份和3条金额恒等式超差记录的业务含义，需要在模型敏感性分析中明确。",
        "- 高相关的规模、HHI与最大对手占比变量不自动删除，需由建模手确定变量组策略。",
        "",
    ]
    write_text("\n".join(report_lines), review_path)
    _write_review_decisions(feature_names, stats, high_frame, review_path)
    logger.info("VALIDATION_STATUS=%s blocking_failures=%d warnings=%d", status, len(overall_failures), summary["warning_count"])
    if overall_failures:
        raise DataQualityError(f"特征验收存在阻断项: {[row['check_id'] for row in overall_failures]}")
    return summary


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    try:
        summary = run_validation(config, args.feature_file)
    except Exception as exc:
        print(f"03_validate_features.py FAILED: {exc}")
        return 1
    print(f"03_validate_features.py SUCCEEDED: status={summary['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
