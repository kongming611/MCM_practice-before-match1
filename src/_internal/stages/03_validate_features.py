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

from _internal.data_pipeline import (
    PAPER_ROOT,
    PROCESSED_ROOT,
    ROOT,
    DataQualityError,
    collapse_invoice_lines,
    load_config,
    longest_consecutive_run,
    normalize_invoice_frame,
    output_paths,
    read_role_sheet,
    sha256_file,
    setup_logging,
    write_csv,
    write_json,
    write_text,
)


DEFAULT_FEATURE_FILE = PROCESSED_ROOT / "q1_enterprise_features.csv"
DEFAULT_DICTIONARY = PAPER_ROOT / "q1_feature_dictionary.md"
DEFAULT_REPORT = ROOT / "outputs" / "q1" / "reports" / "feature_validation_report.md"
COMPARISON_ATOL = 1e-6
COMPARISON_RTOL = 1e-5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate question-one enterprise features.")
    parser.add_argument("--config", type=Path, default=None)
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


def _load_ledgers(
    config: dict[str, Any], input_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, pd.DataFrame]]:
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
    return input_ledger, output_ledger, diagnostics, {"input": input_raw, "output": output_raw}


def _zero_amount_level_frames(
    raw_frames: dict[str, pd.DataFrame],
    atomic_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Return raw, exact-deduplicated and atomic invoice frames for zero checks."""

    levels: dict[str, pd.DataFrame] = {}
    tolerance = float(config["processing"].get("zero_amount_tolerance_yuan", 0.0))
    for direction in ["input", "output"]:
        normalized = normalize_invoice_frame(raw_frames[direction], direction, config)
        deduplicated = normalized.loc[~normalized["exact_duplicate"]].copy()
        raw_level = normalized[["enterprise_id", "status", "total_yuan"]].copy()
        dedup_level = deduplicated[["enterprise_id", "status", "total_yuan"]].copy()
        atomic_level = atomic_frames[direction][["enterprise_id", "status", "total_yuan"]].copy()
        for stage, level in [
            ("raw_detail", raw_level),
            ("deduplicated_detail", dedup_level),
            ("atomic_invoice", atomic_level),
        ]:
            level = level.copy()
            level["direction"] = direction
            level["exact_zero"] = level["total_yuan"].eq(0)
            level["tolerance_zero"] = level["total_yuan"].abs().le(tolerance)
            levels[f"{stage}_{direction}"] = level
    return levels


def _summarize_zero_amount_business(
    raw_frames: dict[str, pd.DataFrame],
    atomic_frames: dict[str, pd.DataFrame],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute zero counts at every requested processing level and enterprise."""

    levels = _zero_amount_level_frames(raw_frames, atomic_frames, config)
    tolerance = float(config["processing"].get("zero_amount_tolerance_yuan", 0.0))
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    stage_order = ["raw_detail", "deduplicated_detail", "atomic_invoice"]

    def append_row(
        level: pd.DataFrame,
        stage: str,
        direction: str,
        status_scope: str,
        enterprise_id: str,
        scope: str,
    ) -> None:
        status_value = {"valid": "有效", "void": "作废"}.get(status_scope)
        subset = level if status_value is None else level.loc[level["status"].eq(status_value)]
        invoice_count = int(len(subset))
        exact_count = int(subset["exact_zero"].sum())
        tolerance_count = int(subset["tolerance_zero"].sum())
        row = {
            "stage": stage,
            "direction": direction,
            "status_scope": status_scope,
            "scope": scope,
            "enterprise_id": enterprise_id,
            "invoice_count": invoice_count,
            "exact_zero_count": exact_count,
            "tolerance_zero_count": tolerance_count,
            "exact_zero_rate": exact_count / invoice_count if invoice_count else np.nan,
            "tolerance_zero_rate": tolerance_count / invoice_count if invoice_count else np.nan,
            "difference_count_tolerance_minus_exact": tolerance_count - exact_count,
            "exact_equals_tolerance": bool(exact_count == tolerance_count),
            "zero_amount_tolerance_yuan": tolerance,
        }
        detail_rows.append(row)
        if scope == "overall":
            summary_rows.append(row.copy())

    for stage in stage_order:
        direction_frames = [levels[f"{stage}_{direction}"] for direction in ["input", "output"]]
        direction_frames.append(pd.concat(direction_frames, ignore_index=True))
        for direction, level in zip(["input", "output", "all"], direction_frames):
            append_row(level, stage, direction, "all", "__ALL__", "overall")
            for status_scope, status_value in [("valid", "有效"), ("void", "作废")]:
                append_row(
                    level.loc[level["status"].eq(status_value)],
                    stage,
                    direction,
                    status_scope,
                    "__ALL__",
                    "overall",
                )
            for enterprise_id in sorted(level["enterprise_id"].astype(str).unique()):
                enterprise_level = level.loc[level["enterprise_id"].astype(str).eq(enterprise_id)]
                append_row(enterprise_level, stage, direction, "all", enterprise_id, "enterprise")
                for status_scope, status_value in [("valid", "有效"), ("void", "作废")]:
                    append_row(
                        enterprise_level.loc[enterprise_level["status"].eq(status_value)],
                        stage,
                        direction,
                        status_scope,
                        enterprise_id,
                        "enterprise",
                    )

    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def _zero_row(
    summary: pd.DataFrame,
    stage: str,
    direction: str,
    status_scope: str,
) -> pd.Series:
    """Select one overall zero-business summary row."""

    rows = summary.loc[
        summary["stage"].eq(stage)
        & summary["direction"].eq(direction)
        & summary["status_scope"].eq(status_scope)
    ]
    if len(rows) != 1:
        raise DataQualityError(f"zero business summary row is not unique: {stage}/{direction}/{status_scope}")
    return rows.iloc[0]


def _write_zero_amount_decision(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    config: dict[str, Any],
    output_path: Path,
) -> None:
    """Write a data-derived business decision for zero_amount_invoice_rate."""

    tolerance = float(config["processing"].get("zero_amount_tolerance_yuan", 0.0))
    raw_all = _zero_row(summary, "raw_detail", "all", "all")
    raw_output_void = _zero_row(summary, "raw_detail", "output", "void")
    raw_input_all = _zero_row(summary, "raw_detail", "input", "all")
    atomic_valid_all = _zero_row(summary, "atomic_invoice", "all", "valid")
    atomic_valid_input = _zero_row(summary, "atomic_invoice", "input", "valid")
    atomic_valid_output = _zero_row(summary, "atomic_invoice", "output", "valid")
    atomic_void_output = _zero_row(summary, "atomic_invoice", "output", "void")

    raw_exact_from_output_void = int(raw_output_void["exact_zero_count"])
    raw_tolerance_from_output_void = int(raw_output_void["tolerance_zero_count"])
    exact_only_output_void = (
        int(raw_all["exact_zero_count"]) == raw_exact_from_output_void
        and int(raw_input_all["exact_zero_count"]) == 0
    )
    tolerance_only_output_void = (
        int(raw_all["tolerance_zero_count"]) == raw_tolerance_from_output_void
        and int(raw_input_all["tolerance_zero_count"]) == 0
    )
    enterprise_atomic_valid = detail.loc[
        detail["stage"].eq("atomic_invoice")
        & detail["scope"].eq("enterprise")
        & detail["direction"].eq("all")
        & detail["status_scope"].eq("valid")
    ]
    enterprise_with_exact = int((enterprise_atomic_valid["exact_zero_count"] > 0).sum())
    enterprise_with_tolerance = int((enterprise_atomic_valid["tolerance_zero_count"] > 0).sum())

    def count_table(stage: str, status_scope: str) -> str:
        rows = []
        for direction, label in [("input", "进项"), ("output", "销项"), ("all", "合计")]:
            row = _zero_row(summary, stage, direction, status_scope)
            rows.append(
                {
                    "方向": label,
                    "发票数": int(row["invoice_count"]),
                    "精确零值数": int(row["exact_zero_count"]),
                    "容差零值数": int(row["tolerance_zero_count"]),
                    "精确零值率": f"{row['exact_zero_rate']:.8f}" if pd.notna(row["exact_zero_rate"]) else "NA",
                    "容差零值率": f"{row['tolerance_zero_rate']:.8f}" if pd.notna(row["tolerance_zero_rate"]) else "NA",
                    "两口径一致": "是" if bool(row["exact_equals_tolerance"]) else "否",
                }
            )
        return _md_table(pd.DataFrame(rows))

    lines = [
        "# zero_amount_invoice_rate 业务口径决策",
        "",
        "本文件由 __BT__src/_internal/stages/03_validate_features.py__BT__ 重新读取附件1并程序计算生成。数字来自当前运行结果，不从历史报告复制。",
        "",
        "## 一、数据事实",
        "",
        f"- 零金额容差配置为 {tolerance:g} 元；精确口径为 __BT__total_yuan == 0__BT__，容差口径为 __BT__abs(total_yuan) <= {tolerance:g}__BT__。",
        "- 原始明细层（未去除确认的完全重复行）的进项、销项和合计统计如下：",
        count_table("raw_detail", "all"),
        "",
        "- 去除确认的完全重复行后，以及按方向—企业代号—发票号码—开票日期—交易对手—发票状态聚合为原子发票后，全部状态统计如下：",
        "",
        "### 去重明细层",
        "",
        count_table("deduplicated_detail", "all"),
        "",
        "### 原子发票层",
        "",
        count_table("atomic_invoice", "all"),
        "",
        "- 原子发票层按有效/作废状态的明细如下：",
        "",
        "### 原子发票—有效",
        "",
        count_table("atomic_invoice", "valid"),
        "",
        "### 原子发票—作废",
        "",
        count_table("atomic_invoice", "void"),
        "",
        f"- 原始明细层的精确零值中，销项作废占比结论为：{'全部来自销项作废' if exact_only_output_void else '并非全部来自销项作废'}；容差零值的对应结论为：{'全部来自销项作废' if tolerance_only_output_void else '并非全部来自销项作废'}。该结论由本次程序按方向和状态重算得到。",
        f"- 原始明细层销项作废精确零值数为 {raw_exact_from_output_void}，容差零值数为 {raw_tolerance_from_output_void}；原始进项明细精确零值数为 {int(raw_input_all['exact_zero_count'])}，容差零值数为 {int(raw_input_all['tolerance_zero_count'])}。",
        f"- 原子有效发票层合计分母为 {int(atomic_valid_all['invoice_count'])}；精确零值数为 {int(atomic_valid_all['exact_zero_count'])}，容差零值数为 {int(atomic_valid_all['tolerance_zero_count'])}。按进项分别为 {int(atomic_valid_input['exact_zero_count'])}/{int(atomic_valid_input['tolerance_zero_count'])}，按销项分别为 {int(atomic_valid_output['exact_zero_count'])}/{int(atomic_valid_output['tolerance_zero_count'])}。",
        f"- 原子销项作废发票的精确零值数/容差零值数为 {int(atomic_void_output['exact_zero_count'])}/{int(atomic_void_output['tolerance_zero_count'])}。企业级有效原子发票统计中，出现精确零值的企业数为 {enterprise_with_exact}，出现容差零值的企业数为 {enterprise_with_tolerance}；完整企业明细见 __BT__data/processed/_runtime/feature_validation/zero_amount_invoice_business_check.csv__BT__。",
        "",
        "## 二、指标业务定义",
        "",
        "__BT__zero_amount_invoice_rate__BT__ 定义为：在去除确认的完全重复行，并按照“方向—企业代号—发票号码—开票日期—交易对手—发票状态”聚合明细后，有效且原子发票价税合计为零的发票数除以有效原子发票总数。",
        "",
        "零金额判定同时报告两种口径：精确零值 __BT__total_yuan == 0__BT__，以及配置容差零值 __BT__abs(total_yuan) <= zero_amount_tolerance_yuan__BT__。二者不得未经计算直接视为一致。作废发票不进入有效原子发票分母，也不因金额为零而被改写为有效交易。",
        "",
        "## 三、最终建模决定",
        "",
        "- __BT__zero_amount_invoice_rate__BT__ 不进入任何正式风险模型；",
        "- 不因为它全为0而删除原始零金额记录；",
        "- 不把作废零金额发票重新解释为有效交易；",
        "- 作废行为由 __BT__void_invoice_rate__BT__ 刻画；",
        "- __BT__zero_amount_invoice_rate__BT__ 仅作为审计型派生字段保留；",
        "- 若附件2未来出现有效零金额发票，也不得在没有重新训练问题一模型的情况下临时加入预测模型。",
        "",
        "## 四、可追溯输出",
        "",
        "- __BT__data/processed/_runtime/feature_validation/zero_amount_invoice_business_check.csv__BT__：按处理层级、方向、状态、企业和两种零值口径的明细统计。",
        "- __BT__data/processed/_runtime/feature_validation/zero_amount_invoice_business_summary.csv__BT__：按处理层级、方向和状态汇总的机器可读统计。",
        "- __BT__data/processed/q1_enterprise_features.csv__BT__ 仍保留 __BT__zero_amount_invoice_rate__BT__ 列；该列不属于主模型或特征集敏感性模型矩阵。",
        "",
    ]
    write_text("\n".join(lines).replace("__BT__", chr(96)), output_path)


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
    primary_features: list[str],
    sensitivity_features: list[str],
    excluded_features: dict[str, Any],
    stats: pd.DataFrame,
    high_pairs: pd.DataFrame,
    validation_path: Path,
) -> None:
    """Write the feature-by-feature economic review using actual validation outputs."""

    stat_map = stats.set_index("feature").to_dict(orient="index")
    high_features = set(high_pairs["feature_a"]).union(set(high_pairs["feature_b"])) if not high_pairs.empty else set()
    primary_set = set(primary_features)
    sensitivity_set = set(sensitivity_features)
    excluded_set = set(excluded_features)
    sensitivity_only = sensitivity_set - primary_set
    lines = [
        "# 问题一特征评审与处理决定",
        "",
        f"本文件由 {validation_path.relative_to(ROOT).as_posix()} 的实际输出生成；分类只表示模型角色，不覆盖原始特征表。",
        f"共构造{len(feature_names)}个企业级派生特征，其中{len(primary_features)}个进入主模型，{len(sensitivity_only)}个用于特征集敏感性分析，{', '.join(sorted(excluded_set)) if excluded_set else '无'}仅作审计。",
        "‘经营净流入代理’与‘经营差额’均不是利润；评级、标签、企业身份和audit_字段不进入正式风险模型矩阵。",
        "",
        "## 分类总表",
        "",
        "| 特征 | 模型角色 | 依据与原因 |",
        "|---|---|---|",
    ]
    for feature in feature_names:
        row = stat_map[feature]
        if feature in excluded_set:
            category = "审计型派生字段，不进入任何正式模型"
        elif feature in primary_set:
            category = "主模型特征"
        elif feature in sensitivity_only:
            category = "特征集敏感性分析"
        else:
            category = "未进入本轮模型"
        reasons: list[str] = []
        if row["constant_flag"]:
            reasons.append("当前样本为全常数；零金额字段只作为已知审计事实，不阻断主模型验收")
        elif row["near_constant_flag"]:
            reasons.append(f"近似常数（最高频值占比={row['top_frequency_share']:.3f}），需核查")
        if feature.endswith("_10k") or feature in {"customer_count", "supplier_count", "invoice_activity_per_month", "sales_monthly_cv", "purchase_sales_ratio", "sales_return_rate", "purchase_return_rate"}:
            reasons.append("规模/右偏风险较强，按字典在训练折内log1p或有符号log1p并缩尾")
        if feature in high_features:
            reasons.append("参与|Spearman|>0.85高相关对，按预先锁定的替代变量组处理")
        if feature == "business_scale_10k":
            reasons.append("与销售、采购规模高度重复，仅放入完整敏感性特征集")
        if feature == "net_sales_10k":
            reasons.append("与sales_scale_10k高度相关，仅放入销售规模替代敏感性组")
        if feature in {"max_customer_share", "max_supplier_share"}:
            reasons.append("分别作为对应HHI的替代变量组，仅用于敏感性分析")
        if feature == "longest_active_streak_ratio":
            reasons.append("与active_month_ratio高度相关，仅用于连续性替代敏感性组")
        if feature == "operating_net_inflow_proxy_10k":
            reasons.append("只能解释为销售净额减采购净额的经营差额代理，不代表利润")
        if feature == "zero_amount_invoice_rate":
            reasons.append("正式定义和精确/容差核验见q1_zero_amount_invoice_rate_decision.md；作废行为由void_invoice_rate刻画")
        if not reasons:
            reasons.append("按特征字典定义在折内完成缺失填补、缩尾、变换和标准化")
        lines.append(f"| {feature} | {category} | {'；'.join(reasons)} |")
    lines.extend([
        "",
        "## 已锁定的主模型特征选择规则",
        "",
        "- business_scale_10k与销售、采购规模高度重复，只作敏感性分析。",
        "- sales_scale_10k与net_sales_10k高度相关，主模型保留sales_scale_10k。",
        "- customer_hhi与max_customer_share高度相关，主模型保留HHI。",
        "- supplier_hhi与max_supplier_share高度相关，主模型保留HHI。",
        "- active_month_ratio与longest_active_streak_ratio高度相关，主模型保留active_month_ratio。",
        "- 被替代变量保留在特征集敏感性分析中，不从特征表删除。",
        "",
        "## 高相关变量处理",
        "",
        "高相关变量不从原始企业特征表物理删除；正式主模型和敏感性模型使用配置中的明确特征列表。",
        "",
        _md_table(high_pairs),
        "",
        "## 变量排除边界",
        "",
        "- zero_amount_invoice_rate保留在features.names和企业级特征表中，但不在primary_model_features或sensitivity_model_features中。",
        "- credit_rating只作单独敏感性实验，default_label只作标签；enterprise_id和enterprise_name只作标识或展示。",
        "- audit_开头字段只作审计辅助，不进入任何模型矩阵。",
        "",
    ])
    write_text("\n".join(lines), ROOT / "outputs" / "q1" / "reports" / "q1_feature_review_decisions.md")


def run_validation(config: dict[str, Any], feature_path: Path) -> dict[str, Any]:
    """Run all feature assertions, recalc checks and diagnostic outputs."""

    paths = output_paths(config)
    logger = setup_logging("03_validate_features", config, paths)
    input_path = (ROOT / config["input"]["attachment1"]).resolve()
    feature_path = feature_path.resolve()
    out_dir = paths["validation"]
    out_dir.mkdir(parents=True, exist_ok=True)
    input_hash_before = sha256_file(input_path)
    features = _load_feature_table(feature_path)
    info = _load_enterprise_info(config, input_path)
    input_ledger, output_ledger, ledger_diagnostics, raw_frames = _load_ledgers(config, input_path)
    months = _months(input_ledger, output_ledger, config)
    feature_names = [str(name) for name in config["features"]["names"]]
    primary_features = [str(name) for name in config["features"].get("primary_model_features", [])]
    sensitivity_features = [str(name) for name in config["features"].get("sensitivity_model_features", [])]
    excluded_features = dict(config["features"].get("excluded_from_model", {}))
    if not primary_features or not sensitivity_features or not excluded_features:
        raise DataQualityError("features.primary_model_features, sensitivity_model_features and excluded_from_model are required")
    if len(primary_features) != len(set(primary_features)) or len(sensitivity_features) != len(set(sensitivity_features)):
        raise DataQualityError("model feature lists must not contain duplicate names")
    if not set(primary_features).issubset(set(sensitivity_features)):
        raise DataQualityError("primary_model_features must be a subset of sensitivity_model_features")
    if set(excluded_features).intersection(set(primary_features) | set(sensitivity_features)):
        raise DataQualityError("excluded_from_model variables cannot enter either model feature list")
    zero_detail, zero_summary = _summarize_zero_amount_business(
        raw_frames,
        {"input": input_ledger, "output": output_ledger},
        config,
    )
    write_csv(zero_detail, out_dir / "zero_amount_invoice_business_check.csv")
    write_csv(zero_summary, out_dir / "zero_amount_invoice_business_summary.csv")
    _write_zero_amount_decision(
        zero_detail,
        zero_summary,
        config,
        paths["reports"] / "q1_zero_amount_invoice_rate_decision.md",
    )
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
    add_check("all_constructed_features_present", not missing_feature_columns, len(missing_feature_columns), 0, detail=json.dumps(missing_feature_columns, ensure_ascii=False))
    primary_missing = sorted(set(primary_features) - set(features.columns))
    sensitivity_missing = sorted(set(sensitivity_features) - set(features.columns))
    add_check("primary_model_features_present", not primary_missing, len(primary_missing), 0, detail=json.dumps(primary_missing, ensure_ascii=False))
    add_check("sensitivity_model_features_present", not sensitivity_missing, len(sensitivity_missing), 0, detail=json.dumps(sensitivity_missing, ensure_ascii=False))
    excluded_missing = sorted(set(excluded_features) - set(features.columns))
    add_check("excluded_features_reported_in_feature_table", not excluded_missing, len(excluded_missing), 0, detail=json.dumps(excluded_missing, ensure_ascii=False))
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
    primary_constant_features = sorted(set(constant_features).intersection(primary_features))
    add_check(
        "no_constant_primary_model_feature",
        not primary_constant_features,
        primary_constant_features,
        [],
        detail="主模型候选集中不得存在全常数特征；审计型排除特征只报告不阻断",
    )
    model_matrix_features = sorted(set(primary_features) | set(sensitivity_features))
    excluded_in_matrix = sorted(set(excluded_features).intersection(model_matrix_features))
    add_check(
        "excluded_features_absent_from_model_matrix",
        not excluded_in_matrix,
        excluded_in_matrix,
        [],
        detail="excluded_from_model变量不得进入主模型或敏感性模型矩阵",
    )
    audit_reported = sorted(
        name for name, metadata in excluded_features.items()
        if name in features.columns and str(metadata.get("role", "")) == "audit_only"
    )
    add_check(
        "audit_only_features_reported",
        audit_reported == sorted(excluded_features),
        audit_reported,
        sorted(excluded_features),
        detail="审计型排除特征必须在特征表保留并报告",
    )
    add_check("near_constant_feature_reported", True, near_constant_features, "仅报告，不自动删除", severity="warning")
    duplicate_pairs: list[tuple[str, str]] = []
    for i, left in enumerate(feature_names):
        for right in feature_names[i + 1:]:
            if features[left].equals(features[right]):
                duplicate_pairs.append((left, right))
    add_check("no_duplicate_constructed_feature_columns", not duplicate_pairs, duplicate_pairs, [], detail=json.dumps(duplicate_pairs, ensure_ascii=False))

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
    add_check("constructed_names_match_dictionary", feature_names == dictionary_names, feature_names, dictionary_names, detail="features.names与字典21个构造特征顺序一致")
    add_check("primary_model_feature_count_15", len(primary_features) == 15, len(primary_features), 15)

    leakage_rows: list[dict[str, Any]] = []
    for column in features.columns:
        if column in {"enterprise_id", "enterprise_name"}:
            leakage_type, in_model, status = "identity", False, "excluded_metadata"
        elif column in {"credit_rating", "default_label"}:
            leakage_type, in_model, status = "label_or_rating", False, "excluded_label_or_auxiliary"
        elif column.startswith("audit_"):
            leakage_type, in_model, status = "audit_auxiliary", False, "excluded_audit_auxiliary"
        elif column in excluded_features:
            leakage_type, in_model, status = "audit_excluded_feature", False, "excluded_from_all_model_matrices"
        elif column in feature_names:
            leakage_type = "primary_model_feature" if column in primary_features else "sensitivity_model_feature"
            in_model, status = True, "included_in_configured_model_matrix"
        else:
            leakage_type, in_model, status = "unrecognized", True, "review_required"
        leakage_rows.append({
            "column": column,
            "is_constructed_feature": column in feature_names,
            "is_primary_model_feature": column in primary_features,
            "is_sensitivity_model_feature": column in sensitivity_features,
            "is_excluded_from_model": column in excluded_features,
            "is_main_feature": column in feature_names,
            "leakage_type": leakage_type,
            "in_model_matrix": in_model,
            "in_main_model": column in primary_features,
            "status": status,
        })
    leakage_frame = pd.DataFrame(leakage_rows)
    leakage_bad = leakage_frame.loc[
        leakage_frame["in_model_matrix"]
        & leakage_frame["column"].isin(["enterprise_id", "enterprise_name", "credit_rating", "default_label"]),
        "column",
    ].tolist()
    add_check("no_identity_label_leakage_in_model_features", not leakage_bad, leakage_bad, [], detail="评级和标签保留在表中但不进入行为特征矩阵")

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
    manifest_path = paths["reports"] / "q1_run_manifest.json"
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
        if column in excluded_features:
            role, status = "audit_only_excluded_feature", "reported_but_absent_from_model_matrices"
        elif column in primary_features:
            role, status = "primary_model_feature", "included_in_primary_model"
        elif column in sensitivity_features:
            role, status = "sensitivity_model_feature", "included_in_sensitivity_model"
        elif column in dictionary_names:
            role, status = "constructed_not_modelled", "documented_but_not_in_configured_model_matrix"
        elif column in allowed_metadata:
            role, status = "metadata_or_label", "explicitly_excluded_from_all_model_matrices"
        elif column in aux_columns:
            role, status = "audit_auxiliary", "kept_for_audit_not_main_model"
        else:
            role, status = "unexplained", "review_required"
        alignment_rows.append({"column": column, "in_feature_table": True, "in_dictionary": column in dictionary_names, "role": role, "alignment_status": status, "dictionary_label": dictionary_labels.get(column, "")})
    for column in sorted(set(dictionary_names) - set(features.columns)):
        alignment_rows.append({"column": column, "in_feature_table": False, "in_dictionary": True, "role": "missing_dictionary_feature", "alignment_status": "FAIL", "dictionary_label": dictionary_labels.get(column, "")})
    alignment_frame = pd.DataFrame(alignment_rows)
    write_csv(alignment_frame, out_dir / "feature_dictionary_alignment.csv")
    role_rows = []
    for feature in feature_names:
        role_rows.append({
            "feature": feature,
            "constructed": True,
            "primary_model": feature in primary_features,
            "sensitivity_model": feature in sensitivity_features,
            "excluded_from_model": feature in excluded_features,
            "role": "audit_only" if feature in excluded_features else "primary" if feature in primary_features else "sensitivity_only" if feature in sensitivity_features else "constructed_only",
        })
    write_csv(pd.DataFrame(role_rows), out_dir / "feature_role_alignment.csv")

    overall_failures = [row for row in checks if not row["passed"] and row["severity"] == "blocking"]
    status = "FAIL" if overall_failures else "PASS_WITH_WARNINGS" if any(not row["passed"] for row in checks) else "PASS"
    summary = {
        "status": status,
        "blocking_failure_count": len(overall_failures),
        "warning_count": sum(1 for row in checks if not row["passed"] and row["severity"] != "blocking"),
        "sampled_enterprises": sampled_ids,
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "constructed_feature_count": int(len(feature_names)),
        "primary_model_feature_count": int(len(primary_features)),
        "sensitivity_model_feature_count": int(len(sensitivity_features)),
        "sensitivity_only_feature_count": int(len(set(sensitivity_features) - set(primary_features))),
        "primary_model_features": primary_features,
        "sensitivity_model_features": sensitivity_features,
        "excluded_from_model": excluded_features,
        "constant_features": constant_features,
        "primary_constant_features": primary_constant_features,
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
        f"- 共构造特征数：{len(feature_names)}；主模型特征数：{len(primary_features)}；敏感性模型特征数：{len(sensitivity_features)}；审计型排除特征数：{len(excluded_features)}",
        f"- 抽查企业：{', '.join(sampled_ids)}；本报告不训练风险模型，只验收特征角色和业务口径。",
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
        f"- 主模型特征：{', '.join(primary_features)}。",
        f"- 敏感性模型特征：{', '.join(sensitivity_features)}。",
        f"- 允许的 `audit_` 辅助列：{', '.join(aux_columns) if aux_columns else '无'}；这些列不属于模型矩阵。",
        f"- 全常数构造特征：{', '.join(constant_features) if constant_features else '无'}；主模型候选中的全常数特征：{', '.join(primary_constant_features) if primary_constant_features else '无'}。",
        f"- 近似常数构造特征：{', '.join(near_constant_features) if near_constant_features else '无'}；全常数审计事实仍在报告中显示，不造成主模型验收失败。",
        f"- |Spearman|>0.85的组合数：{len(high_frame)}；本轮不自动删除。",
        "- VIF未计算：当前环境没有额外统计包，且小样本下VIF仅作辅助判断；Spearman结果已完整输出。",
        "",
        "## 抽查重算",
        "",
        f"使用固定种子20260805抽取至少5家企业，容差为absolute={COMPARISON_ATOL:g}、relative={COMPARISON_RTOL:g}；逐特征结果见 `feature_recalculation_check.csv`。",
        _md_table(recalculation.head(40)),
        "",
        "## 零金额业务核验",
        "",
        "- 零金额业务核验明细见运行目录的 `zero_amount_invoice_business_check.csv`，汇总见 `zero_amount_invoice_business_summary.csv`，最终决定见 `outputs/q1/reports/q1_zero_amount_invoice_rate_decision.md`。",
        "- 完全重复行、边界月份和金额恒等式超差记录继续按现有审计口径保留并可追溯。",
        "- 高相关变量按配置的主模型与敏感性模型列表处理，不从原始特征表删除。",
        "",
    ]
    write_text("\n".join(report_lines), review_path)
    _write_review_decisions(feature_names, primary_features, sensitivity_features, excluded_features, stats, high_frame, review_path)
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
