"""Build enterprise-month and enterprise-level features for question one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _internal.data_pipeline import (
    ROOT,
    PROCESSED_ROOT,
    DataQualityError,
    assert_input_unchanged,
    check_finite,
    collapse_invoice_lines,
    load_config,
    longest_consecutive_run,
    output_paths,
    read_role_sheet,
    relative,
    safe_ratio,
    setup_logging,
    sha256_file,
    stable_sort,
    update_manifest,
    write_csv,
    write_json,
    write_text,
)


FEATURE_NAMES = [
    "business_scale_10k",
    "sales_scale_10k",
    "purchase_scale_10k",
    "net_sales_10k",
    "operating_net_inflow_proxy_10k",
    "sales_growth_trend",
    "sales_monthly_cv",
    "invoice_activity_per_month",
    "sales_return_rate",
    "purchase_return_rate",
    "void_invoice_rate",
    "zero_amount_invoice_rate",
    "customer_count",
    "supplier_count",
    "customer_hhi",
    "supplier_hhi",
    "max_customer_share",
    "max_supplier_share",
    "purchase_sales_ratio",
    "active_month_ratio",
    "longest_active_streak_ratio",
]

BOUNDED_FEATURES = [
    "void_invoice_rate",
    "zero_amount_invoice_rate",
    "customer_hhi",
    "supplier_hhi",
    "max_customer_share",
    "max_supplier_share",
    "active_month_ratio",
    "longest_active_streak_ratio",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build question-one enterprise features.")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args()


def _load_enterprise_info(config: dict[str, Any]) -> tuple[pd.DataFrame, str]:
    """Read and validate enterprise metadata and labels."""

    path = (ROOT / config["input"]["attachment1"]).resolve()
    frame, sheet = read_role_sheet(path, "enterprise", config)
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
    if info["enterprise_id"].isna().any() or info["enterprise_id"].eq("").any():
        raise DataQualityError("企业信息存在空企业代号")
    if info["enterprise_id"].duplicated().any():
        raise DataQualityError("企业信息企业代号重复")
    if len(info) != 123 or info["enterprise_id"].nunique() != 123:
        raise DataQualityError(f"企业信息应为123家，实际为{len(info)}行/{info['enterprise_id'].nunique()}个主键")
    if info["credit_rating"].isna().any() or not info["credit_rating"].isin(["A", "B", "C", "D"]).all():
        raise DataQualityError("信誉评级缺失或存在非A/B/C/D值")
    label_map = {"是": 1, "否": 0}
    info["default_label"] = info["default_label_raw"].map(label_map)
    if info["default_label"].isna().any():
        raise DataQualityError("是否违约标签缺失或存在非是/否值")
    return info.drop(columns=["default_label_raw"]), sheet


def _derive_months(
    ledgers: list[pd.DataFrame],
    config: dict[str, Any],
) -> pd.PeriodIndex:
    """Create the single complete month index used by both invoice directions."""

    dates = pd.concat([ledger["invoice_date"] for ledger in ledgers], ignore_index=True).dropna()
    if dates.empty:
        raise DataQualityError("进销项没有可用日期，无法构造月份集合")
    start = pd.Period(dates.min(), freq="M")
    end = pd.Period(dates.max(), freq="M")
    configured_start = config["processing"].get("analysis_start")
    configured_end = config["processing"].get("analysis_end")
    if configured_start:
        start = max(start, pd.Period(str(configured_start), freq="M"))
    if configured_end:
        end = min(end, pd.Period(str(configured_end), freq="M"))
    if start > end:
        raise DataQualityError(f"analysis_start={start} is after analysis_end={end}")
    return pd.period_range(start=start, end=end, freq="M")


def _monthly_direction(
    ledger: pd.DataFrame,
    direction: str,
    enterprise_ids: pd.Index,
    months: pd.PeriodIndex,
) -> pd.DataFrame:
    """Aggregate atomic invoices to a complete enterprise-month panel."""

    work = ledger.copy()
    work["month"] = work["month"].astype("string")
    work["positive_amount_10k"] = work["total_10k"].where(work["is_valid"] & work["is_positive"], 0.0)
    work["negative_amount_10k"] = (-work["total_10k"]).where(work["is_valid"] & work["is_negative"], 0.0)
    work["net_amount_10k"] = work["total_10k"].where(work["is_valid"], 0.0)
    work["abs_amount_10k"] = work["total_10k"].abs().where(work["is_valid"], 0.0)
    work["valid_positive_count"] = (work["is_valid"] & work["is_positive"]).astype(int)
    work["valid_negative_count"] = (work["is_valid"] & work["is_negative"]).astype(int)
    work["valid_zero_count"] = (work["is_valid"] & work["is_zero"]).astype(int)
    grouped = (
        work.groupby(["enterprise_id", "month"], sort=False, dropna=False)
        .agg(
            all_invoice_count=("invoice_number", "size"),
            valid_invoice_count=("is_valid", "sum"),
            void_invoice_count=("is_void", "sum"),
            positive_invoice_count=("is_positive", "sum"),
            negative_invoice_count=("is_negative", "sum"),
            zero_invoice_count=("is_zero", "sum"),
            valid_positive_invoice_count=("valid_positive_count", "sum"),
            valid_negative_invoice_count=("valid_negative_count", "sum"),
            valid_zero_invoice_count=("valid_zero_count", "sum"),
            positive_amount_10k=("positive_amount_10k", "sum"),
            negative_amount_10k=("negative_amount_10k", "sum"),
            net_amount_10k=("net_amount_10k", "sum"),
            abs_amount_10k=("abs_amount_10k", "sum"),
        )
        .reset_index()
    )
    grid = pd.MultiIndex.from_product(
        [enterprise_ids.tolist(), [str(month) for month in months]],
        names=["enterprise_id", "month"],
    ).to_frame(index=False)
    result = grid.merge(grouped, on=["enterprise_id", "month"], how="left", sort=False)
    numeric = [column for column in result.columns if column not in {"enterprise_id", "month"}]
    result[numeric] = result[numeric].fillna(0.0)
    result["direction"] = direction
    return result


def _build_monthly_panel(
    info: pd.DataFrame,
    input_ledger: pd.DataFrame,
    output_ledger: pd.DataFrame,
    months: pd.PeriodIndex,
) -> pd.DataFrame:
    """Build one row per enterprise-month with both invoice directions."""

    ids = pd.Index(info["enterprise_id"], name="enterprise_id")
    input_monthly = _monthly_direction(input_ledger, "input", ids, months)
    output_monthly = _monthly_direction(output_ledger, "output", ids, months)
    key = ["enterprise_id", "month"]
    input_monthly = input_monthly.drop(columns=["direction"]).rename(
        columns={column: f"in_{column}" for column in input_monthly.columns if column not in key}
    )
    output_monthly = output_monthly.drop(columns=["direction"]).rename(
        columns={column: f"out_{column}" for column in output_monthly.columns if column not in key}
    )
    panel = input_monthly.merge(output_monthly, on=key, how="outer", sort=False)
    count_or_amount = [column for column in panel.columns if column not in key]
    panel[count_or_amount] = panel[count_or_amount].fillna(0.0)
    panel["active_month"] = panel["in_abs_amount_10k"].add(panel["out_abs_amount_10k"]).gt(0)
    return stable_sort(panel, ["enterprise_id", "month"])


def _counterparty_stats(
    ledger: pd.DataFrame,
    direction: str,
    enterprise_ids: pd.Index,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate count, HHI and maximum positive amount share."""

    positive = ledger[
        ledger["is_valid"]
        & ledger["is_positive"]
        & ledger["counterparty_known"]
        & ledger["direction"].eq(direction)
    ]
    if positive.empty:
        empty = pd.Series(0.0, index=enterprise_ids)
        return empty.astype(int), pd.Series(np.nan, index=enterprise_ids), pd.Series(np.nan, index=enterprise_ids)
    amounts = positive.groupby(["enterprise_id", "counterparty_id"], sort=False)["total_10k"].sum()
    shares = amounts / amounts.groupby(level=0).transform("sum")
    count = shares.groupby(level=0).size().reindex(enterprise_ids, fill_value=0).astype(int)
    hhi = shares.pow(2).groupby(level=0).sum().reindex(enterprise_ids)
    maximum = shares.groupby(level=0).max().reindex(enterprise_ids)
    return count, hhi, maximum


def _aggregate_enterprise_features(
    info: pd.DataFrame,
    input_ledger: pd.DataFrame,
    output_ledger: pd.DataFrame,
    monthly: pd.DataFrame,
    months: pd.PeriodIndex,
) -> pd.DataFrame:
    """Calculate exactly the 21 dictionary features plus clearly marked audit fields."""

    ids = pd.Index(info["enterprise_id"], name="enterprise_id")
    all_ledger = pd.concat([input_ledger, output_ledger], ignore_index=True)
    valid = all_ledger[all_ledger["is_valid"]].copy()
    input_valid = valid[valid["direction"].eq("input")]
    output_valid = valid[valid["direction"].eq("output")]

    def grouped_sum(frame: pd.DataFrame, mask: pd.Series, column: str = "total_10k") -> pd.Series:
        selected = frame.loc[mask]
        return selected.groupby("enterprise_id")[column].sum().reindex(ids, fill_value=0.0)

    sales_positive = grouped_sum(output_valid, output_valid["is_positive"])
    purchase_positive = grouped_sum(input_valid, input_valid["is_positive"])
    sales_net = output_valid.groupby("enterprise_id")["total_10k"].sum().reindex(ids, fill_value=0.0)
    purchase_net = input_valid.groupby("enterprise_id")["total_10k"].sum().reindex(ids, fill_value=0.0)
    sales_negative_amount = grouped_sum(output_valid, output_valid["is_negative"], "total_10k").mul(-1)
    purchase_negative_amount = grouped_sum(input_valid, input_valid["is_negative"], "total_10k").mul(-1)
    sales_return = safe_ratio(sales_negative_amount, sales_positive)
    purchase_return = safe_ratio(purchase_negative_amount, purchase_positive)

    all_count = all_ledger.groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)
    valid_count = valid.groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)
    void_count = all_ledger[all_ledger["is_void"]].groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)
    negative_count = valid[valid["is_negative"]].groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)
    zero_count = valid[valid["is_zero"]].groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)
    positive_count = valid[valid["is_positive"]].groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)
    missing_counterparty_count = all_ledger[
        ~all_ledger["counterparty_known"]
    ].groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)

    month_map = {str(period): index for index, period in enumerate(months)}
    sales_monthly = monthly[["enterprise_id", "month", "out_positive_amount_10k"]].copy()
    sales_monthly["_t"] = sales_monthly["month"].map(month_map).astype(float)
    sales_monthly["_log_sales"] = np.log1p(sales_monthly["out_positive_amount_10k"].astype(float))
    t_centered = sales_monthly["_t"] - (len(months) - 1) / 2.0
    denominator = float((t_centered.pow(2)).sum() / len(ids))
    numerator = sales_monthly.assign(_weighted=t_centered * sales_monthly["_log_sales"]).groupby(
        "enterprise_id"
    )["_weighted"].sum().reindex(ids, fill_value=0.0)
    sales_growth = numerator / denominator if denominator > 0 else pd.Series(np.nan, index=ids)
    sales_growth.loc[sales_positive.le(0)] = np.nan
    sales_stats = sales_monthly.groupby("enterprise_id")["out_positive_amount_10k"].agg(["mean", "std"]).reindex(ids)
    sales_cv = sales_stats["std"].div(sales_stats["mean"]).where(sales_stats["mean"].gt(0))
    if len(months) < 2:
        sales_cv[:] = np.nan

    nonzero_valid_count = valid[valid["total_10k"].ne(0)].groupby("enterprise_id").size().reindex(ids, fill_value=0)
    customer_count, customer_hhi, max_customer = _counterparty_stats(output_valid, "output", ids)
    supplier_count, supplier_hhi, max_supplier = _counterparty_stats(input_valid, "input", ids)
    activity = nonzero_valid_count.astype(float) / len(months)
    active_ratio = monthly.groupby("enterprise_id")["active_month"].mean().reindex(ids)
    longest = pd.Series(
        {
            enterprise_id: longest_consecutive_run(group["active_month"].tolist()) / len(months)
            for enterprise_id, group in monthly.groupby("enterprise_id", sort=False)
        },
        dtype="float64",
    ).reindex(ids)

    features = pd.DataFrame(index=ids)
    features["business_scale_10k"] = sales_positive + purchase_positive
    features["sales_scale_10k"] = sales_positive
    features["purchase_scale_10k"] = purchase_positive
    features["net_sales_10k"] = sales_net
    features["operating_net_inflow_proxy_10k"] = sales_net - purchase_net
    features["sales_growth_trend"] = sales_growth
    features["sales_monthly_cv"] = sales_cv
    features["invoice_activity_per_month"] = activity
    features["sales_return_rate"] = sales_return
    features["purchase_return_rate"] = purchase_return
    features["void_invoice_rate"] = safe_ratio(void_count.astype(float), all_count.astype(float))
    features["zero_amount_invoice_rate"] = safe_ratio(zero_count.astype(float), valid_count.astype(float))
    features["customer_count"] = customer_count
    features["supplier_count"] = supplier_count
    features["customer_hhi"] = customer_hhi
    features["supplier_hhi"] = supplier_hhi
    features["max_customer_share"] = max_customer
    features["max_supplier_share"] = max_supplier
    features["purchase_sales_ratio"] = safe_ratio(purchase_positive, sales_positive)
    features["active_month_ratio"] = active_ratio
    features["longest_active_streak_ratio"] = longest
    features = features[FEATURE_NAMES]

    aux = pd.DataFrame(index=ids)
    aux["audit_all_invoice_count"] = all_count
    aux["audit_valid_invoice_count"] = valid_count
    aux["audit_void_invoice_count"] = void_count
    aux["audit_positive_invoice_count"] = positive_count
    aux["audit_negative_invoice_count"] = negative_count
    aux["audit_zero_invoice_count"] = zero_count
    aux["audit_negative_amount_10k"] = sales_negative_amount.add(purchase_negative_amount)
    aux["audit_missing_counterparty_count"] = missing_counterparty_count

    result = info.set_index("enterprise_id").join(features).join(aux).reset_index()
    return stable_sort(result, ["enterprise_id"])


def _feature_quality(
    feature_frame: pd.DataFrame,
    info: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, list[str]]:
    """Check feature finite values, bounds, and key-level assertions."""

    errors: list[str] = []
    assertions: dict[str, bool] = {}
    assertions["row_count_is_123"] = bool(len(feature_frame) == 123)
    assertions["enterprise_id_unique"] = bool(feature_frame["enterprise_id"].is_unique)
    assertions["enterprise_ids_match_info"] = bool(
        set(feature_frame["enterprise_id"]) == set(info["enterprise_id"])
    )
    assertions["default_label_complete"] = bool(feature_frame["default_label"].notna().all())
    assertions["credit_rating_complete"] = bool(feature_frame["credit_rating"].notna().all())
    for name, passed in assertions.items():
        if not passed:
            errors.append(name)

    finite_columns = check_finite(feature_frame, FEATURE_NAMES)
    assertions["no_infinite_numeric_features"] = bool(not finite_columns)
    if finite_columns:
        errors.append(f"infinite_features={finite_columns}")

    quality_rows: list[dict[str, Any]] = []
    for column in FEATURE_NAMES:
        values = pd.to_numeric(feature_frame[column], errors="coerce")
        finite = values.replace([np.inf, -np.inf], np.nan).dropna()
        bounded_bad = 0
        if column in BOUNDED_FEATURES:
            bounded_bad = int((values.dropna().lt(0) | values.dropna().gt(1)).sum())
            if bounded_bad:
                errors.append(f"{column}_out_of_[0,1]={bounded_bad}")
        quality_rows.append(
            {
                "feature": column,
                "row_count": int(len(values)),
                "missing_count": int(values.isna().sum()),
                "missing_rate": float(values.isna().mean()),
                "finite_count": int(len(finite)),
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "unique_count": int(finite.nunique()),
                "bounded_01_violation_count": bounded_bad,
            }
        )
    hhi_bad = (
        feature_frame["customer_hhi"].notna()
        & feature_frame["max_customer_share"].notna()
        & feature_frame["customer_hhi"].lt(feature_frame["max_customer_share"].pow(2) - 1e-10)
    )
    if hhi_bad.any():
        errors.append(f"customer_hhi_below_max_share_square={int(hhi_bad.sum())}")
    hhi_bad = (
        feature_frame["supplier_hhi"].notna()
        & feature_frame["max_supplier_share"].notna()
        & feature_frame["supplier_hhi"].lt(feature_frame["max_supplier_share"].pow(2) - 1e-10)
    )
    if hhi_bad.any():
        errors.append(f"supplier_hhi_below_max_share_square={int(hhi_bad.sum())}")
    assertions["ratio_bounds_valid"] = bool(not any("out_of_[0,1]" in item for item in errors))
    assertions["hhi_bounds_consistent"] = bool(not any("hhi_below" in item for item in errors))
    quality = {
        "status": "FAIL" if errors else "PASS",
        "script": "02_build_features.py",
        "random_seed": int(config["random_seed"]),
        "main_feature_count": len(FEATURE_NAMES),
        "main_feature_names": FEATURE_NAMES,
        "assertions": assertions,
        "errors": errors,
        "open_decisions": [
            "完全重复行当前按config exact_duplicate_policy=keep_first处理；需建模手确认这些行确属重复导入而非合法重复明细。",
            "analysis_start/end为空时当前使用2016-10至2020-02并纳入边界月份；需建模手确认是否剔除不完整边界月。",
            "金额+税额与价税合计超差记录当前保留并仅在审计报告标记；若需剔除或修正，应由建模手给出容差外规则。",
        ],
        "auxiliary_audit_columns": [
            column for column in feature_frame.columns if column.startswith("audit_")
        ],
    }
    return quality, pd.DataFrame(quality_rows), errors


def _report(
    quality: dict[str, Any],
    quality_rows: pd.DataFrame,
    monthly: pd.DataFrame,
    diagnostics: dict[str, Any],
) -> str:
    """Create a concise, deterministic feature quality report."""

    lines = [
        "# 问题一企业级特征质量报告",
        "",
        f"- 状态：**{quality['status']}**",
        f"- 企业行数：{diagnostics['enterprise_rows']}",
        f"- 月份数：{diagnostics['month_count']}",
        f"- 企业—月份行数：{len(monthly)}",
        f"- 主模型特征数：{quality['main_feature_count']}",
        "- 原始特征值已保留；未在本脚本中做全数据对数、缩尾或标准化。",
        "- 负数、作废和零金额统计保留在企业—月份表及 audit_ 前缀辅助字段中。",
        "",
        "## 特征质量",
        "",
        _markdown_table(quality_rows),
        "",
        "## 处理规则记录",
        "",
        "- 有效发票参与金额与交易统计；负数按退货/退款保留；作废仅进入作废统计；零金额仅进入零额统计。",
        "- 发票先按企业—发票号—日期—交易对手—状态聚合，再按企业—月份、企业聚合。",
        "- 月份集合在所有企业和进/销项间一致，缺月补零。",
        "- 比例分母为0时输出NA空值，后续训练Pipeline再填补；不把NA改成0。",
        "- 信誉评级保留为元数据，default_label保留为标签，二者均不进入本脚本的主模型特征列。",
        "",
        "## 尚需建模手确认的口径",
        "",
        *[f"- {item}" for item in quality["open_decisions"]],
        "",
        "## 诊断",
        "",
        _markdown_table(pd.DataFrame([diagnostics])),
        "",
        "## 规格一致性",
        "",
        "- q1_feature_dictionary.md中的21个主特征已逐列生成；无已知口径偏离。",
        "- 额外 audit_ 前缀字段仅用于审计和论文解释，不属于主风险模型输入。",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    """Render a deterministic Markdown table without requiring tabulate."""

    if frame.empty:
        return "_无记录_"
    sample = frame.head(max_rows)
    columns = [str(column) for column in sample.columns]
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in sample.iterrows():
        values = []
        for value in row.tolist():
            if isinstance(value, (list, tuple, dict)):
                values.append(json.dumps(value, ensure_ascii=False, default=str))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n> 仅展示前{max_rows}行，共{len(frame)}行。")
    return "\n".join(lines)


def run_features(config: dict[str, Any]) -> dict[str, Any]:
    """Build and validate the 123-enterprise feature files."""

    paths = output_paths(config)
    logger = setup_logging("02_build_features", config, paths)
    input_path = (ROOT / config["input"]["attachment1"]).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Attachment 1 not found: {input_path}")
    before_hash = sha256_file(input_path)
    quality_path = paths["audit"] / "q1_data_quality.json"
    if not quality_path.exists():
        raise DataQualityError("Run 01_data_audit.py first: q1_data_quality.json is missing")
    audit_quality = json.loads(quality_path.read_text(encoding="utf-8"))
    if not audit_quality.get("can_build_features", False):
        raise DataQualityError("Data audit did not permit feature construction")

    info, _ = _load_enterprise_info(config)
    input_raw, input_sheet = read_role_sheet(input_path, "input", config)
    output_raw, output_sheet = read_role_sheet(input_path, "output", config)
    input_ledger, input_diag = collapse_invoice_lines(input_raw, "input", config)
    output_ledger, output_diag = collapse_invoice_lines(output_raw, "output", config)
    logger.info(
        "status_aliases=%s amount_unit=source_yuan_to_feature_10k denominator_rule=zero_to_NA",
        json.dumps(config["processing"]["status_aliases"], ensure_ascii=False, sort_keys=True),
    )
    logger.info(
        "missing_counterparty_rule=%s month_rule=%s log_and_standardize=deferred_to_training_pipeline",
        config["processing"]["missing_counterparty_policy"],
        config["processing"]["boundary_month_policy"],
    )
    logger.info(
        "negative_void_zero_rule=negative_kept_void_counted_zero_counted_not_deleted "
        "main_feature_count=%d output_amount_unit=10k_yuan",
        len(FEATURE_NAMES),
    )
    logger.info(
        "business_rule=exact_duplicate_keep_first; invoice_key=enterprise_id+invoice_number+date+counterparty+status"
    )
    logger.info(
        "input_raw=%d input_atomic=%d output_raw=%d output_atomic=%d",
        input_diag["raw_rows"],
        input_diag["atomic_rows"],
        output_diag["raw_rows"],
        output_diag["atomic_rows"],
    )
    months = _derive_months([input_ledger, output_ledger], config)
    logger.info("analysis_months=%s..%s count=%d", months[0], months[-1], len(months))
    monthly = _build_monthly_panel(info, input_ledger, output_ledger, months)
    features = _aggregate_enterprise_features(info, input_ledger, output_ledger, monthly, months)

    expected_columns = ["enterprise_id", "enterprise_name", "credit_rating", "default_label"] + FEATURE_NAMES
    missing_columns = [column for column in expected_columns if column not in features.columns]
    if missing_columns:
        raise DataQualityError(f"Feature output missing columns: {missing_columns}")
    quality, quality_rows, errors = _feature_quality(features, info, config)
    diagnostics = {
        "enterprise_rows": int(len(features)),
        "month_count": int(len(months)),
        "monthly_rows": int(len(monthly)),
        "input_sheet": input_sheet,
        "output_sheet": output_sheet,
        "input_raw_rows": input_diag["raw_rows"],
        "input_atomic_rows": input_diag["atomic_rows"],
        "input_exact_duplicate_extra_rows": input_diag["exact_duplicate_extra_rows"],
        "input_business_duplicate_extra_rows": input_diag["business_duplicate_extra_rows"],
        "output_raw_rows": output_diag["raw_rows"],
        "output_atomic_rows": output_diag["atomic_rows"],
        "output_exact_duplicate_extra_rows": output_diag["exact_duplicate_extra_rows"],
        "output_business_duplicate_extra_rows": output_diag["business_duplicate_extra_rows"],
    }
    quality["diagnostics"] = diagnostics
    quality["input_sha256_before"] = before_hash

    feature_dir = paths["features"]
    intermediate_dir = paths["intermediate"]
    feature_outputs = [
        feature_dir / "enterprise_monthly_123.csv",
        feature_dir / "enterprise_features_123.csv",
        feature_dir / "feature_quality_report.md",
        paths["audit"] / "q1_feature_quality.json",
        paths["reports"] / "q1_feature_quality_report.md",
        paths["intermediate"] / "q1_enterprise_monthly.parquet",
        paths["intermediate"] / "q1_clean_invoice_ledger.parquet",
        PROCESSED_ROOT / "q1_enterprise_features.csv",
    ]
    update_manifest(
        config,
        "02_build_features",
        {"attachment1": before_hash},
        feature_outputs,
        verify_only=True,
    )
    write_csv(monthly, feature_dir / "enterprise_monthly_123.csv")
    write_csv(features, feature_dir / "enterprise_features_123.csv")
    write_csv(features, PROCESSED_ROOT / "q1_enterprise_features.csv")
    monthly.to_parquet(intermediate_dir / "q1_enterprise_monthly.parquet", index=False)
    clean_ledger = pd.concat([input_ledger, output_ledger], ignore_index=True)
    clean_ledger = stable_sort(clean_ledger, ["enterprise_id", "direction", "invoice_date", "invoice_number"])
    clean_ledger["invoice_date"] = clean_ledger["invoice_date"].dt.strftime("%Y-%m-%d")
    clean_ledger.to_parquet(intermediate_dir / "q1_clean_invoice_ledger.parquet", index=False)
    report = _report(quality, quality_rows, monthly, diagnostics)
    write_text(report, feature_dir / "feature_quality_report.md")
    write_text(report, paths["reports"] / "q1_feature_quality_report.md")
    write_json(quality, paths["audit"] / "q1_feature_quality.json")
    assert_input_unchanged(input_path, before_hash)
    quality["input_sha256_after"] = sha256_file(input_path)
    write_json(quality, paths["audit"] / "q1_feature_quality.json")

    output_readable = True
    try:
        reloaded = pd.read_csv(feature_dir / "enterprise_features_123.csv", encoding="utf-8-sig")
        monthly_reloaded = pd.read_csv(feature_dir / "enterprise_monthly_123.csv", encoding="utf-8-sig")
    except Exception as exc:
        output_readable = False
        quality["assertions"]["output_csv_re_readable"] = False
        write_json(quality, paths["audit"] / "q1_feature_quality.json")
        raise DataQualityError(f"Output CSV cannot be re-read: {exc}") from exc
    quality["assertions"]["output_csv_re_readable"] = bool(output_readable)
    if reloaded.shape != features.shape or monthly_reloaded.shape != monthly.shape:
        quality["assertions"]["output_shape_matches_memory"] = False
        write_json(quality, paths["audit"] / "q1_feature_quality.json")
        raise DataQualityError("Re-read output shape differs from in-memory result")
    quality["assertions"]["output_shape_matches_memory"] = True
    quality["assertions"]["original_input_unchanged"] = bool(
        sha256_file(input_path) == before_hash
    )
    if errors:
        raise DataQualityError(f"Feature assertions failed: {errors}")
    if not quality["assertions"]["original_input_unchanged"]:
        raise DataQualityError("Original input hash changed during feature construction")
    quality["assertions"]["deterministic_manifest_recorded"] = True
    write_json(quality, paths["audit"] / "q1_feature_quality.json")
    update_manifest(config, "02_build_features", {"attachment1": before_hash}, feature_outputs)
    logger.info(
        "FEATURE_STATUS=%s enterprise_rows=%d monthly_rows=%d feature_columns=%d",
        quality["status"],
        len(features),
        len(monthly),
        len(FEATURE_NAMES),
    )
    return quality


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    try:
        run_features(config)
    except Exception as exc:
        print(f"02_build_features.py FAILED: {exc}")
        return 1
    print("02_build_features.py SUCCEEDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
