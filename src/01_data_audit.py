"""Audit attachment 1 without modifying the source workbook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from q1_common import (
    ROOT,
    DataQualityError,
    assert_input_unchanged,
    load_config,
    normalize_invoice_frame,
    output_paths,
    read_role_sheet,
    relative,
    setup_logging,
    sha256_file,
    stable_sort,
    update_manifest,
    write_csv,
    write_json,
    write_text,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit question-one attachment 1.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "q1.yaml")
    return parser.parse_args()


def _table_rows(
    frame: pd.DataFrame,
    workbook: str,
    sheet: str,
    role: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build table-level and field-level profile rows."""

    field_types = {str(column): str(dtype) for column, dtype in frame.dtypes.items()}
    summary = {
        "workbook": workbook,
        "sheet_name": sheet,
        "role": role,
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "fields": json.dumps([str(column) for column in frame.columns], ensure_ascii=False),
        "field_types": json.dumps(field_types, ensure_ascii=False, sort_keys=True),
    }
    missing_rows = []
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        missing_rows.append(
            {
                "workbook": workbook,
                "sheet_name": sheet,
                "role": role,
                "field": str(column),
                "dtype": str(frame[column].dtype),
                "row_count": int(len(frame)),
                "missing_count": missing,
                "missing_rate": missing / len(frame) if len(frame) else np.nan,
            }
        )
    return summary, missing_rows


def _status_rows(
    frame: pd.DataFrame,
    direction: str,
    workbook: str,
    sheet: str,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Summarize raw/canonical statuses and sign counts."""

    normalized = normalize_invoice_frame(frame, direction, config)
    rows: list[dict[str, Any]] = []
    grouped = normalized.assign(
        status_raw_report=normalized["raw_status"].fillna("<MISSING>"),
        status_canonical_report=normalized["status"].fillna("<MISSING>"),
    ).groupby(
        ["status_raw_report", "status_canonical_report"],
        dropna=False,
        sort=True,
    )
    for (raw_status, canonical_status), group in grouped:
        total = pd.to_numeric(group["total_yuan"], errors="coerce")
        rows.append(
            {
                "workbook": workbook,
                "sheet_name": sheet,
                "direction": direction,
                "status_raw": str(raw_status),
                "status_canonical": str(canonical_status),
                "record_count": int(len(group)),
                "positive_amount_count": int(total.gt(0).sum()),
                "negative_amount_count": int(total.lt(0).sum()),
                "zero_amount_count": int(total.eq(0).sum()),
                "missing_amount_count": int(total.isna().sum()),
            }
        )
    return rows, normalized


def _coverage_rows(
    enterprise: pd.DataFrame,
    input_frame: pd.DataFrame,
    output_frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return one coverage row per enterprise and unmatched identifiers."""

    ids = enterprise["企业代号"].astype("string").str.strip()
    input_ids = input_frame["企业代号"].astype("string").str.strip()
    output_ids = output_frame["企业代号"].astype("string").str.strip()
    input_counts = input_ids.value_counts(dropna=False)
    output_counts = output_ids.value_counts(dropna=False)
    rows = pd.DataFrame({"enterprise_id": ids})
    rows["input_invoice_count"] = rows["enterprise_id"].map(input_counts).fillna(0).astype(int)
    rows["output_invoice_count"] = rows["enterprise_id"].map(output_counts).fillna(0).astype(int)
    rows["has_input"] = rows["input_invoice_count"].gt(0)
    rows["has_output"] = rows["output_invoice_count"].gt(0)
    rows["has_any_invoice"] = rows["has_input"] | rows["has_output"]
    known = set(ids.dropna())
    unmatched_input = sorted(set(input_ids.dropna()) - known)
    unmatched_output = sorted(set(output_ids.dropna()) - known)
    details = {
        "enterprise_count": int(ids.nunique(dropna=True)),
        "input_covered_count": int(rows["has_input"].sum()),
        "output_covered_count": int(rows["has_output"].sum()),
        "any_covered_count": int(rows["has_any_invoice"].sum()),
        "unmatched_input_ids": unmatched_input,
        "unmatched_output_ids": unmatched_output,
        "missing_enterprise_ids_in_info": int(ids.isna().sum()),
    }
    return stable_sort(rows, ["enterprise_id"]), details


def _duplicate_rows(
    frame: pd.DataFrame,
    direction: str,
    workbook: str,
    sheet: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Check exact duplicates, repeated business keys and invoice conflicts."""

    normalized = normalize_invoice_frame(frame, direction, config)
    exact_key = [
        "enterprise_id",
        "invoice_number",
        "date_key",
        "counterparty_id",
        "amount_yuan",
        "tax_yuan",
        "total_yuan",
        "raw_status",
    ]
    exact_sizes = normalized.groupby(exact_key, dropna=False, sort=False).size()
    exact_groups = exact_sizes[exact_sizes.gt(1)]
    business_key = ["enterprise_id", "invoice_number", "date_key", "counterparty_id", "status"]
    key_sizes = normalized.groupby(business_key, dropna=False, sort=False).size()
    duplicate_groups = key_sizes[key_sizes.gt(1)]
    invoice_group = normalized.groupby(
        ["enterprise_id", "invoice_number"], dropna=False, sort=False
    ).agg(
        date_nunique=("date_key", "nunique"),
        counterparty_nunique=("counterparty_id", "nunique"),
        status_nunique=("status", "nunique"),
    )
    conflicts = invoice_group[
        invoice_group[["date_nunique", "counterparty_nunique", "status_nunique"]].gt(1).any(axis=1)
    ]
    return [
        {
            "workbook": workbook,
            "sheet_name": sheet,
            "direction": direction,
            "check_type": "exact_duplicate_rows",
            "group_count": int(len(exact_groups)),
            "extra_row_count": int(normalized["exact_duplicate"].sum()),
            "affected_enterprise_count": int(
                normalized.loc[normalized["exact_duplicate"], "enterprise_id"].nunique()
            ),
            "sample_keys": "",
        },
        {
            "workbook": workbook,
            "sheet_name": sheet,
            "direction": direction,
            "check_type": "business_key_repeated",
            "group_count": int(len(duplicate_groups)),
            "extra_row_count": int((duplicate_groups - 1).sum()),
            "affected_enterprise_count": int(
                duplicate_groups.index.get_level_values("enterprise_id").nunique()
            )
            if len(duplicate_groups)
            else 0,
            "sample_keys": json.dumps(
                [list(key) for key in duplicate_groups.head(5).index.tolist()],
                ensure_ascii=False,
                default=str,
            ),
        },
        {
            "workbook": workbook,
            "sheet_name": sheet,
            "direction": direction,
            "check_type": "invoice_number_metadata_conflict",
            "group_count": int(len(conflicts)),
            "extra_row_count": 0,
            "affected_enterprise_count": int(conflicts.index.get_level_values("enterprise_id").nunique())
            if len(conflicts)
            else 0,
            "sample_keys": json.dumps(
                [list(key) for key in conflicts.head(5).index.tolist()],
                ensure_ascii=False,
                default=str,
            ),
        },
    ]


def _date_rows(
    normalized: pd.DataFrame,
    direction: str,
    workbook: str,
    sheet: str,
) -> dict[str, Any]:
    """Summarize date validity and range."""

    dates = normalized["invoice_date"]
    valid_dates = dates.dropna()
    return {
        "workbook": workbook,
        "sheet_name": sheet,
        "direction": direction,
        "record_count": int(len(normalized)),
        "invalid_date_count": int(normalized["date_invalid"].sum()),
        "min_date": valid_dates.min().strftime("%Y-%m-%d") if len(valid_dates) else "",
        "max_date": valid_dates.max().strftime("%Y-%m-%d") if len(valid_dates) else "",
        "month_count": int(normalized["month"].nunique(dropna=True)),
    }


def _amount_row(
    raw_frame: pd.DataFrame,
    normalized: pd.DataFrame,
    direction: str,
    workbook: str,
    sheet: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Return amount quantiles and configured anomaly counts."""

    total = pd.to_numeric(raw_frame["价税合计"], errors="coerce")
    amount = total.dropna()
    quantiles = amount.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    lower = float(quantiles.loc[0.01]) if len(amount) else np.nan
    upper = float(quantiles.loc[0.99]) if len(amount) else np.nan
    arithmetic = normalized["arithmetic_error"]
    return {
        "workbook": workbook,
        "sheet_name": sheet,
        "direction": direction,
        "record_count": int(len(raw_frame)),
        "non_numeric_total_count": int(total.isna().sum() - raw_frame["价税合计"].isna().sum()),
        "non_finite_total_count": int(np.isinf(total.fillna(np.nan)).sum()),
        "arithmetic_mismatch_count": int(arithmetic.sum()),
        "min_yuan": float(amount.min()) if len(amount) else np.nan,
        "q01_yuan": lower,
        "q05_yuan": float(quantiles.loc[0.05]) if len(amount) else np.nan,
        "q25_yuan": float(quantiles.loc[0.25]) if len(amount) else np.nan,
        "median_yuan": float(quantiles.loc[0.50]) if len(amount) else np.nan,
        "q75_yuan": float(quantiles.loc[0.75]) if len(amount) else np.nan,
        "q95_yuan": float(quantiles.loc[0.95]) if len(amount) else np.nan,
        "q99_yuan": upper,
        "max_yuan": float(amount.max()) if len(amount) else np.nan,
        "below_q01_count": int(total.lt(lower).sum()) if len(amount) else 0,
        "above_q99_count": int(total.gt(upper).sum()) if len(amount) else 0,
    }


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    """Render a small deterministic Markdown table without optional dependencies."""

    if frame.empty:
        return "_无记录_"
    sample = frame.head(max_rows).copy()
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


def run_audit(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the complete attachment-one audit and write all reports."""

    paths = output_paths(config)
    logger = setup_logging("01_data_audit", config, paths)
    input_path = (ROOT / config["input"]["attachment1"]).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Attachment 1 not found: {input_path}")
    before_hash = sha256_file(input_path)
    logger.info("input=%s sha256=%s", relative(input_path), before_hash)
    logger.info(
        "status_aliases=%s exact_duplicate_policy=%s invalid_date_policy=%s amount_tolerance_yuan=%s",
        json.dumps(config["processing"]["status_aliases"], ensure_ascii=False, sort_keys=True),
        config["processing"]["exact_duplicate_policy"],
        config["processing"]["invalid_date_policy"],
        config["processing"]["amount_tolerance_yuan"],
    )

    enterprise, enterprise_sheet = read_role_sheet(input_path, "enterprise", config)
    input_invoice, input_sheet = read_role_sheet(input_path, "input", config)
    output_invoice, output_sheet = read_role_sheet(input_path, "output", config)
    workbook_name = relative(input_path)
    role_frames = [
        (enterprise, enterprise_sheet, "enterprise"),
        (input_invoice, input_sheet, "input_invoice"),
        (output_invoice, output_sheet, "output_invoice"),
    ]

    table_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for frame, sheet, role in role_frames:
        summary, missing = _table_rows(frame, workbook_name, sheet, role)
        table_rows.append(summary)
        missing_rows.extend(missing)

    status_rows: list[dict[str, Any]] = []
    normalized_frames: dict[str, pd.DataFrame] = {}
    for frame, sheet, direction, role in [
        (input_invoice, input_sheet, "input", "input_invoice"),
        (output_invoice, output_sheet, "output", "output_invoice"),
    ]:
        rows, normalized = _status_rows(frame, direction, workbook_name, sheet, config)
        status_rows.extend(rows)
        normalized_frames[direction] = normalized

    coverage, coverage_details = _coverage_rows(enterprise, input_invoice, output_invoice, config)
    duplicate_rows: list[dict[str, Any]] = []
    for frame, sheet, direction in [
        (input_invoice, input_sheet, "input"),
        (output_invoice, output_sheet, "output"),
    ]:
        duplicate_rows.extend(_duplicate_rows(frame, direction, workbook_name, sheet, config))

    date_rows = [
        _date_rows(normalized_frames["input"], "input", workbook_name, input_sheet),
        _date_rows(normalized_frames["output"], "output", workbook_name, output_sheet),
    ]
    amount_rows = [
        _amount_row(input_invoice, normalized_frames["input"], "input", workbook_name, input_sheet, config),
        _amount_row(output_invoice, normalized_frames["output"], "output", workbook_name, output_sheet, config),
    ]

    enterprise_ids = enterprise["企业代号"].astype("string").str.strip()
    errors: list[str] = []
    warnings: list[str] = []
    if enterprise_ids.isna().any() or enterprise_ids.eq("").any():
        errors.append("企业信息表存在空企业代号")
    if enterprise_ids.duplicated().any():
        errors.append("企业信息表企业代号重复")
    if int(enterprise_ids.nunique(dropna=True)) != 123:
        errors.append(f"企业信息表企业数为{enterprise_ids.nunique(dropna=True)}，期望123")
    rating = enterprise["信誉评级"].astype("string").str.strip()
    if rating.isna().any() or rating.eq("").any() or not rating.dropna().isin(["A", "B", "C", "D"]).all():
        errors.append("信誉评级存在缺失或非A/B/C/D值")
    labels = enterprise["是否违约"].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any() or not labels.dropna().isin(["是", "否"]).all():
        errors.append("是否违约存在缺失或非是/否值")
    for direction, normalized in normalized_frames.items():
        if int(normalized["status_unknown"].sum()):
            errors.append(f"{direction}存在未知发票状态")
        if int(normalized["date_invalid"].sum()):
            errors.append(f"{direction}存在无法解析的开票日期")
        for column in ["enterprise_id", "invoice_number", "total_yuan"]:
            if normalized[column].isna().any():
                errors.append(f"{direction}字段{column}存在缺失")
        if int(normalized["arithmetic_error"].sum()):
            warnings.append(f"{direction}存在金额+税额与价税合计超差记录")

    if coverage_details["unmatched_input_ids"] or coverage_details["unmatched_output_ids"]:
        errors.append("发票表存在企业信息表中不存在的企业代号")
    if coverage_details["any_covered_count"] != 123:
        errors.append("并非123家企业均有至少一条进/销项记录")
    if any(row["group_count"] for row in duplicate_rows):
        warnings.append("存在完全重复、重复业务键或同号元数据冲突，详见duplicate_check.csv")
    if any(row["above_q99_count"] or row["below_q01_count"] for row in amount_rows):
        warnings.append("金额存在配置分位数外记录；原始记录未删除")

    status = "FAIL" if errors else ("WARN" if warnings else "PASS")
    quality = {
        "status": status,
        "can_build_features": not errors,
        "script": "01_data_audit.py",
        "random_seed": int(config["random_seed"]),
        "input_file": workbook_name,
        "input_sha256_before": before_hash,
        "input_sha256_after": sha256_file(input_path),
        "sheets": {
            "enterprise": enterprise_sheet,
            "input_invoice": input_sheet,
            "output_invoice": output_sheet,
        },
        "enterprise_count": int(enterprise_ids.nunique(dropna=True)),
        "coverage": coverage_details,
        "errors": errors,
        "warnings": warnings,
        "processing_rules": {
            "status_aliases": config["processing"]["status_aliases"],
            "exact_duplicate_policy": config["processing"]["exact_duplicate_policy"],
            "amount_unit": "yuan_in_source",
            "amount_tolerance_yuan": config["processing"]["amount_tolerance_yuan"],
            "unknown_status_policy": config["processing"]["unknown_status_policy"],
        },
    }

    table_summary = pd.DataFrame(table_rows)
    missing_values = pd.DataFrame(missing_rows)
    status_summary = stable_sort(pd.DataFrame(status_rows), ["direction", "status_canonical", "status_raw"])
    duplicate_check = pd.DataFrame(duplicate_rows)
    enterprise_coverage = coverage
    date_summary = pd.DataFrame(date_rows)
    amount_summary = pd.DataFrame(amount_rows)
    audit_outputs = [
        paths["audit"] / name
        for name in [
            "table_summary.csv",
            "table_profile.csv",
            "missing_values.csv",
            "invoice_status_summary.csv",
            "invoice_status_sign_summary.csv",
            "enterprise_coverage.csv",
            "duplicate_check.csv",
            "duplicate_key_issues.csv",
            "date_range.csv",
            "enterprise_record_counts.csv",
            "amount_anomalies.csv",
            "q1_data_quality.json",
            "data_quality_report.md",
        ]
    ]
    update_manifest(
        config,
        "01_data_audit",
        {"attachment1": before_hash},
        audit_outputs,
        verify_only=True,
    )
    write_csv(table_summary, paths["audit"] / "table_summary.csv")
    write_csv(table_summary, paths["audit"] / "table_profile.csv")
    write_csv(missing_values, paths["audit"] / "missing_values.csv")
    write_csv(status_summary, paths["audit"] / "invoice_status_summary.csv")
    write_csv(status_summary, paths["audit"] / "invoice_status_sign_summary.csv")
    write_csv(enterprise_coverage, paths["audit"] / "enterprise_coverage.csv")
    write_csv(duplicate_check, paths["audit"] / "duplicate_check.csv")
    write_csv(duplicate_check, paths["audit"] / "duplicate_key_issues.csv")
    write_csv(date_summary, paths["audit"] / "date_range.csv")
    write_csv(
        enterprise_coverage[
            ["enterprise_id", "input_invoice_count", "output_invoice_count", "has_any_invoice"]
        ],
        paths["audit"] / "enterprise_record_counts.csv",
    )
    write_csv(amount_summary, paths["audit"] / "amount_anomalies.csv")
    write_json(quality, paths["audit"] / "q1_data_quality.json")

    report = "\n".join(
        [
            "# 问题一数据质量报告",
            "",
            f"- 审计状态：**{status}**",
            f"- 输入文件：{workbook_name}",
            f"- 输入SHA-256：{before_hash}",
            f"- 企业数：{quality['enterprise_count']}",
            f"- 可进入特征构建：**{'是' if quality['can_build_features'] else '否'}**",
            "",
            "## 工作表概况",
            "",
            _markdown_table(table_summary),
            "",
            "## 企业覆盖",
            "",
            _markdown_table(pd.DataFrame([coverage_details])),
            "",
            "## 发票状态与金额符号",
            "",
            _markdown_table(status_summary),
            "",
            "## 日期范围",
            "",
            _markdown_table(date_summary),
            "",
            "## 重复与疑似重复",
            "",
            _markdown_table(duplicate_check),
            "",
            "## 金额分布与异常",
            "",
            _markdown_table(amount_summary),
            "",
            "## 错误与警告",
            "",
            "### 错误",
            "",
            "\n".join(f"- {item}" for item in errors) if errors else "- 无",
            "",
            "### 警告",
            "",
            "\n".join(f"- {item}" for item in warnings) if warnings else "- 无",
            "",
            "原始工作簿只读；任何重复、负数、零额和分位数外金额均未在审计阶段删除。",
            "",
        ]
    )
    write_text(report, paths["audit"] / "data_quality_report.md")
    write_text(report, paths["reports"] / "q1_data_quality_report.md")
    quality["input_sha256_after"] = sha256_file(input_path)
    write_json(quality, paths["audit"] / "q1_data_quality.json")
    assert_input_unchanged(input_path, before_hash)
    update_manifest(
        config,
        "01_data_audit",
        {"attachment1": before_hash},
        audit_outputs,
    )
    logger.info(
        "AUDIT_STATUS=%s enterprise=%d input_rows=%d output_rows=%d warnings=%d errors=%d",
        status,
        quality["enterprise_count"],
        len(input_invoice),
        len(output_invoice),
        len(warnings),
        len(errors),
    )
    if errors:
        raise DataQualityError("Data audit failed; see results/audit/data_quality_report.md")
    return quality


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    try:
        run_audit(config)
    except Exception as exc:
        print(f"01_data_audit.py FAILED: {exc}")
        return 1
    print("01_data_audit.py SUCCEEDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
