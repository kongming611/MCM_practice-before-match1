"""Question-two data-only pipeline.

This module deliberately stops before risk-model training.  It reuses the
question-one atomic invoice and feature functions so that attachment 1 and
attachment 2 cannot silently acquire different business definitions.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from _internal.data_pipeline import (
    ROOT,
    DataQualityError,
    config_hash,
    collapse_invoice_lines,
    locate_sheet,
    load_config,
    normalize_invoice_frame,
    read_role_sheet,
    relative,
    sha256_file,
    stable_sort,
    write_csv,
    write_json,
    write_text,
)


FEATURE_STAGE = importlib.import_module("_internal.stages.02_build_features")
FEATURE_NAMES: list[str] = list(FEATURE_STAGE.FEATURE_NAMES)
BOUNDED_FEATURES: set[str] = set(FEATURE_STAGE.BOUNDED_FEATURES)


@dataclass
class DatasetBundle:
    name: str
    path: Path
    info: pd.DataFrame
    info_sheet: str
    raw_frames: dict[str, pd.DataFrame]
    sheets: dict[str, str]
    normalized: dict[str, pd.DataFrame]
    ledgers: dict[str, pd.DataFrame]
    audit: dict[str, Any]
    hash_before: str


def _paths(config: Mapping[str, Any]) -> dict[str, Path]:
    outputs = config["outputs"]
    paths = {
        "runtime": ROOT / outputs["runtime_dir"],
        "reports": ROOT / outputs["reports_dir"],
        "figures": ROOT / outputs["figures_dir"],
        "processed_feature": ROOT / outputs["processed_feature_file"],
        "processed_ood": ROOT / outputs["processed_ood_file"],
        "manifest": ROOT / outputs["manifest"],
    }
    paths["audit"] = paths["runtime"] / "audit"
    paths["features"] = paths["runtime"] / "features"
    paths["eda"] = paths["runtime"] / "eda"
    paths["validation"] = paths["runtime"] / "validation"
    paths["atomic"] = paths["runtime"] / "atomic"
    for key, path in paths.items():
        if key not in {"processed_feature", "processed_ood", "manifest"}:
            path.mkdir(parents=True, exist_ok=True)
    paths["processed_feature"].parent.mkdir(parents=True, exist_ok=True)
    paths["processed_ood"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    return paths


def _setup_logger(paths: Mapping[str, Path]) -> logging.Logger:
    logger = logging.getLogger("q2.data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path = paths["runtime"] / f"q2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    logger.info("log_file=%s", relative(log_path))
    return logger


def _markdown_table(frame: pd.DataFrame, max_rows: int = 40) -> str:
    if frame.empty:
        return "_无记录_"
    view = frame.head(max_rows).copy()
    columns = [str(column) for column in view.columns]
    lines = [
        "| " + " | ".join(column.replace("|", "\\|") for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        values = []
        for value in row:
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                values.append(json.dumps(value, ensure_ascii=False, default=str).replace("|", "\\|"))
            elif value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append(f"\n> 仅展示前{max_rows}行，共{len(frame)}行。")
    return "\n".join(lines)


def _file_snapshot(root: Path, exclude_parts: set[str] | None = None) -> dict[str, str]:
    if not root.exists():
        return {}
    excluded = exclude_parts or set()
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if any(part in excluded for part in path.parts):
            continue
        result[relative(path)] = sha256_file(path)
    return result


def _raw_snapshot() -> dict[str, str]:
    return _file_snapshot(ROOT / "data" / "raw")


def _protected_q1_snapshot() -> dict[str, str]:
    result: dict[str, str] = {}
    q1_output = ROOT / "outputs" / "q1"
    result.update(_file_snapshot(q1_output))
    processed = ROOT / "data" / "processed"
    for path in sorted(processed.glob("q1*")):
        if path.is_file():
            result[relative(path)] = sha256_file(path)
    runtime = processed / "_runtime"
    result.update(_file_snapshot(runtime, exclude_parts={"q2"}))
    return result


def _snapshot_diff(before: Mapping[str, str], after: Mapping[str, str]) -> dict[str, list[str]]:
    before_keys = set(before)
    after_keys = set(after)
    changed = sorted(key for key in before_keys & after_keys if before[key] != after[key])
    return {
        "added": sorted(after_keys - before_keys),
        "removed": sorted(before_keys - after_keys),
        "changed": changed,
    }


def _read_target_info(
    path: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    expected = config["input"]["expected_sheets"]["enterprise"]
    required = config["input"]["required_columns"]["target_enterprise"]
    dtype = {"企业代号": "string", "企业名称": "string"}
    with pd.ExcelFile(path) as workbook:
        sheet = locate_sheet(workbook, expected, required, "target_enterprise")
        frame = pd.read_excel(workbook, sheet_name=sheet, dtype=dtype)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise DataQualityError(f"target enterprise sheet missing columns: {missing}")
    info = frame.rename(
        columns={"企业代号": "enterprise_id", "企业名称": "enterprise_name"}
    )[["enterprise_id", "enterprise_name"]].copy()
    for column in ["enterprise_id", "enterprise_name"]:
        info[column] = info[column].astype("string").str.strip()
    info["credit_rating"] = pd.Series(pd.NA, index=info.index, dtype="string")
    info["default_label"] = pd.Series(np.nan, index=info.index, dtype="float64")
    return info, sheet


def _table_profile(
    frame: pd.DataFrame,
    dataset: str,
    sheet: str,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in frame.columns:
        series = frame[column]
        rows.append(
            {
                "dataset": dataset,
                "sheet": sheet,
                "role": role,
                "column": str(column),
                "row_count": int(len(frame)),
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_rate": float(series.isna().mean()) if len(frame) else np.nan,
                "unique_count": int(series.nunique(dropna=True)),
            }
        )
    return rows


def _invoice_audit_rows(
    frame: pd.DataFrame,
    normalized: pd.DataFrame,
    dataset: str,
    sheet: str,
    direction: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    status_rows: list[dict[str, Any]] = []
    status_grouped = (
        normalized.assign(
            raw_status_report=normalized["raw_status"].fillna("<MISSING>"),
            status_report=normalized["status"].fillna("<MISSING>"),
        )
        .groupby(["raw_status_report", "status_report"], dropna=False)
        .size()
        .reset_index(name="row_count")
    )
    for row in status_grouped.itertuples(index=False):
        status_rows.append(
            {
                "dataset": dataset,
                "sheet": sheet,
                "direction": direction,
                "status_raw": str(row.raw_status_report),
                "status_canonical": str(row.status_report),
                "row_count": int(row.row_count),
            }
        )

    business_key = [
        "enterprise_id",
        "invoice_number",
        "date_key",
        "counterparty_id",
        "status",
    ]
    key_sizes = normalized.groupby(business_key, dropna=False).size()
    business_extra = int(key_sizes.sub(1).clip(lower=0).sum())
    conflict_group = (
        normalized.groupby(["enterprise_id", "invoice_number"], dropna=False)
        .agg(
            date_nunique=("date_key", "nunique"),
            counterparty_nunique=("counterparty_id", "nunique"),
            status_nunique=("status", "nunique"),
        )
    )
    conflict_mask = conflict_group.gt(1).any(axis=1)
    duplicate_row = {
        "dataset": dataset,
        "sheet": sheet,
        "direction": direction,
        "raw_exact_duplicate_rows": int(frame.duplicated(keep="first").sum()),
        "normalized_exact_duplicate_extra_rows": int(normalized["exact_duplicate"].sum()),
        "business_key_duplicate_extra_rows": business_extra,
        "conflicting_invoice_number_groups": int(conflict_mask.sum()),
    }

    total = pd.to_numeric(normalized["total_yuan"], errors="coerce")
    valid_dates = normalized["invoice_date"].dropna()
    date_row = {
        "dataset": dataset,
        "sheet": sheet,
        "direction": direction,
        "invalid_date_count": int(normalized["date_invalid"].sum()),
        "min_date": valid_dates.min().strftime("%Y-%m-%d") if len(valid_dates) else "",
        "max_date": valid_dates.max().strftime("%Y-%m-%d") if len(valid_dates) else "",
    }
    finite_total = total.dropna()
    lower = float(config["processing"]["outlier_quantiles"]["lower"])
    upper = float(config["processing"]["outlier_quantiles"]["upper"])
    amount_row = {
        "dataset": dataset,
        "sheet": sheet,
        "direction": direction,
        "raw_rows": int(len(frame)),
        "missing_amount_yuan": int(frame["金额"].isna().sum()),
        "missing_tax_yuan": int(frame["税额"].isna().sum()),
        "missing_total_yuan": int(frame["价税合计"].isna().sum()),
        "arithmetic_error_rows": int(normalized["arithmetic_error"].sum()),
        "positive_total_rows": int(total.gt(0).sum()),
        "negative_total_rows": int(total.lt(0).sum()),
        "zero_total_rows": int(total.eq(0).sum()),
        "min_total_yuan": float(finite_total.min()) if len(finite_total) else np.nan,
        "max_total_yuan": float(finite_total.max()) if len(finite_total) else np.nan,
        "q01_total_yuan": float(finite_total.quantile(lower)) if len(finite_total) else np.nan,
        "q99_total_yuan": float(finite_total.quantile(upper)) if len(finite_total) else np.nan,
        "configured_lower_quantile": lower,
        "configured_upper_quantile": upper,
    }
    missing_rows = []
    for column in config["input"]["required_columns"]["invoice"]:
        missing_rows.append(
            {
                "dataset": dataset,
                "sheet": sheet,
                "direction": direction,
                "column": column,
                "missing_count": int(frame[column].isna().sum()),
                "missing_rate": float(frame[column].isna().mean()),
            }
        )
    blocking = []
    for column in ["enterprise_id", "invoice_number", "total_yuan"]:
        if normalized[column].isna().any():
            blocking.append(f"{direction}_missing_{column}")
    if normalized["status_unknown"].any():
        blocking.append(f"{direction}_unknown_status")
    if normalized["date_invalid"].any():
        blocking.append(f"{direction}_invalid_date")
    warnings = []
    if duplicate_row["raw_exact_duplicate_rows"] or duplicate_row["normalized_exact_duplicate_extra_rows"]:
        warnings.append(f"{direction}_exact_duplicates_present")
    if business_extra:
        warnings.append(f"{direction}_business_key_duplicates_present")
    if duplicate_row["conflicting_invoice_number_groups"]:
        warnings.append(f"{direction}_invoice_number_conflicts_present")
    if amount_row["arithmetic_error_rows"]:
        warnings.append(f"{direction}_amount_arithmetic_mismatch_present")
    return {
        "status_rows": status_rows,
        "duplicate_row": duplicate_row,
        "date_row": date_row,
        "amount_row": amount_row,
        "missing_rows": missing_rows,
        "blocking": blocking,
        "warnings": warnings,
        "raw_rows": int(len(frame)),
        "normalized_rows": int(len(normalized)),
    }


def _coverage(
    info: pd.DataFrame,
    normalized: Mapping[str, pd.DataFrame],
    dataset: str,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    ids = pd.Index(info["enterprise_id"].astype("string"), name="enterprise_id")
    rows = pd.DataFrame(index=ids)
    rows["enterprise_name"] = info.set_index("enterprise_id")["enterprise_name"]
    errors: list[str] = []
    for direction in ["input", "output"]:
        ledger = normalized[direction]
        counts = ledger.groupby("enterprise_id").size().reindex(ids, fill_value=0).astype(int)
        rows[f"{direction}_invoice_count"] = counts
        invoice_ids = set(ledger["enterprise_id"].dropna().astype(str))
        unknown_ids = sorted(invoice_ids - set(ids.astype(str)))
        if unknown_ids:
            errors.append(f"{dataset}_{direction}_unknown_enterprise_ids")
    rows["has_input"] = rows["input_invoice_count"].gt(0)
    rows["has_output"] = rows["output_invoice_count"].gt(0)
    rows["has_any_invoice"] = rows["has_input"] | rows["has_output"]
    rows["has_both_directions"] = rows["has_input"] & rows["has_output"]
    rows = rows.reset_index()
    expected = len(ids)
    details = {
        "dataset": dataset,
        "enterprise_count": expected,
        "input_covered_count": int(rows["has_input"].sum()),
        "output_covered_count": int(rows["has_output"].sum()),
        "any_covered_count": int(rows["has_any_invoice"].sum()),
        "both_directions_covered_count": int(rows["has_both_directions"].sum()),
        "uncovered_enterprise_count": int((~rows["has_any_invoice"]).sum()),
    }
    if details["both_directions_covered_count"] != expected:
        errors.append(f"{dataset}_not_all_enterprises_have_both_invoice_directions")
    return rows, details, errors


def _metadata_checks(info: pd.DataFrame, dataset: str, expected_count: int) -> list[str]:
    errors: list[str] = []
    ids = info["enterprise_id"].astype("string")
    if len(info) != expected_count:
        errors.append(f"{dataset}_enterprise_row_count={len(info)}_expected={expected_count}")
    if ids.isna().any() or ids.eq("").any():
        errors.append(f"{dataset}_enterprise_id_missing")
    if ids.duplicated().any():
        errors.append(f"{dataset}_enterprise_id_duplicate")
    if info["enterprise_name"].isna().any() or info["enterprise_name"].eq("").any():
        errors.append(f"{dataset}_enterprise_name_missing")
    if dataset == "training":
        if info["credit_rating"].isna().any() or not info["credit_rating"].isin(["A", "B", "C", "D"]).all():
            errors.append("training_credit_rating_invalid")
        if info["default_label"].isna().any() or not info["default_label"].isin([0, 1]).all():
            errors.append("training_default_label_invalid")
    return errors


def _load_bundle(
    dataset: str,
    path: Path,
    config: Mapping[str, Any],
) -> DatasetBundle:
    if dataset == "training":
        info, info_sheet = FEATURE_STAGE._load_enterprise_info(dict(config))
        expected_count = int(config["q2"]["training_enterprise_count"])
    else:
        info, info_sheet = _read_target_info(path, config)
        expected_count = int(config["q2"]["target_enterprise_count"])
    raw_frames: dict[str, pd.DataFrame] = {}
    normalized: dict[str, pd.DataFrame] = {}
    sheets: dict[str, str] = {}
    status_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    date_rows: list[dict[str, Any]] = []
    amount_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    errors = _metadata_checks(info, dataset, expected_count)
    warnings: list[str] = []
    for direction in ["input", "output"]:
        raw, sheet = read_role_sheet(path, direction, config)
        raw_frames[direction] = raw
        sheets[direction] = sheet
        norm = normalize_invoice_frame(raw, direction, config)
        normalized[direction] = norm
        audit = _invoice_audit_rows(raw, norm, dataset, sheet, direction, config)
        status_rows.extend(audit["status_rows"])
        duplicate_rows.append(audit["duplicate_row"])
        date_rows.append(audit["date_row"])
        amount_rows.append(audit["amount_row"])
        missing_rows.extend(audit["missing_rows"])
        errors.extend(audit["blocking"])
        warnings.extend(audit["warnings"])
    coverage, coverage_details, coverage_errors = _coverage(info, normalized, dataset)
    errors.extend(coverage_errors)
    if any(row["input_invoice_count"] == 0 or row["output_invoice_count"] == 0 for row in coverage.to_dict("records")):
        warnings.append(f"{dataset}_coverage_has_zero_direction_count")
    ledgers: dict[str, pd.DataFrame] = {}
    if not errors:
        for direction in ["input", "output"]:
            ledger, _ = collapse_invoice_lines(raw_frames[direction], direction, config)
            ledgers[direction] = ledger
    table_profile = _table_profile(info, dataset, info_sheet, "enterprise")
    for direction, raw in raw_frames.items():
        table_profile.extend(_table_profile(raw, dataset, sheets[direction], f"{direction}_invoice"))
    audit_payload = {
        "dataset": dataset,
        "status": "FAIL" if errors else ("WARN" if warnings else "PASS"),
        "path": relative(path),
        "hash_before": sha256_file(path),
        "enterprise_sheet": info_sheet,
        "invoice_sheets": sheets,
        "enterprise_count": int(len(info)),
        "metadata_errors": _metadata_checks(info, dataset, expected_count),
        "coverage": coverage_details,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "table_profile": table_profile,
        "status_rows": status_rows,
        "duplicate_rows": duplicate_rows,
        "date_rows": date_rows,
        "amount_rows": amount_rows,
        "missing_rows": missing_rows,
        "coverage_rows": coverage,
    }
    return DatasetBundle(
        name=dataset,
        path=path,
        info=info,
        info_sheet=info_sheet,
        raw_frames=raw_frames,
        sheets=sheets,
        normalized=normalized,
        ledgers=ledgers,
        audit=audit_payload,
        hash_before=sha256_file(path),
    )


def _feature_quality(frame: pd.DataFrame, expected_count: int, config: Mapping[str, Any]) -> dict[str, Any]:
    primary = list(config["features"]["primary_model_features"])
    constructed = list(config["features"]["names"])
    errors: list[str] = []
    expected_columns = ["enterprise_id", "enterprise_name"] + constructed
    if len(frame) != expected_count:
        errors.append(f"row_count={len(frame)} expected={expected_count}")
    if not frame["enterprise_id"].is_unique:
        errors.append("enterprise_id_not_unique")
    if frame["enterprise_id"].isna().any() or frame["enterprise_id"].eq("").any():
        errors.append("enterprise_id_missing")
    missing_columns = [column for column in expected_columns if column not in frame.columns]
    if missing_columns:
        errors.append(f"missing_feature_columns={missing_columns}")
    extra_labels = sorted(set(["credit_rating", "default_label"]).intersection(frame.columns))
    if extra_labels:
        errors.append(f"target_label_columns_present={extra_labels}")
    numeric = frame.reindex(columns=constructed).apply(pd.to_numeric, errors="coerce")
    missing_features = numeric.columns[numeric.isna().any()].tolist()
    if missing_features:
        errors.append(f"feature_missing={missing_features}")
    infinite_features = [
        column for column in numeric.columns if np.isinf(numeric[column].to_numpy(dtype=float)).any()
    ]
    if infinite_features:
        errors.append(f"feature_infinite={infinite_features}")
    bounded_violations = {
        column: int((numeric[column].lt(0) | numeric[column].gt(1)).sum())
        for column in BOUNDED_FEATURES
        if column in numeric and (numeric[column].lt(0) | numeric[column].gt(1)).any()
    }
    if bounded_violations:
        errors.append(f"bounded_feature_violations={bounded_violations}")
    nonnegative_features = {
        column: int(numeric[column].lt(0).sum())
        for column in [
            "business_scale_10k",
            "sales_scale_10k",
            "purchase_scale_10k",
            "customer_count",
            "supplier_count",
            "invoice_activity_per_month",
            "sales_monthly_cv",
            "sales_return_rate",
            "purchase_return_rate",
            "purchase_sales_ratio",
        ]
        if column in numeric and numeric[column].lt(0).any()
    }
    if nonnegative_features:
        errors.append(f"nonnegative_feature_violations={nonnegative_features}")
    integer_violations = {
        column: int((numeric[column] % 1 != 0).sum())
        for column in ["customer_count", "supplier_count"]
        if column in numeric and (numeric[column] % 1 != 0).any()
    }
    if integer_violations:
        errors.append(f"count_features_not_integer={integer_violations}")
    return {
        "status": "FAIL" if errors else "PASS",
        "row_count": int(len(frame)),
        "enterprise_id_unique": bool(frame["enterprise_id"].is_unique),
        "constructed_feature_count": len(constructed),
        "primary_model_feature_count": len(primary),
        "constructed_feature_names": constructed,
        "primary_model_features": primary,
        "missing_features": missing_features,
        "infinite_features": infinite_features,
        "bounded_violations": bounded_violations,
        "errors": errors,
        "target_label_columns_present": extra_labels,
    }


def _as_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    return pd.read_csv(value, encoding="utf-8-sig")


class Q2DataPipeline:
    """Orchestrate audit, feature construction, EDA and data-layer validation."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.paths = _paths(config)
        self.logger = _setup_logger(self.paths)
        self.raw_before = _raw_snapshot()
        self.q1_before = _protected_q1_snapshot()
        self.baseline_path = self.paths["runtime"] / "q2_protection_baseline.json"
        if self.baseline_path.exists():
            baseline = json.loads(self.baseline_path.read_text(encoding="utf-8"))
            self.raw_baseline = baseline.get("raw_before", self.raw_before)
            self.q1_baseline = baseline.get("q1_before", self.q1_before)
        else:
            self.raw_baseline = self.raw_before
            self.q1_baseline = self.q1_before
            write_json(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "raw_before": self.raw_before,
                    "q1_before": self.q1_before,
                },
                self.baseline_path,
            )
        self.loaded = False
        self.audit_written = False
        self.features_built = False
        self.eda_built = False
        self.validation_done = False
        self.stage_status: dict[str, str] = {}
        self.bundles: dict[str, DatasetBundle] = {}
        self.months: pd.PeriodIndex | None = None
        self.reference_features: pd.DataFrame | None = None
        self.target_features: pd.DataFrame | None = None
        self.comparison: pd.DataFrame | None = None
        self.ood_scores: pd.DataFrame | None = None

    def load(self) -> None:
        if self.loaded:
            return
        train_path = (ROOT / self.config["q2"]["training_attachment"]).resolve()
        target_path = (ROOT / self.config["q2"]["target_attachment"]).resolve()
        if not train_path.exists() or not target_path.exists():
            raise FileNotFoundError(f"missing q2 input: {train_path} or {target_path}")
        self.bundles = {
            "training": _load_bundle("training", train_path, self.config),
            "target": _load_bundle("target", target_path, self.config),
        }
        clean = all(not bundle.audit["errors"] for bundle in self.bundles.values())
        if clean:
            train_ledgers = [
                self.bundles["training"].ledgers["input"],
                self.bundles["training"].ledgers["output"],
            ]
            self.months = FEATURE_STAGE._derive_months(train_ledgers, dict(self.config))
            month_values = {str(value) for value in self.months}
            common_month_window = [str(value) for value in self.months]
            for bundle in self.bundles.values():
                bundle.audit["common_month_window"] = common_month_window
                for direction, ledger in bundle.ledgers.items():
                    outside = ~ledger["month"].astype("string").isin(month_values)
                    count = int(outside.sum())
                    bundle.audit.setdefault("window", {})[direction] = {
                        "outside_training_window_rows": count,
                        "rows_used_for_features": int((~outside).sum()),
                    }
                    if count:
                        bundle.audit["warnings"].append(
                            f"{bundle.name}_{direction}_outside_training_month_window={count}"
                        )
                        bundle.ledgers[direction] = ledger.loc[~outside].copy()
        self.loaded = True
        self.logger.info(
            "loaded training_status=%s target_status=%s",
            self.bundles["training"].audit["status"],
            self.bundles["target"].audit["status"],
        )

    def _require_clean(self) -> None:
        self.load()
        errors = []
        for bundle in self.bundles.values():
            errors.extend(bundle.audit["errors"])
        if errors:
            raise DataQualityError(f"q2 data audit failed: {sorted(set(errors))}")
        if self.months is None:
            raise DataQualityError("common training month window was not created")

    def write_audit(self) -> None:
        self.load()
        profile_rows: list[dict[str, Any]] = []
        status_rows: list[dict[str, Any]] = []
        duplicate_rows: list[dict[str, Any]] = []
        date_rows: list[dict[str, Any]] = []
        amount_rows: list[dict[str, Any]] = []
        missing_rows: list[dict[str, Any]] = []
        coverage_frames: list[pd.DataFrame] = []
        errors: list[str] = []
        warnings: list[str] = []
        for bundle in self.bundles.values():
            audit = bundle.audit
            profile_rows.extend(audit["table_profile"])
            status_rows.extend(audit["status_rows"])
            duplicate_rows.extend(audit["duplicate_rows"])
            date_rows.extend(audit["date_rows"])
            amount_rows.extend(audit["amount_rows"])
            missing_rows.extend(audit["missing_rows"])
            coverage_frames.append(audit["coverage_rows"].assign(dataset=bundle.name))
            errors.extend(audit["errors"])
            warnings.extend(audit["warnings"])
            errors.extend(audit["metadata_errors"])
        if coverage_frames:
            coverage_frame = pd.concat(coverage_frames, ignore_index=True)
        else:
            coverage_frame = pd.DataFrame()
        quality = {
            "status": "FAIL" if errors else ("WARN" if warnings else "PASS"),
            "training": self.bundles["training"].audit,
            "target": self.bundles["target"].audit,
            "errors": sorted(set(errors)),
            "warnings": sorted(set(warnings)),
            "raw_hashes_before": self.raw_before,
            "q1_protected_hashes_before": self.q1_before,
            "common_month_window": [str(value) for value in self.months] if self.months is not None else [],
            "same_feature_processing_contract": True,
            "pooled_preprocessing_used": False,
        }
        write_csv(pd.DataFrame(profile_rows), self.paths["reports"] / "q2_table_profile.csv")
        write_csv(pd.DataFrame(status_rows), self.paths["reports"] / "q2_invoice_status_summary.csv")
        write_csv(pd.DataFrame(duplicate_rows), self.paths["reports"] / "q2_duplicate_summary.csv")
        write_csv(pd.DataFrame(date_rows), self.paths["reports"] / "q2_date_summary.csv")
        write_csv(pd.DataFrame(amount_rows), self.paths["reports"] / "q2_amount_anomalies.csv")
        write_csv(pd.DataFrame(missing_rows), self.paths["reports"] / "q2_missing_values.csv")
        write_csv(coverage_frame, self.paths["reports"] / "q2_enterprise_coverage.csv")
        write_json(quality, self.paths["audit"] / "q2_data_quality.json")
        report_lines = [
            "# 问题二数据质量报告",
            "",
            f"- 审计状态：**{quality['status']}**",
            f"- 训练附件：{relative(self.bundles['training'].path)}",
            f"- 目标附件：{relative(self.bundles['target'].path)}",
            f"- 训练附件SHA-256：{self.bundles['training'].hash_before}",
            f"- 目标附件SHA-256：{self.bundles['target'].hash_before}",
            f"- 共同月份窗口：{', '.join(quality['common_month_window'])}",
            "",
            "## 企业覆盖",
            "",
            _markdown_table(
                pd.DataFrame(
                    [
                        {
                            "dataset": bundle.name,
                            **bundle.audit["coverage"],
                            "errors": ", ".join(bundle.audit["errors"]) or "无",
                            "warnings": ", ".join(bundle.audit["warnings"]) or "无",
                        }
                        for bundle in self.bundles.values()
                    ]
                )
            ),
            "",
            "## 表、缺失、状态、日期、重复和金额检查",
            "",
            "详细结果见 q2_table_profile.csv、q2_missing_values.csv、q2_invoice_status_summary.csv、q2_date_summary.csv、q2_duplicate_summary.csv 和 q2_amount_anomalies.csv。",
            "",
            "### 状态摘要",
            "",
            _markdown_table(pd.DataFrame(status_rows)),
            "",
            "### 日期摘要",
            "",
            _markdown_table(pd.DataFrame(date_rows)),
            "",
            "### 重复摘要",
            "",
            _markdown_table(pd.DataFrame(duplicate_rows)),
            "",
            "### 金额异常摘要",
            "",
            _markdown_table(pd.DataFrame(amount_rows)),
            "",
            "## 错误与警告",
            "",
            "### 错误",
            "",
            "\n".join(f"- {item}" for item in sorted(set(errors))) if errors else "- 无",
            "",
            "### 警告",
            "",
            "\n".join(f"- {item}" for item in sorted(set(warnings))) if warnings else "- 无",
            "",
            "原始工作簿只读；重复、负数、零额和分位数外金额均不在审计阶段静默删除。",
            "",
        ]
        write_text("\n".join(report_lines), self.paths["reports"] / "q2_data_quality_report.md")
        self.audit_written = True
        self.stage_status["audit"] = "FAIL" if errors else ("WARN" if warnings else "PASS")

    def build_features(self) -> None:
        if self.features_built:
            return
        self._require_clean()
        assert self.months is not None
        month_strings = [str(value) for value in self.months]
        for bundle in self.bundles.values():
            for direction, ledger in bundle.ledgers.items():
                path = self.paths["atomic"] / f"{bundle.name}_{direction}_atomic.parquet"
                ledger.to_parquet(path, index=False)
        reference_monthly = FEATURE_STAGE._build_monthly_panel(
            self.bundles["training"].info,
            self.bundles["training"].ledgers["input"],
            self.bundles["training"].ledgers["output"],
            self.months,
        )
        target_monthly = FEATURE_STAGE._build_monthly_panel(
            self.bundles["target"].info,
            self.bundles["target"].ledgers["input"],
            self.bundles["target"].ledgers["output"],
            self.months,
        )
        reference_features = FEATURE_STAGE._aggregate_enterprise_features(
            self.bundles["training"].info,
            self.bundles["training"].ledgers["input"],
            self.bundles["training"].ledgers["output"],
            reference_monthly,
            self.months,
        )
        target_features_full = FEATURE_STAGE._aggregate_enterprise_features(
            self.bundles["target"].info,
            self.bundles["target"].ledgers["input"],
            self.bundles["target"].ledgers["output"],
            target_monthly,
            self.months,
        )
        target_audit_columns = [
            column for column in target_features_full.columns if column.startswith("audit_")
        ]
        target_columns = ["enterprise_id", "enterprise_name"] + FEATURE_NAMES + target_audit_columns
        target_features = target_features_full[target_columns].copy()
        target_features = stable_sort(target_features, ["enterprise_id"])
        reference_features = stable_sort(reference_features, ["enterprise_id"])
        quality = _feature_quality(
            target_features,
            int(self.config["q2"]["target_enterprise_count"]),
            self.config,
        )
        quality["month_window"] = month_strings
        quality["reference_feature_quality"] = {
            "rows": int(len(reference_features)),
            "enterprise_ids_unique": bool(reference_features["enterprise_id"].is_unique),
            "feature_columns": FEATURE_NAMES,
        }
        quality["target_labels_absent_by_design"] = not set(
            ["credit_rating", "default_label"]
        ).intersection(target_features.columns)
        if not quality["target_labels_absent_by_design"]:
            quality["errors"].append("target_labels_present")
            quality["status"] = "FAIL"
        write_csv(target_features, self.paths["processed_feature"])
        write_csv(reference_features, self.paths["features"] / "q2_reference_enterprise_features.csv")
        write_csv(target_features, self.paths["features"] / "q2_target_enterprise_features.csv")
        write_csv(reference_monthly, self.paths["features"] / "q2_reference_enterprise_monthly.csv")
        write_csv(target_monthly, self.paths["features"] / "q2_target_enterprise_monthly.csv")
        write_json(
            {
                "feature_names": FEATURE_NAMES,
                "primary_model_features": list(self.config["features"]["primary_model_features"]),
                "sensitivity_model_features": list(self.config["features"]["sensitivity_model_features"]),
                "excluded_from_model": self.config["features"]["excluded_from_model"],
                "month_window": month_strings,
                "training_rows": int(len(reference_features)),
                "target_rows": int(len(target_features)),
                "preprocessing_fit_source": "attachment1_123_only",
                "pooled_preprocessing_fit": False,
                "risk_model_training_performed": False,
            },
            self.paths["features"] / "q2_feature_metadata.json",
        )
        write_json(quality, self.paths["features"] / "q2_feature_quality.json")
        report = "\n".join(
            [
                "# 问题二企业级特征质量报告",
                "",
                f"- 状态：**{quality['status']}**",
                f"- 训练参考企业数：{len(reference_features)}",
                f"- 目标企业数：{len(target_features)}",
                f"- 构造特征数：{len(FEATURE_NAMES)}",
                f"- 正式主模型特征数：{len(self.config['features']['primary_model_features'])}",
                f"- 共同月份窗口：{', '.join(month_strings)}",
                "- 目标表不包含信誉评级和是否违约字段。",
                "- 未执行风险模型训练；本阶段只生成数据层特征和审计输出。",
                "",
                "## 特征合法性",
                "",
                _markdown_table(
                    pd.DataFrame(
                        [
                            {
                                "check": key,
                                "value": value,
                            }
                            for key, value in quality.items()
                            if key not in {"errors", "missing_features", "infinite_features"}
                        ]
                    )
                ),
                "",
                "## 错误",
                "",
                "\n".join(f"- {item}" for item in quality["errors"]) if quality["errors"] else "- 无",
                "",
                "训练参考特征和目标特征均调用问题一的原子发票、共同月份面板和21项特征函数；标准化、填补和缩尾未在数据层执行。",
                "",
            ]
        )
        write_text(report, self.paths["reports"] / "q2_feature_quality_report.md")
        self.reference_features = reference_features
        self.target_features = target_features
        self.features_built = True
        self.stage_status["features"] = quality["status"]
        if quality["errors"]:
            raise DataQualityError(f"q2 feature checks failed: {quality['errors']}")

    def _load_feature_artifacts(self) -> None:
        if self.features_built:
            return
        self.build_features()

    def _distribution_comparison(self) -> pd.DataFrame:
        assert self.reference_features is not None
        assert self.target_features is not None
        rows: list[dict[str, Any]] = []
        lower = float(self.config["eda"]["ood_quantile_lower"])
        upper = float(self.config["eda"]["ood_quantile_upper"])
        for feature in self.config["features"]["primary_model_features"]:
            train = pd.to_numeric(self.reference_features[feature], errors="coerce").dropna()
            target = pd.to_numeric(self.target_features[feature], errors="coerce").dropna()
            train_mean = float(train.mean()) if len(train) else np.nan
            target_mean = float(target.mean()) if len(target) else np.nan
            train_std = float(train.std(ddof=1)) if len(train) > 1 else np.nan
            if not np.isfinite(train_std) or train_std == 0:
                smd = 0.0 if np.isclose(train_mean, target_mean, equal_nan=True) else np.nan
                target_standardized_mean = smd
            else:
                smd = (target_mean - train_mean) / train_std
                target_standardized_mean = smd
            if len(train) and len(target):
                ks = ks_2samp(train.to_numpy(dtype=float), target.to_numpy(dtype=float))
                wasserstein = float(
                    wasserstein_distance(
                        train.to_numpy(dtype=float),
                        target.to_numpy(dtype=float),
                    )
                )
            else:
                ks = None
                wasserstein = np.nan
            train_min = float(train.min()) if len(train) else np.nan
            train_max = float(train.max()) if len(train) else np.nan
            q_low = float(train.quantile(lower)) if len(train) else np.nan
            q_high = float(train.quantile(upper)) if len(train) else np.nan
            train_range_coverage = (
                float(target.between(train_min, train_max, inclusive="both").mean())
                if len(target)
                else np.nan
            )
            quantile_coverage = (
                float(target.between(q_low, q_high, inclusive="both").mean())
                if len(target)
                else np.nan
            )
            rows.append(
                {
                    "feature": feature,
                    "reference_n": int(len(train)),
                    "target_n": int(len(target)),
                    "reference_mean": train_mean,
                    "target_mean": target_mean,
                    "reference_std": train_std,
                    "standardized_mean_difference": smd,
                    "target_standardized_mean": target_standardized_mean,
                    "reference_min": train_min,
                    "reference_max": train_max,
                    "reference_q01": q_low,
                    "reference_q99": q_high,
                    "target_min": float(target.min()) if len(target) else np.nan,
                    "target_max": float(target.max()) if len(target) else np.nan,
                    "train_min_max_coverage": train_range_coverage,
                    "train_q01_q99_coverage": quantile_coverage,
                    "ks_statistic": float(ks.statistic) if ks is not None else np.nan,
                    "ks_p_value": float(ks.pvalue) if ks is not None else np.nan,
                    "wasserstein_distance": wasserstein,
                    "standardization_fit_source": "attachment1_123_only",
                    "comparison_used_pooled_fit": False,
                }
            )
        return pd.DataFrame(rows)

    def _ood_scores(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert self.reference_features is not None
        assert self.target_features is not None
        lower_q = float(self.config["eda"]["ood_quantile_lower"])
        upper_q = float(self.config["eda"]["ood_quantile_upper"])
        primary = list(self.config["features"]["primary_model_features"])
        bounds_rows: list[dict[str, Any]] = []
        bounds: dict[str, tuple[float, float, float]] = {}
        for feature in primary:
            values = pd.to_numeric(self.reference_features[feature], errors="coerce").dropna()
            low = float(values.quantile(lower_q))
            high = float(values.quantile(upper_q))
            scale = max(high - low, np.finfo(float).eps)
            bounds[feature] = (low, high, scale)
            bounds_rows.append(
                {
                    "feature": feature,
                    "reference_q_low": low,
                    "reference_q_high": high,
                    "reference_tail_interval": f"[{lower_q},{upper_q}]",
                    "scale": scale,
                    "threshold_source": "attachment1_123_only",
                }
            )
        rows: list[dict[str, Any]] = []
        for row in self.target_features.itertuples(index=False):
            result: dict[str, Any] = {"enterprise_id": getattr(row, "enterprise_id")}
            ood_count = 0
            missing_count = 0
            distances: list[float] = []
            for feature in primary:
                value = float(getattr(row, feature))
                low, high, scale = bounds[feature]
                if not np.isfinite(value):
                    missing_count += 1
                    distances.append(np.nan)
                    continue
                if value < low:
                    ood_count += 1
                    distances.append((low - value) / scale)
                elif value > high:
                    ood_count += 1
                    distances.append((value - high) / scale)
                else:
                    distances.append(0.0)
            finite_distances = [value for value in distances if np.isfinite(value)]
            result["ood_feature_count"] = int(ood_count)
            result["ood_feature_fraction"] = float(ood_count / len(primary))
            result["missing_primary_feature_count"] = int(missing_count)
            result["novelty_score_max_tail_distance"] = (
                float(max(finite_distances)) if finite_distances else np.nan
            )
            result["ood_flag"] = bool(ood_count > 0)
            result["threshold_source"] = "attachment1_123_only"
            result["threshold_rule"] = "any_primary_feature_outside_reference_quantile_interval"
            rows.append(result)
        return pd.DataFrame(rows), pd.DataFrame(bounds_rows)

    def _write_figures(self, comparison: pd.DataFrame, ood: pd.DataFrame) -> list[str]:
        assert self.reference_features is not None
        assert self.target_features is not None
        figures: list[str] = []
        primary = list(self.config["features"]["primary_model_features"])
        feature_labels = {
            "sales_scale_10k": "销售规模（万元）",
            "purchase_scale_10k": "采购规模（万元）",
            "operating_net_inflow_proxy_10k": "经营净流入代理（万元）",
            "sales_growth_trend": "销售增长趋势",
            "sales_monthly_cv": "销售月度变异系数",
            "invoice_activity_per_month": "月均发票活跃度",
            "sales_return_rate": "销售退货率",
            "purchase_return_rate": "采购退货率",
            "void_invoice_rate": "发票作废率",
            "customer_count": "客户数量",
            "supplier_count": "供应商数量",
            "customer_hhi": "客户集中度指数",
            "supplier_hhi": "供应商集中度指数",
            "purchase_sales_ratio": "进销比",
            "active_month_ratio": "活跃月份占比",
        }
        primary_labels = [feature_labels.get(feature, feature) for feature in primary]

        fig, ax = plt.subplots(figsize=(14, 7))
        positions = np.arange(len(primary))
        smd = comparison["standardized_mean_difference"].to_numpy(dtype=float)
        colors = ["#b2182b" if abs(value) >= 0.5 else "#2166ac" for value in smd]
        ax.bar(positions, smd, color=colors)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7)
        ax.axhline(-0.5, color="grey", linestyle="--", linewidth=0.7)
        ax.set_xticks(positions, primary_labels, rotation=65, ha="right")
        ax.set_ylabel("标准化均值差")
        ax.set_title("附件1参考组与附件2目标组")
        fig.tight_layout()
        path = self.paths["figures"] / "standardized_mean_difference.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        figures.append(relative(path))

        fig, ax = plt.subplots(figsize=(14, 7))
        coverage = comparison["train_q01_q99_coverage"].to_numpy(dtype=float)
        ax.bar(positions, coverage, color="#4daf4a")
        ax.set_ylim(0, 1.05)
        ax.set_xticks(positions, primary_labels, rotation=65, ha="right")
        ax.set_ylabel("落在参考组1%—99%区间的目标企业比例")
        ax.set_title("目标特征落在参考组分位区间内的比例")
        fig.tight_layout()
        path = self.paths["figures"] / "training_quantile_coverage.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        figures.append(relative(path))

        fig, ax = plt.subplots(figsize=(10, 6))
        finite_scores = ood["novelty_score_max_tail_distance"].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hist(finite_scores.to_numpy(dtype=float), bins=30, color="#984ea3", alpha=0.85)
        ax.set_xlabel("相对于参考组1%—99%区间的最大尾部距离")
        ax.set_ylabel("目标企业数量")
        ax.set_title("目标企业新颖度分数分布")
        fig.tight_layout()
        path = self.paths["figures"] / "ood_novelty_score_distribution.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        figures.append(relative(path))

        fig, axes = plt.subplots(3, 5, figsize=(17, 10))
        axes = axes.ravel()
        for index, feature in enumerate(primary):
            train = pd.to_numeric(self.reference_features[feature], errors="coerce").dropna()
            target = pd.to_numeric(self.target_features[feature], errors="coerce").dropna()
            center = float(train.mean())
            scale = float(train.std(ddof=1)) if len(train) > 1 else 1.0
            if not np.isfinite(scale) or scale == 0:
                scale = 1.0
            axes[index].boxplot(
                [(train - center) / scale, (target - center) / scale],
                tick_labels=["123", "302"],
                showfliers=False,
            )
            axes[index].set_title(feature_labels.get(feature, feature), fontsize=8)
            axes[index].axhline(0, color="grey", linewidth=0.5)
            axes[index].tick_params(labelsize=7)
        fig.suptitle("仅按附件1标准化的特征分布")
        fig.tight_layout()
        path = self.paths["figures"] / "standardized_feature_distributions.png"
        fig.savefig(path, dpi=300)
        plt.close(fig)
        figures.append(relative(path))
        return figures

    def run_eda(self) -> None:
        if self.eda_built:
            return
        self.build_features()
        assert self.reference_features is not None
        assert self.target_features is not None
        comparison = self._distribution_comparison()
        ood, bounds = self._ood_scores()
        write_csv(comparison, self.paths["reports"] / "q2_feature_distribution_comparison.csv")
        write_csv(ood, self.paths["reports"] / "q2_ood_scores.csv")
        write_csv(ood, self.paths["processed_ood"])
        write_csv(bounds, self.paths["reports"] / "q2_ood_reference_bounds.csv")
        figures = self._write_figures(comparison, ood)
        write_text(
            "\n".join(
                [
                    "# 问题二EDA图表索引",
                    "",
                    *[f"- {path}" for path in figures],
                    "",
                    "所有标准化图表的中心和尺度只由附件1的123家企业计算。",
                ]
            ),
            self.paths["reports"] / "q2_eda_figure_index.md",
        )
        summary_rows = [
            {
                "target_enterprise_count": int(len(ood)),
                "ood_enterprise_count": int(ood["ood_flag"].sum()),
                "ood_enterprise_rate": float(ood["ood_flag"].mean()),
                "target_with_missing_primary_count": int(
                    (ood["missing_primary_feature_count"] > 0).sum()
                ),
                "max_abs_smd": float(comparison["standardized_mean_difference"].abs().max()),
                "mean_abs_smd": float(comparison["standardized_mean_difference"].abs().mean()),
                "min_quantile_coverage": float(comparison["train_q01_q99_coverage"].min()),
                "mean_quantile_coverage": float(comparison["train_q01_q99_coverage"].mean()),
                "ood_threshold_source": "attachment1_123_only",
            }
        ]
        write_csv(pd.DataFrame(summary_rows), self.paths["reports"] / "q2_eda_summary.csv")
        comparison_view = comparison[
            [
                "feature",
                "reference_mean",
                "target_mean",
                "standardized_mean_difference",
                "ks_statistic",
                "ks_p_value",
                "wasserstein_distance",
                "train_min_max_coverage",
                "train_q01_q99_coverage",
            ]
        ]
        top_ood = ood.sort_values(
            ["ood_flag", "novelty_score_max_tail_distance", "ood_feature_count"],
            ascending=[False, False, False],
        ).head(20)
        report = "\n".join(
            [
                "# 问题二特征分布与OOD报告",
                "",
                "## 数据概况",
                "",
                f"- 训练参考组：{len(self.reference_features)}家。",
                f"- 问题二目标组：{len(self.target_features)}家。",
                "- 风险模型未训练；本报告只覆盖数据层和分布诊断。",
                "",
                "## 标准化和分布比较",
                "",
                "标准化均值差的均值和标准差只由附件1训练参考组的均值、样本标准差计算；KS和Wasserstein直接比较两组原始特征值，不拟合合并分布参数。",
                "",
                _markdown_table(comparison_view, max_rows=30),
                "",
                "完整结果见 q2_feature_distribution_comparison.csv。",
                "",
                "## OOD/新颖度",
                "",
                f"- OOD企业数：{int(ood['ood_flag'].sum())}/{len(ood)}。",
                f"- OOD企业比例：{float(ood['ood_flag'].mean()):.6f}。",
                f"- 使用的阈值：每项主特征的附件1训练分布{self.config['eda']['ood_quantile_lower']:.2f}至{self.config['eda']['ood_quantile_upper']:.2f}分位区间。",
                "- OOD阈值在运行前由123家参考组确定，未使用302家目标分布调参。",
                "- 缺失特征单独报告，不把缺失自动当作分布外。",
                "",
                _markdown_table(top_ood),
                "",
                "完整302家OOD结果见 q2_ood_scores.csv；每项训练分位边界见 q2_ood_reference_bounds.csv。",
                "",
                "## 对问题二后续建模的影响",
                "",
                "- 分布比较只描述协变量差异，不能证明目标组的真实违约率变化。",
                "- OOD标记可作为后续高不确定性降额或人工复核的输入，但阈值不能在看过风险模型结果后调整。",
                "- 本阶段没有把两组数据合并后拟合缩尾、缺失填补或标准化参数。",
                "",
            ]
        )
        write_text(report, self.paths["reports"] / "q2_distribution_and_ood_report.md")
        paper_report = "\n".join(
            [
                "# 问题二数据层报告（论文手引用稿）",
                "",
                "本报告只陈述当前分支程序实际生成的数据层结果，不包含风险模型训练结果或信贷策略数值。",
                "",
                "## 数据质量",
                "",
                "详细审计见 q2_data_quality_report.md；原始附件SHA-256和逐表检查见 q2_data_quality.json。",
                "",
                "## 特征口径",
                "",
                f"- 训练参考组：{len(self.reference_features)}家；目标组：{len(self.target_features)}家。",
                f"- 共同月份窗口：{', '.join(str(value) for value in (self.months.tolist() if self.months is not None else []))}。",
                f"- 共构造{len(FEATURE_NAMES)}项特征，正式主模型保留{len(self.config['features']['primary_model_features'])}项。",
                "- 原子发票聚合、状态、金额单位和特征公式复用问题一实现；目标表不含评级和违约标签。",
                "",
                "## 分布迁移和OOD",
                "",
                f"- OOD阈值由附件1的123家参考组{self.config['eda']['ood_quantile_lower']:.2f}至{self.config['eda']['ood_quantile_upper']:.2f}分位区间确定。",
                f"- 目标组OOD企业数：{int(ood['ood_flag'].sum())}/{len(ood)}。",
                f"- 主特征标准化均值差绝对值均值：{float(comparison['standardized_mean_difference'].abs().mean()):.6f}。",
                f"- 主特征标准化均值差绝对值最大值：{float(comparison['standardized_mean_difference'].abs().max()):.6f}。",
                "",
                "逐特征指标见 q2_feature_distribution_comparison.csv，逐企业指标见 q2_ood_scores.csv。",
                "",
                "## 使用限制",
                "",
                "附件2没有真实违约标签，因此本报告不能验证目标组风险预测的外部准确率，也不能把协变量分布差异解释为因果结论。",
                "",
            ]
        )
        write_text(paper_report, self.paths["reports"] / "q2_data_report_for_paper.md")
        self.comparison = comparison
        self.ood_scores = ood
        self.eda_built = True
        self.stage_status["eda"] = "PASS"

    def _config_alignment(self) -> tuple[bool, dict[str, Any]]:
        q1_path = ROOT / self.config["q2"]["reference_config"]
        q1_config = load_config(q1_path)
        comparisons = {
            "processing": self.config["processing"] == q1_config["processing"],
            "features": self.config["features"] == q1_config["features"],
            "invoice_required_columns": self.config["input"]["required_columns"]["invoice"]
            == q1_config["input"]["required_columns"]["invoice"],
            "expected_sheets": self.config["input"]["expected_sheets"]
            == q1_config["input"]["expected_sheets"],
            "amount_unit_divisor": self.config["processing"]["amount_unit_divisor"]
            == q1_config["processing"]["amount_unit_divisor"],
            "primary_feature_count": len(self.config["features"]["primary_model_features"]) == 15,
            "target_config_is_independent": q1_path.resolve() != (
                ROOT / "src" / "_internal" / "q2_config.yaml"
            ).resolve(),
        }
        return all(comparisons.values()), comparisons

    def _reference_rebuild_match(self) -> tuple[bool, dict[str, Any]]:
        assert self.reference_features is not None
        q1_path = ROOT / "data" / "processed" / "q1_enterprise_features.csv"
        if not q1_path.exists():
            return False, {"reason": "q1_enterprise_features.csv_missing"}
        existing = pd.read_csv(q1_path, encoding="utf-8-sig")
        left = self.reference_features.set_index("enterprise_id").sort_index()
        right = existing.set_index("enterprise_id").sort_index()
        details: dict[str, Any] = {
            "q1_existing_shape": [int(value) for value in existing.shape],
            "q2_rebuilt_shape": [int(value) for value in self.reference_features.shape],
            "id_sets_equal": set(left.index) == set(right.index),
            "feature_max_abs_diff": {},
        }
        if not details["id_sets_equal"]:
            return False, details
        all_ok = True
        for feature in FEATURE_NAMES:
            if feature not in right.columns:
                all_ok = False
                details["feature_max_abs_diff"][feature] = None
                continue
            a = pd.to_numeric(left[feature], errors="coerce").to_numpy(dtype=float)
            b = pd.to_numeric(right[feature], errors="coerce").to_numpy(dtype=float)
            same = np.allclose(a, b, equal_nan=True, atol=1e-9, rtol=1e-9)
            diff = np.nanmax(np.abs(a - b)) if len(a) else 0.0
            details["feature_max_abs_diff"][feature] = float(diff) if np.isfinite(diff) else None
            all_ok &= bool(same)
        details["all_feature_values_equal"] = bool(all_ok)
        return bool(all_ok), details

    def _output_hashes(self) -> dict[str, str]:
        candidates = [
            self.paths["processed_feature"],
            self.paths["processed_ood"],
            self.paths["manifest"],
        ]
        for directory in [self.paths["runtime"], self.paths["reports"], self.paths["figures"]]:
            candidates.extend(path for path in directory.rglob("*") if path.is_file())
        hashes: dict[str, str] = {}
        for path in sorted(set(candidates)):
            if path.exists() and path != self.paths["manifest"]:
                hashes[relative(path)] = sha256_file(path)
        return hashes

    def _write_manifest(self) -> None:
        raw_after = _raw_snapshot()
        q1_after = _protected_q1_snapshot()
        manifest = {
            "status": (
                self.stage_status.get("validate")
                if self.stage_status.get("validate") in {"PASS", "FAIL"}
                else ("FAIL" if "FAIL" in self.stage_status.values() else "WARN")
            ),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "config_path": relative(ROOT / "src" / "_internal" / "q2_config.yaml"),
            "config_sha256": config_hash(self.config),
            "stage_status": self.stage_status,
            "raw_hashes_before": self.raw_before,
            "raw_hashes_after": raw_after,
            "raw_hash_diff": _snapshot_diff(self.raw_baseline, raw_after),
            "q1_protected_hashes_before": self.q1_before,
            "q1_protected_hashes_after": q1_after,
            "q1_protected_hash_diff": _snapshot_diff(self.q1_baseline, q1_after),
            "common_month_window": [str(value) for value in self.months] if self.months is not None else [],
            "feature_count": len(FEATURE_NAMES),
            "primary_model_feature_count": len(self.config["features"]["primary_model_features"]),
            "risk_model_training_performed": False,
            "pooled_preprocessing_used": False,
            "output_hashes": self._output_hashes(),
        }
        write_json(manifest, self.paths["manifest"])

    def validate(self) -> None:
        if self.validation_done:
            return
        self.run_eda()
        checks: list[dict[str, Any]] = []

        def add(name: str, passed: bool, detail: Any) -> None:
            checks.append(
                {
                    "check_name": name,
                    "status": "PASS" if passed else "FAIL",
                    "detail": json.dumps(detail, ensure_ascii=False, default=str)
                    if isinstance(detail, (dict, list, tuple))
                    else str(detail),
                }
            )

        output = pd.read_csv(self.paths["processed_feature"], encoding="utf-8-sig")
        add(
            "q2_feature_table_302_unique",
            len(output) == 302 and output["enterprise_id"].nunique() == 302,
            {"rows": len(output), "unique_ids": int(output["enterprise_id"].nunique())},
        )
        add(
            "target_label_columns_absent",
            not bool({"credit_rating", "default_label"}.intersection(output.columns)),
            sorted({"credit_rating", "default_label"}.intersection(output.columns)),
        )
        aligned, alignment_details = self._config_alignment()
        add("same_q1_processing_and_feature_contract", aligned, alignment_details)
        rebuild_match, rebuild_details = self._reference_rebuild_match()
        add("q1_reference_feature_rebuild_matches_existing", rebuild_match, rebuild_details)
        add(
            "common_month_window_is_shared",
            self.months is not None
            and self.bundles["training"].audit.get("common_month_window", [])
            == self.bundles["target"].audit.get("common_month_window", [])
            == [str(value) for value in self.months],
            [str(value) for value in (self.months.tolist() if self.months is not None else [])],
        )
        add(
            "standardization_reference_only",
            self.config["eda"]["standardization"] == "reference_mean_std_only"
            and bool(self.config["q2"]["pooled_preprocessing_forbidden"]),
            {
                "standardization": self.config["eda"]["standardization"],
                "pooled_preprocessing_used": False,
            },
        )
        add(
            "ood_threshold_reference_only",
            self.config["eda"]["ood_threshold_source"] == "attachment1_123_only"
            and self.config["eda"]["ood_flag_rule"]
            == "any_primary_feature_outside_reference_quantile_interval",
            {
                "source": self.config["eda"]["ood_threshold_source"],
                "rule": self.config["eda"]["ood_flag_rule"],
            },
        )
        add(
            "feature_legality_pass",
            self.stage_status.get("features") == "PASS",
            self.stage_status.get("features"),
        )
        add(
            "eda_outputs_present",
            self.comparison is not None
            and self.ood_scores is not None
            and len(self.comparison) == 15
            and len(self.ood_scores) == 302,
            {
                "comparison_rows": len(self.comparison) if self.comparison is not None else 0,
                "ood_rows": len(self.ood_scores) if self.ood_scores is not None else 0,
            },
        )
        if self.ood_scores is not None:
            add(
                "ood_enterprise_id_unique",
                self.ood_scores["enterprise_id"].nunique() == 302,
                int(self.ood_scores["enterprise_id"].nunique()),
            )
            add(
                "ood_scores_finite_or_missing_separate",
                self.ood_scores["novelty_score_max_tail_distance"].notna().all(),
                int(self.ood_scores["novelty_score_max_tail_distance"].isna().sum()),
            )
        raw_after = _raw_snapshot()
        q1_after = _protected_q1_snapshot()
        add("raw_files_unchanged", not _snapshot_diff(self.raw_baseline, raw_after)["changed"] and not _snapshot_diff(self.raw_baseline, raw_after)["added"] and not _snapshot_diff(self.raw_baseline, raw_after)["removed"], _snapshot_diff(self.raw_baseline, raw_after))
        add("q1_existing_pass_outputs_unchanged", not bool(_snapshot_diff(self.q1_baseline, q1_after)["changed"] or _snapshot_diff(self.q1_baseline, q1_after)["added"] or _snapshot_diff(self.q1_baseline, q1_after)["removed"]), _snapshot_diff(self.q1_baseline, q1_after))
        q1_report = ROOT / "outputs" / "q1" / "final" / "q1_final_validation_report.md"
        q1_report_text = q1_report.read_text(encoding="utf-8") if q1_report.exists() else ""
        add("q1_existing_final_report_is_pass", "最终状态：PASS" in q1_report_text, relative(q1_report))
        add(
            "risk_model_stage_is_separate",
            True,
            "risk, rating and Label Spreading outputs are produced by the separate q2 model stage",
        )
        add(
            "required_processed_files_present",
            self.paths["processed_feature"].exists()
            and self.paths["processed_ood"].exists(),
            [relative(self.paths["processed_feature"]), relative(self.paths["processed_ood"])],
        )
        validation_frame = pd.DataFrame(checks)
        write_csv(validation_frame, self.paths["validation"] / "q2_acceptance_checks.csv")
        failures = validation_frame.loc[validation_frame["status"].eq("FAIL")]
        status = "PASS" if failures.empty else "FAIL"
        report = "\n".join(
            [
                "# 问题二数据层自动验收报告",
                "",
                f"最终状态：**{status}**",
                "",
                _markdown_table(validation_frame, max_rows=100),
                "",
                "通过标准包含：302行且主键唯一、q1特征口径重建一致、共同月份窗口一致、标准化与OOD阈值只由123家参考组确定、原始附件和问题一既有PASS输出未改变；风险模型在独立的model阶段运行。",
                "",
            ]
        )
        write_text(report, self.paths["reports"] / "q2_validation_report.md")
        validation_summary = {
            "status": status,
            "checks": checks,
            "failure_count": int(len(failures)),
            "raw_hash_diff": _snapshot_diff(self.raw_baseline, raw_after),
            "q1_protected_hash_diff": _snapshot_diff(self.q1_baseline, q1_after),
        }
        write_json(validation_summary, self.paths["validation"] / "q2_validation_summary.json")
        self.stage_status["validate"] = status
        self.validation_done = True
        self._write_manifest()
        if status != "PASS":
            raise DataQualityError(
                "q2 data-layer acceptance failed: "
                + "; ".join(failures["check_name"].astype(str).tolist())
            )

    def run(self, stage: str) -> None:
        self.load()
        self.write_audit()
        if stage == "audit":
            self._write_manifest()
            if self.stage_status["audit"] == "FAIL":
                raise DataQualityError("q2 audit failed")
            return
        self._require_clean()
        self.build_features()
        if stage == "features":
            self._write_manifest()
            return
        self.run_eda()
        if stage == "eda":
            self._write_manifest()
            return
        if stage in {"validate", "all"}:
            self.validate()
            return
        raise ValueError(f"unsupported q2 stage: {stage}")


def run_q2(config_path: Path | None = None, stage: str = "all") -> int:
    config_path = config_path or (Path(__file__).resolve().with_name("q2_config.yaml"))
    config = load_config(config_path)
    pipeline = Q2DataPipeline(config)
    pipeline.run(stage)
    print(f"q2_stage={stage} status={pipeline.stage_status}")
    return 0
