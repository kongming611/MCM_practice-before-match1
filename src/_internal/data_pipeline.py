"""Shared, deterministic utilities for question-one data checks and features."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("q1_config.yaml")
PROCESSED_ROOT = ROOT / "data" / "processed"
RUNTIME_ROOT = PROCESSED_ROOT / "_runtime"
OUTPUT_ROOT = ROOT / "outputs" / "q1"
PAPER_ROOT = ROOT / "paper" / "q1"
ENTERPRISE_COLUMNS = ["企业代号", "企业名称", "信誉评级", "是否违约"]
INVOICE_COLUMNS = [
    "企业代号",
    "发票号码",
    "开票日期",
    "金额",
    "税额",
    "价税合计",
    "发票状态",
]
COUNTERPARTY_COLUMNS = {"input": "销方单位代号", "output": "购方单位代号"}
CANONICAL_STATUS = {"valid": "有效", "void": "作废"}
MISSING_COUNTERPARTY = "__MISSING_COUNTERPARTY__"


class DataQualityError(RuntimeError):
    """Raised when a required data-quality assertion fails."""


class SchemaError(DataQualityError):
    """Raised when a workbook cannot be mapped to the required schema."""


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve paths relative to the repository."""

    configured = os.environ.get("Q1_CONFIG_PATH")
    path = config_path or (Path(configured) if configured else DEFAULT_CONFIG_PATH)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(path.resolve())
    return config


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable hash of a configuration, excluding its local path."""

    clean = {k: v for k, v in config.items() if k != "_config_path"}
    payload = json.dumps(clean, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path without embedding a machine-specific path."""

    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Create and return configured output directories."""

    outputs = config["outputs"]
    paths = {
        "audit": resolve_path(outputs["audit_dir"]),
        "features": resolve_path(outputs["feature_dir"]),
        "intermediate": resolve_path(outputs["intermediate_dir"]),
        "reports": resolve_path(outputs["reports_dir"]),
        "logs": resolve_path(outputs["logs_dir"]),
        "validation": resolve_path(outputs["validation_dir"]),
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logging(script_name: str, config: Mapping[str, Any], paths: Mapping[str, Path]) -> logging.Logger:
    """Configure UTF-8 file and concise console logging."""

    log_path = paths["logs"] / f"q1_{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(f"q1.{script_name}")
    logger.setLevel(getattr(logging, str(config.get("logging", {}).get("level", "INFO")).upper()))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    logger.info("log_file=%s", log_path.relative_to(ROOT))
    return logger


def sha256_file(path: Path) -> str:
    """Hash a file in chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a stable UTF-8-SIG CSV without an index."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = df.to_csv(
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
        na_rep="",
        float_format="%.10g",
    )
    path.write_bytes(text.encode("utf-8-sig"))


def write_text(text: str, path: Path) -> None:
    """Write deterministic UTF-8 text."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def write_json(payload: Mapping[str, Any], path: Path) -> None:
    """Write stable, human-readable UTF-8 JSON."""

    write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", path)


def relative(path: Path) -> str:
    """Return a repository-relative path for reports."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def locate_sheet(
    workbook: pd.ExcelFile,
    expected_name: str,
    required_columns: Sequence[str],
    role: str,
) -> str:
    """Find a sheet by configured name or, uniquely, by required columns."""

    names = list(workbook.sheet_names)
    if expected_name in names:
        header = pd.read_excel(workbook, sheet_name=expected_name, nrows=0)
        missing = [column for column in required_columns if column not in header.columns]
        if not missing:
            return expected_name
    candidates: list[str] = []
    for name in names:
        header = pd.read_excel(workbook, sheet_name=name, nrows=0)
        if set(required_columns).issubset(set(header.columns)):
            candidates.append(name)
    if len(candidates) != 1:
        raise SchemaError(
            f"Cannot uniquely locate {role}: expected={expected_name!r}, candidates={candidates}, sheets={names}"
        )
    return candidates[0]


def read_role_sheet(
    path: Path,
    role: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Read one configured workbook role after automatic sheet discovery."""

    required = (
        config["input"]["required_columns"]["enterprise"]
        if role == "enterprise"
        else config["input"]["required_columns"]["invoice"]
    )
    expected = config["input"]["expected_sheets"][f"{role}_invoice"] if role in {"input", "output"} else config["input"]["expected_sheets"]["enterprise"]
    with pd.ExcelFile(path) as workbook:
        sheet = locate_sheet(workbook, expected, required, role)
        dtype = {
            column: "string"
            for column in [
                "企业代号",
                "企业名称",
                "信誉评级",
                "是否违约",
                "发票号码",
                "发票状态",
                "销方单位代号",
                "购方单位代号",
            ]
        }
        # Extra dtype keys are ignored by pandas; numeric amount/date columns remain inferred.
        frame = pd.read_excel(workbook, sheet_name=sheet, dtype=dtype)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"{role} sheet {sheet!r} missing columns: {missing}")
    return frame, sheet


def normalize_status(series: pd.Series, config: Mapping[str, Any]) -> tuple[pd.Series, pd.Series]:
    """Return stripped raw status and canonical status (有效/作废/未知)."""

    raw = series.astype("string").str.strip()
    aliases = config["processing"]["status_aliases"]
    canonical = pd.Series("未知", index=raw.index, dtype="string")
    for key, target in CANONICAL_STATUS.items():
        values = {str(value).strip() for value in aliases.get(key, [])}
        canonical.loc[raw.isin(values)] = target
    canonical.loc[raw.isna() | raw.eq("")] = "未知"
    return raw, canonical


def normalize_invoice_frame(
    frame: pd.DataFrame,
    direction: str,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Normalize raw invoice columns while retaining all source rows."""

    if direction not in {"input", "output"}:
        raise ValueError(f"Unsupported direction: {direction}")
    counterpart = COUNTERPARTY_COLUMNS[direction]
    required = list(config["input"]["required_columns"]["invoice"]) + [counterpart]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SchemaError(f"{direction} invoice missing columns: {missing}")
    renamed = frame.copy()
    renamed = renamed.rename(
        columns={
            "企业代号": "enterprise_id",
            "发票号码": "invoice_number",
            "开票日期": "invoice_date",
            counterpart: "counterparty_id",
            "金额": "amount_yuan",
            "税额": "tax_yuan",
            "价税合计": "total_yuan",
            "发票状态": "raw_status",
        }
    )
    for column in ["enterprise_id", "invoice_number", "counterparty_id"]:
        renamed[column] = renamed[column].astype("string").str.strip()
    renamed["counterparty_known"] = renamed["counterparty_id"].notna() & renamed["counterparty_id"].ne("")
    renamed["counterparty_id"] = renamed["counterparty_id"].fillna(MISSING_COUNTERPARTY).replace("", MISSING_COUNTERPARTY)
    renamed["invoice_date"] = pd.to_datetime(renamed["invoice_date"], errors="coerce")
    for column in ["amount_yuan", "tax_yuan", "total_yuan"]:
        renamed[column] = pd.to_numeric(renamed[column], errors="coerce")
    renamed["raw_status"], renamed["status"] = normalize_status(renamed["raw_status"], config)
    renamed["direction"] = direction
    renamed["date_key"] = renamed["invoice_date"].dt.strftime("%Y-%m-%d")
    renamed["month"] = renamed["invoice_date"].dt.to_period("M").astype("string")
    renamed["date_invalid"] = renamed["invoice_date"].isna()
    renamed["status_unknown"] = renamed["status"].eq("未知")
    renamed["exact_duplicate"] = renamed[
        ["enterprise_id", "invoice_number", "date_key", "counterparty_id", "amount_yuan", "tax_yuan", "total_yuan", "raw_status"]
    ].duplicated(keep="first")
    renamed["is_valid"] = renamed["status"].eq("有效")
    renamed["is_void"] = renamed["status"].eq("作废")
    renamed["is_positive"] = renamed["total_yuan"].gt(0)
    renamed["is_negative"] = renamed["total_yuan"].lt(0)
    renamed["is_zero"] = renamed["total_yuan"].eq(0)
    renamed["arithmetic_error"] = (
        renamed[["amount_yuan", "tax_yuan", "total_yuan"]].notna().all(axis=1)
        & (renamed["amount_yuan"] + renamed["tax_yuan"] - renamed["total_yuan"]).abs().gt(
            float(config["processing"]["amount_tolerance_yuan"])
        )
    )
    return renamed


def collapse_invoice_lines(
    frame: pd.DataFrame,
    direction: str,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse exact duplicates and invoice line items by the documented business key."""

    normalized = normalize_invoice_frame(frame, direction, config)
    diagnostics = {
        "raw_rows": int(len(normalized)),
        "exact_duplicate_extra_rows": int(normalized["exact_duplicate"].sum()),
        "unknown_status_rows": int(normalized["status_unknown"].sum()),
        "invalid_date_rows": int(normalized["date_invalid"].sum()),
        "missing_enterprise_rows": int(normalized["enterprise_id"].isna().sum()),
        "missing_invoice_number_rows": int(normalized["invoice_number"].isna().sum()),
    }
    if diagnostics["unknown_status_rows"] or diagnostics["invalid_date_rows"]:
        raise DataQualityError(
            f"{direction} cannot be collapsed: unknown_status={diagnostics['unknown_status_rows']}, "
            f"invalid_date={diagnostics['invalid_date_rows']}"
        )
    if diagnostics["missing_enterprise_rows"] or diagnostics["missing_invoice_number_rows"]:
        raise DataQualityError(
            f"{direction} has missing business keys: enterprise={diagnostics['missing_enterprise_rows']}, "
            f"invoice_number={diagnostics['missing_invoice_number_rows']}"
        )
    deduped = normalized.loc[~normalized["exact_duplicate"]].copy()
    key = ["enterprise_id", "invoice_number", "date_key", "counterparty_id", "status"]
    grouped = (
        deduped.groupby(key, sort=False, dropna=False, as_index=False)
        .agg(
            invoice_date=("invoice_date", "first"),
            counterparty_known=("counterparty_known", "max"),
            amount_yuan=("amount_yuan", "sum"),
            tax_yuan=("tax_yuan", "sum"),
            total_yuan=("total_yuan", "sum"),
            line_count=("invoice_number", "size"),
        )
    )
    grouped["direction"] = direction
    grouped["total_10k"] = grouped["total_yuan"] / float(config["processing"]["amount_unit_divisor"])
    grouped["is_valid"] = grouped["status"].eq("有效")
    grouped["is_void"] = grouped["status"].eq("作废")
    grouped["is_positive"] = grouped["total_10k"].gt(0)
    grouped["is_negative"] = grouped["total_10k"].lt(0)
    grouped["is_zero"] = grouped["total_10k"].eq(0)
    grouped["month"] = grouped["invoice_date"].dt.to_period("M").astype("string")
    diagnostics["atomic_rows"] = int(len(grouped))
    diagnostics["business_duplicate_extra_rows"] = int(len(deduped) - len(grouped))
    return grouped, diagnostics


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide only where denominator is strictly positive; otherwise return NA."""

    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    mask = denominator.astype("float64").gt(0)
    result.loc[mask] = numerator.astype("float64").loc[mask] / denominator.astype("float64").loc[mask]
    return result


def longest_consecutive_run(values: Iterable[bool]) -> int:
    """Return the longest consecutive True run in a finite sequence."""

    longest = current = 0
    for value in values:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def stable_sort(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Stable-sort a frame for reproducible CSV output."""

    return df.sort_values(list(columns), kind="mergesort").reset_index(drop=True)


def check_finite(df: pd.DataFrame, columns: Sequence[str]) -> list[str]:
    """Return numeric columns containing positive or negative infinity."""

    bad: list[str] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if np.isinf(values).any():
            bad.append(column)
    return bad


def update_manifest(
    config: Mapping[str, Any],
    script_key: str,
    input_hashes: Mapping[str, str],
    output_paths_list: Sequence[Path],
    verify_only: bool = False,
) -> None:
    """Verify a prior identical run and optionally record current output hashes."""

    manifest_path = resolve_path(config["outputs"]["manifest"])
    previous: dict[str, Any] = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    scripts = previous.get("scripts", {})
    prior = scripts.get(script_key)
    current_config_hash = config_hash(config)
    script_path = ROOT / "src" / "_internal" / "stages" / f"{script_key}.py"
    common_path = ROOT / "src" / "_internal" / "data_pipeline.py"
    current_code_hash = hashlib.sha256(
        (script_path.read_bytes() + common_path.read_bytes())
    ).hexdigest()
    if (
        prior
        and prior.get("code_hash") == current_code_hash
        and prior.get("config_hash") == current_config_hash
        and prior.get("input_hashes") == dict(input_hashes)
    ):
        for path in output_paths_list:
            key = relative(path)
            expected = prior.get("output_hashes", {}).get(key)
            if expected and (not path.exists() or sha256_file(path) != expected):
                raise DataQualityError(f"Deterministic rerun check failed before write: {key}")
    if verify_only:
        return
    output_hashes = {relative(path): sha256_file(path) for path in output_paths_list if path.exists()}
    scripts[script_key] = {
        "code_hash": current_code_hash,
        "config_hash": current_config_hash,
        "input_hashes": dict(sorted(input_hashes.items())),
        "output_hashes": dict(sorted(output_hashes.items())),
        "python": platform.python_version(),
        "pandas": pd.__version__,
    }
    write_json({"scripts": scripts}, manifest_path)


def assert_input_unchanged(path: Path, before_hash: str) -> None:
    """Fail if a source workbook changed during a run."""

    after_hash = sha256_file(path)
    if before_hash != after_hash:
        raise DataQualityError(f"Original input was modified: {relative(path)}")
