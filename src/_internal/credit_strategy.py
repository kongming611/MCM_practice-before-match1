"""Shared utilities for the fourth-batch credit decision workflow.

The fourth batch deliberately consumes the already accepted enterprise risk
table.  This module owns only attachment-3 parsing, monotone churn fitting,
MILP construction/validation, and deterministic strategy summaries.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix
from sklearn.isotonic import IsotonicRegression

from _internal.data_pipeline import (
    OUTPUT_ROOT,
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


CHURN_RATINGS = ("A", "B", "C")
ALL_RATINGS = ("A", "B", "C", "D")
SOLVER_TOLERANCE = 1e-8


def _as_float(value: Any) -> float:
    """Convert a scalar to a JSON/CSV-friendly float."""

    return float(value) if value is not None and not pd.isna(value) else float("nan")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalise_rate(series: pd.Series) -> tuple[pd.Series, str, int]:
    """Normalise a rate expressed either as a fraction or a percentage.

    The decision is data-derived and returned explicitly so the audit report
    records whether division by 100 was applied.
    """

    numeric = _numeric(series)
    non_null = numeric.dropna()
    if non_null.empty:
        return numeric.astype(float), "no_numeric_values", 0
    invalid = int(((non_null < 0) | (non_null > 100)).sum())
    if invalid:
        return numeric.astype(float), "invalid_values_outside_0_100", invalid
    if float(non_null.max()) > 1.0:
        return numeric.astype(float) / 100.0, "divided_by_100_because_max_exceeded_1", 0
    return numeric.astype(float), "kept_as_0_1_fraction", 0


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _raw_column_label(raw: pd.DataFrame, index: int) -> str:
    """Preserve the workbook header, including pandas-style unnamed headers."""

    value = _text(raw.iloc[0, index])
    return value if value else f"Unnamed: {index}"


def _find_attachment3_sheet(workbook: Path, expected_sheet: str | None) -> str:
    """Use the configured sheet when valid, otherwise identify a non-empty sheet."""

    excel = pd.ExcelFile(workbook)
    if expected_sheet and expected_sheet in excel.sheet_names:
        candidate = pd.read_excel(workbook, sheet_name=expected_sheet, header=None, nrows=8)
        if not candidate.dropna(how="all").empty:
            return expected_sheet
    candidates: list[str] = []
    for sheet in excel.sheet_names:
        frame = pd.read_excel(workbook, sheet_name=sheet, header=None, nrows=12)
        cells = " ".join(_text(v) for v in frame.to_numpy().ravel())
        if frame.dropna(how="all").empty:
            continue
        if "利率" in cells or "interest" in cells.lower() or "流失" in cells:
            candidates.append(sheet)
    if len(candidates) != 1:
        raise ValueError(f"Unable to uniquely identify attachment-3 sheet: {candidates}")
    return candidates[0]


def _detect_rate_column(raw: pd.DataFrame) -> int:
    scores: list[tuple[int, int]] = []
    for index in range(raw.shape[1]):
        cells = " ".join(_text(v) for v in raw.iloc[:8, index].tolist())
        numeric_count = int(_numeric(raw.iloc[1:, index]).notna().sum())
        score = 0
        if "利率" in cells or "interest" in cells.lower() or "rate" in cells.lower():
            score += 100
        score += min(numeric_count, 50)
        scores.append((score, index))
    if not scores or max(scores)[0] <= 0:
        raise ValueError("Could not identify the interest-rate column in attachment 3")
    return max(scores)[1]


def _detect_rating_columns(raw: pd.DataFrame, rate_column: int) -> dict[str, int]:
    detected: dict[str, int] = {}
    for index in range(raw.shape[1]):
        if index == rate_column:
            continue
        cells = [_text(v) for v in raw.iloc[:8, index].tolist()]
        joined = " ".join(cells)
        match = re.search(r"评级\s*([ABC])", joined)
        if match:
            detected[match.group(1)] = index
            continue
        for rating in CHURN_RATINGS:
            if any(cell.upper() == rating for cell in cells):
                detected[rating] = index
                break
    if set(detected) != set(CHURN_RATINGS):
        remaining = [index for index in range(raw.shape[1]) if index != rate_column and index not in detected.values()]
        for rating, index in zip((rating for rating in CHURN_RATINGS if rating not in detected), remaining):
            detected[rating] = index
    if set(detected) != set(CHURN_RATINGS):
        raise ValueError(f"Could not identify A/B/C churn columns: {detected}")
    return detected


def read_attachment3(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Read, identify, normalize, and audit the attachment-3 workbook."""

    churn_config = config["churn_model"]
    workbook = resolve_path(churn_config["attachment3"])
    if not workbook.exists():
        raise FileNotFoundError(workbook)
    sheet = _find_attachment3_sheet(workbook, str(churn_config.get("expected_sheet", "")))
    raw = pd.read_excel(workbook, sheet_name=sheet, header=None, dtype=object)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    rate_column = _detect_rate_column(raw)
    rating_columns = _detect_rating_columns(raw, rate_column)
    rate_header = _raw_column_label(raw, rate_column)
    rate_numeric_all = _numeric(raw.iloc[:, rate_column])
    data_mask = rate_numeric_all.notna()
    data_indices = list(raw.index[data_mask])
    if not data_indices:
        raise ValueError("Attachment 3 has no numeric interest-rate observations")

    rate_raw = rate_numeric_all.loc[data_indices].reset_index(drop=True)
    interest_rate, rate_action, rate_invalid = _normalise_rate(rate_raw)
    normalized_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    def audit(check: str, value: Any, status: str, detail: str) -> None:
        audit_rows.append({"check_name": check, "value": value, "status": status, "detail": detail})

    audit("workbook_exists", True, "PASS", relative(workbook))
    audit("sheet_detected", sheet, "PASS", f"expected_sheet={churn_config.get('expected_sheet')}")
    audit("raw_interest_rate_column", rate_header, "PASS", f"column_index={rate_column}")
    audit("raw_interest_rate_normalization", rate_action, "PASS" if rate_invalid == 0 else "FAIL", "Conversion recorded explicitly")
    audit("interest_rate_invalid_value_count", rate_invalid, "PASS" if rate_invalid == 0 else "FAIL", "Raw interest-rate numeric range check")

    duplicate_count = int(rate_raw.duplicated().sum())
    strict_increasing = bool(np.all(np.diff(interest_rate.to_numpy(dtype=float)) > SOLVER_TOLERANCE))
    audit("interest_rate_duplicate_count", duplicate_count, "PASS" if duplicate_count == 0 else "FAIL", "Duplicate rates are not permitted")
    audit("interest_rate_strictly_increasing", strict_increasing, "PASS" if strict_increasing else "FAIL", "Observed row order")
    audit("interest_rate_min", float(interest_rate.min()), "PASS" if interest_rate.min() >= 0.04 - SOLVER_TOLERANCE else "FAIL", "Expected 4% to 15%")
    audit("interest_rate_max", float(interest_rate.max()), "PASS" if interest_rate.max() <= 0.15 + SOLVER_TOLERANCE else "FAIL", "Expected 4% to 15%")

    for rating in CHURN_RATINGS:
        column = rating_columns[rating]
        raw_column = _raw_column_label(raw, column)
        label_cells = [_text(raw.iloc[row, column]) for row in range(min(5, len(raw)))]
        raw_values = raw.iloc[data_indices, column].reset_index(drop=True)
        churn_raw = _numeric(raw_values)
        invalid_count = int(raw_values.notna().sum() - churn_raw.notna().sum())
        missing_count = int(churn_raw.isna().sum())
        churn_rate, action, range_invalid = _normalise_rate(churn_raw)
        invalid_count += range_invalid
        in_range = bool(((churn_rate.dropna() >= -SOLVER_TOLERANCE) & (churn_rate.dropna() <= 1 + SOLVER_TOLERANCE)).all())
        complete = invalid_count == 0 and missing_count == 0 and in_range
        audit(f"rating_{rating}_raw_column", raw_column, "PASS", f"column_index={column}; labels={label_cells}")
        audit(f"rating_{rating}_normalization", action, "PASS" if complete else "FAIL", "Conversion recorded explicitly")
        audit(f"rating_{rating}_missing_count", missing_count, "PASS" if missing_count == 0 else "FAIL", "Every observed rate must have churn data")
        audit(f"rating_{rating}_invalid_count", invalid_count, "PASS" if invalid_count == 0 else "FAIL", "Churn rate numeric/range check")
        audit(f"rating_{rating}_min", float(churn_rate.min()) if not churn_rate.dropna().empty else np.nan, "PASS" if in_range else "FAIL", "Normalized churn rate must be in [0, 1]")
        audit(f"rating_{rating}_max", float(churn_rate.max()) if not churn_rate.dropna().empty else np.nan, "PASS" if in_range else "FAIL", "Normalized churn rate must be in [0, 1]")
        for index, rate_value in enumerate(interest_rate):
            normalized_rows.append({
                "source_sheet": sheet,
                "raw_interest_rate_column": rate_header,
                "raw_churn_rate_column": raw_column,
                "credit_rating": rating,
                "interest_rate_raw": float(rate_raw.iloc[index]),
                "churn_rate_raw": float(churn_raw.iloc[index]) if pd.notna(churn_raw.iloc[index]) else np.nan,
                "interest_rate": float(rate_value),
                "churn_rate": float(churn_rate.iloc[index]) if pd.notna(churn_rate.iloc[index]) else np.nan,
                "interest_rate_normalization": rate_action,
                "churn_rate_normalization": action,
            })

    normalized = pd.DataFrame(normalized_rows)
    audit_df = pd.DataFrame(audit_rows)
    audit_pass = bool((audit_df["status"] == "PASS").all())
    meta = {
        "workbook": relative(workbook),
        "workbook_sha256": sha256_file(workbook),
        "sheet": sheet,
        "rate_header": rate_header,
        "rating_columns": {rating: _raw_column_label(raw, column) for rating, column in rating_columns.items()},
        "observed_rate_count": int(interest_rate.nunique()),
        "audit_pass": audit_pass,
    }
    if not audit_pass:
        failed = audit_df.loc[audit_df["status"] != "PASS", "check_name"].tolist()
        raise ValueError(f"Attachment 3 audit failed: {failed}")
    return normalized, audit_df, meta


def fit_churn_curves(normalized: pd.DataFrame, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit independent increasing isotonic curves at observed rate points."""

    fitted_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    for rating in CHURN_RATINGS:
        group = normalized.loc[normalized["credit_rating"] == rating].sort_values("interest_rate").reset_index(drop=True)
        x = group["interest_rate"].to_numpy(dtype=float)
        y = group["churn_rate"].to_numpy(dtype=float)
        if len(x) == 0 or np.isnan(y).any():
            raise ValueError(f"Missing complete churn observations for rating {rating}")
        model = IsotonicRegression(
            increasing=bool(config["churn_model"].get("increasing", True)),
            y_min=0.0,
            y_max=1.0,
            out_of_bounds=str(config["churn_model"].get("out_of_bounds", "clip")),
        )
        fitted = model.fit_transform(x, y)
        diff = y - fitted
        raw_delta = np.diff(y)
        fitted_delta = np.diff(fitted)
        flat_intervals = int(np.sum(np.abs(fitted_delta) <= SOLVER_TOLERANCE))
        fitted_group = group[["source_sheet", "raw_interest_rate_column", "raw_churn_rate_column", "credit_rating", "interest_rate_raw", "churn_rate_raw", "interest_rate", "churn_rate", "interest_rate_normalization", "churn_rate_normalization"]].copy()
        fitted_group["raw_churn_rate"] = fitted_group.pop("churn_rate")
        fitted_group["fitted_churn_rate"] = fitted
        fitted_group["acceptance_probability"] = 1.0 - fitted
        fitted_group["raw_minus_fitted"] = diff
        fitted_group["is_observed_rate"] = True
        fitted_group["fit_method"] = "isotonic_regression_increasing"
        fitted_rows.append(fitted_group)
        metric_rows.append({
            "credit_rating": rating,
            "fit_method": "isotonic_regression_increasing",
            "observed_point_count": len(x),
            "raw_monotonic_violation_count": int(np.sum(raw_delta < -SOLVER_TOLERANCE)),
            "fitted_monotonic_violation_count": int(np.sum(fitted_delta < -SOLVER_TOLERANCE)),
            "mae": float(np.mean(np.abs(diff))),
            "rmse": float(np.sqrt(np.mean(diff ** 2))),
            "maximum_absolute_adjustment": float(np.max(np.abs(diff))),
            "mean_absolute_adjustment": float(np.mean(np.abs(diff))),
            "flat_interval_count": flat_intervals,
            "raw_min": float(np.min(y)),
            "raw_max": float(np.max(y)),
            "fitted_min": float(np.min(fitted)),
            "fitted_max": float(np.max(fitted)),
        })

    fitted_df = pd.concat(fitted_rows, ignore_index=True)
    pivot = fitted_df.pivot(index="interest_rate", columns="credit_rating", values="fitted_churn_rate").reset_index()
    raw_pivot = fitted_df.pivot(index="interest_rate", columns="credit_rating", values="raw_churn_rate").reset_index()
    cross_rows: list[dict[str, Any]] = []
    for _, row in pivot.iterrows():
        raw_row = raw_pivot.loc[np.isclose(raw_pivot["interest_rate"], row["interest_rate"])].iloc[0]
        fitted_ab = float(row["B"] - row["A"])
        fitted_bc = float(row["C"] - row["B"])
        raw_ab = float(raw_row["B"] - raw_row["A"])
        raw_bc = float(raw_row["C"] - raw_row["B"])
        fitted_order = bool(row["A"] <= row["B"] + SOLVER_TOLERANCE and row["B"] <= row["C"] + SOLVER_TOLERANCE)
        raw_order = bool(raw_row["A"] <= raw_row["B"] + SOLVER_TOLERANCE and raw_row["B"] <= raw_row["C"] + SOLVER_TOLERANCE)
        cross_rows.append({
            "interest_rate": float(row["interest_rate"]),
            "raw_A": float(raw_row["A"]),
            "raw_B": float(raw_row["B"]),
            "raw_C": float(raw_row["C"]),
            "fitted_A": float(row["A"]),
            "fitted_B": float(row["B"]),
            "fitted_C": float(row["C"]),
            "raw_A_to_B_difference": raw_ab,
            "raw_B_to_C_difference": raw_bc,
            "fitted_A_to_B_difference": fitted_ab,
            "fitted_B_to_C_difference": fitted_bc,
            "raw_order_A_le_B_le_C": raw_order,
            "fitted_order_A_le_B_le_C": fitted_order,
            "fitted_crossing_flag": not fitted_order,
        })
    cross_df = pd.DataFrame(cross_rows)
    meta = {
        "ratings": list(CHURN_RATINGS),
        "observed_rate_count": int(fitted_df["interest_rate"].nunique()),
        "fitted_values_in_unit_interval": bool(((fitted_df["fitted_churn_rate"] >= 0) & (fitted_df["fitted_churn_rate"] <= 1)).all()),
        "acceptance_values_in_unit_interval": bool(((fitted_df["acceptance_probability"] >= 0) & (fitted_df["acceptance_probability"] <= 1)).all()),
        "all_fitted_curves_monotone": bool((fitted_df.sort_values(["credit_rating", "interest_rate"]).groupby("credit_rating")["fitted_churn_rate"].diff().dropna() >= -SOLVER_TOLERANCE).all()),
        "rating_order_crossing_count": int(cross_df["fitted_crossing_flag"].sum()),
    }
    return fitted_df, pd.DataFrame(metric_rows), cross_df, meta


def churn_report(meta: Mapping[str, Any], audit: pd.DataFrame, metrics: pd.DataFrame, cross: pd.DataFrame) -> str:
    """Render the attachment-3 audit and fitting report."""

    lines = [
        "# 附件3审计与利率—客户流失率单调拟合报告",
        "",
        "本报告由程序从附件3实际数据生成。客户流失率是客户接受贷款利率后流失的情景参数，不是违约率。",
        "",
        "## 1. 数据审计",
        "",
        f"- 工作簿：`{meta['workbook']}`；工作表：`{meta['sheet']}`。",
        f"- 检测到{meta['observed_rate_count']}个实际利率点；利率和流失率统一为0—1小数。",
        "- 百分数转换仅在原始非空数值最大值超过1且不超过100时执行，并在audit CSV中逐项记录；本次转换结果不能由文档预设。",
        "- 利率集合使用附件3实际观测点，不生成区间外或额外精细利率。",
        "",
        "## 2. 拟合方法",
        "",
        "对A、B、C分别拟合独立的 `IsotonicRegression(increasing=True, y_min=0, y_max=1, out_of_bounds='clip')`。拟合值只在实际观测利率点使用；接受概率定义为 `A_g(r)=1-L_g(r)`。",
        "",
        "## 3. 拟合指标",
        "",
        "```text\n" + metrics.to_string(index=False) + "\n```",
        "",
        "## 4. 评级间顺序诊断",
        "",
        f"独立保序拟合后，在相同利率点发现{int(cross['fitted_crossing_flag'].sum())}个A级≤B级≤C级不成立的点。该顺序只作业务合理性诊断，不作为本基准模型硬约束；若存在交叉，基准仍使用独立拟合，不静默调整。",
        "",
        "## 5. 输出解释边界",
        "",
        "后续优化中的收益、损失和组合策略均为参数化情景结果，不是银行真实利润或真实损失预测。基准优化使用独立评级保序拟合曲线。",
        "",
        "## 6. 审计状态",
        "",
        "`attachment3_audit.csv`中的所有强制检查均为PASS。",
    ]
    return "\n".join(lines) + "\n"


def plot_churn_curves(fitted: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Write the two required attachment-3 figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    colors = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 6))
    for rating in CHURN_RATINGS:
        group = fitted[fitted["credit_rating"] == rating].sort_values("interest_rate")
        ax.plot(group["interest_rate"] * 100, group["raw_churn_rate"], "o--", alpha=0.55, color=colors[rating], label=f"{rating} 原始")
        ax.plot(group["interest_rate"] * 100, group["fitted_churn_rate"], "-", color=colors[rating], label=f"{rating} 保序拟合")
    ax.set_title("附件3利率—客户流失率：原始与保序拟合")
    ax.set_xlabel("贷款年利率（%）")
    ax.set_ylabel("客户流失率（0—1）")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.text(0.5, 0.01, "客户流失率不是违约率；拟合使用附件3实际利率点", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = output_dir / "churn_curve_raw_vs_fitted.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 6))
    for rating in CHURN_RATINGS:
        group = fitted[fitted["credit_rating"] == rating].sort_values("interest_rate")
        ax.plot(group["interest_rate"] * 100, group["acceptance_probability"], "o-", color=colors[rating], label=f"评级{rating}")
    ax.set_title("贷款接受概率曲线（由1−客户流失率得到）")
    ax.set_xlabel("贷款年利率（%）")
    ax.set_ylabel("接受概率（0—1）")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.01, "风险与收益为参数化情景；不代表真实客户接受概率预测", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = output_dir / "acceptance_probability_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)
    return paths


def write_churn_outputs(config: Mapping[str, Any], normalized: pd.DataFrame, audit: pd.DataFrame, meta: Mapping[str, Any]) -> dict[str, Path]:
    """Fit attachment 3 and write all churn-model artifacts."""

    fitted, metrics, cross, fit_meta = fit_churn_curves(normalized, config)
    out_dir = RUNTIME_ROOT / "churn_model"
    figure_dir = OUTPUT_ROOT / "figures" / "credit"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "audit": out_dir / "attachment3_audit.csv",
        "normalized": out_dir / "attachment3_normalized.csv",
        "audit_report": out_dir / "attachment3_audit_report.md",
        "fitted": out_dir / "churn_curve_fitted.csv",
        "metrics": out_dir / "churn_curve_metrics.csv",
        "cross": out_dir / "churn_curve_cross_rating_check.csv",
        "manifest": out_dir / "churn_model_manifest.json",
    }
    write_csv(audit, paths["audit"])
    write_csv(normalized, paths["normalized"])
    write_csv(fitted, paths["fitted"])
    write_csv(metrics, paths["metrics"])
    write_csv(cross, paths["cross"])
    write_text(churn_report({**meta, **fit_meta}, audit, metrics, cross), paths["audit_report"])
    figure_paths = plot_churn_curves(fitted, figure_dir)
    manifest = {
        "workflow": "fourth_batch_churn_model",
        "input_workbook": meta["workbook"],
        "input_workbook_sha256": meta["workbook_sha256"],
        "sheet": meta["sheet"],
        "raw_interest_rate_column": meta["rate_header"],
        "raw_rating_columns": meta["rating_columns"],
        "method": "isotonic_regression",
        "increasing": True,
        "out_of_bounds": "clip",
        "use_observed_rate_points_only": True,
        "observed_rate_count": fit_meta["observed_rate_count"],
        "audit_pass": meta["audit_pass"],
        "fitted_values_in_unit_interval": fit_meta["fitted_values_in_unit_interval"],
        "acceptance_values_in_unit_interval": fit_meta["acceptance_values_in_unit_interval"],
        "all_fitted_curves_monotone": fit_meta["all_fitted_curves_monotone"],
        "rating_order_crossing_count": fit_meta["rating_order_crossing_count"],
        "config_sha256": config_hash(config),
        "code_hashes": {
            "src/_internal/stages/06_fit_churn_curves.py": sha256_file(ROOT / "src" / "_internal" / "stages" / "06_fit_churn_curves.py"),
            "src/_internal/credit_strategy.py": sha256_file(ROOT / "src" / "_internal" / "credit_strategy.py"),
        },
        "output_hashes": {
            relative(path): sha256_file(path)
            for key, path in paths.items()
            if key != "manifest"
        } | {relative(path): sha256_file(path) for path in figure_paths},
    }
    write_json(manifest, paths["manifest"])
    return {**paths, "figure_churn": figure_paths[0], "figure_acceptance": figure_paths[1]}


def load_fitted_churn(config: Mapping[str, Any]) -> pd.DataFrame:
    path = RUNTIME_ROOT / "churn_model" / "churn_curve_fitted.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    fitted = pd.read_csv(path)
    required = {"interest_rate", "credit_rating", "fitted_churn_rate", "acceptance_probability"}
    missing = required - set(fitted.columns)
    if missing:
        raise ValueError(f"Fitted churn table missing columns: {sorted(missing)}")
    if not set(fitted["credit_rating"].dropna().unique()).issubset(set(CHURN_RATINGS)):
        raise ValueError("Fitted churn table contains non-A/B/C ratings")
    return fitted.sort_values(["credit_rating", "interest_rate"]).reset_index(drop=True)


@dataclass(frozen=True)
class MilpIndex:
    eligible_ids: tuple[str, ...]
    rates: tuple[float, ...]
    x_index: dict[tuple[str, float], int]
    s_index: dict[tuple[str, float], int]
    variable_count: int


def build_milp_index(eligible_ids: Sequence[str], rates: Sequence[float]) -> MilpIndex:
    x_index: dict[tuple[str, float], int] = {}
    s_index: dict[tuple[str, float], int] = {}
    cursor = 0
    for enterprise_id in eligible_ids:
        for rate in rates:
            key = (str(enterprise_id), float(rate))
            x_index[key] = cursor
            cursor += 1
    for enterprise_id in eligible_ids:
        for rate in rates:
            key = (str(enterprise_id), float(rate))
            s_index[key] = cursor
            cursor += 1
    return MilpIndex(tuple(str(v) for v in eligible_ids), tuple(float(v) for v in rates), x_index, s_index, cursor)


def _constraint_matrix(index: MilpIndex, min_loan: float, max_loan: float, budget: float, budget_definition: str, acceptance: Mapping[tuple[str, float], float]) -> tuple[csr_matrix, np.ndarray, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    row = 0
    for enterprise_id in index.eligible_ids:
        for rate in index.rates:
            rows.append(row)
            cols.append(index.x_index[(enterprise_id, rate)])
            data.append(1.0)
        lower.append(-np.inf)
        upper.append(1.0)
        row += 1
    for enterprise_id in index.eligible_ids:
        for rate in index.rates:
            rows.extend([row, row])
            cols.extend([index.s_index[(enterprise_id, rate)], index.x_index[(enterprise_id, rate)]])
            data.extend([1.0, -max_loan])
            lower.append(-np.inf)
            upper.append(0.0)
            row += 1
            rows.extend([row, row])
            cols.extend([index.s_index[(enterprise_id, rate)], index.x_index[(enterprise_id, rate)]])
            data.extend([-1.0, min_loan])
            lower.append(-np.inf)
            upper.append(0.0)
            row += 1
    for enterprise_id in index.eligible_ids:
        for rate in index.rates:
            rows.append(row)
            cols.append(index.s_index[(enterprise_id, rate)])
            if budget_definition == "expected_disbursement_cap":
                data.append(float(acceptance[(enterprise_id, rate)]))
            else:
                data.append(1.0)
    if budget_definition == "nominal_equality":
        lower.append(float(budget))
        upper.append(float(budget))
    else:
        lower.append(-np.inf)
        upper.append(float(budget))
    matrix = csr_matrix((data, (rows, cols)), shape=(row + 1, index.variable_count))
    return matrix, np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)


def solve_credit_milp(
    enterprise_df: pd.DataFrame,
    churn_df: pd.DataFrame,
    *,
    budget: float,
    lgd: float,
    funding_cost_rate: float,
    risk_column: str,
    budget_definition: str,
    config: Mapping[str, Any],
    scenario_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Solve one explicit credit strategy scenario using scipy.optimize.milp."""

    opt = config["credit_optimization"]
    min_loan = float(opt["min_loan_10k"])
    max_loan = float(opt["max_loan_10k"])
    if budget_definition not in {"nominal_equality", "nominal_cap", "expected_disbursement_cap"}:
        raise ValueError(f"Unknown budget definition: {budget_definition}")
    eligible = enterprise_df.loc[enterprise_df["credit_rating"].isin(CHURN_RATINGS)].copy()
    eligible["risk_value"] = pd.to_numeric(eligible[risk_column], errors="coerce")
    if eligible["risk_value"].isna().any():
        raise ValueError(f"Missing risk values for {risk_column}")
    rates = sorted(float(v) for v in churn_df["interest_rate"].unique())
    churn_lookup = churn_df.set_index(["credit_rating", "interest_rate"])
    index = build_milp_index(eligible["enterprise_id"].astype(str).tolist(), rates)
    unit_rows: list[dict[str, Any]] = []
    acceptance: dict[tuple[str, float], float] = {}
    unit_net: dict[tuple[str, float], float] = {}
    for row in eligible.itertuples(index=False):
        rating = str(row.credit_rating)
        risk = float(row.risk_value)
        for rate in rates:
            curve = churn_lookup.loc[(rating, rate)]
            fitted_churn = float(curve["fitted_churn_rate"])
            accept = float(curve["acceptance_probability"])
            unit_interest = accept * (1.0 - risk) * rate
            unit_loss = accept * risk * lgd
            unit_funding = accept * funding_cost_rate
            unit_return = unit_interest - unit_loss - unit_funding
            key = (str(row.enterprise_id), rate)
            acceptance[key] = accept
            unit_net[key] = unit_return
            unit_rows.append({
                "enterprise_id": str(row.enterprise_id),
                "credit_rating": rating,
                "interest_rate": rate,
                "risk_score": risk,
                "fitted_churn_rate": fitted_churn,
                "acceptance_probability": accept,
                "unit_expected_interest_income": unit_interest,
                "unit_expected_credit_loss": unit_loss,
                "unit_expected_funding_cost": unit_funding,
                "unit_expected_net_return": unit_return,
            })
    objective = np.zeros(index.variable_count, dtype=float)
    for key, variable in index.s_index.items():
        objective[variable] = -unit_net[key]
    integrality = np.zeros(index.variable_count, dtype=int)
    for variable in index.x_index.values():
        integrality[variable] = 1
    lower_bounds = np.zeros(index.variable_count, dtype=float)
    upper_bounds = np.full(index.variable_count, np.inf, dtype=float)
    for variable in index.x_index.values():
        upper_bounds[variable] = 1.0
    for variable in index.s_index.values():
        upper_bounds[variable] = max_loan
    matrix, lower_constraint, upper_constraint = _constraint_matrix(index, min_loan, max_loan, budget, budget_definition, acceptance)
    solver_config = config["solver"]
    start = pd.Timestamp.now(tz="UTC")
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix, lower_constraint, upper_constraint),
        options={
            "time_limit": float(solver_config["time_limit_seconds"]),
            "mip_rel_gap": float(solver_config["mip_rel_gap"]),
        },
    )
    elapsed = float((pd.Timestamp.now(tz="UTC") - start).total_seconds())
    status_code = int(result.status)
    optimal = status_code == 0
    diagnostics = {
        "scenario_id": scenario_id,
        "solver": "scipy.optimize.milp",
        "solver_status_code": status_code,
        "solver_message": str(result.message),
        "is_optimal": optimal,
        "objective_value_minimization": float(result.fun) if result.fun is not None and np.isfinite(result.fun) else np.nan,
        "objective_value_maximization": float(-result.fun) if result.fun is not None and np.isfinite(result.fun) else np.nan,
        "solve_time_seconds": elapsed,
        "mip_gap": float(solver_config["mip_rel_gap"]),
        "variable_count": index.variable_count,
        "integer_variable_count": len(index.x_index),
        "constraint_count": matrix.shape[0],
        "budget": float(budget),
        "budget_definition": budget_definition,
        "lgd": float(lgd),
        "funding_cost_rate": float(funding_cost_rate),
        "risk_column": risk_column,
    }
    if not optimal or result.x is None:
        return pd.DataFrame(), diagnostics

    values = result.x
    selected_rows: list[dict[str, Any]] = []
    for key, variable in index.s_index.items():
        enterprise_id, rate = key
        amount = float(max(0.0, values[variable]))
        x_value = float(values[index.x_index[key]])
        if amount <= SOLVER_TOLERANCE:
            continue
        unit = next(item for item in unit_rows if item["enterprise_id"] == enterprise_id and abs(item["interest_rate"] - rate) <= SOLVER_TOLERANCE)
        selected_rows.append({
            "enterprise_id": enterprise_id,
            "interest_rate": rate,
            "offered_loan_amount_10k": amount,
            "selection_variable": x_value,
            **unit,
        })
    selected = pd.DataFrame(selected_rows)
    diagnostics["selected_offer_count"] = int(len(selected))
    diagnostics["budget_used_10k"] = float(selected["offered_loan_amount_10k"].sum()) if not selected.empty else 0.0
    diagnostics["expected_disbursed_amount_10k"] = float((selected["offered_loan_amount_10k"] * selected["acceptance_probability"]).sum()) if not selected.empty else 0.0
    diagnostics["negative_unit_return_selected_count"] = int((selected["unit_expected_net_return"] < -SOLVER_TOLERANCE).sum()) if not selected.empty else 0
    return selected, diagnostics


def strategy_from_solution(
    enterprise_df: pd.DataFrame,
    selected: pd.DataFrame,
    diagnostics: Mapping[str, Any],
    *,
    risk_column: str,
    conservative_risk_column: str,
    lgd: float,
    funding_cost_rate: float,
    scenario_id: str,
    budget_definition: str,
    representative: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand sparse MILP decisions to a complete 123-enterprise strategy table."""

    selected_by_id = selected.set_index("enterprise_id") if not selected.empty else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for row in enterprise_df.itertuples(index=False):
        enterprise_id = str(row.enterprise_id)
        rating = str(row.credit_rating)
        risk = float(getattr(row, risk_column))
        is_d = rating == "D"
        chosen = (not is_d) and not selected_by_id.empty and enterprise_id in selected_by_id.index
        if is_d:
            reason = "D级企业原则上不予放贷"
        elif chosen:
            reason = "优化模型选择"
        else:
            reason = "优化模型未选择"
        if chosen:
            item = selected_by_id.loc[enterprise_id]
            amount = float(item["offered_loan_amount_10k"])
            rate = float(item["interest_rate"])
            churn = float(item["fitted_churn_rate"])
            acceptance = float(item["acceptance_probability"])
            expected_disbursed = amount * acceptance
            interest = amount * item["unit_expected_interest_income"]
            loss = amount * item["unit_expected_credit_loss"]
            funding = amount * item["unit_expected_funding_cost"]
            net = interest - loss - funding
            unit_net = float(item["unit_expected_net_return"])
        else:
            amount = 0.0
            rate = churn = acceptance = np.nan
            expected_disbursed = interest = loss = funding = net = 0.0
            unit_net = np.nan
        rows.append({
            "enterprise_id": enterprise_id,
            "enterprise_name": row.enterprise_name,
            "credit_rating": rating,
            "risk_model": row.selected_model,
            "risk_score": risk,
            "risk_score_source": risk_column,
            "risk_rank": int(row.logistic_risk_rank),
            "risk_percentile": float(row.logistic_risk_percentile),
            "loan_decision": "放贷" if chosen else "不放贷",
            "decision_reason": reason,
            "offered_loan_amount_10k": amount,
            "offered_loan_amount_yuan": amount * 10000.0,
            "interest_rate": rate,
            "interest_rate_percent": rate * 100.0 if pd.notna(rate) else np.nan,
            "fitted_churn_rate": churn,
            "acceptance_probability": acceptance,
            "expected_disbursed_amount_10k": expected_disbursed,
            "lgd": float(lgd),
            "funding_cost_rate": float(funding_cost_rate),
            "expected_interest_income_10k": interest,
            "expected_credit_loss_10k": loss,
            "expected_funding_cost_10k": funding,
            "expected_net_return_10k": net,
            "expected_net_return_per_offered_unit": unit_net,
            "solver_scenario_id": scenario_id,
            "budget_definition": budget_definition,
            "representative_scenario_flag": bool(representative),
            "conservative_risk_score": float(getattr(row, conservative_risk_column)),
        })
    strategy = pd.DataFrame(rows)
    summary = portfolio_summary(strategy, diagnostics, scenario_id=scenario_id)
    return strategy, summary


def portfolio_summary(strategy: pd.DataFrame, diagnostics: Mapping[str, Any], *, scenario_id: str) -> dict[str, Any]:
    offered = strategy["offered_loan_amount_10k"]
    selected = strategy[strategy["loan_decision"] == "放贷"]
    budget = float(diagnostics.get("budget", 0.0))
    total_offer = float(offered.sum())
    total_expected = float(strategy["expected_disbursed_amount_10k"].sum())
    total_interest = float(strategy["expected_interest_income_10k"].sum())
    total_loss = float(strategy["expected_credit_loss_10k"].sum())
    total_funding = float(strategy["expected_funding_cost_10k"].sum())
    total_net = float(strategy["expected_net_return_10k"].sum())

    def weighted(column: str, weight: pd.Series) -> float:
        if selected.empty or float(weight.sum()) <= SOLVER_TOLERANCE:
            return np.nan
        return float((selected[column] * weight.loc[selected.index]).sum() / weight.loc[selected.index].sum())

    by_rating: dict[str, Any] = {}
    for rating in ALL_RATINGS:
        group = strategy[strategy["credit_rating"] == rating]
        granted = group[group["loan_decision"] == "放贷"]
        by_rating[f"{rating}_enterprise_count"] = int(len(group))
        by_rating[f"{rating}_offered_count"] = int(len(granted))
        by_rating[f"{rating}_loan_amount_10k"] = float(granted["offered_loan_amount_10k"].sum())
    unit_returns = selected["expected_net_return_per_offered_unit"].dropna()
    result = {
        "scenario_id": scenario_id,
        "budget_10k": budget,
        "budget_definition": diagnostics.get("budget_definition"),
        "budget_used_10k": total_offer,
        "budget_unused_10k": budget - total_offer if diagnostics.get("budget_definition") != "expected_disbursement_cap" else budget - total_expected,
        "eligible_enterprise_count": int((strategy["credit_rating"].isin(CHURN_RATINGS)).sum()),
        "offered_enterprise_count": int(len(selected)),
        **by_rating,
        "expected_disbursed_amount_10k": total_expected,
        "weighted_average_interest_rate_by_offer": weighted("interest_rate", selected["offered_loan_amount_10k"]),
        "weighted_average_interest_rate_by_expected_disbursement": weighted("interest_rate", selected["expected_disbursed_amount_10k"]),
        "weighted_average_risk_by_offer": weighted("risk_score", selected["offered_loan_amount_10k"]),
        "weighted_average_risk_by_expected_disbursement": weighted("risk_score", selected["expected_disbursed_amount_10k"]),
        "expected_interest_income_10k": total_interest,
        "expected_credit_loss_10k": total_loss,
        "expected_funding_cost_10k": total_funding,
        "expected_net_return_10k": total_net,
        "minimum_unit_expected_return": float(unit_returns.min()) if not unit_returns.empty else np.nan,
        "negative_unit_return_selected_count": int((unit_returns < -SOLVER_TOLERANCE).sum()) if not unit_returns.empty else 0,
        "solver_status": diagnostics.get("solver_status_code"),
        "solver_message": diagnostics.get("solver_message"),
        "is_optimal": diagnostics.get("is_optimal"),
        "mip_gap": diagnostics.get("mip_gap"),
        "solve_time_seconds": diagnostics.get("solve_time_seconds"),
        "variable_count": diagnostics.get("variable_count"),
        "integer_variable_count": diagnostics.get("integer_variable_count"),
        "constraint_count": diagnostics.get("constraint_count"),
    }
    return result


def validate_strategy(strategy: pd.DataFrame, summary: Mapping[str, Any], fitted: pd.DataFrame, *, budget: float, budget_definition: str, min_loan: float, max_loan: float, tolerance: float) -> list[str]:
    """Return all violated deterministic strategy checks for one scenario."""

    failures: list[str] = []
    if len(strategy) != 123 or strategy["enterprise_id"].nunique() != 123:
        failures.append("strategy_not_123_unique_enterprises")
    selected = strategy[strategy["loan_decision"] == "放贷"]
    if (strategy.loc[strategy["credit_rating"] == "D", "offered_loan_amount_10k"] > tolerance).any():
        failures.append("D_enterprise_has_loan")
    if ((selected["offered_loan_amount_10k"] < min_loan - tolerance) | (selected["offered_loan_amount_10k"] > max_loan + tolerance)).any():
        failures.append("loan_amount_outside_bounds")
    if selected["enterprise_id"].duplicated().any():
        failures.append("enterprise_selected_more_than_once")
    observed_rates = set(float(v) for v in fitted["interest_rate"].unique())
    if not set(float(v) for v in selected["interest_rate"].dropna().unique()).issubset(observed_rates):
        failures.append("interest_rate_not_observed_attachment3_point")
    if ((selected["interest_rate"] < 0.04 - tolerance) | (selected["interest_rate"] > 0.15 + tolerance)).any():
        failures.append("interest_rate_outside_4_to_15_percent")
    if ((strategy["risk_score"] < -tolerance) | (strategy["risk_score"] > 1 + tolerance)).any():
        failures.append("risk_score_outside_0_1")
    if ((strategy["fitted_churn_rate"].dropna() < -tolerance) | (strategy["fitted_churn_rate"].dropna() > 1 + tolerance)).any():
        failures.append("fitted_churn_outside_0_1")
    if ((strategy["acceptance_probability"].dropna() < -tolerance) | (strategy["acceptance_probability"].dropna() > 1 + tolerance)).any():
        failures.append("acceptance_probability_outside_0_1")
    if budget_definition == "nominal_equality" and abs(float(summary["budget_used_10k"]) - budget) > tolerance:
        failures.append("nominal_equality_budget_mismatch")
    if budget_definition == "nominal_cap" and float(summary["budget_used_10k"]) > budget + tolerance:
        failures.append("nominal_cap_budget_exceeded")
    if budget_definition == "expected_disbursement_cap" and float(summary["expected_disbursed_amount_10k"]) > budget + tolerance:
        failures.append("expected_disbursement_cap_exceeded")
    recomputed_net = strategy["expected_interest_income_10k"] - strategy["expected_credit_loss_10k"] - strategy["expected_funding_cost_10k"]
    if not np.allclose(recomputed_net, strategy["expected_net_return_10k"], atol=tolerance, rtol=0):
        failures.append("return_decomposition_mismatch")
    for column in ["offered_loan_amount_10k", "offered_loan_amount_yuan", "expected_disbursed_amount_10k", "expected_interest_income_10k", "expected_credit_loss_10k", "expected_funding_cost_10k", "expected_net_return_10k"]:
        if not np.isfinite(strategy[column].to_numpy(dtype=float)).all():
            failures.append(f"nonfinite_{column}")
    return failures


def jaccard_selected(left: pd.DataFrame, right: pd.DataFrame) -> float:
    left_set = set(left.loc[left["loan_decision"] == "放贷", "enterprise_id"])
    right_set = set(right.loc[right["loan_decision"] == "放贷", "enterprise_id"])
    union = left_set | right_set
    return float(len(left_set & right_set) / len(union)) if union else 1.0


def write_json_lines_report(path: Path, title: str, paragraphs: Iterable[str]) -> None:
    write_text("\n".join([f"# {title}", "", *paragraphs, ""]) , path)
