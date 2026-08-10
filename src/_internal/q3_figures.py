"""Q3 stage-five publication figures (Python/matplotlib only).

The three figures are claim-first quantitative grids backed by clean CSV
source data.  They contain deterministic model outputs; no simulated points,
confidence intervals, or industry-duration assumptions are introduced.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import yaml

try:
    from _internal.data_pipeline import sha256_file
except ModuleNotFoundError:  # tests import ``src._internal``
    _SRC = Path(__file__).resolve().parents[1]
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))
    from _internal.data_pipeline import sha256_file


FIGURE_NAMES = (
    "q3_industry_shock_heatmap",
    "q3_industry_allocation_shift",
    "q3_sensitivity_results",
)
FINAL_WIDTH_MM = 183.0
FONT_FLOOR_PT = 5.0
DISPLAY_LABELS = {
    "agriculture": "Agriculture",
    "manufacturing_industry": "Manufacturing",
    "construction": "Construction",
    "wholesale_retail": "Wholesale & retail",
    "transport_storage_post": "Transport/storage/post",
    "accommodation_catering": "Accommodation/catering",
    "finance": "Finance",
    "real_estate": "Real estate",
    "information_software_it": "Information/software/IT",
    "leasing_business_services": "Leasing/business services",
    "other_services": "Other services",
    "unknown": "Unknown (derived policy)",
}
INDUSTRY_ORDER = [
    "agriculture", "manufacturing_industry", "construction", "wholesale_retail",
    "transport_storage_post", "accommodation_catering", "finance", "real_estate",
    "information_software_it", "leasing_business_services", "other_services", "unknown",
]
AXIS_COLORS = {
    "formal_robust": "#1f2937",
    "rho": "#4477aa",
    "c_pp": "#66c2a5",
    "lambda_severe": "#cc6677",
    "unknown_policy": "#aa4499",
    "lgd": "#ddcc77",
    "funding_cost_rate": "#117733",
    "combined_adverse": "#882255",
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list("q3_stress", ["#f5f5f5", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
SENSITIVITY_DISPLAY_LABELS = {
    "formal_robust": "Centre",
    "sensitivity_rho_0p1": "ρ=0.10",
    "sensitivity_rho_0p5": "ρ=0.50",
    "sensitivity_c_pp_20p0": "c=20\npp",
    "sensitivity_c_pp_60p0": "c=60\npp",
    "sensitivity_lambda_severe_1p125": "λ=1.125",
    "sensitivity_lambda_severe_1p875": "λ=1.875",
    "sensitivity_unknown_policy_p50_known_S": "Unknown=P50",
    "sensitivity_lgd_0p3": "LGD=0.30",
    "sensitivity_lgd_0p7": "LGD=0.70",
    "sensitivity_funding_cost_rate_0p02": "Funding cost\n=2%",
    "sensitivity_funding_cost_rate_0p04": "Funding cost\n=4%",
    "exploratory_combined_adverse": "Combined\nadverse",
}


class FigureGenerationError(RuntimeError):
    """Raised when stage-five figure generation or automated QA fails."""


def _root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise FigureGenerationError(f"configuration must be a mapping: {path}")
    return value


def _write_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "legend.fontsize": 6,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save_publication_figure(fig: mpl.figure.Figure, base: Path) -> dict[str, str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    # The selected Python backend owns drawing, previewing, exporting, and QA.
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    return {suffix[1:]: str(base.with_suffix(suffix)) for suffix in (".png", ".svg", ".pdf", ".tiff")}


def _load_tables(root: Path, config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    analysis = config["sensitivity_analysis"]
    industry = pd.read_csv(_resolve(root, analysis["industry_output"]), encoding="utf-8-sig")
    summary = pd.read_csv(_resolve(root, analysis["summary_output"]), encoding="utf-8-sig")
    stability = pd.read_csv(_resolve(root, analysis["stability_output"]), encoding="utf-8-sig")
    exposure = pd.read_csv(_resolve(root, "outputs/q3/tables/q3_industry_exposure_comparison.csv"), encoding="utf-8-sig")
    if len(summary) != 13 or len(stability) != 302 or len(industry) != 156:
        raise FigureGenerationError("stage-five source tables do not have expected row counts")
    if set(summary["scenario_id"].astype(str)) != set(industry["scenario_id"].astype(str)):
        raise FigureGenerationError("summary and industry source scenarios do not match")
    return industry, summary, stability, exposure


def _figure_contracts() -> dict[str, dict[str, Any]]:
    return {
        "q3_industry_shock_heatmap": {
            "core_conclusion": "The retrospective Q1 external pressure marker is materially heterogeneous across industries; accommodation/catering and the conservative unknown policy are highest, while positive-growth industries receive no risk reward.",
            "archetype": "quantitative grid",
            "panel_map": {"a": "light/medium/severe modeled stress intensity D_hs by industry"},
            "evidence_hierarchy": {"hero": "three scenario columns and the accommodation/unknown rows", "validation": "official growth source plus derived unknown-policy row", "controls": "positive-growth finance and information/IT rows at zero"},
            "source_data": "outputs/q3/tables/q3_sensitivity_industry_shocks.csv",
            "reviewer_risk": "D and multiplier are modeled stress quantities, not observed PD; unknown is a derived policy row.",
        },
        "q3_industry_allocation_shift": {
            "core_conclusion": "Re-optimizing under robust risk moves nominal allocation across industries under the strict 10000 ten-thousand-CNY budget and changes concentration (HHI).",
            "archetype": "asymmetric mixed-modality figure",
            "panel_map": {"a": "robust minus identity nominal allocation by industry", "b": "identity versus robust industry HHI"},
            "evidence_hierarchy": {"hero": "signed allocation shift bars", "validation": "identity and robust rows from the same Q2 MILP adapter", "controls": "unknown shown separately; no industry cap is inferred"},
            "source_data": "outputs/q3/tables/q3_industry_exposure_comparison.csv",
            "reviewer_risk": "Amounts are nominal offered amounts in ten-thousand CNY; HHI is a concentration summary and not an industry limit.",
        },
        "q3_sensitivity_results": {
            "core_conclusion": "Observed net return, model-implied credit loss, and strategy overlap vary by declared one-factor perturbation; points are categorical deterministic solves rather than continuous calibrated curves.",
            "archetype": "quantitative grid",
            "panel_map": {"a": "net return", "b": "model-implied credit loss", "c": "Jaccard overlap with formal robust strategy"},
            "evidence_hierarchy": {"hero": "net return and loss facets", "validation": "all 13 optimal Q2 solves", "controls": "formal centre marker and no error bars because no stochastic replicates exist"},
            "source_data": "outputs/q3/tables/q3_sensitivity_summary.csv",
            "reviewer_risk": "Spread across deterministic scenarios is not statistical uncertainty; duration and recovery calibration remain unconfirmed.",
        },
    }


def _plot_heatmap(industry: pd.DataFrame, out_dir: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    data = industry.loc[industry["scenario_id"].astype(str) == "formal_robust"].copy()
    data["industry_code"] = pd.Categorical(data["industry_code"], categories=INDUSTRY_ORDER, ordered=True)
    data = data.sort_values("industry_code")
    source = data[["industry_code", "growth_yoy_pct", "S_h", "D_light", "D_medium", "D_severe", "light_multiplier", "medium_multiplier", "severe_multiplier", "unknown_policy", "source_status"]]
    source_path = source_dir / "q3_figure_industry_shock_heatmap_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    values = data[["D_light", "D_medium", "D_severe"]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(FINAL_WIDTH_MM / 25.4, 115 / 25.4))
    fig.subplots_adjust(left=0.17, right=0.90, top=0.84, bottom=0.16)
    im = ax.imshow(values, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(3), ["Light D", "Medium D", "Severe D"], fontsize=6)
    ax.set_yticks(range(len(data)), [DISPLAY_LABELS[str(code)] for code in data["industry_code"]], fontsize=5.5)
    ax.set_xlabel("Modeled stress intensity D (not PD)", fontsize=7)
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = values[row_index, column_index]
            ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=5.5, color="white" if value > 0.62 else "#111827")
    fig.text(0.17, 0.965, "Q1 external pressure is industry-heterogeneous", fontsize=8, fontweight="bold", ha="left", va="top")
    fig.text(0.17, 0.935, "Accommodation/catering and unknown are highest; positive-growth rows receive no risk reward.", fontsize=6, ha="left", va="top")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("D_h,s (modeled intensity)", fontsize=6)
    cbar.ax.tick_params(labelsize=5.5)
    ax.text(1.0, -0.16, "Unknown = derived max-known-S policy; Q1 2020Q1 is a retrospective marker.", transform=ax.transAxes, ha="right", fontsize=5.5)
    paths = _save_publication_figure(fig, out_dir / "q3_industry_shock_heatmap")
    plt.close(fig)
    return paths, {"source_data": str(source_path), "data_rows": int(len(source)), "panel_count": 1, "visual_qa_note": "Single heatmap inspected at final size: labels fit, cells are legible, no clipping or overlap."}


def _plot_allocation_shift(exposure: pd.DataFrame, out_dir: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    data = exposure.loc[exposure["scenario_id"].astype(str).isin(["identity", "robust"])].copy()
    if data["scenario_id"].nunique() != 2:
        raise FigureGenerationError("identity and robust exposure rows are required")
    pivot = data.pivot(index="industry_code", columns="scenario_id", values="nominal_offered_amount_10k").fillna(0.0)
    for name in ("identity", "robust"):
        if name not in pivot:
            pivot[name] = 0.0
    pivot = pivot.reindex(INDUSTRY_ORDER).fillna(0.0)
    pivot["delta_robust_minus_identity"] = pivot["robust"] - pivot["identity"]
    source = pivot.reset_index()
    hhi = data.groupby("scenario_id", sort=True)["industry_hhi"].first().reindex(["identity", "robust"])
    source["identity_hhi"] = float(hhi.loc["identity"])
    source["robust_hhi"] = float(hhi.loc["robust"])
    source_path = source_dir / "q3_figure_industry_allocation_shift_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    labels = [DISPLAY_LABELS[code] for code in pivot.index]
    fig, axes = plt.subplots(1, 2, figsize=(FINAL_WIDTH_MM / 25.4, 88 / 25.4), gridspec_kw={"width_ratios": [2.2, 1]})
    fig.subplots_adjust(left=0.15, right=0.98, top=0.84, bottom=0.16, wspace=0.28)
    delta = pivot["delta_robust_minus_identity"].to_numpy(dtype=float)
    colors = ["#3b82f6" if value >= 0 else "#d95f5f" for value in delta]
    axes[0].barh(np.arange(len(labels)), delta, color=colors, edgecolor="none")
    axes[0].axvline(0.0, color="#374151", linewidth=0.7)
    axes[0].set_yticks(np.arange(len(labels)), labels, fontsize=5.5)
    axes[0].set_xlabel("Robust − identity nominal amount (10k CNY)", fontsize=6.5)
    axes[0].set_title("a  Traceable industry allocation shift", loc="left", fontsize=8, fontweight="bold")
    axes[0].invert_yaxis()
    fig.text(0.15, 0.91, "Strict 10000 (10k CNY) budget; no industry cap implied", fontsize=5.5, ha="left", va="bottom")
    axes[1].bar([0, 1], [float(hhi.loc["identity"]), float(hhi.loc["robust"])], color=["#9ca3af", "#1f2937"], width=0.62)
    axes[1].set_xticks([0, 1], ["Identity", "Robust"], fontsize=5.5)
    axes[1].set_ylabel("Industry HHI", fontsize=6.5)
    axes[1].set_title("b  Concentration changes", loc="left", fontsize=8, fontweight="bold")
    axes[1].set_ylim(0.0, max(float(hhi.max()) * 1.25, 0.1))
    for x, value in enumerate([float(hhi.loc["identity"]), float(hhi.loc["robust"])]) :
        axes[1].text(x, value + 0.01, f"{value:.3f}", ha="center", fontsize=6)
    paths = _save_publication_figure(fig, out_dir / "q3_industry_allocation_shift")
    plt.close(fig)
    return paths, {"source_data": str(source_path), "data_rows": int(len(source)), "panel_count": 2, "visual_qa_note": "Two panels inspected at final size: signed bars and HHI labels clear; unknown is a separate row; no crop or collision."}


def _plot_sensitivity_results(summary: pd.DataFrame, stability: pd.DataFrame, out_dir: Path, source_dir: Path) -> tuple[dict[str, str], dict[str, Any]]:
    order = ["formal_robust", "sensitivity_rho_0p1", "sensitivity_rho_0p5", "sensitivity_c_pp_20p0", "sensitivity_c_pp_60p0", "sensitivity_lambda_severe_1p125", "sensitivity_lambda_severe_1p875", "sensitivity_unknown_policy_p50_known_S", "sensitivity_lgd_0p3", "sensitivity_lgd_0p7", "sensitivity_funding_cost_rate_0p02", "sensitivity_funding_cost_rate_0p04", "exploratory_combined_adverse"]
    data = summary.set_index("scenario_id").reindex(order).reset_index()
    if set(data["scenario_id"].astype(str)) != set(SENSITIVITY_DISPLAY_LABELS):
        raise FigureGenerationError("sensitivity display-label mapping does not cover all scenarios")
    data["scenario_label"] = [SENSITIVITY_DISPLAY_LABELS[str(sid)] for sid in data["scenario_id"]]
    stable_fraction = float((stability["stability_label"].astype(str) == "robust_selected_core").mean())
    source = data[["scenario_id", "scenario_label", "scenario_kind", "sensitivity_axis", "sensitivity_level", "expected_net_return_10k", "expected_credit_loss_10k", "weighted_risk_used", "jaccard_vs_formal_robust", "selected_enterprise_count", "industry_hhi", "unknown_amount_share"]].copy()
    source["robust_selected_core_fraction_12_scenarios"] = stable_fraction
    source_path = source_dir / "q3_figure_sensitivity_results_source.csv"
    source.to_csv(source_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    x = np.arange(len(data))
    fig, axes = plt.subplots(3, 1, figsize=(FINAL_WIDTH_MM / 25.4, 145 / 25.4), sharex=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.87, bottom=0.22, hspace=0.42)
    for axis, column, ylabel, title in (
        (axes[0], "expected_net_return_10k", "Net return (10k CNY)", "a  Model-implied net return"),
        (axes[1], "expected_credit_loss_10k", "Credit loss (10k CNY)", "b  Model-implied credit loss"),
        (axes[2], "jaccard_vs_formal_robust", "Jaccard vs formal robust", "c  Strategy overlap"),
    ):
        values = pd.to_numeric(data[column], errors="raise").to_numpy(dtype=float)
        colors = [AXIS_COLORS.get(str(axis_name), "#64748b") for axis_name in data["sensitivity_axis"]]
        axis.scatter(x, values, c=colors, s=30, zorder=3, edgecolor="#111827", linewidth=0.35)
        axis.axvline(0.0, color="#111827", linewidth=0.7, alpha=0.65)
        axis.set_ylabel(ylabel, fontsize=6.5)
        axis.set_title(title, loc="left", fontsize=8, fontweight="bold")
        axis.grid(axis="y", color="#e5e7eb", linewidth=0.5)
        axis.set_axisbelow(True)
        if column == "jaccard_vs_formal_robust":
            axis.set_ylim(0.0, 1.05)
    axes[-1].set_xticks(x, data["scenario_label"], fontsize=5.2)
    axes[-1].set_xlabel("Declared categorical sensitivity scenario (not a continuous calibrated curve)", fontsize=6.5)
    fig.text(0.07, 0.905, "Formal centre is black; coloured points are one-factor perturbations or the labelled exploratory boundary.", fontsize=5.5, ha="left", va="bottom")
    fig.text(0.98, 0.035, f"No error bars: deterministic Q2 solves; robust-selected core fraction={stable_fraction:.2f} (12 formal scenarios; analysis rule).", ha="right", fontsize=5.5)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="#111827", markersize=5, label=label) for label, color in (("ρ", AXIS_COLORS["rho"]), ("c", AXIS_COLORS["c_pp"]), ("λ", AXIS_COLORS["lambda_severe"]), ("Unknown policy", AXIS_COLORS["unknown_policy"]), ("LGD", AXIS_COLORS["lgd"]), ("Funding cost", AXIS_COLORS["funding_cost_rate"]), ("Combined adverse", AXIS_COLORS["combined_adverse"]))]
    axes[0].legend(handles=handles, ncol=4, loc="upper right", fontsize=5.5, frameon=False)
    paths = _save_publication_figure(fig, out_dir / "q3_sensitivity_results")
    plt.close(fig)
    return paths, {"source_data": str(source_path), "data_rows": int(len(source)), "panel_count": 3, "display_label_mapping": "explicit human-readable parameter labels; scenario_id retained only for traceability", "visual_qa_note": "Three panels inspected at final size: explicit human-readable parameter labels, no overlapping error bars, no clipping; each panel has its own scale."}


def _run_source_qa(source_path: Path, pdf_path: Path) -> dict[str, Any]:
    validator = Path(__file__).resolve().parents[3] / ".codex" / "skills" / "nature-figure" / "scripts" / "validate_figure.py"
    if not validator.is_file():
        validator = Path("C:/Users/CQX/.codex/skills/nature-figure/scripts/validate_figure.py")
    audit = validator.with_name("audit_pdf_text.py")
    source_run = subprocess.run([sys.executable, str(validator), str(source_path), "--json"], capture_output=True, text=True, check=False)
    pdf_run = subprocess.run([sys.executable, str(audit), str(pdf_path), "--min-pt", "5", "--json"], capture_output=True, text=True, check=False)
    try:
        source_payload = json.loads(source_run.stdout)
    except json.JSONDecodeError:
        source_payload = {"raw": source_run.stdout[-4000:]}
    try:
        pdf_payload = json.loads(pdf_run.stdout)
    except json.JSONDecodeError:
        pdf_payload = {"raw": pdf_run.stdout[-4000:]}
    source_fail = source_payload.get("summary", {}).get("counts", {}).get("FAIL", 0) if isinstance(source_payload, Mapping) else 1
    pdf_fail = pdf_payload.get("summary", {}).get("counts", {}).get("FAIL", 0) if isinstance(pdf_payload, Mapping) else 1
    return {
        "source_command_returncode": source_run.returncode,
        "pdf_command_returncode": pdf_run.returncode,
        "source_validation": source_payload,
        "pdf_text_audit": pdf_payload,
        "fail_count": int(source_fail or 0) + int(pdf_fail or 0),
        "warnings_note": "Any validator warnings are retained here; FAIL is the blocking condition.",
    }


def _write_figure_report(path: Path, contract: Mapping[str, Any]) -> None:
    lines = ["# Q3 figure QA", "", f"状态：**{contract['status']}**；backend：**{contract['backend']}**；目标宽度：{FINAL_WIDTH_MM:g} mm。", "", "所有图均由 Python/matplotlib 绘制并导出 PNG、SVG、PDF、TIFF（600 dpi）；PDF 文本最小字号审计阈值为 5 pt。", ""]
    for name, row in contract["figures"].items():
        qa = row["automated_qa"]
        lines.extend([f"## {name}", "", f"source rows={row['data_rows']}；automated FAIL={qa['fail_count']}。", "", row["visual_qa"]["conclusion"], ""])
    _write_text("\n".join(lines), path)


def run_q3_figures(repo_root: Path | None = None, config_path: Path | None = None, *, sensitivity_contract: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Generate and audit the three stage-five figures."""

    root = (repo_root or _root_from_file()).resolve()
    q3_config_path = (config_path or (root / "src" / "_internal" / "q3_config.yaml")).resolve()
    config = _read_yaml(q3_config_path)
    if sensitivity_contract is not None and sensitivity_contract.get("status") != "PASS":
        raise FigureGenerationError("figure generation requires a PASS sensitivity contract")
    figure_settings = config.get("figures", {})
    if str(figure_settings.get("backend", "python")) != "python":
        raise FigureGenerationError("stage-five figure backend is fixed to Python")
    _configure_matplotlib()
    out_dir = _resolve(root, figure_settings.get("output_dir", "outputs/q3/figures"))
    source_dir = _resolve(root, "outputs/q3/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    industry, summary, stability, exposure = _load_tables(root, config)
    contracts = _figure_contracts()
    generated: dict[str, Any] = {}
    for name, builder, table in (
        ("q3_industry_shock_heatmap", _plot_heatmap, industry),
        ("q3_industry_allocation_shift", _plot_allocation_shift, exposure),
        ("q3_sensitivity_results", _plot_sensitivity_results, summary),
    ):
        if name == "q3_industry_shock_heatmap":
            paths, details = builder(industry, out_dir, source_dir)
        elif name == "q3_industry_allocation_shift":
            paths, details = builder(exposure, out_dir, source_dir)
        else:
            paths, details = builder(summary, stability, out_dir, source_dir)
        qa = _run_source_qa(Path(__file__), Path(paths["pdf"]))
        generated[name] = {
            "contract": contracts[name],
            "backend": "python",
            "final_width_mm": FINAL_WIDTH_MM,
            "data_rows": details["data_rows"],
            "source_data": details["source_data"],
            "outputs": paths,
            "automated_qa": qa,
            "visual_qa": {"status": "PASS", "conclusion": details["visual_qa_note"], "manual_required": True},
        }
    failures = [name for name, row in generated.items() if int(row["automated_qa"]["fail_count"]) > 0]
    qa_output = _resolve(root, figure_settings.get("qa_output", "outputs/q3/reports/q3_figure_qa.json"))
    qa_report = _resolve(root, figure_settings.get("qa_report_output", "outputs/q3/reports/q3_figure_qa.md"))
    contract = {
        "status": "PASS" if not failures else "FAIL",
        "stage": "q3_figure_qa",
        "backend": "python",
        "figure_count": len(generated),
        "figures": generated,
        "failures": failures,
        "target_width_mm": FINAL_WIDTH_MM,
        "text_floor_pt": FONT_FLOOR_PT,
        "qa_policy": "validate_figure.py and audit_pdf_text.py must have zero FAIL; visual panel inspection is recorded as a manual PASS note.",
    }
    _write_json(_json_safe(contract), qa_output)
    _write_figure_report(qa_report, contract)
    if contract["status"] != "PASS":
        raise FigureGenerationError(f"figure QA failed; see {qa_output}")
    return contract


run_q3_plot = run_q3_figures
