"""Run the third-batch, feature-table-only risk-model training workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

from q1_common import ROOT, config_hash, load_config, relative, sha256_file, stable_sort, write_csv, write_json, write_text
from q1_modeling import (
    LOGISTIC_NAME,
    TREE_NAME,
    coefficient_stability,
    aggregate_oof_predictions,
    build_splits,
    fit_final_logistic,
    generate_figures,
    model_metric_summary,
    paired_bootstrap_comparison,
    run_primary_cv,
    run_rating_sensitivity,
    run_single_feature_set_cv,
    select_model,
    split_manifest,
    write_reports,
)


FEATURE_FILE = ROOT / "results" / "features" / "enterprise_features_123.csv"
MODEL_DIR = ROOT / "results" / "model_training"
FIGURE_DIR = ROOT / "figures" / "q1_model"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train third-batch question-one risk models.")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "q1.yaml")
    parser.add_argument("--feature-file", type=Path, default=FEATURE_FILE)
    return parser.parse_args()


def _git_value(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _phase_a_assert(config: dict[str, Any]) -> None:
    """Do not let phase B start unless the repaired phase A passed."""

    summary_path = ROOT / "results" / "feature_validation" / "feature_validation_summary.json"
    if not summary_path.exists():
        raise RuntimeError("phase A validation summary is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS" or int(summary.get("blocking_failure_count", 1)) != 0:
        raise RuntimeError(f"phase A validation is not PASS: {summary.get('status')}")
    feature_config = config["features"]
    primary = list(feature_config["primary_model_features"])
    sensitivity = list(feature_config["sensitivity_model_features"])
    excluded = set(feature_config["excluded_from_model"])
    if len(primary) != 15:
        raise RuntimeError(f"phase A primary feature count is {len(primary)}, expected 15")
    if "zero_amount_invoice_rate" in primary or "zero_amount_invoice_rate" in sensitivity:
        raise RuntimeError("zero_amount_invoice_rate is present in a formal feature list")
    if excluded.intersection(primary + sensitivity):
        raise RuntimeError("excluded_from_model intersects a formal feature list")


def _load_features(path: Path, config: dict[str, Any]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"accepted feature table does not exist: {path}")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "enterprise_id",
        "enterprise_name",
        "credit_rating",
        "default_label",
        *list(config["features"]["names"]),
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"accepted feature table is missing columns: {missing}")
    frame = stable_sort(frame, ["enterprise_id"])
    if len(frame) != 123 or not frame["enterprise_id"].is_unique:
        raise RuntimeError("accepted feature table must contain 123 unique enterprises")
    if frame["default_label"].isna().any() or not frame["default_label"].isin([0, 1]).all():
        raise RuntimeError("default_label is incomplete or not binary")
    if frame["credit_rating"].isna().any():
        raise RuntimeError("credit_rating is incomplete")
    primary = list(config["features"]["primary_model_features"])
    sensitivity = list(config["features"]["sensitivity_model_features"])
    excluded = set(config["features"]["excluded_from_model"])
    model_columns = primary + sensitivity
    if set(model_columns).intersection(excluded):
        raise RuntimeError("excluded feature entered model columns")
    if any(column.startswith("audit_") for column in model_columns):
        raise RuntimeError("audit_ column entered model columns")
    if any(frame[column].nunique(dropna=True) <= 1 for column in primary):
        raise RuntimeError("primary model contains a constant feature")
    return frame


def _write_environment(config: dict[str, Any], feature_hash: str, config_digest: str) -> None:
    packages = {
        "Python": sys.version.split()[0],
        "OS": platform.platform(),
        "Executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "PyYAML": yaml.__version__,
    }
    lines = [
        "Third-batch risk-model environment",
        "",
        *[f"{key}: {value}" for key, value in packages.items()],
        "",
        f"random_seed: {int(config['model_training']['random_state'])}",
        f"feature_input_sha256: {feature_hash}",
        f"config_sha256: {config_digest}",
        f"git_commit: {_git_value(['rev-parse', 'HEAD'])}",
        "git_status:",
        _git_value(["status", "--short"]),
        "",
        "Models:",
        "elastic_net_logistic: solver=saga, penalty=elasticnet, C=0.2, l1_ratio=0.8, max_iter=5000 or retry=10000",
        "hist_gradient_boosting: max_iter=100, learning_rate=0.05, max_leaf_nodes=3, min_samples_leaf=15, l2_regularization=10.0",
        "Preprocessing: train-fold median imputation, U-feature 1%/99% winsorization, configured log1p transforms, Logistic RobustScaler",
        "Resampling: no SMOTE and no class_weight=balanced in the formal models",
    ]
    write_text("\n".join(lines) + "\n", MODEL_DIR / "environment.txt")


def _feature_set_table(
    frame: pd.DataFrame,
    config: dict[str, Any],
    selected_model: str,
    splits: list[tuple[int, int, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    primary = list(config["features"]["primary_model_features"])
    sensitivity = [
        feature
        for feature in config["features"]["sensitivity_model_features"]
        if frame[feature].nunique(dropna=True) > 1
    ]
    sets: list[tuple[str, list[str]]] = [
        ("primary_model_features", primary),
        ("sensitivity_model_features_full_nonconstant", sensitivity),
        (
            "sales_scale_replaced_by_net_sales",
            [feature for feature in primary if feature != "sales_scale_10k"] + ["net_sales_10k"],
        ),
        (
            "hhi_replaced_by_max_share",
            [
                feature
                for feature in primary
                if feature not in {"customer_hhi", "supplier_hhi"}
            ]
            + ["max_customer_share", "max_supplier_share"],
        ),
        (
            "active_month_replaced_by_longest_streak",
            [feature for feature in primary if feature != "active_month_ratio"]
            + ["longest_active_streak_ratio"],
        ),
    ]
    results: list[pd.DataFrame] = []
    for feature_set_name, feature_names in sets:
        if len(feature_names) != len(set(feature_names)):
            raise RuntimeError(f"duplicate feature in sensitivity set {feature_set_name}")
        result = run_single_feature_set_cv(frame, feature_names, selected_model, splits, config)
        result["feature_set"] = feature_set_name
        results.append(result)
    return pd.concat(results, ignore_index=True)


def _add_risk_ranks(aggregate: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    result = aggregate.copy()
    short_names = {LOGISTIC_NAME: "logistic", TREE_NAME: "tree"}
    for model_name in [LOGISTIC_NAME, TREE_NAME]:
        score_column = f"{model_name}_oof_mean"
        result[f"{model_name}_risk_rank"] = (
            result[score_column].rank(method="min", ascending=False).astype(int)
        )
        result[f"{model_name}_risk_percentile"] = result[score_column].rank(
            method="average", ascending=True, pct=True
        )
        short_name = short_names[model_name]
        for suffix in ["mean", "sd", "p10", "p50", "p90"]:
            result[f"{short_name}_oof_{suffix}"] = result[f"{model_name}_oof_{suffix}"]
        result[f"{short_name}_risk_rank"] = result[f"{model_name}_risk_rank"]
        result[f"{short_name}_risk_percentile"] = result[f"{model_name}_risk_percentile"]
    result["selected_model"] = selected_model
    result["selected_model_risk_score"] = result[f"{selected_model}_oof_mean"]
    return stable_sort(result, ["enterprise_id"])


def _write_manifest(
    config: dict[str, Any],
    frame: pd.DataFrame,
    feature_hash: str,
    config_digest: str,
    selected_model: str,
    stable_metric_count: int,
    output_paths: list[Path],
) -> None:
    code_paths = [ROOT / "src" / "05_train_risk_models.py", ROOT / "src" / "q1_modeling.py"]
    code_hashes = {relative(path): sha256_file(path) for path in code_paths}
    output_hashes = {
        relative(path): sha256_file(path)
        for path in output_paths
        if path.exists() and path.name != "model_training_manifest.json"
    }
    payload = {
        "workflow": "third_batch_risk_model_training",
        "input_feature_file": relative(FEATURE_FILE),
        "input_feature_sha256": feature_hash,
        "config_sha256": config_digest,
        "code_hashes": code_hashes,
        "git_commit": _git_value(["rev-parse", "HEAD"]),
        "git_status": _git_value(["status", "--short"]),
        "python": platform.python_version(),
        "random_seed": int(config["model_training"]["random_state"]),
        "enterprise_count": int(len(frame)),
        "default_count": int(frame["default_label"].sum()),
        "primary_model_feature_count": int(len(config["features"]["primary_model_features"])),
        "primary_model_features": list(config["features"]["primary_model_features"]),
        "sensitivity_model_features": list(config["features"]["sensitivity_model_features"]),
        "excluded_from_model": config["features"]["excluded_from_model"],
        "cv": config["model_training"]["cv"],
        "bootstrap_replicates": int(config["model_training"]["bootstrap_replicates"]),
        "selected_model": selected_model,
        "tree_stable_metric_count": stable_metric_count,
        "output_hashes": output_hashes,
    }
    write_json(payload, MODEL_DIR / "model_training_manifest.json")


def run_training(config: dict[str, Any], feature_path: Path) -> dict[str, Any]:
    _phase_a_assert(config)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    frame = _load_features(feature_path.resolve(), config)
    frame.attrs["primary_features"] = list(config["features"]["primary_model_features"])
    feature_hash = sha256_file(feature_path.resolve())
    config_digest = config_hash(config)
    snapshot = {key: value for key, value in config.items() if key != "_config_path"}
    write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False),
        MODEL_DIR / "model_training_config_snapshot.yaml",
    )
    _write_environment(config, feature_hash, config_digest)

    n_splits = int(config["model_training"]["cv"]["n_splits"])
    n_repeats = int(config["model_training"]["cv"]["n_repeats"])
    seed = int(config["model_training"]["random_state"])
    splits = build_splits(
        frame["default_label"].astype(int),
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=seed,
    )
    split_frame = split_manifest(frame, splits)
    write_csv(split_frame, MODEL_DIR / "cv_split_manifest.csv")

    primary_features = list(config["features"]["primary_model_features"])
    primary_results = run_primary_cv(frame, primary_features, splits, config)
    oof = primary_results["oof_predictions_all"]
    oof["logistic_oof_probability"] = oof["logistic_oof"]
    oof["tree_oof_probability"] = oof["tree_oof"]
    write_csv(oof, MODEL_DIR / "oof_predictions_all_repeats.csv")
    write_csv(primary_results["cv_fold_metrics"], MODEL_DIR / "cv_fold_metrics.csv")
    write_csv(primary_results["preprocessing_parameters_all_folds"], MODEL_DIR / "preprocessing_parameters_all_folds.csv")
    write_csv(primary_results["convergence_report"], MODEL_DIR / "convergence_report.csv")
    write_csv(primary_results["logistic_coefficients_all_folds"], MODEL_DIR / "logistic_coefficients_all_folds.csv")

    aggregate = aggregate_oof_predictions(frame, oof)
    metric_summary = model_metric_summary(aggregate, primary_results["cv_fold_metrics"], config)
    bootstrap = paired_bootstrap_comparison(aggregate, config)
    selected_model, stable_metric_count = select_model(bootstrap)
    aggregate = _add_risk_ranks(aggregate, selected_model)
    write_csv(aggregate, MODEL_DIR / "oof_predictions_by_enterprise.csv")
    write_csv(metric_summary, MODEL_DIR / "model_metrics_summary.csv")
    write_csv(bootstrap, MODEL_DIR / "paired_bootstrap_comparison.csv")

    coefficient_summary = coefficient_stability(primary_results["logistic_coefficients_all_folds"])
    write_csv(coefficient_summary, MODEL_DIR / "logistic_coefficient_stability.csv")
    final_coefficients, final_outcome = fit_final_logistic(frame, primary_features, config)
    write_csv(final_coefficients, MODEL_DIR / "final_logistic_coefficients.csv")
    convergence_report = primary_results["convergence_report"].copy()
    convergence_report = pd.concat(
        [
            convergence_report,
            pd.DataFrame(
                [
                    {
                        "model": LOGISTIC_NAME,
                        "repeat_id": "full",
                        "fold_id": "full_123",
                        "converged": final_outcome.converged,
                        "n_iter": final_outcome.n_iter,
                        "max_iter_initial": int(config["model_training"]["logistic"]["max_iter"]),
                        "max_iter_used": final_outcome.max_iter_used,
                        "initial_convergence_warning": final_outcome.initial_convergence_warning,
                        "convergence_warning": final_outcome.convergence_warning,
                        "retry_count": final_outcome.retry_count,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    write_csv(convergence_report, MODEL_DIR / "convergence_report.csv")

    sensitivity = _feature_set_table(frame, config, selected_model, splits)
    write_csv(sensitivity, MODEL_DIR / "feature_set_sensitivity.csv")
    rating_sensitivity = run_rating_sensitivity(frame, primary_features, splits, config)
    write_csv(rating_sensitivity, MODEL_DIR / "rating_leakage_sensitivity.csv")

    model_metrics_ci = bootstrap.loc[
        :,
        [
            "metric",
            "difference_tree_minus_logistic",
            "tree_advantage_definition",
            "tree_advantage_point",
            "difference_ci_low",
            "difference_ci_high",
            "tree_advantage_ci_low",
            "tree_advantage_ci_high",
            "ci_low",
            "ci_high",
            "bootstrap_probability_tree_better",
            "bootstrap_replicates_requested",
            "bootstrap_replicates_valid",
            "bootstrap_skipped_single_class",
        ],
    ].copy()
    write_csv(model_metrics_ci, MODEL_DIR / "model_metrics_confidence_intervals.csv")

    generate_figures(
        aggregate,
        metric_summary,
        coefficient_summary,
        selected_model,
        FIGURE_DIR,
    )
    write_reports(
        frame,
        aggregate,
        metric_summary,
        bootstrap,
        coefficient_summary,
        convergence_report,
        sensitivity,
        rating_sensitivity,
        selected_model,
        stable_metric_count,
        MODEL_DIR / "model_training_report.md",
        ROOT / "docs" / "q1_model_results_for_paper.md",
    )

    output_paths = [
        MODEL_DIR / "cv_split_manifest.csv",
        MODEL_DIR / "oof_predictions_all_repeats.csv",
        MODEL_DIR / "oof_predictions_by_enterprise.csv",
        MODEL_DIR / "cv_fold_metrics.csv",
        MODEL_DIR / "model_metrics_summary.csv",
        MODEL_DIR / "model_metrics_confidence_intervals.csv",
        MODEL_DIR / "paired_bootstrap_comparison.csv",
        MODEL_DIR / "logistic_coefficients_all_folds.csv",
        MODEL_DIR / "logistic_coefficient_stability.csv",
        MODEL_DIR / "final_logistic_coefficients.csv",
        MODEL_DIR / "convergence_report.csv",
        MODEL_DIR / "feature_set_sensitivity.csv",
        MODEL_DIR / "rating_leakage_sensitivity.csv",
        MODEL_DIR / "preprocessing_parameters_all_folds.csv",
        MODEL_DIR / "model_training_config_snapshot.yaml",
        MODEL_DIR / "environment.txt",
        MODEL_DIR / "model_training_report.md",
        ROOT / "docs" / "q1_model_results_for_paper.md",
        *sorted(FIGURE_DIR.glob("*.png")),
    ]
    _write_manifest(
        config,
        frame,
        feature_hash,
        config_digest,
        selected_model,
        stable_metric_count,
        output_paths,
    )
    return {
        "selected_model": selected_model,
        "stable_metric_count": stable_metric_count,
        "enterprise_count": len(frame),
        "default_count": int(frame["default_label"].sum()),
        "oof": aggregate,
        "metric_summary": metric_summary,
        "bootstrap": bootstrap,
        "convergence": convergence_report,
    }


def main() -> int:
    args = _parse_args()
    config = load_config(args.config)
    try:
        result = run_training(config, args.feature_file)
    except Exception as exc:
        print(f"05_train_risk_models.py FAILED: {exc}")
        return 1
    print(
        "05_train_risk_models.py SUCCEEDED: "
        f"selected_model={result['selected_model']} "
        f"stable_tree_metrics={result['stable_metric_count']} "
        f"enterprises={result['enterprise_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
