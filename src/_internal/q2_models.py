"""Question-two risk, rating and semi-supervised cross-check stage.

The risk model in this module is deliberately narrow: it reuses the locked
question-one elastic-net Logistic Pipeline and the 15 common invoice
features.  The ordinal rating model is a proportional-odds cumulative
Logistic model implemented with scipy, which is already a declared project
dependency.  Label Spreading is run only as a label-masked diagnostic and is
never averaged into the deployed risk score or rating probabilities.
"""

from __future__ import annotations

import json
import math
import pickle
import platform
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from matplotlib import font_manager
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.semi_supervised import LabelSpreading

from _internal import risk_model
from _internal.data_pipeline import (
    ROOT,
    config_hash,
    load_config,
    relative,
    sha256_file,
    stable_sort,
    write_csv,
    write_json,
    write_text,
)


RISK_MODEL_NAME = risk_model.LOGISTIC_NAME
RISK_METRICS = [
    "pr_auc",
    "brier",
    "log_loss",
    "roc_auc",
    "calibration_error_5bin",
    "balanced_accuracy",
    "f1",
    "top20_recall",
    "top20_precision",
]
RATING_ORDER = ["A", "B", "C", "D"]
RATING_TO_CODE = {value: index for index, value in enumerate(RATING_ORDER)}
RATING_METRICS = [
    "macro_f1",
    "balanced_accuracy",
    "ordered_grade_error",
    "within_one_grade_rate",
    "weighted_kappa",
    "multiclass_brier",
    "multiclass_log_loss",
    "probability_ece",
    "expected_absolute_grade_error",
]


def _model_paths() -> dict[str, Path]:
    paths = {
        "runtime": ROOT / "data" / "processed" / "_runtime" / "q2" / "model",
        "tables": ROOT / "outputs" / "q2" / "tables",
        "figures": ROOT / "outputs" / "q2" / "figures" / "model",
        "reports": ROOT / "outputs" / "q2" / "reports",
        "scores": ROOT / "data" / "processed" / "q2_risk_rating_scores.csv",
        "manifest": ROOT / "outputs" / "q2" / "reports" / "q2_risk_rating_run_manifest.json",
    }
    for key in ["runtime", "tables", "figures", "reports"]:
        paths[key].mkdir(parents=True, exist_ok=True)
    paths["scores"].parent.mkdir(parents=True, exist_ok=True)
    return paths


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
            if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
                values.append("")
            else:
                values.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(values) + " |")
    if len(frame) > max_rows:
        lines.append("")
        lines.append(f"（表格仅展示前{max_rows}行，完整结果见CSV。）")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype("string").str.strip().str.lower().isin(["true", "1", "yes", "y"])


def _make_risk_factory(feature_names: Sequence[str], q1_config: Mapping[str, Any]):
    names = list(feature_names)

    def factory(max_iter: int | None) -> Pipeline:
        return Pipeline(
            [
                (
                    "preprocess",
                    risk_model._make_fold_preprocessor(names, RISK_MODEL_NAME, q1_config),
                ),
                (
                    "model",
                    risk_model._make_classifier(
                        RISK_MODEL_NAME,
                        q1_config,
                        max_iter_override=max_iter,
                    ),
                ),
            ]
        )

    return factory


class OrderedLogistic(ClassifierMixin, BaseEstimator):
    """A four-level proportional-odds cumulative Logistic estimator.

    For ordered labels 0 < 1 < 2 < 3, the model is

        P(Y <= h | x) = sigmoid(theta_h - x @ beta), h=0,1,2.

    Threshold gaps are represented by softplus parameters, so the fitted
    cumulative probabilities are ordered by construction.  Only the slope
    vector receives the small, explicitly configured L2 penalty; the model is
    still an ordered Logistic model rather than a one-vs-rest or multinomial
    Logistic model.
    """

    def __init__(
        self,
        l2_penalty: float = 0.1,
        max_iter: int = 2000,
        tolerance: float = 1e-8,
    ) -> None:
        self.l2_penalty = float(l2_penalty)
        self.max_iter = int(max_iter)
        self.tolerance = float(tolerance)

    @staticmethod
    def _thresholds(params: np.ndarray, n_features: int) -> np.ndarray:
        first = float(params[n_features])
        raw_gaps = np.asarray(params[n_features + 1 : n_features + 3], dtype=float)
        gaps = np.logaddexp(0.0, raw_gaps)
        return np.asarray([first, first + gaps[0], first + gaps[0] + gaps[1]], dtype=float)

    @staticmethod
    def _probabilities_from_params(
        params: np.ndarray,
        X: np.ndarray,
    ) -> np.ndarray:
        n_features = X.shape[1]
        beta = np.asarray(params[:n_features], dtype=float)
        thresholds = OrderedLogistic._thresholds(params, n_features)
        eta = X @ beta
        cumulative = expit(thresholds[None, :] - eta[:, None])
        probabilities = np.column_stack(
            [
                cumulative[:, 0],
                cumulative[:, 1] - cumulative[:, 0],
                cumulative[:, 2] - cumulative[:, 1],
                1.0 - cumulative[:, 2],
            ]
        )
        probabilities = np.clip(probabilities, 1e-15, 1.0)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def _objective(self, params: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        probabilities = self._probabilities_from_params(params, X)
        selected = probabilities[np.arange(len(y)), y]
        loss = -float(np.mean(np.log(np.clip(selected, 1e-15, 1.0))))
        beta = params[: X.shape[1]]
        loss += 0.5 * self.l2_penalty * float(np.mean(beta * beta))
        return loss if np.isfinite(loss) else 1e100

    @staticmethod
    def _inverse_softplus(values: np.ndarray) -> np.ndarray:
        values = np.maximum(np.asarray(values, dtype=float), 1e-4)
        result = np.empty_like(values)
        large = values > 20.0
        result[large] = values[large]
        result[~large] = np.log(np.expm1(values[~large]))
        return result

    def fit(self, X: np.ndarray | pd.DataFrame, y: Sequence[int]) -> "OrderedLogistic":
        matrix = np.asarray(X, dtype=float)
        labels = np.asarray(y, dtype=int)
        if matrix.ndim != 2 or len(labels) != len(matrix):
            raise ValueError("ordered Logistic input shape is invalid")
        if set(np.unique(labels)) != set(range(4)):
            raise ValueError("ordered Logistic requires all four labels A/B/C/D in the training fold")
        if not np.isfinite(matrix).all():
            raise ValueError("ordered Logistic input contains non-finite values")
        self.n_features_in_ = int(matrix.shape[1])
        self.classes_ = np.arange(4, dtype=int)
        counts = np.bincount(labels, minlength=4).astype(float)
        cumulative = np.cumsum(counts)[:3] / float(len(labels))
        cumulative = np.clip(cumulative, 1e-5, 1.0 - 1e-5)
        initial_thresholds = np.log(cumulative / (1.0 - cumulative))
        initial_params = np.concatenate(
            [
                np.zeros(self.n_features_in_, dtype=float),
                np.asarray([initial_thresholds[0]], dtype=float),
                self._inverse_softplus(np.diff(initial_thresholds)),
            ]
        )
        result = minimize(
            self._objective,
            initial_params,
            args=(matrix, labels),
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tolerance, "gtol": self.tolerance},
        )
        self.optimization_success_ = bool(result.success)
        self.optimization_message_ = str(result.message)
        self.n_iter_ = int(getattr(result, "nit", 0) or 0)
        self.objective_value_ = float(result.fun)
        self.params_ = np.asarray(result.x, dtype=float)
        self.coef_ = self.params_[: self.n_features_in_].reshape(1, -1)
        self.thresholds_ = self._thresholds(self.params_, self.n_features_in_)
        if not np.all(np.diff(self.thresholds_) > 0):
            raise RuntimeError("ordered Logistic thresholds are not strictly ordered")
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "params_"):
            raise RuntimeError("ordered Logistic estimator is not fitted")
        matrix = np.asarray(X, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
            raise ValueError("ordered Logistic prediction shape is invalid")
        probabilities = self._probabilities_from_params(self.params_, matrix)
        probabilities = np.clip(probabilities, 0.0, 1.0)
        return probabilities / probabilities.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1).astype(int)


def _binary_metric_row(
    model: str,
    scope: str,
    y: Sequence[int],
    probabilities: Sequence[float],
    q1_config: Mapping[str, Any],
    *,
    fold_count: int | None = None,
) -> dict[str, Any]:
    values = risk_model.compute_metrics(
        y,
        probabilities,
        top_risk_fraction=float(q1_config["model_training"]["top_risk_fraction"]),
        calibration_bins=int(q1_config["model_training"]["calibration_bins"]),
    )
    row: dict[str, Any] = {
        "model": model,
        "scope": scope,
        "n_observations": int(values.pop("n")),
        "n_default": int(values.pop("n_default")),
    }
    for metric in RISK_METRICS:
        row[metric] = float(values[metric])
    if fold_count is not None:
        row["fold_count"] = int(fold_count)
    return row


def _aggregate_binary_predictions(
    predictions: pd.DataFrame,
    base: pd.DataFrame,
    prediction_column: str,
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for enterprise_id, group in predictions.groupby("enterprise_id", sort=True):
        values = group[prediction_column].to_numpy(dtype=float)
        rows.append(
            {
                "enterprise_id": enterprise_id,
                f"{prefix}_mean": float(np.mean(values)),
                f"{prefix}_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                f"{prefix}_p05": float(np.quantile(values, 0.05)),
                f"{prefix}_p10": float(np.quantile(values, 0.10)),
                f"{prefix}_p50": float(np.quantile(values, 0.50)),
                f"{prefix}_p90": float(np.quantile(values, 0.90)),
                f"{prefix}_p95": float(np.quantile(values, 0.95)),
                f"{prefix}_min": float(np.min(values)),
                f"{prefix}_max": float(np.max(values)),
                f"{prefix}_test_count": int(len(values)),
            }
        )
    result = pd.DataFrame(rows)
    result = base[["enterprise_id", "enterprise_name"]].merge(
        result,
        on="enterprise_id",
        how="left",
        validate="one_to_one",
    )
    return stable_sort(result, ["enterprise_id"])


def _label_spreading_predictions(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    X_target: pd.DataFrame,
    feature_names: Sequence[str],
    q1_config: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit Label Spreading while passing -1 for every hidden/test label."""

    preprocessor = risk_model._make_fold_preprocessor(
        list(feature_names), RISK_MODEL_NAME, q1_config
    )
    preprocessor.fit(X_train, y_train)
    train_transformed = preprocessor.transform(X_train)
    test_transformed = preprocessor.transform(X_test)
    target_transformed = preprocessor.transform(X_target)
    matrix = np.vstack([train_transformed, test_transformed, target_transformed])
    hidden_labels = np.concatenate(
        [
            y_train.to_numpy(dtype=int),
            np.full(len(X_test), -1, dtype=int),
            np.full(len(X_target), -1, dtype=int),
        ]
    )
    parameters = q2_config["modeling"]["label_spreading"]
    estimator = LabelSpreading(
        kernel=str(parameters["kernel"]),
        n_neighbors=int(parameters["n_neighbors"]),
        alpha=float(parameters["alpha"]),
        max_iter=int(parameters["max_iter"]),
        tol=float(parameters["tolerance"]),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(matrix, hidden_labels)
    classes = np.asarray(estimator.classes_)
    class_one = int(np.where(classes == 1)[0][0])
    distributions = np.asarray(estimator.label_distributions_, dtype=float)
    test_start = len(X_train)
    target_start = len(X_train) + len(X_test)
    test_probability = np.clip(distributions[test_start:target_start, class_one], 0.0, 1.0)
    target_probability = np.clip(distributions[target_start:, class_one], 0.0, 1.0)
    audit = {
        "n_labeled_train_nodes": int(len(X_train)),
        "n_hidden_test_nodes": int(len(X_test)),
        "n_unlabeled_target_nodes": int(len(X_target)),
        "test_labels_passed_to_label_spreading": False,
        "test_label_array_all_minus_one": bool(np.all(hidden_labels[test_start:target_start] == -1)),
        "target_label_array_all_minus_one": bool(np.all(hidden_labels[target_start:] == -1)),
        "warnings": int(len(caught)),
        "classes": classes.astype(int).tolist(),
        "n_iter": int(getattr(estimator, "n_iter_", 0) or 0),
    }
    if not audit["test_label_array_all_minus_one"] or not audit["target_label_array_all_minus_one"]:
        raise AssertionError("Label Spreading label masking failed")
    return test_probability, target_probability, audit


def _run_risk_and_label_spreading(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_names: Sequence[str],
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    q1_config: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    X_reference = reference.loc[:, list(feature_names)]
    X_target = target.loc[:, list(feature_names)]
    y = reference["default_label"].astype(int)
    risk_oof_rows: list[dict[str, Any]] = []
    risk_target_rows: list[dict[str, Any]] = []
    risk_fold_rows: list[dict[str, Any]] = []
    risk_fit_rows: list[dict[str, Any]] = []
    ls_oof_rows: list[dict[str, Any]] = []
    ls_target_rows: list[dict[str, Any]] = []
    ls_fold_rows: list[dict[str, Any]] = []
    ls_audit_rows: list[dict[str, Any]] = []

    for repeat_id, fold_id, train_idx, test_idx in splits:
        X_train = X_reference.iloc[train_idx]
        X_test = X_reference.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        outcome = risk_model.fit_with_retry(
            _make_risk_factory(feature_names, q1_config),
            RISK_MODEL_NAME,
            X_train,
            y_train,
            q1_config,
        )
        risk_test = risk_model.prediction(outcome.estimator, X_test)
        risk_target = risk_model.prediction(outcome.estimator, X_target)
        risk_fold_rows.append(
            {
                "model": RISK_MODEL_NAME,
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_default_test": int(y_test.sum()),
                **{
                    key: value
                    for key, value in _binary_metric_row(
                        RISK_MODEL_NAME,
                        "cv_fold",
                        y_test,
                        risk_test,
                        q1_config,
                    ).items()
                    if key not in {"model", "scope", "n_observations", "n_default"}
                },
            }
        )
        risk_fit_rows.append(
            {
                "model": RISK_MODEL_NAME,
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "converged": bool(outcome.converged),
                "initial_convergence_warning": bool(outcome.initial_convergence_warning),
                "convergence_warning": bool(outcome.convergence_warning),
                "n_iter": int(outcome.n_iter),
                "max_iter_used": int(outcome.max_iter_used),
                "retry_count": int(outcome.retry_count),
            }
        )
        for offset, index in enumerate(test_idx):
            row = reference.iloc[int(index)]
            risk_oof_rows.append(
                {
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "enterprise_id": row["enterprise_id"],
                    "enterprise_name": row["enterprise_name"],
                    "default_label": int(row["default_label"]),
                    "test_label_hidden_from_risk_fit": True,
                    "risk_oof": float(risk_test[offset]),
                }
            )
        for offset, row in target.reset_index(drop=True).iterrows():
            risk_target_rows.append(
                {
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "enterprise_id": row["enterprise_id"],
                    "enterprise_name": row["enterprise_name"],
                    "risk_fold_model": float(risk_target[offset]),
                }
            )

        ls_test, ls_target, ls_audit = _label_spreading_predictions(
            X_train,
            y_train,
            X_test,
            X_target,
            feature_names,
            q1_config,
            q2_config,
        )
        ls_audit.update({"repeat_id": int(repeat_id), "fold_id": int(fold_id)})
        ls_audit_rows.append(ls_audit)
        ls_fold_rows.append(
            {
                "model": "label_spreading",
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "n_default_test": int(y_test.sum()),
                **{
                    key: value
                    for key, value in _binary_metric_row(
                        "label_spreading",
                        "cv_fold",
                        y_test,
                        ls_test,
                        q1_config,
                    ).items()
                    if key not in {"model", "scope", "n_observations", "n_default"}
                },
            }
        )
        for offset, index in enumerate(test_idx):
            row = reference.iloc[int(index)]
            ls_oof_rows.append(
                {
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "enterprise_id": row["enterprise_id"],
                    "enterprise_name": row["enterprise_name"],
                    "default_label": int(row["default_label"]),
                    "test_label_hidden_from_label_spreading": True,
                    "label_spreading_oof": float(ls_test[offset]),
                }
            )
        for offset, row in target.reset_index(drop=True).iterrows():
            ls_target_rows.append(
                {
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "enterprise_id": row["enterprise_id"],
                    "enterprise_name": row["enterprise_name"],
                    "label_spreading_fold_model": float(ls_target[offset]),
                }
            )

    risk_oof = stable_sort(pd.DataFrame(risk_oof_rows), ["repeat_id", "fold_id", "enterprise_id"])
    risk_target = stable_sort(pd.DataFrame(risk_target_rows), ["repeat_id", "fold_id", "enterprise_id"])
    risk_folds = stable_sort(pd.DataFrame(risk_fold_rows), ["repeat_id", "fold_id"])
    risk_fit = stable_sort(pd.DataFrame(risk_fit_rows), ["repeat_id", "fold_id"])
    ls_oof = stable_sort(pd.DataFrame(ls_oof_rows), ["repeat_id", "fold_id", "enterprise_id"])
    ls_target = stable_sort(pd.DataFrame(ls_target_rows), ["repeat_id", "fold_id", "enterprise_id"])
    ls_folds = stable_sort(pd.DataFrame(ls_fold_rows), ["repeat_id", "fold_id"])
    ls_audit = stable_sort(pd.DataFrame(ls_audit_rows), ["repeat_id", "fold_id"])
    expected_predictions = len(splits) // int(q1_config["model_training"]["cv"]["n_splits"])
    for frame, column in [(risk_oof, "risk_oof"), (ls_oof, "label_spreading_oof")]:
        counts = frame.groupby("enterprise_id")[column].size()
        if counts.nunique() != 1 or int(counts.iloc[0]) != expected_predictions:
            raise AssertionError(
                f"each enterprise must have {expected_predictions} repeated OOF predictions"
            )
    return {
        "risk_oof_all": risk_oof,
        "risk_target_fold": risk_target,
        "risk_fold_metrics": risk_folds,
        "risk_fit_diagnostics": risk_fit,
        "label_spreading_oof_all": ls_oof,
        "label_spreading_target_fold": ls_target,
        "label_spreading_fold_metrics": ls_folds,
        "label_spreading_fold_audit": ls_audit,
    }


def _summarize_binary_models(
    reference: pd.DataFrame,
    cv_results: Mapping[str, pd.DataFrame],
    q1_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = reference[["enterprise_id", "enterprise_name"]]
    risk_agg = _aggregate_binary_predictions(
        cv_results["risk_oof_all"], base, "risk_oof", "risk_oof"
    ).merge(reference[["enterprise_id", "default_label"]], on="enterprise_id", validate="one_to_one")
    ls_agg = _aggregate_binary_predictions(
        cv_results["label_spreading_oof_all"], base, "label_spreading_oof", "label_spreading_oof"
    ).merge(reference[["enterprise_id", "default_label"]], on="enterprise_id", validate="one_to_one")
    metrics_rows: list[dict[str, Any]] = []
    for model_name, fold_key, aggregate, probability_column in [
        (RISK_MODEL_NAME, "risk_fold_metrics", risk_agg, "risk_oof_mean"),
        ("label_spreading", "label_spreading_fold_metrics", ls_agg, "label_spreading_oof_mean"),
    ]:
        fold_frame = cv_results[fold_key]
        fold_row: dict[str, Any] = {
            "model": model_name,
            "scope": "cv_fold_mean_sd",
            "n_observations": int(len(fold_frame)),
            "n_default": int(fold_frame["n_default_test"].sum()),
            "fold_count": int(len(fold_frame)),
        }
        for metric in RISK_METRICS:
            fold_row[metric] = float(fold_frame[metric].mean())
            fold_row[f"{metric}_sd"] = float(fold_frame[metric].std(ddof=1))
        metrics_rows.append(fold_row)
        metrics_rows.append(
            _binary_metric_row(
                model_name,
                "enterprise_aggregated_oof",
                aggregate["default_label"].to_numpy(dtype=int),
                aggregate[probability_column].to_numpy(dtype=float),
                q1_config,
                fold_count=int(len(fold_frame)),
            )
        )
    metrics = pd.DataFrame(metrics_rows)
    oof_comparison = risk_agg.merge(
        ls_agg,
        on=["enterprise_id", "enterprise_name", "default_label"],
        validate="one_to_one",
        suffixes=("_risk", "_ls"),
    )
    oof_comparison["risk_difference_abs"] = (
        oof_comparison["risk_oof_mean"] - oof_comparison["label_spreading_oof_mean"]
    ).abs()
    oof_comparison["risk_rank"] = oof_comparison["risk_oof_mean"].rank(
        method="first", ascending=False
    ).astype(int)
    oof_comparison["label_spreading_rank"] = oof_comparison["label_spreading_oof_mean"].rank(
        method="first", ascending=False
    ).astype(int)
    oof_comparison["rank_difference"] = (
        oof_comparison["label_spreading_rank"] - oof_comparison["risk_rank"]
    )
    top_k = max(1, int(math.ceil(len(oof_comparison) * float(q1_config["model_training"]["top_risk_fraction"]))))
    risk_top = set(oof_comparison.nsmallest(top_k, "risk_rank")["enterprise_id"])
    ls_top = set(oof_comparison.nsmallest(top_k, "label_spreading_rank")["enterprise_id"])
    spearman = oof_comparison["risk_oof_mean"].corr(
        oof_comparison["label_spreading_oof_mean"], method="spearman"
    )
    summary = pd.DataFrame(
        [
            {
                "scope": "123_enterprise_aggregated_oof",
                "risk_rank_spearman": float(spearman),
                "top20_count": int(top_k),
                "top20_overlap_count": int(len(risk_top.intersection(ls_top))),
                "top20_overlap_fraction_of_risk_top20": float(len(risk_top.intersection(ls_top)) / top_k),
                "mean_abs_risk_difference": float(oof_comparison["risk_difference_abs"].mean()),
                "median_abs_risk_difference": float(oof_comparison["risk_difference_abs"].median()),
                "max_abs_risk_difference": float(oof_comparison["risk_difference_abs"].max()),
                "high_disagreement_count_threshold_0_20": int((oof_comparison["risk_difference_abs"] >= 0.20).sum()),
            }
        ]
    )
    return metrics, risk_agg, ls_agg, oof_comparison.merge(summary, how="cross")


def _paired_bootstrap_comparison(
    oof_comparison: pd.DataFrame,
    q1_config: Mapping[str, Any],
    q2_config: Mapping[str, Any],
    seed: int,
) -> pd.DataFrame:
    """Compare the two risk scores by enterprise-level paired bootstrap."""

    y = oof_comparison["default_label"].to_numpy(dtype=int)
    risk = oof_comparison["risk_oof_mean"].to_numpy(dtype=float)
    label_spreading = oof_comparison["label_spreading_oof_mean"].to_numpy(dtype=float)
    replicates = int(q2_config["modeling"]["comparison_bootstrap_replicates"])
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = {metric: [] for metric in RISK_METRICS}
    skipped = 0
    for _ in range(replicates):
        indices = rng.integers(0, len(y), size=len(y))
        y_sample = y[indices]
        if len(np.unique(y_sample)) < 2:
            skipped += 1
            continue
        risk_metrics = risk_model.compute_metrics(
            y_sample,
            risk[indices],
            top_risk_fraction=float(q1_config["model_training"]["top_risk_fraction"]),
            calibration_bins=int(q1_config["model_training"]["calibration_bins"]),
        )
        ls_metrics = risk_model.compute_metrics(
            y_sample,
            label_spreading[indices],
            top_risk_fraction=float(q1_config["model_training"]["top_risk_fraction"]),
            calibration_bins=int(q1_config["model_training"]["calibration_bins"]),
        )
        for metric in RISK_METRICS:
            values[metric].append(float(ls_metrics[metric] - risk_metrics[metric]))
    rows: list[dict[str, Any]] = []
    for metric in RISK_METRICS:
        differences = np.asarray(values[metric], dtype=float)
        higher_is_better = metric in {"pr_auc", "roc_auc", "balanced_accuracy", "f1", "top20_recall", "top20_precision"}
        better_probability = (
            float(np.mean(differences > 0.0))
            if higher_is_better
            else float(np.mean(differences < 0.0))
        ) if len(differences) else float("nan")
        rows.append(
            {
                "comparison": "label_spreading_minus_elastic_net_logistic",
                "metric": metric,
                "elastic_net_logistic_value": float(
                    risk_model.compute_metrics(
                        y,
                        risk,
                        top_risk_fraction=float(q1_config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(q1_config["model_training"]["calibration_bins"]),
                    )[metric]
                ),
                "label_spreading_value": float(
                    risk_model.compute_metrics(
                        y,
                        label_spreading,
                        top_risk_fraction=float(q1_config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(q1_config["model_training"]["calibration_bins"]),
                    )[metric]
                ),
                "difference_point": float(
                    risk_model.compute_metrics(
                        y,
                        label_spreading,
                        top_risk_fraction=float(q1_config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(q1_config["model_training"]["calibration_bins"]),
                    )[metric]
                    - risk_model.compute_metrics(
                        y,
                        risk,
                        top_risk_fraction=float(q1_config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(q1_config["model_training"]["calibration_bins"]),
                    )[metric]
                ),
                "difference_ci_low": float(np.quantile(differences, 0.025)) if len(differences) else float("nan"),
                "difference_ci_high": float(np.quantile(differences, 0.975)) if len(differences) else float("nan"),
                "label_spreading_better_probability": better_probability,
                "direction": "higher_is_better" if higher_is_better else "lower_is_better",
                "bootstrap_replicates_requested": replicates,
                "bootstrap_replicates_valid": int(len(differences)),
                "bootstrap_skipped_single_class": int(skipped),
                "enterprise_sample_size": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def _rating_metric_values(y: Sequence[int], probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    p = np.clip(p, 0.0, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    predicted = np.argmax(p, axis=1)
    correct = (predicted == labels).astype(float)
    confidence = p.max(axis=1)
    bin_edges = np.linspace(0.0, 1.0, 6)
    ece = 0.0
    for index in range(len(bin_edges) - 1):
        left, right = bin_edges[index], bin_edges[index + 1]
        mask = (confidence >= left) & (
            (confidence < right) if index < len(bin_edges) - 2 else (confidence <= right)
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    one_hot = np.eye(4, dtype=float)[labels]
    kappa = cohen_kappa_score(labels, predicted, labels=[0, 1, 2, 3], weights="quadratic")
    kappa = float(kappa) if np.isfinite(kappa) else float("nan")
    return {
        "macro_f1": float(f1_score(labels, predicted, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "ordered_grade_error": float(mean_absolute_error(labels, predicted)),
        "within_one_grade_rate": float(np.mean(np.abs(labels - predicted) <= 1)),
        "weighted_kappa": kappa,
        "multiclass_brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "multiclass_log_loss": float(log_loss(labels, p, labels=[0, 1, 2, 3])),
        "probability_ece": float(ece),
        "expected_absolute_grade_error": float(np.mean(np.sum(p * np.abs(np.arange(4)[None, :] - labels[:, None]), axis=1))),
    }


def _rating_pipeline(
    feature_names: Sequence[str],
    q1_config: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> Pipeline:
    parameters = q2_config["modeling"]["ordered_logistic"]
    return Pipeline(
        [
            (
                "preprocess",
                risk_model._make_fold_preprocessor(list(feature_names), RISK_MODEL_NAME, q1_config),
            ),
            (
                "model",
                OrderedLogistic(
                    l2_penalty=float(parameters["l2_penalty"]),
                    max_iter=int(parameters["max_iter"]),
                    tolerance=float(parameters["tolerance"]),
                ),
            ),
        ]
    )


def _rating_code_series(reference: pd.DataFrame) -> pd.Series:
    values = reference["credit_rating"].astype("string").map(RATING_TO_CODE)
    if values.isna().any():
        raise ValueError("reference rating labels contain values outside A/B/C/D")
    return values.astype(int)


def _run_rating_cv(
    reference: pd.DataFrame,
    feature_names: Sequence[str],
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    q1_config: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> dict[str, Any]:
    X = reference.loc[:, list(feature_names)]
    y = _rating_code_series(reference)
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    for repeat_id, fold_id, train_idx, test_idx in splits:
        estimator = _rating_pipeline(feature_names, q1_config, q2_config)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])
        probabilities = estimator.predict_proba(X.iloc[test_idx])
        fold_values = _rating_metric_values(y.iloc[test_idx], probabilities)
        fold_rows.append(
            {
                "scope": "cv_fold",
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **fold_values,
            }
        )
        model = estimator.named_steps["model"]
        fit_rows.append(
            {
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "optimization_success": bool(model.optimization_success_),
                "optimization_message": model.optimization_message_,
                "n_iter": int(model.n_iter_),
                "objective_value": float(model.objective_value_),
                "threshold_1": float(model.thresholds_[0]),
                "threshold_2": float(model.thresholds_[1]),
                "threshold_3": float(model.thresholds_[2]),
            }
        )
        for offset, index in enumerate(test_idx):
            row = reference.iloc[int(index)]
            item: dict[str, Any] = {
                "repeat_id": int(repeat_id),
                "fold_id": int(fold_id),
                "enterprise_id": row["enterprise_id"],
                "enterprise_name": row["enterprise_name"],
                "true_rating": row["credit_rating"],
                "true_rating_code": int(y.iloc[int(index)]),
                "rating_test_label_hidden_from_fit": True,
            }
            for code, rating in enumerate(RATING_ORDER):
                item[f"rating_prob_{rating}"] = float(probabilities[offset, code])
            item["predicted_rating"] = RATING_ORDER[int(np.argmax(probabilities[offset]))]
            prediction_rows.append(item)
    prediction_frame = stable_sort(pd.DataFrame(prediction_rows), ["repeat_id", "fold_id", "enterprise_id"])
    fold_frame = stable_sort(pd.DataFrame(fold_rows), ["repeat_id", "fold_id"])
    fit_frame = stable_sort(pd.DataFrame(fit_rows), ["repeat_id", "fold_id"])
    probability_columns = [f"rating_prob_{rating}" for rating in RATING_ORDER]
    aggregate_rows: list[dict[str, Any]] = []
    for enterprise_id, group in prediction_frame.groupby("enterprise_id", sort=True):
        row = {
            "enterprise_id": enterprise_id,
            "enterprise_name": group["enterprise_name"].iloc[0],
            "true_rating": group["true_rating"].iloc[0],
            "true_rating_code": int(group["true_rating_code"].iloc[0]),
            "rating_test_count": int(len(group)),
        }
        for column in probability_columns:
            row[column] = float(group[column].mean())
        values = np.asarray([row[column] for column in probability_columns], dtype=float)
        values = values / values.sum()
        for index, column in enumerate(probability_columns):
            row[column] = float(values[index])
        row["predicted_rating"] = RATING_ORDER[int(np.argmax(values))]
        aggregate_rows.append(row)
    aggregate = stable_sort(pd.DataFrame(aggregate_rows), ["enterprise_id"])
    aggregate_metrics = _rating_metric_values(
        aggregate["true_rating_code"].to_numpy(dtype=int),
        aggregate[probability_columns].to_numpy(dtype=float),
    )
    metric_rows: list[dict[str, Any]] = []
    fold_metric_row: dict[str, Any] = {
        "scope": "cv_fold_mean_sd",
        "n_observations": int(len(fold_frame)),
        "fold_count": int(len(fold_frame)),
    }
    for metric in RATING_METRICS:
        fold_metric_row[metric] = float(fold_frame[metric].mean())
        fold_metric_row[f"{metric}_sd"] = float(fold_frame[metric].std(ddof=1))
    metric_rows.append(fold_metric_row)
    metric_rows.append(
        {
            "scope": "enterprise_aggregated_oof",
            "n_observations": int(len(aggregate)),
            "fold_count": int(len(fold_frame)),
            **aggregate_metrics,
        }
    )
    return {
        "predictions_all": prediction_frame,
        "aggregate": aggregate,
        "fold_metrics": fold_frame,
        "metrics": pd.DataFrame(metric_rows),
        "fit_diagnostics": fit_frame,
        "probability_columns": probability_columns,
    }


def _fit_final_risk(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_names: Sequence[str],
    q1_config: Mapping[str, Any],
    fold_target: pd.DataFrame,
) -> dict[str, Any]:
    outcome = risk_model.fit_with_retry(
        _make_risk_factory(feature_names, q1_config),
        RISK_MODEL_NAME,
        reference.loc[:, list(feature_names)],
        reference["default_label"].astype(int),
        q1_config,
    )
    final_probability = risk_model.prediction(
        outcome.estimator,
        target.loc[:, list(feature_names)],
    )
    instability = _aggregate_binary_predictions(
        fold_target,
        target[["enterprise_id", "enterprise_name"]],
        "risk_fold_model",
        "risk_instability",
    )
    rows: list[dict[str, Any]] = []
    preprocessor = outcome.estimator.named_steps["preprocess"]
    classifier = outcome.estimator.named_steps["model"]
    for feature, coefficient in zip(preprocessor.get_feature_names_out(), classifier.coef_[0]):
        value = float(coefficient)
        rows.append(
            {
                "training_scope": "full_123_final_risk_pipeline",
                "feature": str(feature),
                "coefficient_standardized": value,
                "intercept": float(classifier.intercept_[0]),
                "nonzero": bool(abs(value) > 1e-12),
                "sign": "positive" if value > 1e-12 else "negative" if value < -1e-12 else "zero",
                "converged": bool(outcome.converged),
                "n_iter": int(outcome.n_iter),
                "max_iter_used": int(outcome.max_iter_used),
                "convergence_warning": bool(outcome.convergence_warning),
            }
        )
    return {
        "outcome": outcome,
        "pipeline": outcome.estimator,
        "probability": pd.DataFrame(
            {
                "enterprise_id": target["enterprise_id"].to_numpy(),
                "enterprise_name": target["enterprise_name"].to_numpy(),
                "main_risk_score": final_probability,
            }
        ),
        "instability": instability,
        "coefficients": pd.DataFrame(rows),
    }


def _fit_final_label_spreading(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_names: Sequence[str],
    q1_config: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> pd.DataFrame:
    # For deployment all 302 nodes are unlabeled target graph nodes, so the
    # graph is fit directly rather than treating them as a held-out test fold.
    preprocessor = risk_model._make_fold_preprocessor(list(feature_names), RISK_MODEL_NAME, q1_config)
    X_reference = reference.loc[:, list(feature_names)]
    X_target = target.loc[:, list(feature_names)]
    preprocessor.fit(X_reference, reference["default_label"].astype(int))
    matrix = np.vstack([preprocessor.transform(X_reference), preprocessor.transform(X_target)])
    labels = np.concatenate(
        [reference["default_label"].to_numpy(dtype=int), np.full(len(target), -1, dtype=int)]
    )
    parameters = q2_config["modeling"]["label_spreading"]
    estimator = LabelSpreading(
        kernel=str(parameters["kernel"]),
        n_neighbors=int(parameters["n_neighbors"]),
        alpha=float(parameters["alpha"]),
        max_iter=int(parameters["max_iter"]),
        tol=float(parameters["tolerance"]),
    )
    estimator.fit(matrix, labels)
    classes = np.asarray(estimator.classes_)
    class_one = int(np.where(classes == 1)[0][0])
    probability = np.clip(np.asarray(estimator.label_distributions_)[len(reference) :, class_one], 0.0, 1.0)
    if len(probability) != len(target):
        raise AssertionError("final Label Spreading target prediction count is not 302")
    return pd.DataFrame(
        {
            "enterprise_id": target["enterprise_id"].to_numpy(),
            "enterprise_name": target["enterprise_name"].to_numpy(),
            "label_spreading_risk_score": probability,
            "label_spreading_final_labeled_nodes": len(reference),
            "label_spreading_final_unlabeled_target_nodes": len(target),
            "label_spreading_test_labels_hidden": True,
        }
    )


def _fit_final_rating(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_names: Sequence[str],
    q1_config: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> dict[str, Any]:
    y = _rating_code_series(reference)
    estimator = _rating_pipeline(feature_names, q1_config, q2_config)
    estimator.fit(reference.loc[:, list(feature_names)], y)
    probabilities = estimator.predict_proba(target.loc[:, list(feature_names)])
    model = estimator.named_steps["model"]
    probability_columns = [f"rating_prob_{rating}" for rating in RATING_ORDER]
    result = pd.DataFrame(
        {
            "enterprise_id": target["enterprise_id"].to_numpy(),
            "enterprise_name": target["enterprise_name"].to_numpy(),
            **{column: probabilities[:, index] for index, column in enumerate(probability_columns)},
        }
    )
    result["predicted_rating"] = result[probability_columns].to_numpy().argmax(axis=1)
    result["predicted_rating"] = result["predicted_rating"].map(dict(enumerate(RATING_ORDER)))
    coefficient_rows = [
        {
            "parameter": f"beta_{feature}",
            "value": float(value),
            "parameter_type": "slope",
        }
        for feature, value in zip(feature_names, model.coef_[0])
    ]
    coefficient_rows.extend(
        {
            "parameter": f"threshold_{index + 1}",
            "value": float(value),
            "parameter_type": "ordered_threshold",
        }
        for index, value in enumerate(model.thresholds_)
    )
    return {
        "pipeline": estimator,
        "probabilities": result,
        "probability_columns": probability_columns,
        "parameters": pd.DataFrame(coefficient_rows),
        "fit_diagnostics": {
            "optimization_success": bool(model.optimization_success_),
            "optimization_message": model.optimization_message_,
            "n_iter": int(model.n_iter_),
            "objective_value": float(model.objective_value_),
            "thresholds_strictly_ordered": bool(np.all(np.diff(model.thresholds_) > 0)),
        },
    }


def _rating_probability_checks(
    scores: pd.DataFrame,
    probability_columns: Sequence[str],
    tolerance: float,
) -> pd.DataFrame:
    values = scores.loc[:, list(probability_columns)].to_numpy(dtype=float)
    sums = values.sum(axis=1)
    minima = values.min(axis=1)
    scores = scores.copy()
    scores["rating_probability_sum"] = sums
    scores["rating_probability_min"] = minima
    scores["rating_row_nonnegative_ok"] = minima >= -tolerance
    scores["rating_row_sum_ok"] = np.abs(sums - 1.0) <= tolerance
    scores["rating_probability_check"] = np.where(
        scores["rating_row_nonnegative_ok"] & scores["rating_row_sum_ok"], "PASS", "FAIL"
    )
    if not bool(scores["rating_row_nonnegative_ok"].all()) or not bool(scores["rating_row_sum_ok"].all()):
        raise AssertionError("at least one target rating probability row failed nonnegative/sum-to-one validation")
    return scores


def _add_uncertainty_flags(
    scores: pd.DataFrame,
    q2_config: Mapping[str, Any],
) -> pd.DataFrame:
    parameters = q2_config["modeling"]["instability"]
    scores = scores.copy()
    scores["risk_instability_interval_width"] = scores["risk_instability_p95"] - scores["risk_instability_p05"]
    scores["risk_disagreement_abs"] = (
        scores["main_risk_score"] - scores["label_spreading_risk_score"]
    ).abs()
    probability_columns = [f"rating_prob_{rating}" for rating in RATING_ORDER]
    probability_values = scores[probability_columns].to_numpy(dtype=float)
    entropy = -np.sum(probability_values * np.log(np.clip(probability_values, 1e-15, 1.0)), axis=1) / math.log(4.0)
    scores["rating_entropy_normalized"] = entropy
    scores["uncertainty_reason_ood"] = scores["ood_flag"].astype(bool)
    scores["uncertainty_reason_interval_width"] = scores["risk_instability_interval_width"] >= float(parameters["interval_width_threshold"])
    scores["uncertainty_reason_model_disagreement"] = scores["risk_disagreement_abs"] >= float(parameters["risk_disagreement_threshold"])
    scores["uncertainty_reason_rating_entropy"] = scores["rating_entropy_normalized"] >= float(parameters["rating_entropy_threshold"])

    def reason_row(row: pd.Series) -> str:
        reasons = []
        if bool(row["uncertainty_reason_ood"]):
            reasons.append("OOD")
        if bool(row["uncertainty_reason_interval_width"]):
            reasons.append("risk_interval_width")
        if bool(row["uncertainty_reason_model_disagreement"]):
            reasons.append("label_spreading_disagreement")
        if bool(row["uncertainty_reason_rating_entropy"]):
            reasons.append("rating_entropy")
        return "|".join(reasons) if reasons else "none"

    scores["uncertainty_reasons"] = scores.apply(reason_row, axis=1)
    scores["high_uncertainty_flag"] = (scores["uncertainty_reasons"] != "none").astype(int)
    scores["risk_interval_is_external_confidence_interval"] = False
    scores["risk_interval_definition"] = "repeated_fold_model_instability_5th_to_95th_percentile; not an external confidence interval"
    scores["risk_interpretation"] = "historical invoice behavior relative default tendency; 302 firms have no observed default labels"
    return scores


def _build_target_scores(
    reference: pd.DataFrame,
    target: pd.DataFrame,
    ood: pd.DataFrame,
    risk_final: Mapping[str, Any],
    ls_final: pd.DataFrame,
    rating_final: Mapping[str, Any],
    q2_config: Mapping[str, Any],
) -> pd.DataFrame:
    scores = target[["enterprise_id", "enterprise_name"]].copy()
    scores = scores.merge(risk_final["probability"], on=["enterprise_id", "enterprise_name"], validate="one_to_one")
    scores = scores.merge(risk_final["instability"], on=["enterprise_id", "enterprise_name"], validate="one_to_one")
    scores = scores.merge(ls_final, on=["enterprise_id", "enterprise_name"], validate="one_to_one")
    scores = scores.merge(rating_final["probabilities"], on=["enterprise_id", "enterprise_name"], validate="one_to_one")
    ood_columns = [
        "enterprise_id",
        "ood_feature_count",
        "ood_feature_fraction",
        "missing_primary_feature_count",
        "novelty_score_max_tail_distance",
        "ood_flag",
        "threshold_source",
        "threshold_rule",
    ]
    scores = scores.merge(ood.loc[:, [column for column in ood_columns if column in ood.columns]], on="enterprise_id", how="left", validate="one_to_one")
    scores["ood_flag"] = _as_bool_series(scores["ood_flag"])
    scores["main_risk_rank"] = scores["main_risk_score"].rank(method="first", ascending=False).astype(int)
    scores["label_spreading_risk_rank"] = scores["label_spreading_risk_score"].rank(method="first", ascending=False).astype(int)
    scores["rank_difference_label_spreading_minus_main"] = scores["label_spreading_risk_rank"] - scores["main_risk_rank"]
    top_k = max(1, int(math.ceil(len(scores) * 0.20)))
    scores["main_top20_candidate"] = (scores["main_risk_rank"] <= top_k).astype(int)
    scores["label_spreading_top20_candidate"] = (scores["label_spreading_risk_rank"] <= top_k).astype(int)
    probability_columns = rating_final["probability_columns"]
    scores = _rating_probability_checks(
        scores,
        probability_columns,
        float(q2_config["modeling"]["probability_check_tolerance"]),
    )
    scores = _add_uncertainty_flags(scores, q2_config)
    return stable_sort(scores, ["main_risk_rank", "enterprise_id"])


def _target_disagreement_summary(scores: pd.DataFrame) -> pd.DataFrame:
    spearman = scores["main_risk_score"].corr(scores["label_spreading_risk_score"], method="spearman")
    top_main = set(scores.loc[scores["main_top20_candidate"].eq(1), "enterprise_id"])
    top_ls = set(scores.loc[scores["label_spreading_top20_candidate"].eq(1), "enterprise_id"])
    return pd.DataFrame(
        [
            {
                "scope": "302_target_deployment_diagnostic",
                "risk_rank_spearman": float(spearman),
                "top20_count": int(len(top_main)),
                "top20_overlap_count": int(len(top_main.intersection(top_ls))),
                "top20_overlap_fraction_of_main_top20": float(len(top_main.intersection(top_ls)) / max(1, len(top_main))),
                "mean_abs_risk_difference": float(scores["risk_disagreement_abs"].mean()),
                "median_abs_risk_difference": float(scores["risk_disagreement_abs"].median()),
                "max_abs_risk_difference": float(scores["risk_disagreement_abs"].max()),
                "high_disagreement_count_threshold_0_20": int((scores["risk_disagreement_abs"] >= 0.20).sum()),
                "ood_count": int(scores["ood_flag"].sum()),
                "high_uncertainty_count": int(scores["high_uncertainty_flag"].sum()),
                "rating_probability_failed_rows": int((scores["rating_probability_check"] != "PASS").sum()),
            }
        ]
    )


def _configure_plot_fonts() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False


def _write_figures(
    paths: Mapping[str, Path],
    risk_oof: pd.DataFrame,
    ls_oof: pd.DataFrame,
    rating_aggregate: pd.DataFrame,
    target_scores: pd.DataFrame,
    probability_columns: Sequence[str],
) -> list[str]:
    _configure_plot_fonts()
    generated: list[str] = []
    y = risk_oof["default_label"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, frame, column, color in [
        ("弹性网Logistic主模型", risk_oof, "risk_oof_mean", "#1f77b4"),
        ("Label Spreading交叉检查", ls_oof, "label_spreading_oof_mean", "#d62728"),
    ]:
        probabilities = frame[column].to_numpy(dtype=float)
        fraction, mean_predicted = calibration_curve(y, probabilities, n_bins=5, strategy="quantile")
        ax.plot(mean_predicted, fraction, "o-", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="理想校准")
    ax.set_title("123家企业聚合OOF风险概率校准")
    ax.set_xlabel("平均预测风险值")
    ax.set_ylabel("实际违约比例")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.01, "302家无真实标签；风险值只表示历史发票行为相对倾向", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    calibration_path = paths["figures"] / "q2_risk_probability_calibration_oof.png"
    fig.savefig(calibration_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(relative(calibration_path))

    fig, ax = plt.subplots(figsize=(8, 7))
    ood_colors = np.where(target_scores["ood_flag"].to_numpy(dtype=bool), "#d62728", "#1f77b4")
    ax.scatter(
        target_scores["main_risk_score"],
        target_scores["label_spreading_risk_score"],
        c=ood_colors,
        s=np.where(target_scores["high_uncertainty_flag"].to_numpy(dtype=int) == 1, 48, 20),
        alpha=0.75,
        edgecolors="none",
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("弹性网Logistic主风险值")
    ax.set_ylabel("Label Spreading诊断风险值")
    ax.set_title("302家企业主模型与Label Spreading风险分歧")
    ax.text(0.02, 0.97, "红色=OOD；大点=高不确定性", transform=ax.transAxes, va="top", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    disagreement_path = paths["figures"] / "q2_risk_ranking_disagreement.png"
    fig.savefig(disagreement_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(relative(disagreement_path))

    interval_view = target_scores.nsmallest(min(50, len(target_scores)), "main_risk_rank").sort_values("main_risk_rank")
    positions = np.arange(len(interval_view))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        positions,
        interval_view["risk_instability_p50"],
        yerr=np.vstack(
            [
                interval_view["risk_instability_p50"] - interval_view["risk_instability_p05"],
                interval_view["risk_instability_p95"] - interval_view["risk_instability_p50"],
            ]
        ),
        fmt="o",
        color="#1f77b4",
        ecolor="#9EADBA",
        capsize=2,
    )
    ax.set_title("主风险值与重复折模型不稳定性区间（前50名）")
    ax.set_xlabel("主模型风险排名")
    ax.set_ylabel("风险值及5%—95%重复折区间")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.25)
    fig.text(0.5, 0.01, "区间是模型不稳定性诊断，不是真实外部置信区间", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    interval_path = paths["figures"] / "q2_risk_model_instability_interval.png"
    fig.savefig(interval_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(relative(interval_path))

    true_values = rating_aggregate["true_rating_code"].to_numpy(dtype=int)
    predicted_values = rating_aggregate[probability_columns].to_numpy(dtype=float).argmax(axis=1)
    matrix = confusion_matrix(true_values, predicted_values, labels=[0, 1, 2, 3])
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(4), RATING_ORDER)
    ax.set_yticks(range(4), RATING_ORDER)
    ax.set_xlabel("OOF预测评级")
    ax.set_ylabel("真实评级（123家代理验证）")
    ax.set_title("有序Logistic评级混淆矩阵（企业聚合OOF）")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.tight_layout()
    confusion_path = paths["figures"] / "q2_rating_confusion_matrix_oof.png"
    fig.savefig(confusion_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(relative(confusion_path))

    probability_view = target_scores.sort_values("main_risk_rank").loc[:, list(probability_columns)].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, 9))
    image = ax.imshow(probability_view, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="概率")
    ax.set_xticks(range(4), RATING_ORDER)
    ax.set_xlabel("评级")
    ax.set_ylabel("按主风险排序的302家企业")
    ax.set_title("302家企业有序Logistic A/B/C/D概率")
    fig.tight_layout()
    probability_path = paths["figures"] / "q2_rating_probability_heatmap_302.png"
    fig.savefig(probability_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    generated.append(relative(probability_path))
    return generated


def _validate_locked_contract(
    q2_config: Mapping[str, Any],
    q1_config: Mapping[str, Any],
) -> list[str]:
    seed = int(q2_config["random_seed"])
    q1_seed = int(q1_config["model_training"]["random_state"])
    primary_q2 = list(q2_config["features"]["primary_model_features"])
    primary_q1 = list(q1_config["features"]["primary_model_features"])
    if seed != 20260805 or q1_seed != 20260805:
        raise ValueError("question-two and question-one risk seeds must both be 20260805")
    if primary_q2 != primary_q1 or len(primary_q2) != 15:
        raise ValueError("question-two risk features do not exactly match the locked 15 question-one features")
    logistic = q1_config["model_training"]["logistic"]
    locked = {
        "solver": "saga",
        "penalty": "elasticnet",
        "C": 0.2,
        "l1_ratio": 0.8,
    }
    for key, expected in locked.items():
        actual = logistic[key]
        if isinstance(expected, float):
            if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"locked q1 Logistic parameter {key} changed: {actual}")
        elif actual != expected:
            raise ValueError(f"locked q1 Logistic parameter {key} changed: {actual}")
    cv = q1_config["model_training"]["cv"]
    if int(cv["n_splits"]) != 5 or int(cv["n_repeats"]) != 10:
        raise ValueError("question-two risk validation must be 5-fold x 10-repeat")
    return primary_q2


def _load_model_inputs(q2_config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reference_path = ROOT / "data" / "processed" / "_runtime" / "q2" / "features" / "q2_reference_enterprise_features.csv"
    target_path = ROOT / q2_config["outputs"]["processed_feature_file"]
    ood_path = ROOT / q2_config["outputs"]["processed_ood_file"]
    missing = [str(path) for path in [reference_path, target_path, ood_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("q2 model inputs are missing; run q2.py --stage all first: " + ", ".join(missing))
    reference = pd.read_csv(reference_path, encoding="utf-8-sig")
    target = pd.read_csv(target_path, encoding="utf-8-sig")
    ood = pd.read_csv(ood_path, encoding="utf-8-sig")
    if len(reference) != int(q2_config["q2"]["training_enterprise_count"]):
        raise ValueError(f"expected 123 reference enterprises, got {len(reference)}")
    if len(target) != int(q2_config["q2"]["target_enterprise_count"]):
        raise ValueError(f"expected 302 target enterprises, got {len(target)}")
    if {"credit_rating", "default_label"}.difference(reference.columns):
        raise ValueError("reference feature table lacks required labels")
    if {"credit_rating", "default_label"}.intersection(target.columns):
        raise ValueError("target feature table contains forbidden labels")
    if target["enterprise_id"].nunique() != 302 or reference["enterprise_id"].nunique() != 123:
        raise ValueError("enterprise identifiers are not unique")
    return reference, target, ood


def _write_model_report(
    paths: Mapping[str, Path],
    q2_config: Mapping[str, Any],
    q1_config: Mapping[str, Any],
    reference: pd.DataFrame,
    target: pd.DataFrame,
    feature_names: Sequence[str],
    risk_metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    rating_metrics: pd.DataFrame,
    disagreement_summary: pd.DataFrame,
    target_scores: pd.DataFrame,
    rating_final: Mapping[str, Any],
    risk_final: Mapping[str, Any],
    figures: Sequence[str],
) -> None:
    risk_view = risk_metrics.loc[risk_metrics["scope"].eq("enterprise_aggregated_oof"), ["model"] + RISK_METRICS].copy()
    rating_view = rating_metrics.loc[rating_metrics["scope"].eq("enterprise_aggregated_oof"), ["scope"] + RATING_METRICS].copy()
    bootstrap_view = bootstrap.loc[:, [
        "metric",
        "elastic_net_logistic_value",
        "label_spreading_value",
        "difference_point",
        "difference_ci_low",
        "difference_ci_high",
        "label_spreading_better_probability",
    ]].copy()
    high_uncertain = target_scores.loc[target_scores["high_uncertainty_flag"].eq(1)]
    ood_count = int(target_scores["ood_flag"].sum())
    nonconverged = int((~risk_final["outcome"].converged))
    rating_check_failures = int((target_scores["rating_probability_check"] != "PASS").sum())
    risk_parameter = q1_config["model_training"]["logistic"]
    instability_parameter = q2_config["modeling"]["instability"]
    report_lines = [
        "# 问题二风险、评级与半监督交叉检查模型报告",
        "",
        "本报告由本轮程序重新计算生成；未引用 project_plan.md 中的历史指标。",
        "",
        "## 1. 运行边界与输入",
        "",
        f"- 参考企业：{len(reference)}家；目标企业：{len(target)}家。参考企业违约{int(reference['default_label'].sum())}家，违约率{reference['default_label'].mean():.2%}。",
        f"- 固定随机种子：{int(q2_config['random_seed'])}；风险验证：5折×10次重复分层，共50个外层测试折；每家参考企业10次OOF预测。",
        f"- 风险主模型固定为问题一弹性网Logistic：solver={risk_parameter['solver']}、penalty={risk_parameter['penalty']}、C={risk_parameter['C']}、l1_ratio={risk_parameter['l1_ratio']}；没有树模型、随机森林或综合评价主模型。",
        f"- 风险和评级均只使用以下15项共同发票特征：{', '.join(feature_names)}。没有使用信誉评级作为风险输入，也没有把违约标签放进特征。",
        "- `requirements.txt` 未新增有序Logistic专门依赖；有序模型由 scipy 的优化器实现比例优势累计Logistic，scipy 已在原 requirements.txt 中锁定。",
        "",
        "## 2. 风险主模型：123家企业级OOF",
        "",
        _markdown_table(risk_view),
        "",
        "PR-AUC、Brier、LogLoss、ROC-AUC、5分位箱校准误差和Top20召回均来自123家企业的重复折预测先按企业取均值后的正式OOF；50个折本身不是50组独立企业。",
        "",
        "### 主模型与Label Spreading的企业级配对 bootstrap",
        "",
        "差值统一为 Label Spreading − 弹性网Logistic；Brier、LogLoss和校准误差按越低越好解释。bootstrap按企业抽样，不把50个重叠折当成独立样本。",
        _markdown_table(bootstrap_view),
        "",
        "## 3. 302家风险部署与不稳定性区间",
        "",
        f"- 已在全部123家企业上拟合最终风险Pipeline，并对302家企业输出 `main_risk_score`。全样本最终模型收敛：{bool(risk_final['outcome'].converged)}，迭代次数：{int(risk_final['outcome'].n_iter)}。",
        f"- 每家302企业另用50个重复折模型产生风险分布，输出{int(instability_parameter['lower_quantile'] * 100)}%—{int(instability_parameter['upper_quantile'] * 100)}%分位区间、标准差和宽度。该区间是重复拟合的模型不稳定性诊断，不是真实外部置信区间，也不是302家真实违约率的置信区间。",
        "- 302家没有真实违约标签，因此本报告禁止报告302家预测准确率、ROC-AUC或任何外部准确性结论；风险值只能解释为历史发票行为对应的相对违约倾向。",
        "",
        "## 4. 有序Logistic评级模型",
        "",
        "评级按风险从低到高编码为 A < B < C < D，使用比例优势累计模型 P(Y≤h|x)=sigmoid(theta_h−x beta)，而不是多分类Logistic。阈值通过正间隔参数化，确保累计概率有序。",
        _markdown_table(rating_view),
        "",
        f"- 302家每一行的A/B/C/D概率均做了非负和求和为1的逐行核验，失败行数：{rating_check_failures}。",
        f"- 最终模型阈值严格有序：{rating_final['fit_diagnostics']['thresholds_strictly_ordered']}；优化收敛：{rating_final['fit_diagnostics']['optimization_success']}。",
        "",
        "## 5. Label Spreading标签遮蔽交叉检查",
        "",
        "每个5折×10次测试折中，测试企业的违约标签在Label Spreading的标签数组中固定为-1；302家无标签企业也固定为-1并作为图节点。KNN邻居数、alpha、最大迭代次数和容差均在评测前由配置登记，未使用测试标签调参。Label Spreading只用于风险排序、分歧、分布支持和不确定性标记，不替换或平均主模型风险，也不进入评级概率。",
        _markdown_table(disagreement_summary),
        "",
        "## 6. OOD与高不确定性",
        "",
        f"- 302家中OOD企业数：{ood_count}；高不确定性企业数：{len(high_uncertain)}。",
        f"- 运行前登记的触发阈值：重复折区间宽度≥{instability_parameter['interval_width_threshold']}、主模型与Label Spreading风险绝对差≥{instability_parameter['risk_disagreement_threshold']}、归一化评级熵≥{instability_parameter['rating_entropy_threshold']}，或数据层OOD标记为真。",
        "- 高不确定性标记是诊断信号，建议人工复核或保守授信；它不等于企业已经违约。完整逐企业清单见 q2_risk_rating_scores.csv。",
        "",
        "## 7. 输出文件",
        "",
        *[f"- `{path}`" for path in figures],
        "- `data/processed/q2_risk_rating_scores.csv`：302家风险、模型不稳定性区间、A/B/C/D概率、Label Spreading分歧、OOD和高不确定性标记。",
        "- `outputs/q2/tables/q2_model_metrics.csv`：风险主模型和Label Spreading的折均值/标准差及企业聚合OOF指标。",
        "- `outputs/q2/tables/q2_model_comparison_bootstrap.csv`：1000次企业级配对bootstrap的差值区间和Label Spreading更优概率。",
        "- `outputs/q2/tables/q2_rating_metrics.csv`：有序评级折均值/标准差及企业聚合OOF指标。",
        "- `outputs/q2/tables/q2_risk_oof_enterprise.csv`、`q2_label_spreading_oof_enterprise.csv`、`q2_rating_oof_enterprise.csv`：123家企业级代理OOF。",
        "- `outputs/q2/reports/q2_risk_rating_run_manifest.json`：配置、输入哈希、模型边界和输出哈希。",
        "",
        "## 8. 局限",
        "",
        "- 123家有标签企业是附件2无标签迁移的代理验证集；302家没有真实违约和真实评级标签，不能把代理OOF指标外推成302家外部验证。",
        "- 发票行为与历史违约倾向是统计关联，不是因果关系；无违约发生时间和外部观察期时，风险值不应称为严格的一年期PD。",
        "- 有序评级指标反映附件1的历史评级标签，评级本身与违约高度绑定，因此评级模型只用于评级概率和诊断，不能回灌风险主模型。",
        "",
    ]
    write_text("\n".join(report_lines), paths["reports"] / "q2_risk_rating_model_report.md")
    write_text("\n".join(report_lines), paths["reports"] / "q2_model_report_for_paper.md")


def _output_hashes(paths: Mapping[str, Path]) -> dict[str, str]:
    candidates = [paths["scores"]]
    for key in ["runtime", "tables", "figures"]:
        candidates.extend(path for path in paths[key].rglob("*") if path.is_file())
    candidates.extend(
        paths["reports"] / name
        for name in ["q2_risk_rating_model_report.md", "q2_model_report_for_paper.md"]
    )
    return {
        relative(path): sha256_file(path)
        for path in sorted(set(candidates))
        if path.exists() and path.name != paths["manifest"].name
    }


def run_q2_models(config_path: Path | None = None) -> int:
    """Run the locked question-two risk/rating/diagnostic stage."""

    q2_path = config_path or (Path(__file__).resolve().with_name("q2_config.yaml"))
    q2_config = load_config(q2_path)
    q1_path = ROOT / q2_config["q2"]["reference_config"]
    q1_config = load_config(q1_path)
    feature_names = _validate_locked_contract(q2_config, q1_config)
    reference, target, ood = _load_model_inputs(q2_config)
    paths = _model_paths()
    seed = int(q2_config["random_seed"])
    splits = risk_model.build_splits(
        reference["default_label"].astype(int),
        n_splits=int(q1_config["model_training"]["cv"]["n_splits"]),
        n_repeats=int(q1_config["model_training"]["cv"]["n_repeats"]),
        random_state=seed,
    )
    cv_results = _run_risk_and_label_spreading(
        reference,
        target,
        feature_names,
        splits,
        q1_config,
        q2_config,
    )
    risk_metrics, risk_oof_agg, ls_oof_agg, oof_comparison = _summarize_binary_models(
        reference,
        cv_results,
        q1_config,
    )
    bootstrap = _paired_bootstrap_comparison(
        oof_comparison,
        q1_config,
        q2_config,
        seed,
    )
    rating_results = _run_rating_cv(
        reference,
        feature_names,
        splits,
        q1_config,
        q2_config,
    )
    risk_final = _fit_final_risk(
        reference,
        target,
        feature_names,
        q1_config,
        cv_results["risk_target_fold"],
    )
    ls_final = _fit_final_label_spreading(
        reference,
        target,
        feature_names,
        q1_config,
        q2_config,
    )
    rating_final = _fit_final_rating(
        reference,
        target,
        feature_names,
        q1_config,
        q2_config,
    )
    target_scores = _build_target_scores(
        reference,
        target,
        ood,
        risk_final,
        ls_final,
        rating_final,
        q2_config,
    )
    target_summary = _target_disagreement_summary(target_scores)
    oof_summary_columns = [
        "scope",
        "risk_rank_spearman",
        "top20_count",
        "top20_overlap_count",
        "top20_overlap_fraction_of_risk_top20",
        "mean_abs_risk_difference",
        "median_abs_risk_difference",
        "max_abs_risk_difference",
        "high_disagreement_count_threshold_0_20",
    ]
    oof_summary = oof_comparison.loc[:, oof_summary_columns].drop_duplicates().reset_index(drop=True)
    disagreement_summary = pd.concat(
        [oof_summary, target_summary],
        ignore_index=True,
        sort=False,
    )
    risk_oof_enterprise = risk_oof_agg.copy()
    ls_oof_enterprise = ls_oof_agg.copy()
    rating_oof_enterprise = rating_results["aggregate"].copy()
    probability_columns = rating_results["probability_columns"]
    figures = _write_figures(
        paths,
        risk_oof_enterprise,
        ls_oof_enterprise,
        rating_oof_enterprise,
        target_scores,
        probability_columns,
    )

    write_csv(risk_metrics, paths["tables"] / "q2_model_metrics.csv")
    write_csv(bootstrap, paths["tables"] / "q2_model_comparison_bootstrap.csv")
    write_csv(rating_results["metrics"], paths["tables"] / "q2_rating_metrics.csv")
    write_csv(disagreement_summary, paths["tables"] / "q2_model_disagreement_summary.csv")
    write_csv(cv_results["risk_fold_metrics"], paths["tables"] / "q2_risk_cv_fold_metrics.csv")
    write_csv(cv_results["label_spreading_fold_metrics"], paths["tables"] / "q2_label_spreading_cv_fold_metrics.csv")
    write_csv(cv_results["risk_fit_diagnostics"], paths["tables"] / "q2_risk_fit_diagnostics.csv")
    write_csv(cv_results["label_spreading_fold_audit"], paths["tables"] / "q2_label_spreading_fold_audit.csv")
    write_csv(rating_results["fold_metrics"], paths["tables"] / "q2_rating_cv_fold_metrics.csv")
    write_csv(rating_results["fit_diagnostics"], paths["tables"] / "q2_rating_fit_diagnostics.csv")
    write_csv(risk_oof_enterprise, paths["tables"] / "q2_risk_oof_enterprise.csv")
    write_csv(ls_oof_enterprise, paths["tables"] / "q2_label_spreading_oof_enterprise.csv")
    write_csv(oof_comparison, paths["tables"] / "q2_model_disagreement_123.csv")
    write_csv(rating_oof_enterprise, paths["tables"] / "q2_rating_oof_enterprise.csv")
    write_csv(cv_results["risk_oof_all"], paths["runtime"] / "q2_risk_oof_all.csv")
    write_csv(cv_results["label_spreading_oof_all"], paths["runtime"] / "q2_label_spreading_oof_all.csv")
    write_csv(cv_results["risk_target_fold"], paths["runtime"] / "q2_risk_target_fold_predictions.csv")
    write_csv(cv_results["label_spreading_target_fold"], paths["runtime"] / "q2_label_spreading_target_fold_predictions.csv")
    write_csv(rating_results["predictions_all"], paths["runtime"] / "q2_rating_oof_all.csv")
    write_csv(risk_final["coefficients"], paths["tables"] / "q2_final_risk_coefficients.csv")
    write_csv(rating_final["parameters"], paths["tables"] / "q2_final_ordered_logistic_parameters.csv")
    write_csv(target_scores, paths["scores"])
    write_csv(
        pd.DataFrame(
            confusion_matrix(
                rating_oof_enterprise["true_rating_code"],
                rating_oof_enterprise[probability_columns].to_numpy(dtype=float).argmax(axis=1),
                labels=[0, 1, 2, 3],
            ),
            index=RATING_ORDER,
            columns=RATING_ORDER,
        ).reset_index(names="true_rating"),
        paths["tables"] / "q2_rating_confusion_matrix.csv",
    )
    write_json(
        {
            "risk_final_fit": {
                "converged": bool(risk_final["outcome"].converged),
                "n_iter": int(risk_final["outcome"].n_iter),
                "max_iter_used": int(risk_final["outcome"].max_iter_used),
                "retry_count": int(risk_final["outcome"].retry_count),
            },
            "ordered_logistic_final_fit": rating_final["fit_diagnostics"],
            "target_rating_probability_validation": {
                "rows": int(len(target_scores)),
                "failed_rows": int((target_scores["rating_probability_check"] != "PASS").sum()),
                "minimum_probability": float(target_scores["rating_probability_min"].min()),
                "maximum_abs_sum_error": float(np.abs(target_scores["rating_probability_sum"] - 1.0).max()),
            },
        },
        paths["runtime"] / "q2_model_fit_summary.json",
    )
    with (paths["runtime"] / "q2_final_risk_pipeline.pkl").open("wb") as handle:
        pickle.dump(risk_final["pipeline"], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with (paths["runtime"] / "q2_final_ordered_rating_pipeline.pkl").open("wb") as handle:
        pickle.dump(rating_final["pipeline"], handle, protocol=pickle.HIGHEST_PROTOCOL)

    _write_model_report(
        paths,
        q2_config,
        q1_config,
        reference,
        target,
        feature_names,
        risk_metrics,
        bootstrap,
        rating_results["metrics"],
        disagreement_summary,
        target_scores,
        rating_final,
        risk_final,
        figures,
    )
    input_paths = [
        ROOT / q2_config["q2"]["training_attachment"],
        ROOT / q2_config["q2"]["target_attachment"],
        ROOT / q2_config["outputs"]["processed_feature_file"],
        ROOT / q2_config["outputs"]["processed_ood_file"],
        q1_path,
        q2_path,
    ]
    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "q2_config_sha256": config_hash(q2_config),
        "q1_config_sha256": config_hash(q1_config),
        "random_seed": seed,
        "cv": {"n_splits": 5, "n_repeats": 10, "split_count": len(splits), "stratification_target": "default_label"},
        "counts": {"reference_enterprises": int(len(reference)), "reference_defaults": int(reference["default_label"].sum()), "target_enterprises": int(len(target))},
        "risk_model": {"name": RISK_MODEL_NAME, "features": list(feature_names), "q1_locked_parameters": q1_config["model_training"]["logistic"], "final_fit_scope": "all_123_reference_enterprises"},
        "ordered_logistic": {"model_type": "proportional_odds_cumulative_logistic", "parameters": q2_config["modeling"]["ordered_logistic"], "rating_order": RATING_ORDER, "features": list(feature_names)},
        "label_spreading": {"parameters": q2_config["modeling"]["label_spreading"], "purpose": "cross_check_only", "test_labels_hidden": True, "target_nodes_unlabeled": True, "not_averaged_into_main_risk": True},
        "comparison_bootstrap": {"replicates_requested": int(q2_config["modeling"]["comparison_bootstrap_replicates"]), "enterprise_level_paired": True, "comparison": "label_spreading_minus_elastic_net_logistic"},
        "risk_interval": {"definition": "repeated_fold_model_instability_interval", "lower_quantile": q2_config["modeling"]["instability"]["lower_quantile"], "upper_quantile": q2_config["modeling"]["instability"]["upper_quantile"], "external_confidence_interval": False},
        "target_accuracy_reported": False,
        "requirements_changed": False,
        "input_hashes": {relative(path): sha256_file(path) for path in input_paths if path.exists()},
        "output_hashes": _output_hashes(paths),
    }
    write_json(manifest, paths["manifest"])
    print(f"q2_model_stage=PASS target_scores={relative(paths['scores'])} report={relative(paths['reports'] / 'q2_risk_rating_model_report.md')}")
    return 0
