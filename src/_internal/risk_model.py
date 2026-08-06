"""Leakage-safe third-batch risk-model training for question one.

The module only consumes the already accepted enterprise feature table.  It
does not rebuild features from invoice workbooks and it never uses the legacy
model-comparison scripts.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import yaml
from matplotlib import font_manager
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.utils.validation import check_is_fitted

from _internal.data_pipeline import (
    OUTPUT_ROOT,
    PAPER_ROOT,
    PROCESSED_ROOT,
    RUNTIME_ROOT,
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


LOGISTIC_NAME = "elastic_net_logistic"
TREE_NAME = "hist_gradient_boosting"
MODEL_NAMES = [LOGISTIC_NAME, TREE_NAME]
MAIN_METRICS = ["pr_auc", "brier", "log_loss", "top20_recall"]
MODEL_LABELS = {
    LOGISTIC_NAME: "弹性网Logistic",
    TREE_NAME: "HistGradientBoosting",
}
FEATURE_LABELS = {
    "business_scale_10k": "经营规模",
    "sales_scale_10k": "销售规模",
    "purchase_scale_10k": "采购规模",
    "net_sales_10k": "净销售额",
    "operating_net_inflow_proxy_10k": "经营净流入代理",
    "sales_growth_trend": "销售增长趋势",
    "sales_monthly_cv": "销售月度波动CV",
    "invoice_activity_per_month": "月均发票活跃度",
    "sales_return_rate": "销售退货率",
    "purchase_return_rate": "采购退货率",
    "void_invoice_rate": "作废发票率",
    "customer_count": "客户数量",
    "supplier_count": "供应商数量",
    "customer_hhi": "客户HHI",
    "supplier_hhi": "供应商HHI",
    "max_customer_share": "最大客户占比",
    "max_supplier_share": "最大供应商占比",
    "purchase_sales_ratio": "进销比",
    "active_month_ratio": "活跃月份比例",
    "longest_active_streak_ratio": "最长连续交易比例",
}
U_FEATURES = {
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
    "customer_count",
    "supplier_count",
    "purchase_sales_ratio",
}
SIGNED_LOG_FEATURES = {
    "operating_net_inflow_proxy_10k",
    "net_sales_10k",
}
NONNEGATIVE_LOG_FEATURES = {
    "business_scale_10k",
    "sales_scale_10k",
    "purchase_scale_10k",
    "sales_monthly_cv",
    "invoice_activity_per_month",
    "sales_return_rate",
    "purchase_return_rate",
    "customer_count",
    "supplier_count",
    "purchase_sales_ratio",
}


class Q1FoldPreprocessor(BaseEstimator, TransformerMixin):
    """Fit imputation, winsorization, transforms and scaling on one train fold."""

    def __init__(
        self,
        feature_names: Sequence[str],
        winsor_features: Sequence[str],
        signed_log_features: Sequence[str],
        nonnegative_log_features: Sequence[str],
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
        scale: bool = True,
    ) -> None:
        self.feature_names = feature_names
        self.winsor_features = winsor_features
        self.signed_log_features = signed_log_features
        self.nonnegative_log_features = nonnegative_log_features
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.scale = scale

    def _as_frame(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [name for name in self.feature_names if name not in X.columns]
            if missing:
                raise ValueError(f"preprocessor input is missing features: {missing}")
            frame = X.loc[:, list(self.feature_names)].copy()
        else:
            array = np.asarray(X, dtype=float)
            if array.ndim != 2 or array.shape[1] != len(self.feature_names):
                raise ValueError("preprocessor array shape does not match feature_names")
            frame = pd.DataFrame(array, columns=list(self.feature_names))
        for column in self.feature_names:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        return frame.astype(float)

    def _transform_without_scaling(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        frame = self._as_frame(X)
        for feature in self.feature_names:
            frame[feature] = frame[feature].fillna(self.imputation_values_[feature])
            if feature in self.winsor_features:
                frame[feature] = frame[feature].clip(
                    lower=self.lower_bounds_[feature],
                    upper=self.upper_bounds_[feature],
                )
        for feature in self.signed_log_features:
            if feature in frame:
                values = frame[feature].to_numpy(dtype=float)
                frame[feature] = np.sign(values) * np.log1p(np.abs(values))
        for feature in self.nonnegative_log_features:
            if feature in frame:
                values = frame[feature].to_numpy(dtype=float)
                frame[feature] = np.log1p(np.maximum(values, 0.0))
        return frame.to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any = None) -> "Q1FoldPreprocessor":
        frame = self._as_frame(X)
        self.feature_names_in_ = np.asarray(self.feature_names, dtype=object)
        self.imputation_values_ = {}
        for feature in self.feature_names:
            median = float(frame[feature].median(skipna=True))
            if not np.isfinite(median):
                raise ValueError(f"feature {feature} has no finite training median")
            self.imputation_values_[feature] = median
        filled = frame.copy()
        for feature, median in self.imputation_values_.items():
            filled[feature] = filled[feature].fillna(median)
        winsor_set = set(self.winsor_features)
        self.lower_bounds_ = {}
        self.upper_bounds_ = {}
        for feature in self.feature_names:
            if feature in winsor_set:
                values = filled[feature].to_numpy(dtype=float)
                lower = float(np.quantile(values, self.lower_quantile))
                upper = float(np.quantile(values, self.upper_quantile))
                if not np.isfinite(lower) or not np.isfinite(upper):
                    raise ValueError(f"feature {feature} has non-finite training quantiles")
                if lower > upper:
                    lower, upper = upper, lower
                self.lower_bounds_[feature] = lower
                self.upper_bounds_[feature] = upper
            else:
                self.lower_bounds_[feature] = -np.inf
                self.upper_bounds_[feature] = np.inf
        transformed = self._transform_without_scaling(frame)
        self.scaler_ = RobustScaler() if self.scale else None
        if self.scaler_ is not None:
            self.scaler_.fit(transformed)
        return self

    def transform(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        check_is_fitted(self, ["imputation_values_", "lower_bounds_", "upper_bounds_"])
        transformed = self._transform_without_scaling(X)
        if self.scaler_ is not None:
            transformed = self.scaler_.transform(transformed)
        if not np.isfinite(transformed).all():
            raise ValueError("preprocessor produced a non-finite value")
        return transformed

    def get_feature_names_out(self, input_features: Iterable[str] | None = None) -> np.ndarray:
        return np.asarray(list(self.feature_names), dtype=object)

    def parameter_rows(self, model_name: str, repeat_id: int, fold_id: int) -> list[dict[str, Any]]:
        """Return fitted fold statistics for audit and leakage review."""

        check_is_fitted(self, ["imputation_values_", "lower_bounds_", "upper_bounds_"])
        centers = getattr(self.scaler_, "center_", np.full(len(self.feature_names), np.nan))
        scales = getattr(self.scaler_, "scale_", np.full(len(self.feature_names), np.nan))
        rows: list[dict[str, Any]] = []
        for index, feature in enumerate(self.feature_names):
            rows.append(
                {
                    "model": model_name,
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "feature": feature,
                    "imputation_median": self.imputation_values_[feature],
                    "winsor_lower_bound": self.lower_bounds_[feature],
                    "winsor_upper_bound": self.upper_bounds_[feature],
                    "log_transform": (
                        "signed_log1p"
                        if feature in self.signed_log_features
                        else "nonnegative_log1p"
                        if feature in self.nonnegative_log_features
                        else "none"
                    ),
                    "scaler": "RobustScaler" if self.scaler_ is not None else "none",
                    "scaler_center": float(centers[index]) if np.isfinite(centers[index]) else np.nan,
                    "scaler_scale": float(scales[index]) if np.isfinite(scales[index]) else np.nan,
                }
            )
        return rows


@dataclass
class FitOutcome:
    estimator: Pipeline | ColumnTransformer
    convergence_warning: bool
    initial_convergence_warning: bool
    converged: bool
    n_iter: int
    max_iter_used: int
    retry_count: int


def _make_fold_preprocessor(
    feature_names: Sequence[str],
    model_name: str,
    config: Mapping[str, Any],
) -> Q1FoldPreprocessor:
    preprocessing = config["model_training"]["preprocessing"]
    return Q1FoldPreprocessor(
        feature_names=list(feature_names),
        winsor_features=[feature for feature in feature_names if feature in U_FEATURES],
        signed_log_features=[feature for feature in feature_names if feature in SIGNED_LOG_FEATURES],
        nonnegative_log_features=[feature for feature in feature_names if feature in NONNEGATIVE_LOG_FEATURES],
        lower_quantile=float(preprocessing["winsor_lower"]),
        upper_quantile=float(preprocessing["winsor_upper"]),
        scale=model_name == LOGISTIC_NAME,
    )


def _make_classifier(
    model_name: str,
    config: Mapping[str, Any],
    *,
    max_iter_override: int | None = None,
) -> Any:
    seed = int(config["model_training"]["random_state"])
    if model_name == LOGISTIC_NAME:
        params = config["model_training"]["logistic"]
        return LogisticRegression(
            solver=str(params["solver"]),
            penalty=str(params["penalty"]),
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=int(max_iter_override or params["max_iter"]),
            random_state=seed,
            n_jobs=1,
        )
    if model_name == TREE_NAME:
        params = config["model_training"]["tree"]
        return HistGradientBoostingClassifier(
            max_iter=int(params["max_iter"]),
            learning_rate=float(params["learning_rate"]),
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            l2_regularization=float(params["l2_regularization"]),
            random_state=seed,
        )
    raise ValueError(f"unknown model: {model_name}")


def make_pipeline(
    feature_names: Sequence[str],
    model_name: str,
    config: Mapping[str, Any],
) -> Pipeline:
    """Create a model pipeline; no data-dependent operation occurs here."""

    return Pipeline(
        [
            ("preprocess", _make_fold_preprocessor(feature_names, model_name, config)),
            ("model", _make_classifier(model_name, config)),
        ]
    )


def make_rating_pipeline(
    feature_names: Sequence[str],
    model_name: str,
    config: Mapping[str, Any],
    *,
    max_iter_override: int | None = None,
) -> Pipeline:
    """Create a leakage-labelled behavior plus credit-rating sensitivity pipeline."""

    numeric = _make_fold_preprocessor(feature_names, model_name, config)
    preprocess = ColumnTransformer(
        [
            ("behavior", numeric, list(feature_names)),
            ("rating", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["credit_rating"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", _make_classifier(model_name, config, max_iter_override=max_iter_override)),
        ]
    )


def fit_with_retry(
    factory: Callable[[int | None], Pipeline],
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: Mapping[str, Any],
) -> FitOutcome:
    """Fit once and retry logistic convergence with a larger max_iter."""

    base_max_iter = (
        int(config["model_training"]["logistic"]["max_iter"])
        if model_name == LOGISTIC_NAME
        else int(config["model_training"]["tree"]["max_iter"])
    )
    attempts = [base_max_iter, base_max_iter * 2] if model_name == LOGISTIC_NAME else [base_max_iter]
    initial_warning = False
    last_estimator: Pipeline | None = None
    last_warning = False
    for attempt_index, max_iter in enumerate(attempts):
        estimator = factory(max_iter)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X_train, y_train)
        convergence_warning = any(
            isinstance(item.message, ConvergenceWarning) for item in caught
        )
        if attempt_index == 0:
            initial_warning = convergence_warning
        last_estimator = estimator
        last_warning = convergence_warning
        if not convergence_warning:
            break
    if last_estimator is None:
        raise RuntimeError("model fit did not produce an estimator")
    fitted_model = last_estimator.named_steps["model"]
    raw_n_iter = getattr(fitted_model, "n_iter_", base_max_iter)
    n_iter = int(np.max(np.asarray(raw_n_iter, dtype=int)))
    return FitOutcome(
        estimator=last_estimator,
        convergence_warning=last_warning,
        initial_convergence_warning=initial_warning,
        converged=not last_warning,
        n_iter=n_iter,
        max_iter_used=int(attempts[0] if not initial_warning else attempts[-1]),
        retry_count=1 if initial_warning and len(attempts) > 1 else 0,
    )


def prediction(estimator: Pipeline, X: pd.DataFrame) -> np.ndarray:
    values = estimator.predict_proba(X)[:, 1].astype(float)
    return np.clip(values, 0.0, 1.0)


def calibration_error(y_true: Sequence[int], probabilities: Sequence[float], n_bins: int) -> float:
    """Compute a sample-weighted absolute calibration error over quantile bins."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    bins = np.array_split(order, min(n_bins, len(order)))
    errors = []
    for indices in bins:
        if len(indices) == 0:
            continue
        errors.append(len(indices) / len(y) * abs(float(y[indices].mean()) - float(p[indices].mean())))
    return float(np.sum(errors)) if errors else float("nan")


def compute_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    top_risk_fraction: float,
    calibration_bins: int,
) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    if len(np.unique(y)) < 2:
        raise ValueError("metric computation requires both classes")
    threshold_pred = (p >= 0.5).astype(int)
    top_k = max(1, int(np.ceil(len(y) * top_risk_fraction)))
    order = np.argsort(-p, kind="mergesort")
    top_indices = order[:top_k]
    total_defaults = int(y.sum())
    top_defaults = int(y[top_indices].sum())
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, p)),
        "calibration_error_5bin": float(calibration_error(y, p, calibration_bins)),
        "balanced_accuracy": float(balanced_accuracy_score(y, threshold_pred)),
        "f1": float(f1_score(y, threshold_pred, zero_division=0)),
        "threshold_0_5_precision": float(precision_score(y, threshold_pred, zero_division=0)),
        "threshold_0_5_recall": float(recall_score(y, threshold_pred, zero_division=0)),
        "top20_recall": float(top_defaults / total_defaults) if total_defaults else float("nan"),
        "top20_precision": float(top_defaults / top_k),
        "threshold": 0.5,
        "top_risk_fraction": top_risk_fraction,
        "n": float(len(y)),
        "n_default": float(total_defaults),
    }


def build_splits(
    y: Sequence[int],
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[tuple[int, int, np.ndarray, np.ndarray]]:
    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    splits: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for index, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
        repeat_id = index // n_splits + 1
        fold_id = index % n_splits + 1
        splits.append((repeat_id, fold_id, train_idx, test_idx))
    expected = n_splits * n_repeats
    if len(splits) != expected:
        raise AssertionError(f"expected {expected} splits, got {len(splits)}")
    return splits


def split_manifest(
    frame: pd.DataFrame,
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for repeat_id, fold_id, train_idx, test_idx in splits:
        for role, indices in [("train", train_idx), ("test", test_idx)]:
            for index in indices:
                row = frame.iloc[int(index)]
                rows.append(
                    {
                        "repeat_id": repeat_id,
                        "fold_id": fold_id,
                        "enterprise_id": row["enterprise_id"],
                        "split_role": role,
                        "default_label": int(row["default_label"]),
                    }
                )
    return stable_sort(pd.DataFrame(rows), ["repeat_id", "fold_id", "split_role", "enterprise_id"])


def _fold_metric_row(
    model_name: str,
    repeat_id: int,
    fold_id: int,
    y_test: Sequence[int],
    probabilities: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = compute_metrics(
        y_test,
        probabilities,
        top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
        calibration_bins=int(config["model_training"]["calibration_bins"]),
    )
    return {
        "model": model_name,
        "repeat_id": repeat_id,
        "fold_id": fold_id,
        "n_train": np.nan,
        "n_test": int(metrics.pop("n")),
        "n_default_test": int(metrics.pop("n_default")),
        **metrics,
    }


def run_primary_cv(
    frame: pd.DataFrame,
    primary_features: Sequence[str],
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, pd.DataFrame]:
    """Fit both algorithms on identical outer folds and collect all artifacts."""

    feature_frame = frame.loc[:, list(primary_features)]
    oof_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []

    for repeat_id, fold_id, train_idx, test_idx in splits:
        X_train = feature_frame.iloc[train_idx]
        X_test = feature_frame.iloc[test_idx]
        y_train = frame["default_label"].iloc[train_idx].astype(int)
        y_test = frame["default_label"].iloc[test_idx].astype(int)
        predictions_by_model: dict[str, np.ndarray] = {}
        outcomes: dict[str, FitOutcome] = {}
        for model_name in MODEL_NAMES:
            factory = lambda max_iter, names=list(primary_features), name=model_name: make_pipeline(
                names, name, config
            ) if name != LOGISTIC_NAME else Pipeline(
                [
                    ("preprocess", _make_fold_preprocessor(names, name, config)),
                    ("model", _make_classifier(name, config, max_iter_override=max_iter)),
                ]
            )
            outcome = fit_with_retry(factory, model_name, X_train, y_train, config)
            outcomes[model_name] = outcome
            predictions_by_model[model_name] = prediction(outcome.estimator, X_test)
            metric_row = _fold_metric_row(
                model_name,
                repeat_id,
                fold_id,
                y_test,
                predictions_by_model[model_name],
                config,
            )
            metric_row["n_train"] = int(len(train_idx))
            fold_metric_rows.append(metric_row)
            convergence_rows.append(
                {
                    "model": model_name,
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "converged": outcome.converged,
                    "n_iter": outcome.n_iter,
                    "max_iter_initial": int(config["model_training"][("logistic" if model_name == LOGISTIC_NAME else "tree")]["max_iter"]),
                    "max_iter_used": outcome.max_iter_used,
                    "initial_convergence_warning": outcome.initial_convergence_warning,
                    "convergence_warning": outcome.convergence_warning,
                    "retry_count": outcome.retry_count,
                }
            )
            preprocess = outcome.estimator.named_steps["preprocess"]
            preprocessing_rows.extend(preprocess.parameter_rows(model_name, repeat_id, fold_id))
            if model_name == LOGISTIC_NAME:
                classifier = outcome.estimator.named_steps["model"]
                coefficients = classifier.coef_[0].astype(float)
                for feature, coefficient in zip(preprocess.get_feature_names_out(), coefficients):
                    coefficient_rows.append(
                        {
                            "model": model_name,
                            "repeat_id": repeat_id,
                            "fold_id": fold_id,
                            "feature": str(feature),
                            "coefficient_standardized": float(coefficient),
                            "nonzero": bool(abs(float(coefficient)) > 1e-12),
                            "sign": "positive" if coefficient > 1e-12 else "negative" if coefficient < -1e-12 else "zero",
                            "converged": outcome.converged,
                            "n_iter": outcome.n_iter,
                            "max_iter_used": outcome.max_iter_used,
                            "convergence_warning": outcome.convergence_warning,
                        }
                    )
        for index in test_idx:
            row = frame.iloc[int(index)]
            oof_rows.append(
                {
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "enterprise_id": row["enterprise_id"],
                    "enterprise_name": row["enterprise_name"],
                    "credit_rating": row["credit_rating"],
                    "default_label": int(row["default_label"]),
                    "logistic_oof": float(predictions_by_model[LOGISTIC_NAME][list(test_idx).index(index)]),
                    "tree_oof": float(predictions_by_model[TREE_NAME][list(test_idx).index(index)]),
                }
            )

    oof = stable_sort(pd.DataFrame(oof_rows), ["repeat_id", "fold_id", "enterprise_id"])
    fold_metrics = stable_sort(pd.DataFrame(fold_metric_rows), ["model", "repeat_id", "fold_id"])
    coefficients = stable_sort(pd.DataFrame(coefficient_rows), ["repeat_id", "fold_id", "feature"])
    convergence = stable_sort(pd.DataFrame(convergence_rows), ["model", "repeat_id", "fold_id"])
    preprocessing = stable_sort(
        pd.DataFrame(preprocessing_rows),
        ["model", "repeat_id", "fold_id", "feature"],
    )
    expected_test_count = len(splits) // int(config["model_training"]["cv"]["n_splits"])
    counts = oof.groupby("enterprise_id").size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != expected_test_count:
        raise AssertionError(
            f"each enterprise must have {expected_test_count} outer test predictions; observed={counts.to_dict()}"
        )
    if not np.isfinite(oof[["logistic_oof", "tree_oof"]].to_numpy(dtype=float)).all():
        raise AssertionError("OOF predictions contain NaN or infinity")
    if ((oof[["logistic_oof", "tree_oof"]] < 0) | (oof[["logistic_oof", "tree_oof"]] > 1)).any().any():
        raise AssertionError("OOF probabilities are outside [0,1]")
    return {
        "oof_predictions_all": oof,
        "cv_fold_metrics": fold_metrics,
        "logistic_coefficients_all_folds": coefficients,
        "convergence_report": convergence,
        "preprocessing_parameters_all_folds": preprocessing,
    }


def aggregate_oof_predictions(
    frame: pd.DataFrame,
    oof: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate the ten outer-test predictions for each enterprise."""

    base = frame.loc[
        :,
        ["enterprise_id", "enterprise_name", "credit_rating", "default_label"],
    ].copy()
    rows: list[dict[str, Any]] = []
    for enterprise_id, group in oof.groupby("enterprise_id", sort=True):
        row = {
            "enterprise_id": enterprise_id,
            "enterprise_name": group["enterprise_name"].iloc[0],
            "credit_rating": group["credit_rating"].iloc[0],
            "default_label": int(group["default_label"].iloc[0]),
        }
        for model_name, column in [
            (LOGISTIC_NAME, "logistic_oof"),
            (TREE_NAME, "tree_oof"),
        ]:
            values = group[column].to_numpy(dtype=float)
            row[f"{model_name}_oof_mean"] = float(np.mean(values))
            row[f"{model_name}_oof_sd"] = float(np.std(values, ddof=1))
            row[f"{model_name}_oof_p10"] = float(np.quantile(values, 0.10))
            row[f"{model_name}_oof_p50"] = float(np.quantile(values, 0.50))
            row[f"{model_name}_oof_p90"] = float(np.quantile(values, 0.90))
            row[f"{model_name}_test_count"] = int(len(values))
        rows.append(row)
    aggregate = pd.DataFrame(rows)
    aggregate = base.drop(columns=["enterprise_name", "credit_rating", "default_label"]).merge(
        aggregate,
        on="enterprise_id",
        how="left",
        validate="one_to_one",
    )
    return stable_sort(aggregate, ["enterprise_id"])


def model_metric_summary(
    aggregate: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Return fold summaries plus formal enterprise-aggregated OOF metrics."""

    rows: list[dict[str, Any]] = []
    metric_names = [
        "pr_auc",
        "brier",
        "log_loss",
        "roc_auc",
        "calibration_error_5bin",
        "balanced_accuracy",
        "f1",
        "threshold_0_5_precision",
        "threshold_0_5_recall",
        "top20_recall",
        "top20_precision",
    ]
    for model_name in MODEL_NAMES:
        model_folds = fold_metrics.loc[fold_metrics["model"].eq(model_name)]
        fold_row: dict[str, Any] = {
            "model": model_name,
            "scope": "cv_fold_mean_sd",
            "n_observations": int(len(model_folds)),
        }
        for metric in metric_names:
            fold_row[metric] = float(model_folds[metric].mean())
            fold_row[f"{metric}_sd"] = float(model_folds[metric].std(ddof=1))
        rows.append(fold_row)
        probabilities = aggregate[f"{model_name}_oof_mean"].to_numpy(dtype=float)
        metrics = compute_metrics(
            aggregate["default_label"].to_numpy(dtype=int),
            probabilities,
            top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
            calibration_bins=int(config["model_training"]["calibration_bins"]),
        )
        formal_row: dict[str, Any] = {
            "model": model_name,
            "scope": "enterprise_aggregated_oof",
            "n_observations": int(len(aggregate)),
        }
        for metric in metric_names:
            formal_row[metric] = metrics[metric]
            formal_row[f"{metric}_sd"] = np.nan
        rows.append(formal_row)
    return pd.DataFrame(rows)


def paired_bootstrap_comparison(
    aggregate: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Compare tree and logistic by enterprise-level paired bootstrap."""

    y = aggregate["default_label"].to_numpy(dtype=int)
    logistic = aggregate[f"{LOGISTIC_NAME}_oof_mean"].to_numpy(dtype=float)
    tree = aggregate[f"{TREE_NAME}_oof_mean"].to_numpy(dtype=float)
    replicates = int(config["model_training"]["bootstrap_replicates"])
    seed = int(config["model_training"]["random_state"])
    rng = np.random.default_rng(seed)
    valid_rows: dict[str, list[float]] = {metric: [] for metric in MAIN_METRICS}
    skipped = 0
    for _ in range(replicates):
        indices = rng.integers(0, len(y), size=len(y))
        y_sample = y[indices]
        if len(np.unique(y_sample)) < 2:
            skipped += 1
            continue
        logistic_metrics = compute_metrics(
            y_sample,
            logistic[indices],
            top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
            calibration_bins=int(config["model_training"]["calibration_bins"]),
        )
        tree_metrics = compute_metrics(
            y_sample,
            tree[indices],
            top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
            calibration_bins=int(config["model_training"]["calibration_bins"]),
        )
        for metric in MAIN_METRICS:
            raw_difference = tree_metrics[metric] - logistic_metrics[metric]
            advantage = raw_difference if metric in {"pr_auc", "top20_recall"} else -raw_difference
            valid_rows[metric].append(float(advantage))
    rows: list[dict[str, Any]] = []
    for metric in MAIN_METRICS:
        values = np.asarray(valid_rows[metric], dtype=float)
        direction = "higher_is_better" if metric in {"pr_auc", "top20_recall"} else "lower_is_better"
        raw_values = values if direction == "higher_is_better" else -values
        rows.append(
            {
                "metric": metric,
                "logistic_value": float(
                    compute_metrics(
                        y,
                        logistic,
                        top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(config["model_training"]["calibration_bins"]),
                    )[metric]
                ),
                "tree_value": float(
                    compute_metrics(
                        y,
                        tree,
                        top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(config["model_training"]["calibration_bins"]),
                    )[metric]
                ),
                "difference_tree_minus_logistic": float(
                    compute_metrics(
                        y,
                        tree,
                        top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(config["model_training"]["calibration_bins"]),
                    )[metric]
                    - compute_metrics(
                        y,
                        logistic,
                        top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
                        calibration_bins=int(config["model_training"]["calibration_bins"]),
                    )[metric]
                ),
                "tree_advantage_definition": "tree-logistic" if direction == "higher_is_better" else "logistic-tree",
                "tree_advantage_point": float(np.mean(values)) if len(values) else np.nan,
                "difference_ci_low": float(np.quantile(raw_values, 0.025)) if len(raw_values) else np.nan,
                "difference_ci_high": float(np.quantile(raw_values, 0.975)) if len(raw_values) else np.nan,
                "tree_advantage_ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "tree_advantage_ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
                "bootstrap_probability_tree_better": float(np.mean(values > 0)) if len(values) else np.nan,
                "bootstrap_replicates_requested": replicates,
                "bootstrap_replicates_valid": int(len(values)),
                "bootstrap_skipped_single_class": skipped,
                "direction": direction,
                "enterprise_sample_size": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def select_model(bootstrap: pd.DataFrame) -> tuple[str, int]:
    """Apply the pre-registered four-metric, three-of-four replacement rule."""

    stable = (
        (bootstrap["ci_low"] > 0)
        & (bootstrap["bootstrap_probability_tree_better"] >= 0.95)
    )
    count = int(stable.sum())
    return (TREE_NAME if count >= 3 else LOGISTIC_NAME), count


def run_single_feature_set_cv(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    model_name: str,
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Run one final-model feature-set experiment on the locked outer splits."""

    predictions: list[dict[str, Any]] = []
    X = frame.loc[:, list(feature_names)]
    y = frame["default_label"].astype(int)
    for repeat_id, fold_id, train_idx, test_idx in splits:
        factory = lambda max_iter, names=list(feature_names), name=model_name: Pipeline(
            [
                ("preprocess", _make_fold_preprocessor(names, name, config)),
                ("model", _make_classifier(name, config, max_iter_override=max_iter)),
            ]
        )
        outcome = fit_with_retry(factory, model_name, X.iloc[train_idx], y.iloc[train_idx], config)
        probabilities = prediction(outcome.estimator, X.iloc[test_idx])
        for offset, index in enumerate(test_idx):
            predictions.append(
                {
                    "repeat_id": repeat_id,
                    "fold_id": fold_id,
                    "enterprise_id": frame.iloc[int(index)]["enterprise_id"],
                    "probability": float(probabilities[offset]),
                }
            )
    prediction_frame = pd.DataFrame(predictions)
    aggregate = (
        prediction_frame.groupby("enterprise_id", sort=True)["probability"]
        .mean()
        .rename("probability")
        .reset_index()
        .merge(frame[["enterprise_id", "default_label"]], on="enterprise_id", validate="one_to_one")
    )
    metrics = compute_metrics(
        aggregate["default_label"],
        aggregate["probability"],
        top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
        calibration_bins=int(config["model_training"]["calibration_bins"]),
    )
    row = {
        "experiment": "feature_set_sensitivity",
        "feature_set": "",
        "model": model_name,
        "scope": "enterprise_aggregated_oof",
        "n_features": len(feature_names),
    }
    row.update({metric: metrics[metric] for metric in metrics if metric not in {"n", "n_default", "threshold", "top_risk_fraction"}})
    row["feature_names"] = "|".join(feature_names)
    return pd.DataFrame([row])


def run_rating_sensitivity(
    frame: pd.DataFrame,
    primary_features: Sequence[str],
    splits: Sequence[tuple[int, int, np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    """Run a clearly labelled rating-leakage sensitivity experiment."""

    rows: list[dict[str, Any]] = []
    X = frame.loc[:, list(primary_features) + ["credit_rating"]]
    y = frame["default_label"].astype(int)
    for model_name in MODEL_NAMES:
        predictions: list[dict[str, Any]] = []
        for repeat_id, fold_id, train_idx, test_idx in splits:
            factory = lambda max_iter, names=list(primary_features), name=model_name: make_rating_pipeline(
                names, name, config, max_iter_override=max_iter
            )
            outcome = fit_with_retry(factory, model_name, X.iloc[train_idx], y.iloc[train_idx], config)
            probabilities = prediction(outcome.estimator, X.iloc[test_idx])
            for offset, index in enumerate(test_idx):
                predictions.append(
                    {
                        "enterprise_id": frame.iloc[int(index)]["enterprise_id"],
                        "probability": float(probabilities[offset]),
                    }
                )
        prediction_frame = pd.DataFrame(predictions)
        aggregate = (
            prediction_frame.groupby("enterprise_id", sort=True)["probability"]
            .mean()
            .rename("probability")
            .reset_index()
            .merge(frame[["enterprise_id", "default_label"]], on="enterprise_id", validate="one_to_one")
        )
        metrics = compute_metrics(
            aggregate["default_label"],
            aggregate["probability"],
            top_risk_fraction=float(config["model_training"]["top_risk_fraction"]),
            calibration_bins=int(config["model_training"]["calibration_bins"]),
        )
        row = {
            "experiment": "rating_leakage_sensitivity_only",
            "model": model_name,
            "scope": "enterprise_aggregated_oof",
            "n_behavior_features": len(primary_features),
            "rating_used": True,
        }
        row.update({metric: metrics[metric] for metric in metrics if metric not in {"n", "n_default", "threshold", "top_risk_fraction"}})
        rows.append(row)
    return pd.DataFrame(rows)


def coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    """Summarize standardized logistic coefficients across outer folds."""

    rows: list[dict[str, Any]] = []
    for feature, group in coefficients.groupby("feature", sort=True):
        values = group["coefficient_standardized"].to_numpy(dtype=float)
        positive = float(np.mean(values > 1e-12))
        negative = float(np.mean(values < -1e-12))
        zero = float(np.mean(np.abs(values) <= 1e-12))
        rows.append(
            {
                "feature": feature,
                "feature_label": FEATURE_LABELS.get(feature, feature),
                "coefficient_median": float(np.median(values)),
                "coefficient_q25": float(np.quantile(values, 0.25)),
                "coefficient_q75": float(np.quantile(values, 0.75)),
                "absolute_coefficient_median": float(np.median(np.abs(values))),
                "nonzero_rate": float(np.mean(np.abs(values) > 1e-12)),
                "positive_rate": positive,
                "negative_rate": negative,
                "zero_rate": zero,
                "sign_stability": float(max(positive, negative)),
                "dominant_sign": "positive" if positive >= negative and positive >= zero else "negative" if negative >= zero else "zero",
                "fold_count": int(len(values)),
            }
        )
    result = pd.DataFrame(rows)
    result["absolute_coefficient_median_rank"] = (
        result["absolute_coefficient_median"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return stable_sort(result, ["absolute_coefficient_median_rank", "feature"])


def fit_final_logistic(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, FitOutcome]:
    """Fit the all-123 display model after OOF evaluation is complete."""

    X = frame.loc[:, list(feature_names)]
    y = frame["default_label"].astype(int)
    factory = lambda max_iter, names=list(feature_names): Pipeline(
        [
            ("preprocess", _make_fold_preprocessor(names, LOGISTIC_NAME, config)),
            ("model", _make_classifier(LOGISTIC_NAME, config, max_iter_override=max_iter)),
        ]
    )
    outcome = fit_with_retry(factory, LOGISTIC_NAME, X, y, config)
    estimator = outcome.estimator
    preprocess = estimator.named_steps["preprocess"]
    classifier = estimator.named_steps["model"]
    rows: list[dict[str, Any]] = []
    for feature, coefficient in zip(preprocess.get_feature_names_out(), classifier.coef_[0]):
        coefficient_float = float(coefficient)
        rows.append(
            {
                "training_scope": "full_123_display_model",
                "feature": str(feature),
                "feature_label": FEATURE_LABELS.get(str(feature), str(feature)),
                "coefficient_standardized": coefficient_float,
                "intercept": float(classifier.intercept_[0]),
                "nonzero": bool(abs(coefficient_float) > 1e-12),
                "sign": "positive" if coefficient_float > 1e-12 else "negative" if coefficient_float < -1e-12 else "zero",
                "converged": outcome.converged,
                "n_iter": outcome.n_iter,
                "max_iter_used": outcome.max_iter_used,
                "convergence_warning": outcome.convergence_warning,
            }
        )
    return pd.DataFrame(rows), outcome


def _configure_plot_fonts() -> None:
    preferred = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in preferred if name in available), "DejaVu Sans")
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False


def _figure_note(fig: Any) -> None:
    fig.text(
        0.5,
        0.01,
        "风险值表示历史违约倾向，不是严格一年期违约概率",
        ha="center",
        fontsize=9,
        color="#555555",
    )


def generate_figures(
    aggregate: pd.DataFrame,
    metric_summary: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    selected_model: str,
    figure_dir: Path,
) -> None:
    """Generate all required OOF-based figures."""

    _configure_plot_fonts()
    figure_dir.mkdir(parents=True, exist_ok=True)
    y = aggregate["default_label"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, column, color in [
        (LOGISTIC_NAME, "elastic_net_logistic_oof_mean", "#1f77b4"),
        (TREE_NAME, "hist_gradient_boosting_oof_mean", "#d62728"),
    ]:
        precision, recall, _ = precision_recall_curve(y, aggregate[column].to_numpy(dtype=float))
        ax.plot(recall, precision, label=f"{MODEL_LABELS[model_name]} (AP={average_precision_score(y, aggregate[column]):.3f})", color=color)
    ax.set_title("企业聚合OOF精确率—召回率曲线")
    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.legend()
    ax.grid(alpha=0.25)
    _figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "pr_curve_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, column, color in [
        (LOGISTIC_NAME, "elastic_net_logistic_oof_mean", "#1f77b4"),
        (TREE_NAME, "hist_gradient_boosting_oof_mean", "#d62728"),
    ]:
        false_positive, true_positive, _ = roc_curve(y, aggregate[column].to_numpy(dtype=float))
        ax.plot(false_positive, true_positive, label=f"{MODEL_LABELS[model_name]} (AUC={roc_auc_score(y, aggregate[column]):.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title("企业聚合OOF ROC曲线")
    ax.set_xlabel("假阳性率")
    ax.set_ylabel("真阳性率")
    ax.legend()
    ax.grid(alpha=0.25)
    _figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "roc_curve_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, column, color in [
        (LOGISTIC_NAME, "elastic_net_logistic_oof_mean", "#1f77b4"),
        (TREE_NAME, "hist_gradient_boosting_oof_mean", "#d62728"),
    ]:
        fraction, mean_predicted = calibration_curve(
            y,
            aggregate[column].to_numpy(dtype=float),
            n_bins=5,
            strategy="quantile",
        )
        ax.plot(mean_predicted, fraction, "o-", label=MODEL_LABELS[model_name], color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="理想校准")
    ax.set_title("企业聚合OOF概率校准曲线（5分位箱）")
    ax.set_xlabel("平均预测风险值")
    ax.set_ylabel("实际违约比例")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.25)
    _figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "calibration_curve_oof.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    formal = metric_summary.loc[metric_summary["scope"].eq("enterprise_aggregated_oof")].set_index("model")
    display_metrics = ["pr_auc", "brier", "log_loss", "top20_recall"]
    labels = ["PR-AUC", "Brier", "LogLoss", "Top20召回"]
    x = np.arange(len(display_metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    for offset, model_name, color in [(-width / 2, LOGISTIC_NAME, "#1f77b4"), (width / 2, TREE_NAME, "#d62728")]:
        ax.bar(x + offset, [formal.loc[model_name, metric] for metric in display_metrics], width, label=MODEL_LABELS[model_name], color=color)
    ax.set_xticks(x, labels)
    ax.set_title("两类模型企业聚合OOF主要指标比较")
    ax.set_ylabel("指标值（Brier/LogLoss越低越好）")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "model_metric_comparison.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ordered = coefficient_summary.sort_values("absolute_coefficient_median", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.errorbar(
        ordered["coefficient_median"],
        np.arange(len(ordered)),
        xerr=[
            ordered["coefficient_median"] - ordered["coefficient_q25"],
            ordered["coefficient_q75"] - ordered["coefficient_median"],
        ],
        fmt="o",
        color="#4472C4",
        ecolor="#9EADBA",
        capsize=3,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(ordered)), ordered["feature_label"])
    ax.set_title("Logistic标准化系数稳定性（中位数与四分位区间）")
    ax.set_xlabel("标准化系数")
    ax.set_ylabel("特征")
    ax.grid(axis="x", alpha=0.25)
    _figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "logistic_coefficient_stability.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 6))
    for model_name, column, color in [
        (LOGISTIC_NAME, "elastic_net_logistic_oof_mean", "#1f77b4"),
        (TREE_NAME, "hist_gradient_boosting_oof_mean", "#d62728"),
    ]:
        ax.hist(aggregate[column], bins=12, alpha=0.45, label=MODEL_LABELS[model_name], color=color)
    ax.set_title("企业聚合OOF风险值分布")
    ax.set_xlabel("风险值")
    ax.set_ylabel("企业数")
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "oof_probability_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    score_column = f"{selected_model}_oof_mean"
    ranked = aggregate.sort_values(score_column, ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = np.where(ranked["default_label"].to_numpy(dtype=int) == 1, "#C00000", "#4472C4")
    ax.scatter(np.arange(1, len(ranked) + 1), ranked[score_column], c=colors, s=28, alpha=0.85)
    ax.set_title(f"按{MODEL_LABELS[selected_model]}OOF风险排序的企业违约标签")
    ax.set_xlabel("风险排名（1为最高）")
    ax.set_ylabel("风险值")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    _figure_note(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(figure_dir / "risk_rank_by_default.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _format_metric(value: Any) -> str:
    return "NA" if pd.isna(value) else f"{float(value):.4f}"


def _metric_markdown(summary: pd.DataFrame) -> str:
    rows = summary.loc[summary["scope"].eq("enterprise_aggregated_oof")]
    lines = [
        "| 模型 | PR-AUC | Brier | LogLoss | ROC-AUC | 校准误差 | Balanced Accuracy | F1 | Top20召回 | Top20精确率 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS.get(str(row["model"]), str(row["model"])),
                    _format_metric(row["pr_auc"]),
                    _format_metric(row["brier"]),
                    _format_metric(row["log_loss"]),
                    _format_metric(row["roc_auc"]),
                    _format_metric(row["calibration_error_5bin"]),
                    _format_metric(row["balanced_accuracy"]),
                    _format_metric(row["f1"]),
                    _format_metric(row["top20_recall"]),
                    _format_metric(row["top20_precision"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_reports(
    frame: pd.DataFrame,
    aggregate: pd.DataFrame,
    metric_summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    coefficient_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    sensitivity: pd.DataFrame,
    rating_sensitivity: pd.DataFrame,
    selected_model: str,
    stable_metric_count: int,
    output_report: Path,
    paper_report: Path,
) -> None:
    """Write the program-result-only handoff reports."""

    formal = metric_summary.loc[metric_summary["scope"].eq("enterprise_aggregated_oof")]
    zero_summary_path = RUNTIME_ROOT / "feature_validation" / "zero_amount_invoice_business_summary.csv"
    zero_summary = pd.read_csv(zero_summary_path, encoding="utf-8-sig") if zero_summary_path.exists() else pd.DataFrame()
    zero_valid = zero_summary.loc[
        zero_summary["stage"].eq("atomic_invoice")
        & zero_summary["direction"].eq("all")
        & zero_summary["status_scope"].eq("valid")
    ] if not zero_summary.empty else pd.DataFrame()
    if not zero_valid.empty:
        zero_fact_text = (
            f"程序实际核验的有效原子发票分母为{int(zero_valid['invoice_count'].iloc[0])}，精确零值数为{int(zero_valid['exact_zero_count'].iloc[0])}，"
            f"0.01元容差零值数为{int(zero_valid['tolerance_zero_count'].iloc[0])}；两种口径的差异使该字段只保留为审计字段。"
        )
    else:
        zero_fact_text = "零金额业务核验见阶段A输出；该字段只保留为审计字段。"
    top_positive = coefficient_summary.loc[
        (coefficient_summary["coefficient_median"] > 0)
        & (coefficient_summary["positive_rate"] >= coefficient_summary["negative_rate"])
    ].sort_values(["sign_stability", "absolute_coefficient_median"], ascending=False).head(3)
    top_negative = coefficient_summary.loc[
        (coefficient_summary["coefficient_median"] < 0)
        & (coefficient_summary["negative_rate"] >= coefficient_summary["positive_rate"])
    ].sort_values(["sign_stability", "absolute_coefficient_median"], ascending=False).head(3)
    top_risk = aggregate.sort_values(f"{selected_model}_oof_mean", ascending=False).head(5)
    low_risk = aggregate.sort_values(f"{selected_model}_oof_mean", ascending=True).head(5)
    convergence_bad = convergence.loc[~convergence["converged"]]
    bootstrap_lines = [
        "| 指标 | 树模型相对Logistic差值 | 树模型优势定义 | 95%区间 | 树模型更优概率 |",
        "|---|---:|---|---|---:|",
    ]
    for _, row in bootstrap.iterrows():
        bootstrap_lines.append(
            f"| {row['metric']} | {_format_metric(row['difference_tree_minus_logistic'])} | "
            f"{row['tree_advantage_definition']} | [{_format_metric(row['difference_ci_low'])}, {_format_metric(row['difference_ci_high'])}] | "
            f"{_format_metric(row['bootstrap_probability_tree_better'])} |"
        )
    coefficient_positive_text = "；".join(
        f"{row['feature']}（中位数={row['coefficient_median']:.4f}，正号比例={row['positive_rate']:.2%}）"
        for _, row in top_positive.iterrows()
    ) or "无满足条件的稳定正向特征"
    coefficient_negative_text = "；".join(
        f"{row['feature']}（中位数={row['coefficient_median']:.4f}，负号比例={row['negative_rate']:.2%}）"
        for _, row in top_negative.iterrows()
    ) or "无满足条件的稳定负向特征"
    high_text = "；".join(
        f"{row['enterprise_id']}（{row['enterprise_name']}，风险值={row[f'{selected_model}_oof_mean']:.4f}）"
        for _, row in top_risk.iterrows()
    )
    low_text = "；".join(
        f"{row['enterprise_id']}（{row['enterprise_name']}，风险值={row[f'{selected_model}_oof_mean']:.4f}）"
        for _, row in low_risk.iterrows()
    )
    convergence_text = (
        "所有正式折均收敛。"
        if convergence_bad.empty
        else f"仍有{len(convergence_bad)}个未收敛记录，详见convergence_report.csv。"
    )
    rating_lines = []
    for _, row in rating_sensitivity.iterrows():
        rating_lines.append(
            f"{MODEL_LABELS.get(str(row['model']), str(row['model']))}: PR-AUC={_format_metric(row['pr_auc'])}, "
            f"Brier={_format_metric(row['brier'])}, LogLoss={_format_metric(row['log_loss'])}, "
            f"Top20召回={_format_metric(row['top20_recall'])}"
        )
    rating_text = "；".join(rating_lines) if rating_lines else "未运行"
    sensitivity_lines = ["| 特征集 | 模型 | 特征数 | PR-AUC | Brier | LogLoss | Top20召回 |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in sensitivity.iterrows():
        sensitivity_lines.append(
            f"| {row['feature_set']} | {MODEL_LABELS.get(str(row['model']), str(row['model']))} | {int(row['n_features'])} | "
            f"{_format_metric(row['pr_auc'])} | {_format_metric(row['brier'])} | {_format_metric(row['log_loss'])} | {_format_metric(row['top20_recall'])} |"
        )

    report_lines = [
        "# 第三批问题一风险模型训练报告",
        "",
        "本报告只使用data/processed/q1_enterprise_features.csv中已经通过阶段A验收的企业级特征；没有调用历史模型脚本，也没有从原始Excel重新构造特征。",
        "",
        "## 1. 样本、标签和输入契约",
        "",
        f"- 企业样本数：{len(frame)}；违约数：{int(frame['default_label'].sum())}；未违约数：{int((1 - frame['default_label']).sum())}；违约率：{frame['default_label'].mean():.2%}。",
        f"- 每家企业外层测试次数：{int(aggregate[f'{LOGISTIC_NAME}_test_count'].iloc[0])}；重复分层交叉验证为5折×10次，共{len(convergence[(convergence['model'].eq(LOGISTIC_NAME)) & (convergence['repeat_id'].astype(str) != 'full')])}个Logistic正式折。",
        "- 企业代号、企业名称和信誉评级仅用于标识/展示；default_label仅作标签；audit_字段不进入模型矩阵。",
        "",
        "## 2. 特征角色",
        "",
        "- 共构造21个企业级派生特征，其中15个进入主模型，5个用于特征集敏感性分析，zero_amount_invoice_rate仅作审计。",
        f"- 本轮主模型实际使用：{', '.join([str(item) for item in frame.attrs.get('primary_features', [])]) or '见model_training_config_snapshot.yaml'}。",
        f"- zero_amount_invoice_rate的精确/容差核验与排除理由见outputs/q1/reports/q1_zero_amount_invoice_rate_decision.md；{zero_fact_text}",
        "",
        "## 3. 交叉验证与折内预处理",
        "",
        "- 两类算法使用完全相同的50个企业级外层切分；所有中位数、1%/99%缩尾边界、log1p变换和RobustScaler均在训练折Pipeline内拟合。",
        "- 主模型为未使用SMOTE、未使用class_weight=balanced的弹性网Logistic；对照为预先锁定参数的HistGradientBoostingClassifier。",
        "",
        "## 4. 企业聚合OOF主要指标",
        "",
        _metric_markdown(metric_summary),
        "",
        "PR-AUC、Brier、LogLoss和Top20召回用于主要比较；阈值0.5的指标只作辅助结果。",
        "",
        "## 5. 配对bootstrap和主模型选择",
        "",
        f"- Bootstrap按123家企业抽样，要求至少{int(bootstrap['bootstrap_replicates_requested'].iloc[0])}次；跳过单一类别样本次数为{int(bootstrap['bootstrap_skipped_single_class'].iloc[0])}。",
        *bootstrap_lines,
        "",
        f"- 按运行前锁定的规则，树模型在4项主要评价中稳定优于Logistic的项数为{stable_metric_count}；最终主模型为{MODEL_LABELS[selected_model]}，树模型作为{'主模型' if selected_model == TREE_NAME else '稳健性对照'}。",
        "",
        "## 6. Logistic系数稳定性",
        "",
        f"- 稳定正向风险特征：{coefficient_positive_text}。",
        f"- 稳定负向风险特征：{coefficient_negative_text}。",
        f"- {convergence_text} 全样本最终展示系数另存于final_logistic_coefficients.csv；全样本拟合只用于解释和后续部署，不用于评价自身性能。",
        "",
        "## 7. 特征集敏感性与评级敏感性",
        "",
        *sensitivity_lines,
        "",
        f"- 评级实验明确标记为rating_leakage_sensitivity_only，仅用于说明评级与标签绑定，不替代行为主模型：{rating_text}。",
        "",
        "## 8. 典型企业和概率解释",
        "",
        f"- 选定模型的最高风险企业示例：{high_text}。",
        f"- 选定模型的最低风险企业示例：{low_text}。",
        "- 这些风险值是历史发票行为下的相对违约倾向，不是严格一年期违约概率；相关关系不解释为因果关系。",
        "",
        "## 9. 局限和可复现文件",
        "",
        "- 样本只有123家企业且违约标签不均衡；外层重复折彼此重叠，不能当成50组独立样本。",
        "- 没有违约发生日期和明确观察期，可能存在时间信息限制；评级是人工先验且附件2缺失，因此主模型不使用评级。",
        "- 主要输出运行文件见data/processed/_runtime/model_training；正式表格见outputs/q1/tables，图表见outputs/q1/figures/model；输入特征哈希、配置哈希、代码版本和环境见运行目录中的manifest和environment文件。",
        "",
    ]
    write_text("\n".join(report_lines), output_report)

    paper_lines = [
        "# 问题一第三批风险模型结果（论文手文档）",
        "",
        "以下内容全部由第三批训练程序实际输出生成；风险值应解释为历史违约倾向，不应称为严格一年期违约概率。",
        "",
        "## 1. 样本和标签结构",
        "",
        f"附件1企业级样本共{len(frame)}家，违约{int(frame['default_label'].sum())}家、未违约{int((1 - frame['default_label']).sum())}家，违约比例为{frame['default_label'].mean():.2%}。模型评价以企业为单位，不以发票明细为切分单位。",
        "",
        "## 2. 最终使用的特征集",
        "",
        "共构造21个企业级派生特征，其中15个进入主模型，5个用于特征集敏感性分析，zero_amount_invoice_rate仅作审计。主模型实际列表写入运行目录的model_training_config_snapshot.yaml；zero_amount_invoice_rate的业务核验见outputs/q1/reports/q1_zero_amount_invoice_rate_decision.md。",
        "",
        "## 3. 企业级重复分层交叉验证设计",
        "",
        "使用固定随机种子20260805的5折、10次重复分层交叉验证，共50个配对外层测试折；每家企业恰好获得10次外层测试预测。预处理严格封装在Pipeline内并只在训练折拟合。",
        "",
        "## 4. 两类模型主要指标",
        "",
        _metric_markdown(metric_summary),
        "",
        "比较首先在相同的15项主模型特征上完成。主指标为PR-AUC、Brier和LogLoss，并以Top20风险企业召回率作为主要审查指标；准确率未作为主指标。",
        "",
        "## 5. Bootstrap差异区间",
        "",
        *bootstrap_lines,
        "",
        "区间和树模型更优概率来自按企业配对bootstrap，不是按50个重叠测试折抽样。",
        "",
        "## 6. 最终主模型选择",
        "",
        f"按照预先锁定的“4项主要评价中至少3项稳定优于”的规则，树模型稳定胜出{stable_metric_count}项，因此最终主模型为{MODEL_LABELS[selected_model]}；未达到标准时弹性网Logistic保留为主模型，树模型为稳健性对照。",
        "",
        "## 7. 信誉评级敏感性（非正式结果）",
        "",
        f"评级实验只用于说明人工评级与标签的绑定关系，明确标记为rating_leakage_sensitivity_only，不替代行为主模型，也不作为附件2正式预测依据：{rating_text}。",
        "",
        "## 8. Logistic系数的稳定方向",
        "",
        f"正向风险系数中最稳定的特征为：{coefficient_positive_text}。负向风险系数中最稳定的特征为：{coefficient_negative_text}。系数方向是模型条件关联，不是因果证明。",
        "",
        "## 9. 典型高风险和低风险企业",
        "",
        f"按最终主模型OOF均值，典型高风险企业为：{high_text}。典型低风险企业为：{low_text}。",
        "",
        "## 10. 概率校准情况",
        "",
        "校准曲线使用123家企业的聚合OOF预测，五分位箱校准误差见上表和model_metrics_summary.csv；不得用全样本拟合值替代OOF结果。",
        "",
        "## 11. 模型局限",
        "",
        "样本量和违约事件数有限，50个重复折存在重叠；没有违约日期和观察期，风险值不等于一年期PD；附件2没有人工评级，评级敏感性结果不能迁移为正式预测；行为特征与违约之间只作相关解释。",
        "",
        "## 12. 论文图表路径",
        "",
        "可直接引用的图表位于outputs/q1/figures/model/pr_curve_oof.png、roc_curve_oof.png、calibration_curve_oof.png、model_metric_comparison.png、logistic_coefficient_stability.png、oof_probability_distribution.png和risk_rank_by_default.png。",
        "",
        "## 13. 后续信贷优化读取字段",
        "",
        f"后续信贷优化应读取data/processed/q1_risk_scores.csv中的{'logistic_oof_mean' if selected_model == LOGISTIC_NAME else 'tree_oof_mean'}对应字段，或读取selected_model_risk_score；credit_rating只作展示，不能重新进入行为主模型。",
        "",
    ]
    write_text("\n".join(paper_lines), paper_report)
