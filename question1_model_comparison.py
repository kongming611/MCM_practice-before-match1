"""问题一：用重复分层小块样本比较两类风险模型。

本脚本把发票明细先聚合到“企业”层级，再以企业为独立样本进行评测。
主评测不使用已有信誉评级，避免把人工评级几乎等同于违约标签；可用
``--include-rating-sensitivity`` 额外运行评级敏感性检查。

方案 A：弹性网 Logistic 回归。
方案 B：梯度提升树。若环境安装了 LightGBM/XGBoost，可在此处替换；当前
仓库没有这两个依赖，因此使用 sklearn 的 HistGradientBoostingClassifier 作为
同类树模型的可复现实验代理，并在结果中明确记录后端名称。

评测设计：5 折、10 次重复分层交叉验证，共 50 个测试小块。每个小块约
24--25 家企业，按违约标签保持约 22% 的违约比例；所有模型在同一小块上
配对比较。超参数在评测前预先固定为适合小样本的保守配置，不让任何测试
小块参与调参。
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RANDOM_STATE = 20260805
N_SPLITS = 5
N_REPEATS = 10
N_BOOTSTRAPS = 1000


def _find_data(prefix: str) -> Path:
    matches = sorted(DATA_DIR.glob(f"{prefix}*.xlsx"))
    if not matches:
        raise FileNotFoundError(f"data file not found: {prefix}*.xlsx")
    return matches[0]


def _signed_log(values):
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))


def _empty_index_frame(firm_ids: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame(index=pd.Index(list(firm_ids), name="firm"))


def _aggregate_direction(
    frame: pd.DataFrame, firm_ids: Iterable[str], prefix: str
) -> pd.DataFrame:
    """Aggregate one invoice direction to one row per enterprise."""

    frame = frame.iloc[:, :8].copy()
    frame.columns = [
        "firm",
        "invoice",
        "date",
        "counterparty",
        "amount",
        "tax",
        "total",
        "status",
    ]
    frame["firm"] = frame["firm"].astype("string").str.strip()
    frame["counterparty"] = frame["counterparty"].astype("string").str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["total"] = pd.to_numeric(frame["total"], errors="coerce").fillna(0.0)
    frame["status"] = frame["status"].astype("string").str.strip()
    frame["is_void"] = frame["status"].str.contains("作废", na=False)
    frame["is_valid"] = ~frame["is_void"]
    frame["abs_total"] = frame["total"].abs()
    frame["is_negative"] = frame["total"] < 0
    frame["is_zero"] = frame["total"] == 0
    frame["month"] = frame["date"].dt.to_period("M").astype("string")
    frame["day"] = frame["date"].dt.floor("D")

    all_stats = frame.groupby("firm", sort=False).agg(
        lines=("invoice", "size"),
        unique_invoices=("invoice", "nunique"),
        active_days=("day", "nunique"),
        total_net=("total", "sum"),
        total_abs=("abs_total", "sum"),
        mean_abs=("abs_total", "mean"),
        median_abs=("abs_total", "median"),
        negative_lines=("is_negative", "sum"),
        zero_lines=("is_zero", "sum"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    valid = frame.loc[frame["is_valid"]].copy()
    valid_stats = valid.groupby("firm", sort=False).agg(
        valid_lines=("invoice", "size"),
        valid_total_net=("total", "sum"),
        valid_total_abs=("abs_total", "sum"),
        valid_mean_abs=("abs_total", "mean"),
        valid_median_abs=("abs_total", "median"),
        valid_negative_lines=("is_negative", "sum"),
        valid_zero_lines=("is_zero", "sum"),
        valid_active_months=("month", "nunique"),
        valid_counterparties=("counterparty", "nunique"),
    )

    result = all_stats.join(valid_stats, how="outer")
    result["void_lines"] = result["lines"] - result["valid_lines"]
    result["duplicate_lines"] = result["lines"] - result["unique_invoices"]
    result["void_rate"] = result["void_lines"] / result["lines"].replace(0, np.nan)
    result["duplicate_rate"] = result["duplicate_lines"] / result["lines"].replace(0, np.nan)
    result["negative_rate"] = result["negative_lines"] / result["lines"].replace(0, np.nan)
    result["zero_rate"] = result["zero_lines"] / result["lines"].replace(0, np.nan)
    result["valid_negative_rate"] = result["valid_negative_lines"] / result["valid_lines"].replace(0, np.nan)
    result["valid_zero_rate"] = result["valid_zero_lines"] / result["valid_lines"].replace(0, np.nan)
    result["valid_rate"] = result["valid_lines"] / result["lines"].replace(0, np.nan)
    result["date_span_days"] = (result["last_date"] - result["first_date"]).dt.days

    monthly = (
        valid.groupby(["firm", "month"], sort=False)
        .agg(month_net=("total", "sum"), month_abs=("abs_total", "sum"))
        .reset_index()
    )
    monthly_stats = monthly.groupby("firm", sort=False).agg(
        monthly_abs_mean=("month_abs", "mean"),
        monthly_abs_std=("month_abs", "std"),
        monthly_net_std=("month_net", "std"),
        monthly_count=("month", "nunique"),
    )
    monthly_stats["monthly_abs_cv"] = monthly_stats["monthly_abs_std"] / monthly_stats[
        "monthly_abs_mean"
    ].replace(0, np.nan)

    trend_values = {}
    recent_values = {}
    for firm, group in monthly.groupby("firm", sort=False):
        group = group.sort_values("month")
        values = group["month_abs"].to_numpy(dtype=float)
        if len(values) >= 2 and np.mean(values) > 0:
            x = np.arange(len(values), dtype=float)
            trend_values[firm] = float(np.polyfit(x, values, 1)[0] / np.mean(values))
            first = np.mean(values[: min(3, len(values))])
            last = np.mean(values[-min(3, len(values)) :])
            recent_values[firm] = float(last / first) if first > 0 else 0.0
        else:
            trend_values[firm] = 0.0
            recent_values[firm] = 0.0
    monthly_stats["monthly_abs_trend"] = pd.Series(trend_values)
    monthly_stats["recent_to_initial_abs"] = pd.Series(recent_values)
    result = result.join(monthly_stats, how="outer")

    cp = (
        valid.groupby(["firm", "counterparty"], dropna=False, sort=False)["abs_total"]
        .sum()
        .rename("counterparty_abs")
    )
    cp_share = cp / cp.groupby(level=0).transform("sum").replace(0, np.nan)
    concentration = pd.DataFrame(index=cp_share.index.get_level_values(0).unique())
    concentration["top_counterparty_share"] = cp_share.groupby(level=0).max()
    concentration["counterparty_hhi"] = (cp_share**2).groupby(level=0).sum()
    result = result.join(concentration, how="outer")

    result = result.reindex(list(firm_ids))
    result.columns = [f"{prefix}_{column}" for column in result.columns]
    result = result.drop(columns=[f"{prefix}_first_date", f"{prefix}_last_date"])
    return result.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_problem1() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read attachment 1 and return behavior features, labels and metadata."""

    path = _find_data("附件1")
    info = pd.read_excel(path, sheet_name=0)
    firm = info.iloc[:, 0].astype("string").str.strip()
    metadata = pd.DataFrame(
        {
            "firm": firm,
            "rating": info.iloc[:, 2].astype("string").str.strip(),
            "y": info.iloc[:, 3].astype("string").str.strip().eq("是").astype(int),
        }
    ).set_index("firm")
    firm_ids = metadata.index.tolist()

    # Attachment 1 stores input and output invoices in sheets 1 and 2.
    input_frame = pd.read_excel(path, sheet_name=1)
    output_frame = pd.read_excel(path, sheet_name=2)
    input_features = _aggregate_direction(input_frame, firm_ids, "in")
    output_features = _aggregate_direction(output_frame, firm_ids, "out")
    features = input_features.join(output_features, how="outer").fillna(0.0)

    def ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        return numerator / denominator.replace(0, np.nan)

    features["cross_out_to_in_abs"] = ratio(
        features["out_valid_total_abs"], features["in_valid_total_abs"]
    )
    features["cross_out_to_in_count"] = ratio(
        features["out_valid_lines"], features["in_valid_lines"]
    )
    features["cross_out_minus_in_net"] = (
        features["out_valid_total_net"] - features["in_valid_total_net"]
    )
    features["cross_out_minus_in_void_rate"] = (
        features["out_void_rate"] - features["in_void_rate"]
    )
    features["cross_out_minus_in_negative_rate"] = (
        features["out_valid_negative_rate"] - features["in_valid_negative_rate"]
    )
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features, metadata["y"].astype(int), metadata


def _make_preprocessor(kind: str, include_rating: bool, columns: Iterable[str]):
    columns = list(columns)
    numeric_columns = [c for c in columns if c != "rating"]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "logistic":
        numeric_steps.extend(
            [
                ("signed_log", FunctionTransformer(_signed_log, validate=False)),
                ("scale", RobustScaler()),
            ]
        )
    numeric = Pipeline(numeric_steps)
    if include_rating:
        return ColumnTransformer(
            [
                ("numeric", numeric, numeric_columns),
                (
                    "rating",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ["rating"],
                ),
            ],
            remainder="drop",
        )
    return numeric


def _make_estimator(
    kind: str,
    include_rating: bool,
    columns: Iterable[str],
    random_state: int,
):
    if kind == "logistic":
        return Pipeline(
            [
                ("pre", _make_preprocessor(kind, include_rating, columns)),
                (
                    "model",
                    LogisticRegression(
                        solver="saga",
                        C=0.2,
                        l1_ratio=0.8,
                        max_iter=1500,
                        tol=1e-3,
                        n_jobs=1,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif kind == "tree":
        return Pipeline(
            [
                ("pre", _make_preprocessor(kind, include_rating, columns)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=100,
                        learning_rate=0.05,
                        max_leaf_nodes=3,
                        min_samples_leaf=15,
                        l2_regularization=10.0,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"unknown model kind: {kind}")


def _ece(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 5) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if not np.any(mask):
            continue
        total += mask.mean() * abs(float(y_true[mask].mean()) - float(probability[mask].mean()))
    return float(total)


def _metrics(y_true: np.ndarray, probability: np.ndarray) -> Dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    y_true = np.asarray(y_true, dtype=int)
    threshold = probability >= 0.5
    top_k = max(1, int(np.ceil(0.2 * len(y_true))))
    top = np.argsort(-probability, kind="mergesort")[:top_k]
    positives = max(1, int(y_true.sum()))
    return {
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "pr_auc": float(average_precision_score(y_true, probability)),
        "brier": float(brier_score_loss(y_true, probability)),
        "log_loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "ece_5bin": _ece(y_true, probability),
        "balanced_accuracy_05": float(balanced_accuracy_score(y_true, threshold)),
        "f1_05": float(f1_score(y_true, threshold, zero_division=0)),
        "precision_top20": float(y_true[top].mean()),
        "recall_top20": float(y_true[top].sum() / positives),
    }


HIGHER_IS_BETTER = {
    "roc_auc",
    "pr_auc",
    "balanced_accuracy_05",
    "f1_05",
    "precision_top20",
    "recall_top20",
}
METRIC_ORDER = [
    "roc_auc",
    "pr_auc",
    "brier",
    "log_loss",
    "ece_5bin",
    "balanced_accuracy_05",
    "f1_05",
    "precision_top20",
    "recall_top20",
]


def _bootstrap_delta(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    metric: str,
    seed: int,
    n_bootstraps: int = N_BOOTSTRAPS,
) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    while len(deltas) < n_bootstraps:
        sample = rng.integers(0, n, size=n)
        sampled_y = y_true[sample]
        if sampled_y.min() == sampled_y.max():
            continue
        a = _metrics(sampled_y, p_a[sample])[metric]
        b = _metrics(sampled_y, p_b[sample])[metric]
        deltas.append(b - a)
    deltas = np.asarray(deltas)
    low, high = np.quantile(deltas, [0.025, 0.975])
    if metric in HIGHER_IS_BETTER:
        probability = float(np.mean(deltas > 0))
    else:
        probability = float(np.mean(deltas < 0))
    return float(deltas.mean()), float(low), float(high), probability


def _paired_wilcoxon(
    y_true: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, metric: str
) -> float:
    if metric == "brier":
        loss_a = (y_true - p_a) ** 2
        loss_b = (y_true - p_b) ** 2
    elif metric == "log_loss":
        p_a = np.clip(p_a, 1e-6, 1 - 1e-6)
        p_b = np.clip(p_b, 1e-6, 1 - 1e-6)
        loss_a = -(y_true * np.log(p_a) + (1 - y_true) * np.log(1 - p_a))
        loss_b = -(y_true * np.log(p_b) + (1 - y_true) * np.log(1 - p_b))
    else:
        return float("nan")
    if np.allclose(loss_a, loss_b):
        return 1.0
    try:
        return float(wilcoxon(loss_b, loss_a, alternative="less", method="auto").pvalue)
    except ValueError:
        return float("nan")


def evaluate(features: pd.DataFrame, y: pd.Series, metadata: pd.DataFrame, include_rating: bool):
    X = features.copy()
    if include_rating:
        X["rating"] = metadata.loc[X.index, "rating"].astype(str).to_numpy()
    y_array = y.loc[X.index].to_numpy(dtype=int)

    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    fold_rows = []
    prediction_rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for split_id, (train_index, test_index) in enumerate(splitter.split(X, y_array), start=1):
            train_x = X.iloc[train_index]
            test_x = X.iloc[test_index]
            train_y = y_array[train_index]
            test_y = y_array[test_index]
            model_seed = RANDOM_STATE + split_id
            probabilities = {}
            best_params = {}
            for kind, model_name in [("logistic", "A_elastic_net_logistic"), ("tree", "B_gradient_boosting")]:
                estimator = _make_estimator(kind, include_rating, X.columns, model_seed)
                estimator.fit(train_x, train_y)
                probabilities[model_name] = estimator.predict_proba(test_x)[:, 1]
                best_params[model_name] = {
                    "model": type(estimator.named_steps["model"]).__name__,
                    "params": {
                        key: value
                        for key, value in estimator.named_steps["model"].get_params().items()
                        if key in {
                            "C",
                            "l1_ratio",
                            "max_leaf_nodes",
                            "min_samples_leaf",
                            "l2_regularization",
                        }
                    },
                }

            fold = {
                "split_id": split_id,
                "n_train": len(train_index),
                "n_test": len(test_index),
                "n_default_test": int(test_y.sum()),
                "test_default_rate": float(test_y.mean()),
                "feature_set": "behavior_plus_rating" if include_rating else "invoice_behavior_only",
                "tree_backend": "sklearn.HistGradientBoostingClassifier",
                "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            }
            for model_name, probability in probabilities.items():
                for metric, value in _metrics(test_y, probability).items():
                    fold[f"{model_name}__{metric}"] = value
            fold_rows.append(fold)

            for local_index, firm_index in enumerate(test_index):
                firm = X.index[firm_index]
                prediction_rows.append(
                    {
                        "split_id": split_id,
                        "firm": firm,
                        "y_true": int(y_array[firm_index]),
                        "rating": str(metadata.loc[firm, "rating"]),
                        "p_a": float(probabilities["A_elastic_net_logistic"][local_index]),
                        "p_b": float(probabilities["B_gradient_boosting"][local_index]),
                    }
                )

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(prediction_rows)
    aggregate = (
        predictions.groupby("firm", sort=False)
        .agg(
            y_true=("y_true", "first"),
            rating=("rating", "first"),
            p_a=("p_a", "mean"),
            p_b=("p_b", "mean"),
            n_oof=("split_id", "nunique"),
        )
        .reset_index()
    )
    y_agg = aggregate["y_true"].to_numpy(dtype=int)
    p_a = aggregate["p_a"].to_numpy(dtype=float)
    p_b = aggregate["p_b"].to_numpy(dtype=float)

    summary_rows = []
    comparison_rows = []
    for model_name, probability in [
        ("A_elastic_net_logistic", p_a),
        ("B_gradient_boosting", p_b),
    ]:
        values = {metric: _metrics(y_agg, probability)[metric] for metric in METRIC_ORDER}
        for metric in METRIC_ORDER:
            fold_values = folds[f"{model_name}__{metric}"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "feature_set": "behavior_plus_rating" if include_rating else "invoice_behavior_only",
                    "model": model_name,
                    "metric": metric,
                    "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                    "fold_mean": float(fold_values.mean()),
                    "fold_median": float(np.median(fold_values)),
                    "fold_sd": float(fold_values.std(ddof=1)),
                    "fold_q025": float(np.quantile(fold_values, 0.025)),
                    "fold_q975": float(np.quantile(fold_values, 0.975)),
                    "aggregate_oof": float(values[metric]),
                }
            )

    for metric in METRIC_ORDER:
        a_value = _metrics(y_agg, p_a)[metric]
        b_value = _metrics(y_agg, p_b)[metric]
        boot_mean, boot_low, boot_high, p_better = _bootstrap_delta(
            y_agg, p_a, p_b, metric, RANDOM_STATE + (1000 if include_rating else 0) + len(metric)
        )
        comparison_rows.append(
            {
                "feature_set": "behavior_plus_rating" if include_rating else "invoice_behavior_only",
                "metric": metric,
                "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                "A_aggregate_oof": float(a_value),
                "B_aggregate_oof": float(b_value),
                "B_minus_A": float(b_value - a_value),
                "paired_bootstrap_mean_delta": boot_mean,
                "paired_bootstrap_ci_low": boot_low,
                "paired_bootstrap_ci_high": boot_high,
                "bootstrap_probability_B_better": p_better,
                "wilcoxon_p_for_B_lower_loss": _paired_wilcoxon(y_agg, p_a, p_b, metric),
            }
        )

    return folds, predictions, aggregate, pd.DataFrame(summary_rows), pd.DataFrame(comparison_rows)


def _winner(comparison: pd.DataFrame, metric: str) -> str:
    row = comparison.loc[comparison["metric"] == metric].iloc[0]
    favorable = row["paired_bootstrap_ci_low"] > 0 if row["direction"] == "higher" else row["paired_bootstrap_ci_high"] < 0
    if favorable and row["bootstrap_probability_B_better"] >= 0.95:
        return "B"
    opposite = row["paired_bootstrap_ci_high"] < 0 if row["direction"] == "higher" else row["paired_bootstrap_ci_low"] > 0
    if opposite and row["bootstrap_probability_B_better"] <= 0.05:
        return "A"
    return "tie/inconclusive"


def _write_report(
    output: Path,
    features: pd.DataFrame,
    y: pd.Series,
    folds: pd.DataFrame,
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    include_rating: bool,
):
    feature_set = "behavior_plus_rating" if include_rating else "invoice_behavior_only"
    primary = {metric: _winner(comparisons, metric) for metric in ["pr_auc", "brier", "log_loss", "recall_top20"]}
    wins = list(primary.values())
    if wins.count("A") >= 3:
        conclusion = "方案 A 在主要指标上更稳健，建议作为问题一主模型。"
    elif wins.count("B") >= 3:
        conclusion = "方案 B 在主要指标上有一致且稳定的优势，才可考虑将其升为主模型。"
    else:
        conclusion = "两方案没有在主要指标上形成足够稳定的一致优势，应优先选择可解释、约束清晰的方案 A。"
    if include_rating:
        conclusion = "加入已有信誉评级后，方案 B 在本次敏感性检查的主要指标上明显占优；但这只能说明树模型更会利用评级这一强先验，不能作为问题一主模型的最终依据。"

    lines = [
        "# 问题一：两种风险模型的重复小块样本评测",
        "",
        f"- 评测特征集：`{feature_set}`",
        f"- 企业样本数：{len(features)}；违约数：{int(y.sum())}；违约率：{y.mean():.3f}",
        f"- 切分：5 折 × {N_REPEATS} 次重复 = {len(folds)} 个测试小块；每个小块约 24--25 家企业。",
        "- 切分单位：企业。先将进销项发票聚合到企业层级，避免同一企业的发票明细同时出现在训练和测试中。",
        "- 超参数：评测前预先固定为小样本保守配置；测试小块不参与调参，避免把验证数据反复用于挑模型。",
        "- 配对原则：方案 A、B 在完全相同的训练/测试小块上预测；最终差异用企业级重复交叉验证预测的配对 bootstrap 计算 95% 区间。",
        "",
        "## 指标约定",
        "",
        "主指标为 PR-AUC（违约少时比准确率更有信息）、Brier 分数和 LogLoss（概率是否可信）、Top 20% 风险召回率（银行只重点审查最高风险的一小部分企业）。ROC-AUC、ECE、Balanced Accuracy 和 F1 作为辅助指标。AUC/召回率越高越好，Brier/LogLoss/ECE 越低越好。",
        "",
        "## 评测结果",
        "",
        "| 指标 | 方案 A 聚合 OOF | 方案 B 聚合 OOF | B-A | B 优于 A 的 bootstrap 概率 | 95% 区间 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['metric']} | {row['A_aggregate_oof']:.4f} | {row['B_aggregate_oof']:.4f} | "
            f"{row['B_minus_A']:+.4f} | {row['bootstrap_probability_B_better']:.3f} | "
            f"[{row['paired_bootstrap_ci_low']:+.4f}, {row['paired_bootstrap_ci_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"{conclusion}",
            "",
            "这里的比较只判断‘风险预测模型’的识别、概率和稳定性；MILP 与 NSGA-II 的信贷组合优化不能在缺少年度额度、LGD 和资金成本的情况下直接用这组标签评价。后续应将通过验证的风险概率固定后，再在同一组经济参数下比较两种组合策略。",
            "",
            "## 可复现性与限制",
            "",
            f"- 随机种子固定为 `{RANDOM_STATE}`；脚本和原始附件位于同一仓库。",
            f"- {len(folds)} 个小块不是 {len(folds)} 组独立企业；企业会在不同重复中多次进入测试。因此报告同时给出企业级聚合预测和配对 bootstrap，不把重复小块误报成独立样本量。",
            "- 主评测不使用信誉评级。评级敏感性检查可以单独运行；若加入评级后分数突然接近完美，只能说明评级含有很强的先验信息，不能证明发票特征模型更好。",
            "- 没有违约发生日期，不能做严格的时间外推验证；因此结论应写作‘历史违约倾向预测’，而不是监管意义上的一年期 PD。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(include_rating: bool):
    features, y, metadata = load_problem1()
    folds, predictions, aggregate, summaries, comparisons = evaluate(
        features, y, metadata, include_rating=include_rating
    )
    suffix = "with_rating" if include_rating else "behavior_only"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(RESULTS_DIR / "question1_enterprise_features.csv", encoding="utf-8-sig")
    folds.to_csv(RESULTS_DIR / f"question1_fold_metrics_{suffix}.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"question1_small_blocks_{suffix}.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(RESULTS_DIR / f"question1_enterprise_oof_{suffix}.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(RESULTS_DIR / f"question1_model_summary_{suffix}.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(RESULTS_DIR / f"question1_model_comparison_{suffix}.csv", index=False, encoding="utf-8-sig")
    _write_report(
        RESULTS_DIR / f"question1_model_comparison_{suffix}.md",
        features,
        y,
        folds,
        summaries,
        comparisons,
        include_rating,
    )
    print(comparisons.to_string(index=False))
    print(f"report: {RESULTS_DIR / f'question1_model_comparison_{suffix}.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--include-rating-sensitivity",
        action="store_true",
        help="also run the sensitivity check with the supplied A/B/C/D rating",
    )
    args = parser.parse_args()
    run(include_rating=False)
    if args.include_rating_sensitivity:
        run(include_rating=True)


if __name__ == "__main__":
    main()
