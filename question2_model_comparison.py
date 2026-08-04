"""问题二：在真实信息条件下比较监督迁移与半监督方案。

问题二的附件2没有违约结果，因此不能直接在302家企业上计算准确率、PR-AUC
或Brier分数。本脚本采用标签遮蔽式验证：

* 从附件1的123家有标签企业中做5折、10次重复的分层企业级切分；
* 每个测试小块约24--25家，测试标签在建模时完全隐藏；
* 方案A只使用其余有标签企业；
* 方案B使用其余有标签企业，并把附件2的302家企业作为无标签图节点；
* 两种方案在完全相同的测试小块上配对比较。

这会回答“在问题二的信息条件下，哪一种风险量化方案更可靠”。贷款组合
优化本身仍不在这里比较，因为题目没有给出LGD、资金成本和真实贷款结果；
在这些参数缺失时强行比较组合收益会把假设误报成事实。

方案A：弹性网Logistic监督迁移。
方案B：基于K近邻图的Label Spreading半监督标签传播，作为“聚类/标签传播”
方案的可复现实验实现。参数在评测前固定，不用测试小块调参。
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import FunctionTransformer, RobustScaler
from sklearn.semi_supervised import LabelSpreading

from question1_model_comparison import (
    DATA_DIR,
    HIGHER_IS_BETTER,
    METRIC_ORDER,
    N_BOOTSTRAPS,
    N_REPEATS,
    N_SPLITS,
    RANDOM_STATE,
    RESULTS_DIR,
    _aggregate_direction,
    _bootstrap_delta,
    _find_data,
    _make_estimator,
    _metrics,
    _paired_wilcoxon,
    _signed_log,
    load_problem1,
)


MODEL_A = "A_supervised_transfer"
MODEL_B = "B_label_propagation"
PRIMARY_METRICS = ["pr_auc", "brier", "log_loss", "recall_top20"]
GRAPH_NEIGHBORS = 10
GRAPH_ALPHA = 0.2
GRAPH_MAX_ITER = 50
N_DEPLOYMENT_BLOCKS = 10


def _build_features(
    path: Path, input_sheet: int, output_sheet: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build the same enterprise-level invoice features used in question one."""

    info = pd.read_excel(path, sheet_name=0)
    firm = info.iloc[:, 0].astype("string").str.strip()
    metadata = pd.DataFrame(
        {
            "firm": firm,
            "name": info.iloc[:, 1].astype("string").str.strip(),
        }
    ).set_index("firm")
    firm_ids = metadata.index.tolist()

    input_features = _aggregate_direction(
        pd.read_excel(path, sheet_name=input_sheet), firm_ids, "in"
    )
    output_features = _aggregate_direction(
        pd.read_excel(path, sheet_name=output_sheet), firm_ids, "out"
    )
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
    return features, metadata


def load_problem2() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read attachment 2, which contains 302 enterprises without labels."""

    path = _find_data("附件2")
    # Attachment 2 stores input invoices in sheet 2 and output invoices in sheet 1.
    return _build_features(path, input_sheet=2, output_sheet=1)


def _graph_preprocessor() -> Pipeline:
    """A fixed robust transform shared by every label-propagation fit."""

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("signed_log", FunctionTransformer(_signed_log, validate=False)),
            ("scale", RobustScaler()),
        ]
    )


def _label_spreading_probabilities(
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    test_x: pd.DataFrame,
    external_x: pd.DataFrame,
) -> np.ndarray:
    """Predict the positive-label probability for a hidden test block.

    The test rows participate as unlabeled nodes in the graph, which is the
    transductive setting used by semi-supervised learning. Their labels are
    never supplied. The preprocessing fit uses only the labeled training rows
    and the fixed 302-row unlabeled deployment pool.
    """

    preprocessor = _graph_preprocessor()
    preprocessor.fit(pd.concat([train_x, external_x], axis=0))
    context = pd.concat([train_x, test_x, external_x], axis=0)
    transformed = preprocessor.transform(context)

    labels = np.concatenate(
        [
            np.asarray(train_y, dtype=int),
            np.full(len(test_x) + len(external_x), -1, dtype=int),
        ]
    )
    model = LabelSpreading(
        kernel="knn",
        n_neighbors=GRAPH_NEIGHBORS,
        alpha=GRAPH_ALPHA,
        max_iter=GRAPH_MAX_ITER,
    )
    model.fit(transformed, labels)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    start = len(train_x)
    stop = start + len(test_x)
    return np.clip(model.label_distributions_[start:stop, positive_column], 1e-6, 1 - 1e-6)


def _fit_full_deployment(
    labeled_x: pd.DataFrame,
    labels: pd.Series,
    unlabeled_x: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit both schemes using all known labels and all 302 deployment rows."""

    y = labels.to_numpy(dtype=int)
    supervised = _make_estimator(
        "logistic", False, labeled_x.columns, RANDOM_STATE
    )
    supervised.fit(labeled_x, y)
    p_a = supervised.predict_proba(unlabeled_x)[:, 1]

    preprocessor = _graph_preprocessor()
    preprocessor.fit(pd.concat([labeled_x, unlabeled_x], axis=0))
    transformed = preprocessor.transform(pd.concat([labeled_x, unlabeled_x], axis=0))
    graph_labels = np.concatenate([y, np.full(len(unlabeled_x), -1, dtype=int)])
    model = LabelSpreading(
        kernel="knn",
        n_neighbors=GRAPH_NEIGHBORS,
        alpha=GRAPH_ALPHA,
        max_iter=GRAPH_MAX_ITER,
    )
    model.fit(transformed, graph_labels)
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    p_b = model.label_distributions_[len(labeled_x) :, positive_column]
    return p_a, np.clip(p_b, 1e-6, 1 - 1e-6)


def _top_mask(probability: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    count = max(1, int(np.ceil(fraction * len(probability))))
    order = np.argsort(-np.asarray(probability), kind="mergesort")
    result = np.zeros(len(probability), dtype=bool)
    result[order[:count]] = True
    return result


def evaluate(
    features: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    external_features: pd.DataFrame,
    n_bootstraps: int = N_BOOTSTRAPS,
):
    """Run paired repeated small-block validation."""

    y_array = y.loc[features.index].to_numpy(dtype=int)
    splitter = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE
    )
    fold_rows = []
    prediction_rows = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for split_id, (train_index, test_index) in enumerate(
            splitter.split(features, y_array), start=1
        ):
            repeat_id = (split_id - 1) // N_SPLITS + 1
            fold_id = (split_id - 1) % N_SPLITS + 1
            train_x = features.iloc[train_index]
            test_x = features.iloc[test_index]
            train_y = y_array[train_index]
            test_y = y_array[test_index]

            model_a = _make_estimator(
                "logistic", False, features.columns, RANDOM_STATE + split_id
            )
            model_a.fit(train_x, train_y)
            p_a = model_a.predict_proba(test_x)[:, 1]
            p_b = _label_spreading_probabilities(
                train_x, train_y, test_x, external_features
            )

            fold = {
                "split_id": split_id,
                "repeat_id": repeat_id,
                "fold_id": fold_id,
                "n_train_labeled": len(train_index),
                "n_test_hidden": len(test_index),
                "n_external_unlabeled": len(external_features),
                "n_default_test": int(test_y.sum()),
                "test_default_rate": float(test_y.mean()),
                "scheme_b_context_nodes": len(test_index) + len(external_features),
                "graph_backend": "sklearn.LabelSpreading(kernel=knn)",
                "graph_params": json.dumps(
                    {
                        "n_neighbors": GRAPH_NEIGHBORS,
                        "alpha": GRAPH_ALPHA,
                        "max_iter": GRAPH_MAX_ITER,
                    },
                    sort_keys=True,
                ),
            }
            for model_name, probability in [(MODEL_A, p_a), (MODEL_B, p_b)]:
                for metric, value in _metrics(test_y, probability).items():
                    fold[f"{model_name}__{metric}"] = value
            fold_rows.append(fold)

            for local_index, firm_index in enumerate(test_index):
                firm = features.index[firm_index]
                prediction_rows.append(
                    {
                        "split_id": split_id,
                        "repeat_id": repeat_id,
                        "fold_id": fold_id,
                        "firm": firm,
                        "y_true": int(y_array[firm_index]),
                        "rating": str(metadata.loc[firm, "rating"]),
                        "p_a": float(p_a[local_index]),
                        "p_b": float(p_b[local_index]),
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
            p_a_sd=("p_a", "std"),
            p_b_sd=("p_b", "std"),
            n_oof=("split_id", "nunique"),
        )
        .reset_index()
    )
    aggregate[["p_a_sd", "p_b_sd"]] = aggregate[["p_a_sd", "p_b_sd"]].fillna(0.0)
    y_aggregate = aggregate["y_true"].to_numpy(dtype=int)
    p_a_aggregate = aggregate["p_a"].to_numpy(dtype=float)
    p_b_aggregate = aggregate["p_b"].to_numpy(dtype=float)

    summary_rows = []
    for model_name, probability in [
        (MODEL_A, p_a_aggregate),
        (MODEL_B, p_b_aggregate),
    ]:
        aggregate_metrics = _metrics(y_aggregate, probability)
        for metric in METRIC_ORDER:
            fold_values = folds[f"{model_name}__{metric}"].to_numpy(dtype=float)
            summary_rows.append(
                {
                    "model": model_name,
                    "metric": metric,
                    "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                    "fold_mean": float(fold_values.mean()),
                    "fold_median": float(np.median(fold_values)),
                    "fold_sd": float(fold_values.std(ddof=1)),
                    "fold_q025": float(np.quantile(fold_values, 0.025)),
                    "fold_q975": float(np.quantile(fold_values, 0.975)),
                    "aggregate_oof": float(aggregate_metrics[metric]),
                }
            )

    comparison_rows = []
    for metric in METRIC_ORDER:
        a_value = _metrics(y_aggregate, p_a_aggregate)[metric]
        b_value = _metrics(y_aggregate, p_b_aggregate)[metric]
        boot_mean, boot_low, boot_high, p_better = _bootstrap_delta(
            y_aggregate,
            p_a_aggregate,
            p_b_aggregate,
            metric,
            RANDOM_STATE + len(metric) + 2000,
            n_bootstraps=n_bootstraps,
        )
        comparison_rows.append(
            {
                "metric": metric,
                "direction": "higher" if metric in HIGHER_IS_BETTER else "lower",
                "A_aggregate_oof": float(a_value),
                "B_aggregate_oof": float(b_value),
                "B_minus_A": float(b_value - a_value),
                "paired_bootstrap_mean_delta": boot_mean,
                "paired_bootstrap_ci_low": boot_low,
                "paired_bootstrap_ci_high": boot_high,
                "bootstrap_probability_B_better": p_better,
                "wilcoxon_p_for_B_lower_loss": _paired_wilcoxon(
                    y_aggregate, p_a_aggregate, p_b_aggregate, metric
                ),
            }
        )

    return (
        folds,
        predictions,
        aggregate,
        pd.DataFrame(summary_rows),
        pd.DataFrame(comparison_rows),
    )


def _winner(row: pd.Series) -> str:
    if row["direction"] == "higher":
        b_stable = row["paired_bootstrap_ci_low"] > 0 and row["bootstrap_probability_B_better"] >= 0.95
        a_stable = row["paired_bootstrap_ci_high"] < 0 and row["bootstrap_probability_B_better"] <= 0.05
    else:
        b_stable = row["paired_bootstrap_ci_high"] < 0 and row["bootstrap_probability_B_better"] >= 0.95
        a_stable = row["paired_bootstrap_ci_low"] > 0 and row["bootstrap_probability_B_better"] <= 0.05
    if b_stable:
        return "B"
    if a_stable:
        return "A"
    return "tie/inconclusive"


def _balanced_deployment_blocks(
    features: pd.DataFrame, n_blocks: int = N_DEPLOYMENT_BLOCKS
) -> pd.DataFrame:
    """Create deterministic, feature-balanced blocks for the 302 deployments.

    The block key uses only pre-model transaction scale and counterparty
    complexity. Within each stratum, firm IDs are sorted and assigned
    round-robin. No labels, predictions or random numbers enter the selection.
    """

    volume = np.log1p(
        np.maximum(
            features["in_valid_total_abs"] + features["out_valid_total_abs"], 0.0
        )
    )
    complexity = np.log1p(
        np.maximum(
            features["in_valid_counterparties"]
            + features["out_valid_counterparties"],
            0.0,
        )
    )
    frame = pd.DataFrame(
        {
            "firm": features.index.astype(str),
            "volume_log": volume.to_numpy(dtype=float),
            "complexity_log": complexity.to_numpy(dtype=float),
        },
        index=features.index,
    )
    # Keep the enterprise identifier as an ordinary column while sorting;
    # pandas rejects a label that is simultaneously an index level and a column.
    frame.index = np.arange(len(frame))
    frame["volume_bin"] = pd.qcut(
        frame["volume_log"].rank(method="first"), q=4, labels=False
    ).astype(int)
    frame["complexity_bin"] = pd.qcut(
        frame["complexity_log"].rank(method="first"), q=4, labels=False
    ).astype(int)
    frame["stratum"] = frame["volume_bin"] * 4 + frame["complexity_bin"]
    frame["unlabeled_block"] = 0

    assignment_offset = 0
    for _, group in frame.sort_values(["stratum", "firm"]).groupby("stratum", sort=True):
        indices = group.index.to_list()
        for index in indices:
            # Continue the round-robin cursor across strata so the ten
            # deployment blocks remain nearly equal in total size.
            frame.loc[index, "unlabeled_block"] = assignment_offset % n_blocks + 1
            assignment_offset += 1
    frame["unlabeled_block"] = frame["unlabeled_block"].map(lambda value: f"U{int(value):02d}")
    return frame.reset_index(drop=True)


def _domain_support_distance(
    labeled_features: pd.DataFrame, deployment_features: pd.DataFrame
) -> Tuple[np.ndarray, float]:
    """Measure distance from each deployment row to the labeled feature support."""

    preprocessor = _graph_preprocessor()
    preprocessor.fit(labeled_features)
    labeled_z = preprocessor.transform(labeled_features)
    deployment_z = preprocessor.transform(deployment_features)

    train_neighbors = min(6, len(labeled_features))
    train_model = NearestNeighbors(n_neighbors=train_neighbors).fit(labeled_z)
    train_distances = train_model.kneighbors(labeled_z, return_distance=True)[0]
    if train_distances.shape[1] > 1:
        train_support = train_distances[:, 1:].mean(axis=1)
    else:
        train_support = train_distances[:, 0]

    target_neighbors = min(5, len(labeled_features))
    target_model = NearestNeighbors(n_neighbors=target_neighbors).fit(labeled_z)
    target_distances = target_model.kneighbors(deployment_z, return_distance=True)[0]
    target_support = target_distances.mean(axis=1)
    return target_support, float(np.quantile(train_support, 0.95))


def deployment_diagnostics(
    labeled_features: pd.DataFrame,
    y: pd.Series,
    deployment_features: pd.DataFrame,
    deployment_metadata: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score all 302 firms and summarize deterministic deployment blocks."""

    p_a, p_b = _fit_full_deployment(labeled_features, y, deployment_features)
    distance, support_threshold = _domain_support_distance(
        labeled_features, deployment_features
    )
    deployment = deployment_metadata.copy()
    deployment["p_a"] = p_a
    deployment["p_b"] = p_b
    deployment["abs_p_difference"] = np.abs(p_a - p_b)
    deployment["rank_a"] = pd.Series(p_a, index=deployment.index).rank(
        method="first", ascending=False
    ).astype(int)
    deployment["rank_b"] = pd.Series(p_b, index=deployment.index).rank(
        method="first", ascending=False
    ).astype(int)
    deployment["top20_a"] = _top_mask(p_a)
    deployment["top20_b"] = _top_mask(p_b)
    deployment["domain_distance_5nn"] = distance
    deployment["train_support_p95"] = support_threshold
    deployment["outside_labeled_support"] = distance > support_threshold

    block_keys = _balanced_deployment_blocks(deployment_features)
    deployment = deployment.reset_index(names="firm")
    deployment = deployment.merge(block_keys, on="firm", how="left", validate="one_to_one")

    rows = []
    for block, group in deployment.groupby("unlabeled_block", sort=True):
        if len(group) >= 3 and group["p_a"].nunique() > 1 and group["p_b"].nunique() > 1:
            correlation = float(spearmanr(group["p_a"], group["p_b"]).statistic)
        else:
            correlation = float("nan")
        rows.append(
            {
                "unlabeled_block": block,
                "n_firms": len(group),
                "mean_p_a": float(group["p_a"].mean()),
                "mean_p_b": float(group["p_b"].mean()),
                "median_abs_p_difference": float(group["abs_p_difference"].median()),
                "p90_abs_p_difference": float(group["abs_p_difference"].quantile(0.9)),
                "top20_a_count": int(group["top20_a"].sum()),
                "top20_b_count": int(group["top20_b"].sum()),
                "top20_overlap_count": int((group["top20_a"] & group["top20_b"]).sum()),
                "top20_disagreement_count": int((group["top20_a"] ^ group["top20_b"]).sum()),
                "spearman_a_b": correlation,
                "mean_domain_distance_5nn": float(group["domain_distance_5nn"].mean()),
                "outside_support_share": float(group["outside_labeled_support"].mean()),
            }
        )
    return deployment, pd.DataFrame(rows)


def _write_report(
    output: Path,
    features: pd.DataFrame,
    y: pd.Series,
    external_features: pd.DataFrame,
    folds: pd.DataFrame,
    summaries: pd.DataFrame,
    comparisons: pd.DataFrame,
    deployment_blocks: pd.DataFrame,
    n_bootstraps: int,
):
    primary_winners = {
        metric: _winner(comparisons.loc[comparisons["metric"] == metric].iloc[0])
        for metric in PRIMARY_METRICS
    }
    if list(primary_winners.values()).count("A") >= 3:
        conclusion = "方案A在主要指标上形成更稳定的优势，建议作为问题二风险量化主方案；方案B保留为分布外风险交叉检查。"
    elif list(primary_winners.values()).count("B") >= 3:
        conclusion = "方案B在主要指标上形成一致且稳定的优势，才有依据考虑将其升为问题二风险量化主方案。"
    else:
        conclusion = "两方案没有形成至少三项主要指标上的稳定优势，不能仅凭点估计替换主方案；建议采用方案A，并用方案B标记分布外和高分歧企业。"

    lines = [
        "# 问题二：两种方案的标签遮蔽式重复小块评测",
        "",
        "## 评测边界",
        "",
        f"附件1有标签企业：{len(features)}家，其中违约{int(y.sum())}家（{y.mean():.3f}）；附件2无标签企业：{len(external_features)}家。",
        "附件2没有真实违约结果，所以不能把302家直接分成有真值的训练集和测试集。本报告用附件1做标签遮蔽式外部验证：每个隐藏测试小块的标签只用于事后评分，不参与两种方案建模。",
        "比较的是问题二的共同风险量化层，而不是把缺失的LGD、资金成本和贷款结果用人为参数补出来比较组合收益。",
        "",
        "## 小块设计",
        "",
        f"- 企业级5折×{N_REPEATS}次重复分层交叉验证，共{len(folds)}个配对测试小块；每块约24--25家企业，测试违约比例由分层切分保持。",
        "- 每个小块中，方案A只看剩余98--99家有标签企业；方案B看相同的有标签训练集，并把302家附件2企业作为无标签图节点。",
        "- 先聚合发票到企业，再切分；同一企业的发票明细不会跨训练/测试。随机种子固定，参数在评测前固定，测试小块不参与调参。",
        f"- 50个小块不是50组独立样本；每家有标签企业被测试{N_REPEATS}次。差异先按企业聚合，再做{n_bootstraps}次企业级配对bootstrap。",
        "",
        "## 方案参数",
        "",
        "- 方案A：与问题一相同的保守弹性网Logistic（C=0.2，L1比例=0.8），只输入两份附件共有的发票行为特征。",
        f"- 方案B：K近邻图Label Spreading，邻居数={GRAPH_NEIGHBORS}、alpha={GRAPH_ALPHA}、最大迭代={GRAPH_MAX_ITER}；这是半监督聚类/标签传播方案的可复现实验实现。",
        "- 方案B是传导式方法：问题二上线时应把302家作为整体图重新运行，不应把单家企业脱离其余无标签企业单独解释。",
        "",
        "## 指标与判定",
        "",
        "主指标为PR-AUC（稀有违约排序）、Brier和LogLoss（概率可信度）、Top 20%风险召回率（有限审查名额下的发现能力）。ROC-AUC、ECE、Balanced Accuracy和F1作为辅助指标。",
        "B方案只有在某项指标的配对bootstrap 95%区间不跨0，且B优于A的bootstrap概率不低于0.95时才算该项稳定胜出；至少四项主要指标中三项稳定胜出，才允许替换A。",
        "",
        "## 评测结果",
        "",
        "| 指标 | 方案A聚合OOF | 方案B聚合OOF | B-A | B优于A概率 | 95%配对区间 | 稳定结论 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for _, row in comparisons.iterrows():
        lines.append(
            f"| {row['metric']} | {row['A_aggregate_oof']:.4f} | {row['B_aggregate_oof']:.4f} | "
            f"{row['B_minus_A']:+.4f} | {row['bootstrap_probability_B_better']:.3f} | "
            f"[{row['paired_bootstrap_ci_low']:+.4f}, {row['paired_bootstrap_ci_high']:+.4f}] | "
            f"{_winner(row)} |"
        )
    lines.extend(
        [
            "",
            "### 主要指标逐块稳定性",
            "",
            "| 方案 | 指标 | 小块均值 | 小块中位数 | 小块标准差 | 2.5%分位 | 97.5%分位 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in summaries.loc[summaries["metric"].isin(PRIMARY_METRICS)].iterrows():
        lines.append(
            f"| {row['model']} | {row['metric']} | {row['fold_mean']:.4f} | "
            f"{row['fold_median']:.4f} | {row['fold_sd']:.4f} | "
            f"{row['fold_q025']:.4f} | {row['fold_q975']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## 302家部署小块的稳定性检查",
            "",
            "302家没有违约真值，下面的表不能被解释为准确率。它只检查两种方案在真实部署池中的分歧、排序一致性和训练分布支持。小块按交易规模和交易对手数量分层，再按企业代号排序轮转分配，未使用标签、预测结果或随机抽样。",
            "",
            "| 小块 | 企业数 | A平均风险 | B平均风险 | 中位绝对分歧 | P90绝对分歧 | Top20分歧数 | A/B秩相关 | 超出训练支持比例 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in deployment_blocks.iterrows():
        correlation = "NA" if pd.isna(row["spearman_a_b"]) else f"{row['spearman_a_b']:.3f}"
        lines.append(
            f"| {row['unlabeled_block']} | {int(row['n_firms'])} | {row['mean_p_a']:.4f} | "
            f"{row['mean_p_b']:.4f} | {row['median_abs_p_difference']:.4f} | "
            f"{row['p90_abs_p_difference']:.4f} | {int(row['top20_disagreement_count'])} | "
            f"{correlation} | {row['outside_support_share']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## 结论",
            "",
            conclusion,
            "",
            "这里的结论是‘风险量化方案在标签遮蔽验证中的相对表现’，不是302家企业未来一定违约的断言。问题二的评级概率、利率流失率和MILP/鲁棒组合优化应在选定风险方案后，用同一套经济参数继续计算。",
            "",
            "## 限制",
            "",
            "- 标签遮蔽验证仍来自附件1，不能替代真正的附件2外部违约验证；附件2缺少违约和评级真值。",
            "- 没有违约发生日期，结果应称为‘历史发票行为下的相对违约倾向’，不能称为严格的一年期监管PD。",
            "- Label Spreading的概率是图上的标签分布，未必天然校准；因此Brier、LogLoss和校准误差必须与排序指标一起看。",
            "- 贷款组合优劣还需要LGD、资金成本、利率流失函数和实际贷款结果；本报告不把这些未知量伪造为已知。",
            "",
            "## 可复现文件",
            "",
            "- `question2_model_comparison.py`：评测脚本。",
            "- `results/question2_small_blocks.csv`：50个隐藏测试小块的逐企业预测。",
            "- `results/question2_fold_metrics.csv`：每个小块的指标。",
            "- `results/question2_model_comparison.csv`：企业级聚合后的配对比较、bootstrap区间和Wilcoxon检验。",
            "- `results/question2_deployment_blocks.csv`：302家确定性部署小块汇总。",
            "- `results/question2_deployment_scores.csv`：302家方案A/B风险分数、分歧和分布支持诊断。",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(n_bootstraps: int = N_BOOTSTRAPS) -> None:
    labeled_features, y, labeled_metadata = load_problem1()
    deployment_features, deployment_metadata = load_problem2()
    deployment_features = deployment_features.reindex(columns=labeled_features.columns, fill_value=0.0)

    folds, predictions, aggregate, summaries, comparisons = evaluate(
        labeled_features,
        y,
        labeled_metadata,
        deployment_features,
        n_bootstraps=n_bootstraps,
    )
    deployment_scores, deployment_blocks = deployment_diagnostics(
        labeled_features,
        y,
        deployment_features,
        deployment_metadata,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    labeled_features.to_csv(
        RESULTS_DIR / "question2_labeled_enterprise_features.csv",
        encoding="utf-8-sig",
    )
    deployment_features.to_csv(
        RESULTS_DIR / "question2_unlabeled_enterprise_features.csv",
        encoding="utf-8-sig",
    )
    folds.to_csv(
        RESULTS_DIR / "question2_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        RESULTS_DIR / "question2_small_blocks.csv", index=False, encoding="utf-8-sig"
    )
    aggregate.to_csv(
        RESULTS_DIR / "question2_enterprise_oof.csv", index=False, encoding="utf-8-sig"
    )
    summaries.to_csv(
        RESULTS_DIR / "question2_model_summary.csv", index=False, encoding="utf-8-sig"
    )
    comparisons.to_csv(
        RESULTS_DIR / "question2_model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    deployment_scores.to_csv(
        RESULTS_DIR / "question2_deployment_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    deployment_blocks.to_csv(
        RESULTS_DIR / "question2_deployment_blocks.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _write_report(
        RESULTS_DIR / "question2_model_comparison.md",
        labeled_features,
        y,
        deployment_features,
        folds,
        summaries,
        comparisons,
        deployment_blocks,
        n_bootstraps,
    )
    print(comparisons.to_string(index=False))
    print(deployment_blocks.to_string(index=False))
    print(f"report: {RESULTS_DIR / 'question2_model_comparison.md'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-bootstraps",
        type=int,
        default=N_BOOTSTRAPS,
        help=f"paired enterprise bootstrap repetitions (default: {N_BOOTSTRAPS})",
    )
    args = parser.parse_args()
    if args.n_bootstraps < 100:
        raise SystemExit("--n-bootstraps must be at least 100 for a useful interval")
    run(n_bootstraps=args.n_bootstraps)


if __name__ == "__main__":
    main()
