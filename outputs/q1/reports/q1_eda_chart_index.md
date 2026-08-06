# 问题一EDA图表索引

- `outputs/q1/figures/eda/class_distribution.png`：违约与未违约企业数量；数据源：data/processed/q1_enterprise_features.csv:default_label。
- `outputs/q1/figures/eda/rating_distribution.png`：A/B/C/D评级企业数量；数据源：data/processed/q1_enterprise_features.csv:credit_rating。
- `outputs/q1/figures/eda/rating_default_distribution.png`：评级—是否违约交叉分布（企业数）；数据源：data/processed/_runtime/feature_validation/eda_rating_default_contingency.csv。
- `outputs/q1/figures/eda/rating_default_rate.png`：各评级违约率（括号显示样本数）；数据源：data/processed/_runtime/feature_validation/eda_rating_default_contingency.csv。
- `outputs/q1/figures/eda/main_feature_distributions.png`：主要特征总体分布；数据源：data/processed/q1_enterprise_features.csv:main features。
- `outputs/q1/figures/eda/main_features_by_default.png`：主要特征按违约标签箱线图；数据源：data/processed/q1_enterprise_features.csv:main features/default_label。
- `outputs/q1/figures/eda/feature_correlation_heatmap.png`：候选特征Spearman相关性；数据源：data/processed/q1_enterprise_features.csv:main features。
