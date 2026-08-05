# 问题一企业级特征验收报告

- 状态：**PASS**
- 特征表：`results/features/enterprise_features_123.csv`，形状=123行×33列
- 共构造特征数：21；主模型特征数：15；敏感性模型特征数：20；审计型排除特征数：1
- 抽查企业：E110, E118, E12, E14, E47, E64, E68, E70, E88, E96；本报告不训练风险模型，只验收特征角色和业务口径。

## 验收检查

| check_id | passed | severity | observed | expected | detail |
|---|---|---|---|---|---|
| row_count_123 | True | blocking | 123 | 123 |  |
| one_row_per_enterprise | True | blocking | 123 | 123 |  |
| enterprise_id_unique | True | blocking | 0 | 0 |  |
| all_info_enterprises_present | True | blocking | 0 | 0 | [] |
| no_extra_enterprises | True | blocking | 0 | 0 | [] |
| default_label_complete | True | blocking | 0 | 0 |  |
| credit_rating_complete | True | blocking | 0 | 0 |  |
| all_constructed_features_present | True | blocking | 0 | 0 | [] |
| primary_model_features_present | True | blocking | 0 | 0 | [] |
| sensitivity_model_features_present | True | blocking | 0 | 0 | [] |
| excluded_features_reported_in_feature_table | True | blocking | 0 | 0 | [] |
| dictionary_defined_features_present | True | blocking | 0 | 0 | [] |
| unexplained_extra_columns | True | warning | 0 | 0 | [] |
| numeric_features_no_nan | True | blocking | 0 | 0 | [] |
| numeric_features_no_infinity | True | blocking | 0 | 0 | [] |
| all_numeric_output_columns_no_nan | True | blocking | 0 | 0 | [] |
| all_numeric_output_columns_no_infinity | True | blocking | 0 | 0 | [] |
| no_constant_primary_model_feature | True | blocking | [] | [] | 主模型候选集中不得存在全常数特征；审计型排除特征只报告不阻断 |
| excluded_features_absent_from_model_matrix | True | blocking | [] | [] | excluded_from_model变量不得进入主模型或敏感性模型矩阵 |
| audit_only_features_reported | True | blocking | ["zero_amount_invoice_rate"] | ["zero_amount_invoice_rate"] | 审计型排除特征必须在特征表保留并报告 |
| near_constant_feature_reported | True | warning | ["zero_amount_invoice_rate"] | 仅报告，不自动删除 |  |
| no_duplicate_constructed_feature_columns | True | blocking | [] | [] | [] |
| ratio_features_in_0_1 | True | blocking | {} | {} | {} |
| amount_count_features_nonnegative | True | blocking | {} | {} | {} |
| count_features_integer_valued | True | blocking | {} | {} | {} |
| amount_units_match_dictionary | True | blocking | True | True | 企业级金额字段统一为万元，变量名带_10k |
| constructed_names_match_dictionary | True | blocking | ["business_scale_10k", "sales_scale_10k", "purchase_scale_10k", "net_sales_10k", "operating_net_inflow_proxy_10k", "sales_growth_trend", "sales_monthly_cv", "invoice_activity_per_month", "sales_return_rate", "purchase_return_rate", "void_invoice_rate", "zero_amount_invoice_rate", "customer_count", "supplier_count", "customer_hhi", "supplier_hhi", "max_customer_share", "max_supplier_share", "purchase_sales_ratio", "active_month_ratio", "longest_active_streak_ratio"] | ["business_scale_10k", "sales_scale_10k", "purchase_scale_10k", "net_sales_10k", "operating_net_inflow_proxy_10k", "sales_growth_trend", "sales_monthly_cv", "invoice_activity_per_month", "sales_return_rate", "purchase_return_rate", "void_invoice_rate", "zero_amount_invoice_rate", "customer_count", "supplier_count", "customer_hhi", "supplier_hhi", "max_customer_share", "max_supplier_share", "purchase_sales_ratio", "active_month_ratio", "longest_active_streak_ratio"] | features.names与字典21个构造特征顺序一致 |
| primary_model_feature_count_15 | True | blocking | 15 | 15 |  |
| no_identity_label_leakage_in_model_features | True | blocking | [] | [] | 评级和标签保留在表中但不进入行为特征矩阵 |
| raw_invoice_recalculation | True | blocking | 0 | 0 | deterministic_sample=['E110', 'E118', 'E12', 'E14', 'E47', 'E64', 'E68', 'E70', 'E88', 'E96'] |
| raw_input_unchanged | True | blocking | 450df5f7184aa43b3b1cddaa4387cceb91d4881a666e667b7c84c480a8b2b600 | 450df5f7184aa43b3b1cddaa4387cceb91d4881a666e667b7c84c480a8b2b600 |  |
| feature_output_re_readable | True | blocking | (123, 33) | (123, 33) |  |
| fixed_random_seed_20260805 | True | blocking | 20260805 | 20260805 |  |
| upstream_reproducibility_manifest_present | True | warning | D:\MCM_practice-before-match1\results\reports\q1_run_manifest.json | present |  |

## 阻断项

_无阻断项_

## 警告项与口径说明

_无警告项_

- 主模型特征：sales_scale_10k, purchase_scale_10k, operating_net_inflow_proxy_10k, sales_growth_trend, sales_monthly_cv, invoice_activity_per_month, sales_return_rate, purchase_return_rate, void_invoice_rate, customer_count, supplier_count, customer_hhi, supplier_hhi, purchase_sales_ratio, active_month_ratio。
- 敏感性模型特征：business_scale_10k, sales_scale_10k, purchase_scale_10k, net_sales_10k, operating_net_inflow_proxy_10k, sales_growth_trend, sales_monthly_cv, invoice_activity_per_month, sales_return_rate, purchase_return_rate, void_invoice_rate, customer_count, supplier_count, customer_hhi, supplier_hhi, max_customer_share, max_supplier_share, purchase_sales_ratio, active_month_ratio, longest_active_streak_ratio。
- 允许的 `audit_` 辅助列：audit_all_invoice_count, audit_valid_invoice_count, audit_void_invoice_count, audit_positive_invoice_count, audit_negative_invoice_count, audit_zero_invoice_count, audit_negative_amount_10k, audit_missing_counterparty_count；这些列不属于模型矩阵。
- 全常数构造特征：zero_amount_invoice_rate；主模型候选中的全常数特征：无。
- 近似常数构造特征：zero_amount_invoice_rate；全常数审计事实仍在报告中显示，不造成主模型验收失败。
- |Spearman|>0.85的组合数：11；本轮不自动删除。
- VIF未计算：当前环境没有额外统计包，且小样本下VIF仅作辅助判断；Spearman结果已完整输出。

## 抽查重算

使用固定种子20260805抽取至少5家企业，容差为absolute=1e-06、relative=1e-05；逐特征结果见 `feature_recalculation_check.csv`。
| enterprise_id | feature | feature_table_value | recalculated_value | absolute_error | relative_error | absolute_tolerance | relative_tolerance | passed |
|---|---|---|---|---|---|---|---|---|
| E110 | sales_scale_10k | 20.282751 | 20.282751 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | purchase_scale_10k | 0.104 | 0.104 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | net_sales_10k | 20.282751 | 20.282751 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | operating_net_inflow_proxy_10k | 20.178751 | 20.178751000000002 | 3.552713678800501e-15 | 1.7606211993995571e-16 | 1e-06 | 1e-05 | True |
| E110 | audit_valid_invoice_count | 65.0 | 65.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | void_invoice_rate | 0.2441860465 | 0.2441860465116279 | 1.1627893092835961e-11 | 4.7618990763405315e-11 | 1e-06 | 1e-05 | True |
| E110 | audit_negative_amount_10k | -0.0 | -0.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | sales_return_rate | -0.0 | -0.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | purchase_return_rate | -0.0 | -0.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | active_month_ratio | 0.512195122 | 0.5121951219512195 | 4.8780424144467815e-11 | 9.523797093965261e-11 | 1e-06 | 1e-05 | True |
| E110 | customer_count | 34.0 | 34.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | supplier_count | 1.0 | 1.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | max_customer_share | 0.2800434221 | 0.28004342211764066 | 1.76406667051765e-11 | 6.299261226309839e-11 | 1e-06 | 1e-05 | True |
| E110 | max_supplier_share | 1.0 | 1.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | customer_hhi | 0.154411712 | 0.15441171196620398 | 3.3796021536858234e-11 | 2.1886954751760172e-10 | 1e-06 | 1e-05 | True |
| E110 | supplier_hhi | 1.0 | 1.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E110 | sales_monthly_cv | 1.750780904 | 1.7507809038794955 | 1.205044952712342e-10 | 6.882899796080606e-11 | 1e-06 | 1e-05 | True |
| E118 | sales_scale_10k | 28.14582 | 28.14582 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | purchase_scale_10k | 6.771279 | 6.771279 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | net_sales_10k | 28.09412 | 28.09412 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | operating_net_inflow_proxy_10k | 21.322841 | 21.322841 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | audit_valid_invoice_count | 163.0 | 163.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | void_invoice_rate | 0.05780346821 | 0.057803468208092484 | 1.907515811971905e-12 | 3.3000023546024956e-11 | 1e-06 | 1e-05 | True |
| E118 | audit_negative_amount_10k | 0.0517 | 0.0517 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | sales_return_rate | 0.001836862454 | 0.0018368624541761443 | 1.7614425543155399e-13 | 9.589409106162393e-11 | 1e-06 | 1e-05 | True |
| E118 | purchase_return_rate | -0.0 | -0.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | active_month_ratio | 0.8536585366 | 0.8536585365853658 | 1.4634182754491576e-11 | 1.7142899797824824e-11 | 1e-06 | 1e-05 | True |
| E118 | customer_count | 117.0 | 117.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | supplier_count | 9.0 | 9.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E118 | max_customer_share | 0.2471947877 | 0.24719478771625764 | 1.6257634127825327e-11 | 6.576851510136161e-11 | 1e-06 | 1e-05 | True |
| E118 | max_supplier_share | 0.496878064 | 0.49687806395217216 | 4.782785278933943e-11 | 9.625672021886526e-11 | 1e-06 | 1e-05 | True |
| E118 | customer_hhi | 0.1045492824 | 0.10454928239934788 | 6.521172490892013e-13 | 6.237414873822236e-12 | 1e-06 | 1e-05 | True |
| E118 | supplier_hhi | 0.351508157 | 0.35150815697631627 | 2.368372165051369e-11 | 6.737744538461362e-11 | 1e-06 | 1e-05 | True |
| E118 | sales_monthly_cv | 1.825882772 | 1.8258827715555284 | 4.444715706597435e-10 | 2.434283172368656e-10 | 1e-06 | 1e-05 | True |
| E12 | sales_scale_10k | 24400.30205 | 24400.302051 | 1.0000003385357559e-06 | 4.098311309780511e-11 | 1e-06 | 1e-05 | True |
| E12 | purchase_scale_10k | 9571.356283 | 9571.356283000001 | 1.8189894035458565e-12 | 1.9004510434708438e-16 | 1e-06 | 1e-05 | True |
| E12 | net_sales_10k | 24300.30205 | 24300.302051 | 1.0000003385357559e-06 | 4.1151765787855954e-11 | 1e-06 | 1e-05 | True |
| E12 | operating_net_inflow_proxy_10k | 14755.60078 | 14755.600781 | 9.999985195463523e-07 | 6.77707762940949e-11 | 1e-06 | 1e-05 | True |
| E12 | audit_valid_invoice_count | 2020.0 | 2020.0 | 0.0 | 0.0 | 1e-06 | 1e-05 | True |
| E12 | void_invoice_rate | 0.02931283037 | 0.029312830370014416 | 1.4415552085367267e-14 | 4.917830145846563e-13 | 1e-06 | 1e-05 | True |

## 零金额业务核验

- 零金额业务核验明细见 `zero_amount_invoice_business_check.csv`，汇总见 `zero_amount_invoice_business_summary.csv`，最终决定见 `docs/q1_zero_amount_invoice_rate_decision.md`。
- 完全重复行、边界月份和金额恒等式超差记录继续按现有审计口径保留并可追溯。
- 高相关变量按配置的主模型与敏感性模型列表处理，不从原始特征表删除。
