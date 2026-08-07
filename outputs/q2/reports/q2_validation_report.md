# 问题二数据层自动验收报告

最终状态：**PASS**

| check_name | status | detail |
| --- | --- | --- |
| q2_feature_table_302_unique | PASS | {"rows": 302, "unique_ids": 302} |
| target_label_columns_absent | PASS | [] |
| same_q1_processing_and_feature_contract | PASS | {"processing": true, "features": true, "invoice_required_columns": true, "expected_sheets": true, "amount_unit_divisor": true, "primary_feature_count": true, "target_config_is_independent": true} |
| q1_reference_feature_rebuild_matches_existing | PASS | {"q1_existing_shape": [123, 33], "q2_rebuilt_shape": [123, 33], "id_sets_equal": true, "feature_max_abs_diff": {"business_scale_10k": 0.00043000001460313797, "sales_scale_10k": 3.5999983083456755e-05, "purchase_scale_10k": 3.400002606213093e-05, "net_sales_10k": 4.0000013541430235e-05, "operating_net_inflow_proxy_10k": 3.100000321865082e-05, "sales_growth_trend": 4.841088641072133e-11, "sales_monthly_cv": 4.934221919938864e-10, "invoice_activity_per_month": 3.170731588397757e-07, "sales_return_rate": 3.9382441752167097e-11, "purchase_return_rate": 4.4704240309556553e-11, "void_invoice_rate": 4.578314116709947e-11, "zero_amount_invoice_rate": 0.0, "customer_count": 0.0, "supplier_count": 0.0, "customer_hhi": 4.9258763734627564e-11, "supplier_hhi": 4.9563686488340863e-11, "max_customer_share": 4.978850665082746e-11, "max_supplier_share": 4.990963198281406e-11, "purchase_sales_ratio": 4.527931807274399e-09, "active_month_ratio": 4.8780479655619047e-11, "longest_active_streak_ratio": 4.8780479655619047e-11}, "all_feature_values_equal": true} |
| common_month_window_is_shared | PASS | ["2016-10", "2016-11", "2016-12", "2017-01", "2017-02", "2017-03", "2017-04", "2017-05", "2017-06", "2017-07", "2017-08", "2017-09", "2017-10", "2017-11", "2017-12", "2018-01", "2018-02", "2018-03", "2018-04", "2018-05", "2018-06", "2018-07", "2018-08", "2018-09", "2018-10", "2018-11", "2018-12", "2019-01", "2019-02", "2019-03", "2019-04", "2019-05", "2019-06", "2019-07", "2019-08", "2019-09", "2019-10", "2019-11", "2019-12", "2020-01", "2020-02"] |
| standardization_reference_only | PASS | {"standardization": "reference_mean_std_only", "pooled_preprocessing_used": false} |
| ood_threshold_reference_only | PASS | {"source": "attachment1_123_only", "rule": "any_primary_feature_outside_reference_quantile_interval"} |
| feature_legality_pass | PASS | PASS |
| eda_outputs_present | PASS | {"comparison_rows": 15, "ood_rows": 302} |
| ood_enterprise_id_unique | PASS | 302 |
| ood_scores_finite_or_missing_separate | PASS | 0 |
| raw_files_unchanged | PASS | {"added": [], "removed": [], "changed": []} |
| q1_existing_pass_outputs_unchanged | PASS | {"added": [], "removed": [], "changed": []} |
| q1_existing_final_report_is_pass | PASS | outputs/q1/final/q1_final_validation_report.md |
| risk_model_stage_is_separate | PASS | risk, rating and Label Spreading outputs are produced by the separate q2 model stage |
| required_processed_files_present | PASS | ["data/processed/q2_enterprise_features.csv", "data/processed/q2_ood_scores.csv"] |

通过标准包含：302行且主键唯一、q1特征口径重建一致、共同月份窗口一致、标准化与OOD阈值只由123家参考组确定、原始附件和问题一既有PASS输出未改变；风险模型在独立的model阶段运行。
