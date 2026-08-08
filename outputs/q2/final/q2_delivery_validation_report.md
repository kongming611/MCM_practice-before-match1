# 问题二论文材料包最终验收

最终状态：**PASS**

- 验收时间：2026-08-08T13:57:45+08:00
- 检查项：19，通过：19，失败：0。
- 正式索引文件数：39。
- 本验收只确认仓库内模型、结果、图表和论文材料一致；另一位建模同学的人工复核与签字仍须单独完成。

| check_name | status | detail |
|---|---|---|
| required_result_files_present | PASS | ["data/processed/q2_enterprise_features.csv", "data/processed/q2_risk_rating_scores.csv", "data/processed/q2_credit_strategy.csv", "outputs/q2/tables/q2_model_metrics.csv", "outputs/q2/tables/q2_rating_metrics.csv", "outputs/q2/tables/q2_risk_oof_enterprise.csv", "outputs/q2/tables/q2_rating_oof_enterprise.csv", "outputs/q2/tables/q2_optimization_sensitivity.csv", "outputs/q2/tables/q2_strategy_validation.csv", "outputs/q2/tables/q2_model_comparison_bootstrap.csv"] |
| q2_feature_table_302_unique | PASS | {"rows": 302, "unique_ids": 302} |
| q2_score_table_302_unique | PASS | {"rows": 302, "unique_ids": 302} |
| rating_probabilities_legal | PASS | {"minimum": 9.834305362e-05, "maximum_row_sum_error": 1.3000001075624823e-10} |
| risk_diagnostics_in_unit_interval | PASS | {"minimum": 4.016551567e-05, "maximum": 0.9925803723} |
| q2_strategy_302_unique | PASS | {"rows": 302, "unique_ids": 302} |
| strict_nominal_budget_10000_10k | PASS | {"offered_sum_10k": 10000.0, "difference_10k": 0.0} |
| loan_amount_bounds | PASS | {"loaned_count": 164} |
| selected_rates_legal_and_observed | PASS | {"selected_rate_min": 0.0985, "selected_rate_max": 0.1425} |
| D_probability_rejects_have_zero_loan | PASS | {"D_reject_count": 12} |
| all_24_optimization_scenarios_pass | PASS | {"scenario_count": 24, "optimal_count": 24, "pass_count": 24} |
| strategy_validation_all_pass | PASS | {"rows": 24, "pass_count": 24} |
| enterprise_OOF_contract | PASS | {"risk_oof_rows": 123, "rating_oof_rows": 123} |
| paired_bootstrap_contract | PASS | {"metric_rows": 9, "replicates": [1000]} |
| data_validation_report_pass | PASS | outputs/q2/reports/q2_validation_report.md |
| paper_package_files_present | PASS | ["paper/q2/README.md", "paper/q2/q2_v1_problem_analysis.md", "paper/q2/q2_v2_data_and_model_report.md", "paper/q2/q2_v3_results_and_validation_report.md", "paper/q2/q2_final_paper_materials.md", "paper/q2/q2_assumptions_and_limitations.md", "paper/q2/q2_submission_checklist.md"] |
| eleven_nonempty_png_figures | PASS | {"figure_count": 11, "invalid": []} |
| figure_source_data_present | PASS | ["data/processed/q1_enterprise_features.csv", "data/processed/q2_enterprise_features.csv", "data/processed/q2_risk_rating_scores.csv", "outputs/q2/reports/q2_feature_distribution_comparison.csv", "outputs/q2/reports/q2_ood_scores.csv", "outputs/q2/tables/q2_credit_strategy.csv", "outputs/q2/tables/q2_label_spreading_oof_enterprise.csv", "outputs/q2/tables/q2_optimization_sensitivity.csv", "outputs/q2/tables/q2_rating_confusion_matrix.csv", "outputs/q2/tables/q2_risk_oof_enterprise.csv"] |
| final_paper_materials_match_primary_numbers | PASS | {"required_fragments": ["164", "10000", "103.824198", "2122.868113", "302家没有真实违约标签"]} |

论文手总入口：`paper/q2/README.md`。
