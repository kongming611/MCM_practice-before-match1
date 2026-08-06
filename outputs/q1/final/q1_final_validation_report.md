# 问题一最终验收报告

最终状态：PASS

| 检查 | 状态 | 证据 |
|---|---|---|
| required_files_exist | PASS | all first-to-final delivery inputs exist |
| phase_a_feature_validation_pass | PASS | PASS |
| zero_amount_excluded_from_models | PASS | zero_amount_invoice_rate absent from formal feature lists |
| risk_table_123_unique | PASS | (123, 36) |
| selected_model_elastic_net_logistic | PASS | {'elastic_net_logistic': 123} |
| risk_values_legal | PASS | selected_model_risk_score in [0,1] |
| risk_test_count_10_per_enterprise | PASS | cv_split_manifest test rows |
| attachment3_audit_pass | PASS | all attachment-3 checks PASS |
| churn_fitted_complete | PASS | (87, 15) |
| churn_and_acceptance_legal | PASS | fitted churn and acceptance in [0,1] |
| churn_curves_monotone | PASS | churn nondecreasing and acceptance nonincreasing |
| baseline_milp_optimal | PASS | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| baseline_strategy_123_unique | PASS | (123, 28) |
| D_enterprises_zero_loan | PASS | D-level business rule |
| baseline_amount_rate_constraints | PASS | selected amount 10-100万元 and rate 4%-15% |
| baseline_return_decomposition | PASS | interest - loss - funding = net return |
| baseline_numeric_finite | PASS | monetary strategy fields finite |
| baseline_nominal_budget_equality | PASS | 4950 |
| all_official_scenarios_optimal | PASS | 30 scenarios |
| all_scenario_validation_failures_empty | PASS | scenario constraint checks |
| sensitivity_scenario_count | PASS | 30 |
| stability_table_123_unique | PASS | (123, 16) |
| required_figures_nonempty | PASS | all requested q1_credit figures |
| final_excel_re_readable | PASS | ['Assumptions', 'BaselinePortfolio', 'BaselineStrategy', 'BudgetSensitivity', 'ChurnFitted', 'ChurnRaw', 'ModelMetrics', 'ParameterSensitivity', 'README', 'RiskScores', 'StrategyStability', 'Validation'] |
| final_docs_no_placeholders | PASS | final paper/assumptions/checklist documents |
| final_manifest_selected_model | PASS | elastic_net_logistic |
| final_manifest_risk_column | PASS | selected_model_risk_score |
| final_config_hash_recorded | PASS | configuration hash matches current config |
| final_code_hashes_match | PASS | core fourth-batch code hashes |
| output_index_hashes_match | PASS | final output index hashes |

所有官方MILP场景必须达到最优状态；收益、损失和风险均按情景解释。
