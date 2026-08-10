# Q3 stage-five sensitivity validation

本阶段是确定性参数敏感性分析，不是随机模拟；所有策略仍由 Q2 `_run_scenario` 求解。风险仍是历史发票行为相对违约倾向，不是真实 PD。

行业压力数据为包含 2020-03 的 2020Q1 回顾性外部压力标尺；行业持续期/全年恢复路径尚待团队确认，本阶段没有进行持续期校准。

状态：**PASS**；场景数：13；确定性重复：True。

选择频率稳定性仅使用正式中心+11个单因素场景（12个）；排除 exploratory_combined_adverse。类别计数：{'robust_selected_core': 162, 'consistently_not_selected': 132, 'selection_sensitive': 8}。这些类别是选择频率分析标签，不是企业真实稳健性或违约结论。

持续期校准不可用：Q0只有回顾性2020Q1压力标尺，行业持续期/恢复路径尚需团队确认。

| scenario_id | axis | c_pp | rho | severe lambda | unknown policy | LGD | funding | net return (10k) | credit loss (10k) | Jaccard | HHI | unknown share |
|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| exploratory_combined_adverse | combined_adverse | 20.000 | 0.500 | 1.875 | max_known_S | 0.700 | 0.040 | 31.744025 | 81.265600 | 0.9759 | 0.3417 | 0.5450 |
| formal_robust | formal_robust | 40.000 | 0.250 | 1.500 | max_known_S | 0.500 | 0.030 | 89.426840 | 68.691327 | 1.0000 | 0.3419 | 0.5450 |
| sensitivity_c_pp_20p0 | c_pp | 20.000 | 0.250 | 1.500 | max_known_S | 0.500 | 0.030 | 86.848460 | 70.022260 | 0.9759 | 0.3464 | 0.5500 |
| sensitivity_c_pp_60p0 | c_pp | 60.000 | 0.250 | 1.500 | max_known_S | 0.500 | 0.030 | 91.965515 | 67.468903 | 1.0000 | 0.3419 | 0.5450 |
| sensitivity_funding_cost_rate_0p02 | funding_cost_rate | 40.000 | 0.250 | 1.500 | max_known_S | 0.500 | 0.020 | 110.286573 | 78.408646 | 1.0000 | 0.3419 | 0.5450 |
| sensitivity_funding_cost_rate_0p04 | funding_cost_rate | 40.000 | 0.250 | 1.500 | max_known_S | 0.500 | 0.040 | 71.914481 | 59.641049 | 1.0000 | 0.3419 | 0.5450 |
| sensitivity_lambda_severe_1p125 | lambda_severe | 40.000 | 0.250 | 1.125 | max_known_S | 0.500 | 0.030 | 90.558880 | 68.171692 | 1.0000 | 0.3419 | 0.5450 |
| sensitivity_lambda_severe_1p875 | lambda_severe | 40.000 | 0.250 | 1.875 | max_known_S | 0.500 | 0.030 | 88.389201 | 69.022384 | 0.9879 | 0.3412 | 0.5450 |
| sensitivity_lgd_0p3 | lgd | 40.000 | 0.250 | 1.500 | max_known_S | 0.300 | 0.030 | 119.637836 | 50.248935 | 0.9939 | 0.3365 | 0.5400 |
| sensitivity_lgd_0p7 | lgd | 40.000 | 0.250 | 1.500 | max_known_S | 0.700 | 0.030 | 64.617460 | 77.725409 | 1.0000 | 0.3419 | 0.5450 |
| sensitivity_rho_0p1 | rho | 40.000 | 0.100 | 1.500 | max_known_S | 0.500 | 0.030 | 97.862429 | 65.111719 | 0.9759 | 0.3519 | 0.5550 |
| sensitivity_rho_0p5 | rho | 40.000 | 0.500 | 1.500 | max_known_S | 0.500 | 0.030 | 76.533642 | 71.797297 | 0.9701 | 0.3269 | 0.5300 |
| sensitivity_unknown_policy_p50_known_S | unknown_policy | 40.000 | 0.250 | 1.500 | p50_known_S | 0.500 | 0.030 | 95.729399 | 66.118758 | 0.9641 | 0.3568 | 0.5600 |

## Acceptance checks

```json
{
  "all_numeric_finite": true,
  "all_optimal": true,
  "all_validation_pass": true,
  "budget_exact_all": true,
  "centre_parameters_exact": true,
  "centre_reproduces_stage4_robust": true,
  "combined_adverse_labeled": true,
  "deterministic_repeat_all": true,
  "duration_noncalibration_disclosed": true,
  "enterprise_ids_exact": true,
  "exploratory_adverse_excluded_from_stability": true,
  "formal_selected_count_matches": true,
  "long_rows_13x302": true,
  "no_duration_calibration_claim": true,
  "no_fallback": true,
  "q2_artifacts_unchanged": true,
  "q2_score_components_unchanged": true,
  "risk_override_only_all": true,
  "risk_upper_clip_recorded": true,
  "scenario_count_13": true,
  "scenario_ids_unique": true,
  "single_factor_only": true,
  "stability_rows_302": true,
  "stability_rule_recorded": true,
  "stability_uses_12_formal_scenarios": true,
  "summary_rows_13": true,
  "unknown_p50_actual": true
}
```
