# Q3 稳健信贷策略验证

状态：**PASS**

本阶段仅通过 Q2 `_run_scenario` 重用既有 MILP、评级概率、D 规则、不确定性上限、附件3 A/B/C 流失曲线、LGD、资金成本、利率点与预算约束；Q3 只覆盖 `main_risk_score`。风险是历史发票行为相对违约倾向，不是真实 PD。

稳健定义：对每户在阶段3的离散风险区间取 `max(identity, light, medium, severe)`；在当前单调压力变换下它等于 severe，因此两者相同不代表额外独立信息。

## 五场景组合

| scenario | solve | validation | offered(10k CNY) | expected loss(10k CNY) | expected net(10k CNY) | selected | solve seconds |
|---|---|---|---:|---:|---:|---:|---:|
| identity | optimal | PASS | 10000.000000 | 62.959317 | 103.824198 | 164 | 0.211985 |
| light | optimal | PASS | 10000.000000 | 65.435860 | 97.727195 | 164 | 0.226693 |
| medium | optimal | PASS | 10000.000000 | 67.468903 | 91.965515 | 164 | 0.198381 |
| severe | optimal | PASS | 10000.000000 | 68.691327 | 89.426840 | 164 | 0.205443 |
| robust | optimal | PASS | 10000.000000 | 68.691327 | 89.426840 | 164 | 0.205672 |

## 固定Q2策略与稳健重优化

固定 Q2 决策在 robust 候选经济学下的模型隐含 expected loss=75.177425、expected net=88.589708（不是实际观测损失）；候选经济学来源为 Q2 `_run_scenario` 的 robust candidate 表。稳健重优化 net=89.426840，相对固定策略净收益变化=0.837132，信用损失变化=-6.486098。

## 行业资金迁移

行业表的 nominal_share 对每个场景严格求和为1，HHI 为行业名义额度占比平方和；未知行业单列，不增加没有依据的行业上限。

```json
{
  "checks": {
    "D_rule_and_uncertainty_policy_unchanged": true,
    "budget_equality_all": true,
    "deterministic_repeat_solve_all": true,
    "exact_id_name_join": true,
    "fallback_false_all": true,
    "five_scenarios_present": true,
    "fixed_q2_budget_identity": true,
    "fixed_q2_candidate_match": true,
    "fixed_q2_net_decomposition": true,
    "identity_reproduces_q2": true,
    "industry_hhi_identity": true,
    "industry_share_identity": true,
    "mapping_pass": true,
    "preflight_pass": true,
    "q2_artifacts_unchanged": true,
    "rating_probabilities_and_pD_unchanged": true,
    "risk_only_override": true,
    "risk_used_matches_override": true,
    "robust_risk_is_max": true,
    "robust_severe_risk_equal": true,
    "robust_severe_strategy_equal": true,
    "scenario_ids_unique": true,
    "scenario_rows_302": true,
    "solver_called_all": true,
    "solver_optimal_all": true,
    "stress_pass": true,
    "validation_pass_all": true
  },
  "hashes": {
    "q2_optimization_config_sha256": "fa49a1b01a6907a2c017615afd1ef514a29239ff2b0c95576d6b8c0ad47735d8",
    "q2_optimizer_code_sha256": "509f91c1766e9acd5554b7ffdabb8ae62caeae8a5d77e0f21613e5c4aa73acae",
    "q2_risk_rating_scores": "798ba162525373f73ba2f5a9d4516497512e640031e9c14e1f94906f5075cd3c",
    "q2_strategy_after": "0da2b58eb4d07cd7c908983c79f241981473fb8f9a6e3525ef3f85cc0f68fc47",
    "q2_strategy_before": "0da2b58eb4d07cd7c908983c79f241981473fb8f9a6e3525ef3f85cc0f68fc47",
    "q3_config_sha256": "377751ea17f800a20a0e7b088947affedd9ed99c4a42124dbbf0cdd92398aafb",
    "q3_robust_code_sha256": "8b6f4fe6817e0e1804cfe6677d2e946cb438cc8c801c7ae29a545f397795e44a"
  }
}
```

输出：`data/processed/q3_robust_credit_strategy.csv`、`q3_scenario_credit_strategies.csv`、`q3_portfolio_scenario_summary.csv`、`q3_strategy_adjustments.csv`、`q3_industry_exposure_comparison.csv`、`q3_fixed_q2_strategy_under_robust_risk.csv`。
