# 问题二：概率评级信贷组合优化报告

主场景状态：**PASS / optimal**。

## 1. 适配层边界与复用证据

本轮新增 `src/_internal/q2_credit_optimization.py`，不调用或修改问题一确定评级的 `credit_strategy.py` 求解逻辑。问题二只使用评级概率；问题一仅提供已经验收的附件3 A/B/C 保序客户流失率曲线。

| item | value |
| --- | --- |
| q1 churn source | outputs/q1/tables/churn_curve_fitted.csv |
| q1 churn SHA256 | 6e0f767efd6eee863e0ff70b66af7f3851ea1ace854c42feedc842981ba74313 |
| q1 churn byte hash matches manifest | False |
| q1 churn semantic match final workbook | True |
| q1 final validation | PASS |
| curve method | isotonic_regression_increasing |
| observed rate points | 29 |
| D churn curve used | False |
| q2 threshold config | src/_internal/q2_config.yaml |
| OOD reference source | attachment1_123_only |
| OOD reference quantiles | 0.01 / 0.99 |
| q1 credit_strategy code hash | ad7b7a85e73cb84ad6a1ef74cb29ed59240c7c44508b2ea547f371188da44474 |

## 2. 概率评级经济学与目标

对每个企业和附件3实际利率点，使用 `A_i(r)=sum_g pi_ig[1-L_g(r)]`，其中 `g` 只取 A、B、C。`pi_iD` 不外推、不插值客户流失率曲线，因此 `A_i(r) <= 1-pi_iD`。目标是最大化 `sum A_i(r_k) s_ik [(1-p_i)r_k-c_f-p_i*LGD]`。金额单位为万元；`x_ik` 为二元选择变量，`s_ik` 为连续名义额度。

客户流失率/接受概率是附件3的客户行为关系，不是违约率。预期信用损失只由锁定风险分数 `p_i` 与 LGD 计算；本报告不把客户流失率解释成违约概率。

## 3. 主场景规则（预先锁定）

| rule | threshold | action |
| --- | --- | --- |
| D probability | pD >= 0.8 | reject; no D churn curve |
| OOD | q2 ood_flag OR novelty >= 1.0 | cap max loan at 50.0 10k CNY |
| risk interval width | width >= 0.2 | cap max loan at 50.0 10k CNY |
| risk disagreement | abs difference >= 0.2 | cap max loan at 50.0 10k CNY |
| rating entropy | normalized entropy >= 0.75 | cap max loan at 50.0 10k CNY |
| budget | 10000 10k CNY | strict nominal equality in primary |

阈值来源：`q2_optimization_config.yaml` 的预注册政策；`ood_flag` 来自问题二仅用123家参考分布的1%/99%特征区间规则，风险区间、风险分歧和评级熵阈值沿用问题二模型配置中已登记的稳定性诊断。OOD和不确定性只触发额度上限，不改变风险分数；D概率规则才触发拒贷。敏感性表覆盖这些阈值和规则。

## 4. 主场景组合摘要

| metric | value |
| --- | --- |
| solve status | optimal |
| validation status | PASS |
| total enterprises | 302 |
| eligible enterprises | 290 |
| D rejects | 12 |
| uncertainty caps | 246 |
| selected enterprises | 164 |
| nominal offered (10k CNY) | 10000.000000 |
| expected disbursed (10k CNY) | 2122.868113 |
| expected interest (10k CNY) | 230.469558 |
| expected credit loss (10k CNY) | 62.959317 |
| expected funding cost (10k CNY) | 63.686043 |
| expected net return (10k CNY) | 103.824198 |
| weighted risk used | 0.059315 |
| weighted pD | 0.021362 |
| weighted acceptance A | 0.212287 |
| strict budget difference (10k CNY) | 0.0000000000 |

预算恒等式：`sum_i offered_i = 10000.0000000000 10k CNY`, 目标为 `10000.0000000000 10k CNY`，差额 `0.0000000000`。`fallback_used=False`；若主场景不可行，以上状态会明确为 `infeasible`，不会用 nominal cap 替代。

## 5. 约束与求解器诊断

主场景约束包括：每家至多一个实际利率点；获贷额度为0或10--100万元并受不确定性上限约束；D概率拒贷；评级概率、风险、接受概率均在合法区间；主场景名义额度严格等于10000万元。模型为连续额度+二元利率选择的MILP，求解器为已有 `scipy.optimize.milp`/HiGHS，未新增有序Logistic或优化依赖。

| diagnostic | value |
| --- | --- |
| solver_status | optimal |
| solver_status_code | 0 |
| solver_message | Optimization terminated successfully. (HiGHS Status 7: Optimal) |
| is_optimal | True |
| solver_called | True |
| variable_count | 16820 |
| integer_variable_count | 8410 |
| constraint_count | 17111 |
| solve_time_seconds | 0.354972 |
| capacity_max_10k | 16700 |
| budget_projection_adjustment_10k | 0 |

## 6. 敏感性分析

以下为单因素敏感性：每次只改变一项，其他参数保持主场景值。`nominal_equality` 的任何不可行结果保留为不可行，不转成上限；`nominal_cap` 和 `expected_disbursement_cap` 仅作为口径敏感性，不替代主场景。

| sensitivity_axis | sensitivity_level | risk_variant | lgd | funding_cost_rate | budget_definition | solve_status | is_optimal | selected_enterprise_count | nominal_offered_amount_10k | expected_disbursed_amount_10k | expected_net_return_10k | D_reject_count | uncertainty_cap_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risk_variant | mean | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| risk_variant | p90 | p90 | 0.5 | 0.03 | nominal_equality | optimal | True | 166 | 10000 | 1640.54 | 70.7024 | 12 | 246 |
| lgd | 0.3 | mean | 0.3 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2574.65 | 131.472 | 12 | 246 |
| lgd | 0.5 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| lgd | 0.7 | mean | 0.7 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 1779.69 | 80.6159 | 12 | 246 |
| funding_cost_rate | 0.02 | mean | 0.5 | 0.02 | nominal_equality | optimal | True | 164 | 10000 | 2507.59 | 126.489 | 12 | 246 |
| funding_cost_rate | 0.03 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| funding_cost_rate | 0.04 | mean | 0.5 | 0.04 | nominal_equality | optimal | True | 164 | 10000 | 1734.48 | 84.659 | 12 | 246 |
| d_probability_threshold | 0.7 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 22 | 237 |
| d_probability_threshold | 0.8 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| d_probability_threshold | 0.9 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 6 | 251 |
| uncertainty_cap_10k | 30.0 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 238 | 10000 | 1910.93 | 81.5102 | 12 | 246 |
| uncertainty_cap_10k | 50.0 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| uncertainty_cap_10k | 70.0 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 130 | 10000 | 2244.39 | 112.709 | 12 | 246 |
| ood_novelty_threshold | 0.5 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| ood_novelty_threshold | 1.0 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| ood_novelty_threshold | 1.5 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| interval_width_threshold | 0.15 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 165 | 10000 | 2123.64 | 103.802 | 12 | 249 |
| interval_width_threshold | 0.2 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| interval_width_threshold | 0.25 | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 243 |
| budget_definition | nominal_equality | mean | 0.5 | 0.03 | nominal_equality | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| budget_definition | nominal_cap | mean | 0.5 | 0.03 | nominal_cap | optimal | True | 164 | 10000 | 2122.87 | 103.824 | 12 | 246 |
| budget_definition | expected_disbursement_cap | mean | 0.5 | 0.03 | expected_disbursement_cap | optimal | True | 208 | 12400 | 2398.66 | 108.509 | 12 | 246 |

## 7. 不确定性与使用边界

问题二的302家没有真实标签，因此没有报告、也不能计算302家预测准确率。风险值只解释为历史发票行为对应的相对违约倾向；重复折模型得到的区间是不稳定性区间，不是真实外部置信区间。评级概率是模型概率，需结合D概率拒贷和不确定性额度上限使用。

完整302家策略表、候选利率经济学、预算恒等式、组合分解、敏感性和诊断均以CSV落盘。

## 8. 输出文件

| role | path |
| --- | --- |
| 302-enterprise primary strategy | data/processed/q2_credit_strategy.csv |
| 302-enterprise strategy table copy | outputs/q2/tables/q2_credit_strategy.csv |
| primary candidate economics at observed rates | outputs/q2/tables/q2_primary_candidate_economics.csv |
| portfolio summary and one-factor sensitivity | outputs/q2/tables/q2_portfolio_summary.csv |
| budget identity and budget-口径 diagnostics | outputs/q2/tables/q2_budget_identity.csv |
| solver diagnostics | outputs/q2/tables/q2_solver_diagnostics.csv |
| constraint validation | outputs/q2/tables/q2_strategy_validation.csv |
| risk-allocation figure | outputs/q2/figures/credit/q2_credit_allocation_risk.png |
| sensitivity figure | outputs/q2/figures/credit/q2_credit_sensitivity_net_return.png |
| model report | outputs/q2/reports/q2_credit_optimization_report.md |

