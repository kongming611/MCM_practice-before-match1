# 问题一最终结果（第四批收尾）

本文档由第四批程序根据实际CSV和求解结果生成。风险值表示依据历史发票行为得到的相对违约倾向，不是严格一年期违约概率；客户流失率不是违约率；所有收益和损失均为参数化情景结果。

## 1. 完整建模闭环

附件1发票审计与企业级特征构造 → 第三批企业级OOF风险模型 → 读取selected_model_risk_score → 附件3利率—客户流失率独立保序拟合 → scipy.optimize.milp信贷组合优化 → 预算、LGD、资金成本、风险和预算口径敏感性 → 企业策略稳定性与最终交付。

## 2. 第三批风险模型摘要

企业样本为123家，评级A/B/C/D数量为{'A': 27, 'B': 38, 'C': 34, 'D': 24}；selected_model统一为`elastic_net_logistic`。正式企业聚合OOF指标如下：

                 model   pr_auc    brier  log_loss  roc_auc  top20_recall
  elastic_net_logistic 0.681437 0.109099  0.360483 0.864583      0.703704
hist_gradient_boosting 0.690278 0.104472  0.346308 0.874228      0.629630

正式优化只使用`selected_model_risk_score`，因为它是已选主模型的企业聚合OOF风险值；不使用全样本拟合概率、树模型概率或default_label。

## 3. 利率—流失率拟合

对每个评级g∈{A,B,C}，使用实际观测利率点拟合：`L_g(r)=IsotonicRegression(r, churn_rate)`，其中`increasing=True`、`y_min=0`、`y_max=1`、`out_of_bounds='clip'`。接受概率为`A_g(r)=1-L_g(r)`。

credit_rating                     fit_method  observed_point_count  raw_monotonic_violation_count  fitted_monotonic_violation_count      mae     rmse  maximum_absolute_adjustment  mean_absolute_adjustment  flat_interval_count  raw_min  raw_max  fitted_min  fitted_max
            A isotonic_regression_increasing                    29                              3                                 0 0.000689 0.001819                     0.005746                  0.000689                    3        0 0.922061           0    0.922061
            B isotonic_regression_increasing                    29                              2                                 0 0.000623 0.001949                     0.007189                  0.000623                    2        0 0.885865           0    0.885865
            C isotonic_regression_increasing                    29                              2                                 0 0.000521 0.001406                     0.004001                  0.000521                    2        0 0.895165           0    0.895165

拟合前A/B/C原始单调违反次数分别为3/2/2，拟合后均为0；最大绝对调整量分别为0.00574563/0.00718907/0.00400116。评级间A≤B≤C只作业务诊断；本数据独立拟合后存在交叉，基准没有静默添加评级联合硬约束。

## 4. 信贷优化模型

对每家A/B/C企业和附件3实际利率点定义二元报价变量x_ik和名义额度变量s_ik。每家企业至多选一个利率：Σ_k x_ik≤1；金额联动为10x_ik≤s_ik≤100x_ik；未获贷时所有s_ik=0。D级企业不建立可放贷变量，最终额度为0且利率留空。利率只能取附件3实际观测点，且处于4%—15%；风险值、流失率和接受概率均限制在0—1。

单位名义授信的情景预期净收益为：`u_ik=A_g(r_k)[(1-p_i)r_k-p_i*LGD-c_f]`。总目标最大化`Σu_ik*s_ik`，并分别输出情景预期利息收入、信用损失、资金成本和净收益，逐行复核分解恒等式。预算口径分别为`Σs=B`、`Σs≤B`和`ΣA_g(r_k)s_ik≤B`；基准使用第一种。

## 5. 预算与代表性基准策略

问题一没有给定年度预算，因此不声称存在唯一银行策略。基准使用`representative_parameterized_scenario`：预算=0.50×B_max、LGD=0.50、资金成本率=0.03、风险=selected_model_risk_score、预算口径=nominal_equality。LGD和资金成本率是示例中心情景，不是题目给定值或数据估计值。

候选企业数为99，B_max=9900万元，代表性预算为4950万元。基准获贷50家，名义总额度4950万元，额度加权平均利率12.3995%，组合加权风险0.0651395。
基准预期实际放款1059.522856万元，情景预期信用损失31.01850538万元，情景预期净收益55.18716253万元。

## 6. 敏感性分析

预算从0.10到1.00×B_max变化时，获贷数量从10家变化到99家，情景净收益从16.0121万元变化到37.8756万元；逐点结果和企业迁移见budget_sensitivity_summary.csv及budget_transition_details.csv。

scenario_family  lgd  funding_cost_rate  offered_enterprise_count  expected_credit_loss_10k  expected_net_return_10k  jaccard_to_baseline
            lgd  0.3               0.03                        50                 27.534707                69.849230                  1.0
            lgd  0.5               0.03                        50                 31.018505                55.187163                  1.0
            lgd  0.7               0.03                        50                 39.281174                43.388651                  1.0
   funding_cost  0.5               0.02                        50                 39.154099                67.150041                  1.0
   funding_cost  0.5               0.03                        50                 31.018505                55.187163                  1.0
   funding_cost  0.5               0.04                        50                 28.028591                45.280422                  1.0

  scenario_family risk_variant         budget_definition  offered_enterprise_count  expected_disbursed_amount_10k  expected_credit_loss_10k  expected_net_return_10k  jaccard_to_baseline
             risk         mean          nominal_equality                        50                    1059.522856                 31.018505                55.187163             1.000000
             risk          p90          nominal_equality                        50                     979.243305                 35.519351                46.647178             0.923077
budget_definition         mean          nominal_equality                        50                    1059.522856                 31.018505                55.187163             1.000000
budget_definition         mean               nominal_cap                        50                    1059.522856                 31.018505                55.187163             1.000000
budget_definition         mean expected_disbursement_cap                        73                    1361.670275                 51.520420                63.367434             0.684932

mean与p90的比较是风险不确定性敏感性；p90场景称为`conservative_risk_scenario`，不称为真实风险上界。三种预算定义分别标注为nominal_equality、nominal_cap和expected_disbursement_cap，不能混合解释。LGD×资金成本使用3×3联合情景，不展开全部笛卡尔积。

## 7. 策略稳定性

在30个正式敏感性场景中，稳健核心企业数量为50，不稳定企业数量为69。获贷频率最高的企业为E14, E2, E20, E23, E3；不稳定企业示例为E106, E69, E72, E86, E89。频率阈值和额度标准差阈值属于本项目分析规则，不是题目给定规则。完整结果见enterprise_strategy_stability.csv。

## 8. 图表与附录

可直接放入论文的图表位于`figures/q1_credit/`，包括流失率拟合、接受概率、基准策略、预算、LGD/资金成本、风险mean/p90、预算定义、联合情景和企业稳定性图。完整123家企业策略表应作为论文附录或补充材料，主文只展示代表性汇总。

## 9. 局限与解释边界

样本仅123家企业，第三批风险值是历史发票行为的相对违约倾向，不是严格一年期PD；附件3流失率是历史统计关系，不是违约率；没有把客户流失与违约混为同一事件。LGD、资金成本和预算均需业务方进一步校准；本轮输出的利润、损失、放款和接受均为参数化情景结果，不代表真实银行利润或真实损失预测。评级只用于附件3曲线匹配、D级业务约束和展示，不进入违约风险模型。

## 10. 主要文件

- 完整交付Excel：`results/final/q1_final_delivery.xlsx`。
- 完整策略：`results/credit_strategy/baseline_enterprise_strategy.csv`。
- 最终验证：`results/final/q1_final_validation_report.md`。
