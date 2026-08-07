# 问题二论文手交付检查清单

本清单以当前分支 `outputs/q2`、`data/processed/q2_*`、运行清单和验收表为准；不使用未在当前分支重建的历史结果。

## 1. 数据和特征

- [x] 参考组123家、目标组302家；目标表不含评级和违约标签。
- [x] 共同月份窗口为2016-10至2020-02，共41个月；金额单位从元除以10000转换为万元。
- [x] 共构造21项特征，正式风险/评级模型使用同口径15项共同发票特征。
- [x] `same_q1_processing_and_feature_contract=PASS`，问题一参考特征重建验收为`all_feature_values_equal=true`。
- [x] 标准化和OOD阈值只由123家参考组确定；未使用302家目标分布拟合预处理器。
- [x] 已交付123/302分布差异、SMD、训练支持覆盖率和50家OOD企业结果。
- [x] 原始数据审计`WARN`已在假设与局限中披露；错误数为0，警告不被写成“数据完全无异常”。

证据：`outputs/q2/reports/q2_validation_report.md`、`data/processed/_runtime/q2/validation/q2_acceptance_checks.csv`、`outputs/q2/reports/q2_feature_distribution_comparison.csv`、`outputs/q2/reports/q2_ood_scores.csv`。

## 2. 风险、评级和交叉检查模型

- [x] 风险主模型为问题一锁定的弹性网Logistic，未把评级或目标标签加入风险输入。
- [x] 采用5折×10次企业级重复分层OOF；报告PR-AUC、Brier、LogLoss、ROC-AUC、校准误差和Top20指标。
- [x] 302家风险值仅解释为历史发票行为的相对违约倾向；未报告302家外部准确率或真实PD。
- [x] 有序评级概率模型明确使用A<B<C<D和四级概率 \(\pi_A,\pi_B,\pi_C,\pi_D\)。
- [x] 302家评级概率非负、逐行和为1；概率核验失败行数为0，阈值严格有序。
- [x] Label Spreading固定遮蔽测试标签，只作排序分歧、支持范围和不确定性诊断，未替换或平均主风险。
- [x] 已报告123家OOF和302家部署的Label Spreading交叉检查及分歧数量。
- [x] 已明确区分历史违约倾向、评级概率、客户流失率、贷款接受概率和真实违约概率。

证据：`outputs/q2/reports/q2_model_report_for_paper.md`、`outputs/q2/reports/q2_risk_rating_run_manifest.json`、`outputs/q2/tables/q2_model_metrics.csv`、`outputs/q2/tables/q2_rating_metrics.csv`、`outputs/q2/tables/q2_model_disagreement_summary.csv`、`outputs/q2/tables/q2_label_spreading_oof_enterprise.csv`、`data/processed/q2_risk_rating_scores.csv`。

## 3. 流失率、接受率和MILP

- [x] 使用当前q1最终交付中的附件3 A/B/C递增保序流失曲线；每个评级29个观测利率点，D级不拟合、不外推。
- [x] 接受概率按 \(A_i(r)=\sum_{g=A,B,C}\pi_{ig}[1-L_g(r)]\) 计算，D级概率质量没有重新归一化到A/B/C。
- [x] 主场景参数已登记：LGD=0.50、资金成本率=0.03、D概率阈值=0.80、不确定性上限50万元、名义预算等式10000万元。
- [x] 主场景求解状态为`optimal`，优化验收为`PASS`，无fallback。
- [x] 名义授信总额为10000万元=1亿元，预算差额为0；不得把预期实际发放额2122.868113万元写成1亿元。
- [x] 利息、信用损失、资金成本和净收益分解恒等式通过，目标函数重算差额为0。
- [x] D概率规则拒贷12家，额度上限标记246家，获贷164家；主场景所有选定利率均为附件3实际观测点。
- [x] 24个主/敏感性情景均为`optimal`且`validation_status=PASS`；预算口径没有混合解释。

证据：`outputs/q2/reports/q2_credit_optimization_report.md`、`outputs/q2/reports/q2_credit_optimization_manifest.json`、`outputs/q2/tables/q2_primary_candidate_economics.csv`、`outputs/q2/tables/q2_portfolio_summary.csv`、`outputs/q2/tables/q2_optimization_sensitivity.csv`、`outputs/q2/tables/q2_budget_identity.csv`、`outputs/q2/tables/q2_solver_diagnostics.csv`、`outputs/q2/tables/q2_strategy_validation.csv`。

## 4. 论文正文、附录和图表

- [x] 正文只保留主场景汇总、敏感性汇总和E217/E126/E127/E154/E400代表性企业。
- [x] 302家完整策略作为附录A引用：`outputs/q2/tables/q2_credit_strategy.csv`；工作流副本为`data/processed/q2_credit_strategy.csv`。
- [x] 302家逐企业概率、风险、OOD和不确定性清单引用：`data/processed/q2_risk_rating_scores.csv`。
- [x] 四份论文手交付文件已规划为：
  - `paper/q2/q2_model_results_for_paper.md`
  - `paper/q2/q2_final_results_for_paper.md`
  - `paper/q2/q2_assumptions_and_limitations.md`
  - `paper/q2/q2_submission_checklist.md`
- [x] 已列出EDA、模型和信贷图表路径；正文图表只引用当前`outputs/q2/figures`文件。
- [x] 已说明模型报告和策略表的正式来源，不引用`project_plan.md`中的历史指标。

图表索引：

- [x] `outputs/q2/figures/eda/standardized_mean_difference.png`
- [x] `outputs/q2/figures/eda/training_quantile_coverage.png`
- [x] `outputs/q2/figures/eda/ood_novelty_score_distribution.png`
- [x] `outputs/q2/figures/eda/standardized_feature_distributions.png`
- [x] `outputs/q2/figures/model/q2_risk_probability_calibration_oof.png`
- [x] `outputs/q2/figures/model/q2_risk_ranking_disagreement.png`
- [x] `outputs/q2/figures/model/q2_risk_model_instability_interval.png`
- [x] `outputs/q2/figures/model/q2_rating_confusion_matrix_oof.png`
- [x] `outputs/q2/figures/model/q2_rating_probability_heatmap_302.png`
- [x] `outputs/q2/figures/credit/q2_credit_allocation_risk.png`
- [x] `outputs/q2/figures/credit/q2_credit_sensitivity_net_return.png`

## 5. 交付前必须保留的复核备注

- [x] q2数据层验收报告为`PASS`，q1既有最终验收报告为`PASS`，原始附件和q1保护输出未被本轮q2数据流程改写。
- [x] q2优化manifest记录的q1流失曲线源文件与旧q1 manifest字节哈希匹配为`False`，但与q1最终交付工作簿语义匹配为`True`；交付文档已如实保留该差异，不写成“字节哈希完全一致”。
- [x] 所有预期损失、预期发放、预期净收益、客户流失率和贷款接受概率都标注为模型/参数情景结果。
- [x] 未将任何未重建的第二问历史数字、302家真实准确率、真实评级分布或真实违约概率写入论文手材料。
