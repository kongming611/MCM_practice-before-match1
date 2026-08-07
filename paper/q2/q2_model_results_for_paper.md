# 问题二模型结果（论文手交付稿）

本稿只使用当前分支在 2026-08-07 生成的 `outputs/q2` 正式报告、CSV、运行清单和图表；不引用 `project_plan.md` 或其他未在本轮重建的历史结果。所有风险值均解释为“历史发票行为对应的相对违约倾向”，不是已由302家企业真实结果验证的一年期违约概率。

## 1. 样本、同口径特征迁移与验收

附件1为123家有标签参考企业，其中历史违约27家、未违约96家，参考组违约标签比例为21.95%；附件2为302家无信贷记录企业，目标表没有 `credit_rating` 和 `default_label` 字段。两组均按同一企业级流程处理发票，金额字段使用“价税合计”并除以10000换算为万元，共同月份窗口为2016-10至2020-02（41个月）。

本轮共构造21项企业级特征，正式风险模型和评级概率模型均只使用以下15项共同发票特征：

`sales_scale_10k`、`purchase_scale_10k`、`operating_net_inflow_proxy_10k`、`sales_growth_trend`、`sales_monthly_cv`、`invoice_activity_per_month`、`sales_return_rate`、`purchase_return_rate`、`void_invoice_rate`、`customer_count`、`supplier_count`、`customer_hhi`、`supplier_hhi`、`purchase_sales_ratio`、`active_month_ratio`。

其中 `zero_amount_invoice_rate` 仅作审计，不进入主模型；其余6项构造特征也没有进入正式15项矩阵。作废、有效、负数、重复记录和分母为零的处理沿用问题一配置，标准化均值/标准差和OOD分位点只由123家参考组拟合。

当前分支数据层验收结果为 `PASS`：302行且企业代号唯一；问题一参考特征重建与既有企业特征表的验收项为 `all_feature_values_equal=true`；共同月份、主特征数、输入字段契约和目标标签缺失均通过；没有使用混合两组数据拟合预处理器。对应证据为 `outputs/q2/reports/q2_validation_report.md`、`data/processed/_runtime/q2/validation/q2_acceptance_checks.csv` 和 `outputs/q2/reports/q2_run_manifest.json`。

## 2. 123家与302家特征分布差异

分布比较以123家参考组的均值、标准差和1%—99%分位区间为基准，不把302家目标组用于拟合标准化或OOD阈值。

| 指标 | 当前输出 |
|---|---:|
| 目标企业数 | 302 |
| 主特征数 | 15 |
| 目标组缺失主特征企业数 | 0 |
| 目标组OOD企业数 | 50/302 = 16.5563% |
| 主特征绝对SMD均值 | 0.121391 |
| 主特征绝对SMD最大值 | 0.594619 |
| 目标组最小训练1%—99%覆盖率 | 0.950331 |
| 目标组平均训练1%—99%覆盖率 | 0.984989 |

绝对SMD最大的特征是 `purchase_sales_ratio`：参考组均值0.789425，目标组均值1.791150，SMD=0.594619；其后较大的差异包括 `supplier_hhi`（SMD=-0.221070）、`sales_scale_10k`（-0.142349）和 `sales_return_rate`（-0.131591）。这些结果说明目标组存在协变量分布迁移，但不能推出302家的真实违约率、真实评级分布或外部预测准确率发生了同方向变化。逐特征结果见 `outputs/q2/reports/q2_feature_distribution_comparison.csv`，逐企业结果见 `outputs/q2/reports/q2_ood_scores.csv`。

## 3. 风险主模型及123家代理验证

风险主模型继承问题一锁定的弹性网Logistic：`solver=saga`、`penalty=elasticnet`、`C=0.2`、`l1_ratio=0.8`、最大迭代5000。评价采用企业级5折×10次重复分层交叉验证，共50个外层测试折；每家参考企业获得10次外层OOF预测，正式指标先按企业取均值。

| 模型/用途 | PR-AUC | Brier | LogLoss | ROC-AUC | 5分位校准误差 | Top20召回 | Top20精确率 | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 弹性网Logistic（主模型） | 0.681437 | 0.109099 | 0.360483 | 0.864583 | 0.084905 | 0.703704 | 0.760000 | 0.523810 |
| Label Spreading（交叉检查） | 0.553560 | 0.132646 | 0.553135 | 0.778164 | 0.084427 | 0.629630 | 0.680000 | 0.653061 |

主模型全123家最终拟合收敛，迭代372次；`q2_final_risk_coefficients.csv` 中非零标准化系数方向为：`sales_scale_10k`=-1.064066、`operating_net_inflow_proxy_10k`=-0.133042、`sales_monthly_cv`=0.682389、`purchase_return_rate`=0.001450、`void_invoice_rate`=0.121879、`customer_hhi`=0.177310、`supplier_hhi`=-0.194453、`purchase_sales_ratio`=0.113558。系数表示条件关联，不作因果解释。

对302家部署的风险区间为50个重复折模型的5%—95%分位区间，是模型不稳定性诊断而非外部置信区间。302家没有真实违约标签，因此本轮没有报告其ROC-AUC、PR-AUC、Brier、LogLoss或真实准确率。

## 4. 有序Logistic评级概率模型

评级按风险从低到高编码为 \(A<B<C<D\)，使用与风险模型相同的15项特征，但评级是因变量而不是风险模型输入。模型为比例优势累计Logistic：

\[
P(Y_i\le h\mid x_i)=\sigma(\theta_h-x_i^\mathsf{T}\gamma),\quad h=1,2,3,
\]

并通过有序阈值得到

\[
\pi_{iA}=F_{i1},\quad \pi_{iB}=F_{i2}-F_{i1},\quad
\pi_{iC}=F_{i3}-F_{i2},\quad \pi_{iD}=1-F_{i3}.
\]

123家企业级聚合OOF评级指标如下：

| 指标 | 数值 |
|---|---:|
| Macro-F1 | 0.440997 |
| Balanced Accuracy | 0.428409 |
| Ordered grade error | 0.634146 |
| Within-one-grade rate | 0.951220 |
| Weighted kappa | 0.637358 |
| Multiclass Brier | 0.613946 |
| Multiclass LogLoss | 1.054340 |
| Probability ECE | 0.140670 |
| Expected absolute grade error | 0.725401 |

302家目标企业的四级概率逐行核验失败数为0，最大概率和误差为 \(2.22\times10^{-16}\)，最小单项概率为 \(9.8343\times10^{-5}\)，最终阈值严格有序且优化收敛。模型预测评级计数为A=56、B=117、C=95、D=34；这些是模型预测类别，不是302家真实评级观测。

## 5. Label Spreading交叉检查

Label Spreading使用KNN核、10个邻居、`alpha=0.2`、最大迭代100、容差0.001。每个测试折的测试标签固定遮蔽为-1，302家节点始终无标签；该模型只用于排序分歧、特征支持和不确定性诊断，不替换、平均或回灌主风险分数和评级概率。

| 范围 | 风险排序Spearman | 主模型Top20与LS重叠 | 平均绝对分歧 | 中位绝对分歧 | 最大绝对分歧 | 分歧≥0.2企业数 |
|---|---:|---:|---:|---:|---:|---:|
| 123家企业级OOF | 0.662393 | 20/25=0.800000 | 0.124592 | 0.085499 | 0.482394 | 23 |
| 302家部署诊断 | 0.569732 | 38/61=0.622951 | 0.153495 | 0.107137 | 0.696539 | 76 |

302家部署诊断中OOD企业数为50，模型阶段按已登记触发项得到的 `high_uncertainty_flag` 为257，评级概率核验失败数为0。优化阶段依据独立登记的优化阈值产生的实际额度上限标记为246家；二者是不同阶段的计数，不能混写。

## 6. 概率混合客户流失率与接受概率

问题二复用当前问题一已验收的附件3 A/B/C保序流失曲线：来源为 `outputs/q1/tables/churn_curve_fitted.csv`，每个评级29个观测利率点，允许点范围为4%—15%，拟合方法为递增Isotonic Regression；D级不拟合、不外推客户流失曲线。

| 评级 | 原始单调违反次数 | 拟合后违反次数 | MAE | RMSE | 最大绝对调整 |
|---|---:|---:|---:|---:|---:|
| A | 3 | 0 | 0.000689 | 0.001819 | 0.005746 |
| B | 2 | 0 | 0.000623 | 0.001949 | 0.007189 |
| C | 2 | 0 | 0.000521 | 0.001406 | 0.004001 |

对企业 \(i\) 和利率 \(r\)，混合客户流失质量与贷款接受概率分别为

\[
M_i(r)=\pi_{iA}L_A(r)+\pi_{iB}L_B(r)+\pi_{iC}L_C(r),
\]
\[
A_i(r)=\pi_{iA}[1-L_A(r)]+\pi_{iB}[1-L_B(r)]+\pi_{iC}[1-L_C(r)]
       =1-\pi_{iD}-M_i(r).
\]

例如，E126在9.85%利率下的四级评级概率为(0.673036, 0.271424, 0.051050, 0.004490)，A/B/C拟合流失率为(0.708302, 0.673527, 0.658839)，混合客户流失质量为0.693158，贷款接受概率为0.302352。E154在14.25%下的四级概率为(0.624235, 0.307848, 0.062360, 0.005557)，拟合流失率为(0.880180, 0.843909, 0.840532)，混合客户流失质量为0.861651，贷款接受概率为0.132792。接受概率不是 \(1-p_i\)，D级概率质量也没有被重新归一化到A/B/C。

## 7. 五种概率/比例的写作边界

| 名称 | 本轮定义 | 不能写成 |
|---|---|---|
| 历史违约倾向 | 发票行为弹性网Logistic输出，优化主场景使用重复折风险均值 `risk_mean` | 已验证的一年期PD或302家真实违约率 |
| 评级概率 | 有序Logistic输出的 \(\pi_{iA},\pi_{iB},\pi_{iC},\pi_{iD}\) | 违约概率 |
| 客户流失率 | 附件3按评级和利率拟合的 \(L_g(r)\) | 违约率或损失率 |
| 贷款接受概率 | 概率混合后的 \(A_i(r)\)，同时受D级质量和利率影响 | \(1-p_i\) 或已被真实申请数据验证的审批率 |
| 真实违约概率 | 302家无真实违约标签，本轮不可识别 | 由模型小数位数替代的现实频率 |

## 8. 论文图表和正式表路径

模型图表：

- `outputs/q2/figures/model/q2_risk_probability_calibration_oof.png`
- `outputs/q2/figures/model/q2_risk_ranking_disagreement.png`
- `outputs/q2/figures/model/q2_risk_model_instability_interval.png`
- `outputs/q2/figures/model/q2_rating_confusion_matrix_oof.png`
- `outputs/q2/figures/model/q2_rating_probability_heatmap_302.png`

分布图表：

- `outputs/q2/figures/eda/standardized_mean_difference.png`
- `outputs/q2/figures/eda/training_quantile_coverage.png`
- `outputs/q2/figures/eda/ood_novelty_score_distribution.png`
- `outputs/q2/figures/eda/standardized_feature_distributions.png`

主要结果表：`data/processed/q2_risk_rating_scores.csv`、`outputs/q2/tables/q2_model_metrics.csv`、`outputs/q2/tables/q2_rating_metrics.csv`、`outputs/q2/tables/q2_model_disagreement_summary.csv`、`outputs/q2/tables/q2_label_spreading_oof_enterprise.csv`、`outputs/q2/tables/q2_final_risk_coefficients.csv`、`outputs/q2/tables/q2_final_ordered_logistic_parameters.csv`。
