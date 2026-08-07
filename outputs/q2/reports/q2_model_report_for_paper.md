# 问题二风险、评级与半监督交叉检查模型报告

本报告由本轮程序重新计算生成；未引用 project_plan.md 中的历史指标。

## 1. 运行边界与输入

- 参考企业：123家；目标企业：302家。参考企业违约27家，违约率21.95%。
- 固定随机种子：20260805；风险验证：5折×10次重复分层，共50个外层测试折；每家参考企业10次OOF预测。
- 风险主模型固定为问题一弹性网Logistic：solver=saga、penalty=elasticnet、C=0.2、l1_ratio=0.8；没有树模型、随机森林或综合评价主模型。
- 风险和评级均只使用以下15项共同发票特征：sales_scale_10k, purchase_scale_10k, operating_net_inflow_proxy_10k, sales_growth_trend, sales_monthly_cv, invoice_activity_per_month, sales_return_rate, purchase_return_rate, void_invoice_rate, customer_count, supplier_count, customer_hhi, supplier_hhi, purchase_sales_ratio, active_month_ratio。没有使用信誉评级作为风险输入，也没有把违约标签放进特征。
- `requirements.txt` 未新增有序Logistic专门依赖；有序模型由 scipy 的优化器实现比例优势累计Logistic，scipy 已在原 requirements.txt 中锁定。

## 2. 风险主模型：123家企业级OOF

| model | pr_auc | brier | log_loss | roc_auc | calibration_error_5bin | balanced_accuracy | f1 | top20_recall | top20_precision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elastic_net_logistic | 0.681436927876804 | 0.10909883165575679 | 0.3604834302603501 | 0.8645833333333334 | 0.08490481566388915 | 0.6828703703703703 | 0.5238095238095238 | 0.7037037037037037 | 0.76 |
| label_spreading | 0.5535603208018097 | 0.13264588190280954 | 0.5531354903126673 | 0.7781635802469137 | 0.08442665645642353 | 0.7650462962962963 | 0.6530612244897959 | 0.6296296296296297 | 0.68 |

PR-AUC、Brier、LogLoss、ROC-AUC、5分位箱校准误差和Top20召回均来自123家企业的重复折预测先按企业取均值后的正式OOF；50个折本身不是50组独立企业。

### 主模型与Label Spreading的企业级配对 bootstrap

差值统一为 Label Spreading − 弹性网Logistic；Brier、LogLoss和校准误差按越低越好解释。bootstrap按企业抽样，不把50个重叠折当成独立样本。
| metric | elastic_net_logistic_value | label_spreading_value | difference_point | difference_ci_low | difference_ci_high | label_spreading_better_probability |
| --- | --- | --- | --- | --- | --- | --- |
| pr_auc | 0.681436927876804 | 0.5535603208018097 | -0.12787660707499426 | -0.2691638638024328 | 0.025590787896180713 | 0.06 |
| brier | 0.10909883165575679 | 0.13264588190280954 | 0.023547050247052753 | -0.0027262890449715766 | 0.05074411304408423 | 0.047 |
| log_loss | 0.3604834302603501 | 0.5531354903126673 | 0.19265206005231716 | 0.033596018291158505 | 0.3749975247999275 | 0.009 |
| roc_auc | 0.8645833333333334 | 0.7781635802469137 | -0.08641975308641969 | -0.17481119162640915 | -0.006483777044265346 | 0.014 |
| calibration_error_5bin | 0.08490481566388915 | 0.08442665645642353 | -0.0004781592074656199 | -0.05898250112763059 | 0.08638570542493179 | 0.414 |
| balanced_accuracy | 0.6828703703703703 | 0.7650462962962963 | 0.08217592592592593 | 0.005784406565656511 | 0.16667416741674163 | 0.982 |
| f1 | 0.5238095238095238 | 0.6530612244897959 | 0.12925170068027203 | -0.010977580056412816 | 0.28634860921112626 | 0.96 |
| top20_recall | 0.7037037037037037 | 0.6296296296296297 | -0.07407407407407407 | -0.21883680555555554 | 0.050065789473684195 | 0.075 |
| top20_precision | 0.76 | 0.68 | -0.07999999999999996 | -0.24 | 0.040000000000000036 | 0.075 |

## 3. 302家风险部署与不稳定性区间

- 已在全部123家企业上拟合最终风险Pipeline，并对302家企业输出 `main_risk_score`。全样本最终模型收敛：True，迭代次数：372。
- 每家302企业另用50个重复折模型产生风险分布，输出5%—95%分位区间、标准差和宽度。该区间是重复拟合的模型不稳定性诊断，不是真实外部置信区间，也不是302家真实违约率的置信区间。
- 302家没有真实违约标签，因此本报告禁止报告302家预测准确率、ROC-AUC或任何外部准确性结论；风险值只能解释为历史发票行为对应的相对违约倾向。

## 4. 有序Logistic评级模型

评级按风险从低到高编码为 A < B < C < D，使用比例优势累计模型 P(Y≤h|x)=sigmoid(theta_h−x beta)，而不是多分类Logistic。阈值通过正间隔参数化，确保累计概率有序。
| scope | macro_f1 | balanced_accuracy | ordered_grade_error | within_one_grade_rate | weighted_kappa | multiclass_brier | multiclass_log_loss | probability_ece | expected_absolute_grade_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| enterprise_aggregated_oof | 0.4409971811134602 | 0.42840915606008484 | 0.6341463414634146 | 0.9512195121951219 | 0.6373583174998363 | 0.6139458030895732 | 1.0543404261653548 | 0.14067019005607032 | 0.7254009026488389 |

- 302家每一行的A/B/C/D概率均做了非负和求和为1的逐行核验，失败行数：0。
- 最终模型阈值严格有序：True；优化收敛：True。

## 5. Label Spreading标签遮蔽交叉检查

每个5折×10次测试折中，测试企业的违约标签在Label Spreading的标签数组中固定为-1；302家无标签企业也固定为-1并作为图节点。KNN邻居数、alpha、最大迭代次数和容差均在评测前由配置登记，未使用测试标签调参。Label Spreading只用于风险排序、分歧、分布支持和不确定性标记，不替换或平均主模型风险，也不进入评级概率。
| scope | risk_rank_spearman | top20_count | top20_overlap_count | top20_overlap_fraction_of_risk_top20 | mean_abs_risk_difference | median_abs_risk_difference | max_abs_risk_difference | high_disagreement_count_threshold_0_20 | top20_overlap_fraction_of_main_top20 | ood_count | high_uncertainty_count | rating_probability_failed_rows |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 123_enterprise_aggregated_oof | 0.6623931072732199 | 25 | 20 | 0.8 | 0.1245917944863466 | 0.08549875586804324 | 0.48239360313829277 | 23 |  |  |  |  |
| 302_target_deployment_diagnostic | 0.5697318252209812 | 61 | 38 |  | 0.1534947377421901 | 0.10713736672284302 | 0.6965394019738208 | 76 | 0.6229508196721312 | 50.0 | 257.0 | 0.0 |

## 6. OOD与高不确定性

- 302家中OOD企业数：50；高不确定性企业数：257。
- 运行前登记的触发阈值：重复折区间宽度≥0.2、主模型与Label Spreading风险绝对差≥0.2、归一化评级熵≥0.75，或数据层OOD标记为真。
- 高不确定性标记是诊断信号，建议人工复核或保守授信；它不等于企业已经违约。完整逐企业清单见 q2_risk_rating_scores.csv。

## 7. 输出文件

- `outputs/q2/figures/model/q2_risk_probability_calibration_oof.png`
- `outputs/q2/figures/model/q2_risk_ranking_disagreement.png`
- `outputs/q2/figures/model/q2_risk_model_instability_interval.png`
- `outputs/q2/figures/model/q2_rating_confusion_matrix_oof.png`
- `outputs/q2/figures/model/q2_rating_probability_heatmap_302.png`
- `data/processed/q2_risk_rating_scores.csv`：302家风险、模型不稳定性区间、A/B/C/D概率、Label Spreading分歧、OOD和高不确定性标记。
- `outputs/q2/tables/q2_model_metrics.csv`：风险主模型和Label Spreading的折均值/标准差及企业聚合OOF指标。
- `outputs/q2/tables/q2_model_comparison_bootstrap.csv`：1000次企业级配对bootstrap的差值区间和Label Spreading更优概率。
- `outputs/q2/tables/q2_rating_metrics.csv`：有序评级折均值/标准差及企业聚合OOF指标。
- `outputs/q2/tables/q2_risk_oof_enterprise.csv`、`q2_label_spreading_oof_enterprise.csv`、`q2_rating_oof_enterprise.csv`：123家企业级代理OOF。
- `outputs/q2/reports/q2_risk_rating_run_manifest.json`：配置、输入哈希、模型边界和输出哈希。

## 8. 局限

- 123家有标签企业是附件2无标签迁移的代理验证集；302家没有真实违约和真实评级标签，不能把代理OOF指标外推成302家外部验证。
- 发票行为与历史违约倾向是统计关联，不是因果关系；无违约发生时间和外部观察期时，风险值不应称为严格的一年期PD。
- 有序评级指标反映附件1的历史评级标签，评级本身与违约高度绑定，因此评级模型只用于评级概率和诊断，不能回灌风险主模型。
