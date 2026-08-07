# 问题二特征分布与OOD报告

## 数据概况

- 训练参考组：123家。
- 问题二目标组：302家。
- 风险模型未训练；本报告只覆盖数据层和分布诊断。

## 标准化和分布比较

标准化均值差的均值和标准差只由附件1训练参考组的均值、样本标准差计算；KS和Wasserstein直接比较两组原始特征值，不拟合合并分布参数。

| feature | reference_mean | target_mean | standardized_mean_difference | ks_statistic | ks_p_value | wasserstein_distance | train_min_max_coverage | train_q01_q99_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sales_scale_10k | 12586.993647138212 | 5615.864685225165 | -0.1423485490001257 | 0.13831906530985844 | 0.06294503242807813 | 8012.719898607555 | 0.9966887417218543 | 0.9933774834437086 |
| purchase_scale_10k | 9021.734771398373 | 3128.2084931258278 | -0.0906346719157909 | 0.143164809131535 | 0.04927831188100327 | 6235.10696543103 | 1.0 | 0.9801324503311258 |
| operating_net_inflow_proxy_10k | 3472.916913227643 | 2399.064446036424 | -0.04081027151781565 | 0.08415441770311749 | 0.5314679484578643 | 4413.98228225228 | 1.0 | 0.9735099337748344 |
| sales_growth_trend | 0.03288871232595146 | 0.032669744404126705 | -0.004415731008132957 | 0.05820276745813818 | 0.9072504239425422 | 0.004897216293422425 | 0.9966887417218543 | 0.9933774834437086 |
| sales_monthly_cv | 1.5311710986848612 | 1.463505331594622 | -0.07619548583753566 | 0.10730630485112791 | 0.2447488204190341 | 0.0843182996916015 | 0.9900662251655629 | 0.9735099337748344 |
| invoice_activity_per_month | 69.35693039857227 | 54.527055403004354 | -0.10281515886293437 | 0.0850428040704248 | 0.5182315286526722 | 20.186934088691558 | 1.0 | 0.9933774834437086 |
| sales_return_rate | 0.03264055242375516 | 0.021452570972664637 | -0.13159141868353222 | 0.05645291552253273 | 0.9243784317334498 | 0.012894642563759351 | 1.0 | 1.0 |
| purchase_return_rate | 0.014633081775617457 | 0.010688540999609168 | -0.09318473536807026 | 0.04651914068809565 | 0.9856663315398194 | 0.0040032335286537615 | 1.0 | 0.9966887417218543 |
| void_invoice_rate | 0.07845961918936264 | 0.07211036599565007 | -0.08275752030070883 | 0.06000646099176227 | 0.8876429239370367 | 0.010595116163848284 | 0.9966887417218543 | 0.9768211920529801 |
| customer_count | 219.58536585365854 | 167.22847682119206 | -0.06665184015709473 | 0.04926506218704568 | 0.9747494144373298 | 55.32711462876236 | 1.0 | 0.9900662251655629 |
| supplier_count | 242.4227642276423 | 247.14238410596028 | 0.010065639167190518 | 0.12749690410811398 | 0.1050779531461343 | 50.271065525224756 | 0.9966887417218543 | 0.9966887417218543 |
| customer_hhi | 0.22409688280599735 | 0.2435814414467179 | 0.08443904139747802 | 0.06883648306681756 | 0.7692480130222723 | 0.02430559196002702 | 0.9933774834437086 | 0.9867549668874173 |
| supplier_hhi | 0.32114262368040647 | 0.25509301913962795 | -0.22107004383070988 | 0.14432240348893555 | 0.046538708137069576 | 0.06605186587582418 | 0.9735099337748344 | 0.9735099337748344 |
| purchase_sales_ratio | 0.7894253593097963 | 1.7911505032951653 | 0.5946190867402417 | 0.0759435740052764 | 0.659583194633332 | 1.1006352411204587 | 0.9701986754966887 | 0.9503311258278145 |
| active_month_ratio | 0.7709696609161215 | 0.7856565982878372 | 0.07926282898289826 | 0.08773488397135631 | 0.4787676591394281 | 0.019864923249458627 | 1.0 | 0.9966887417218543 |

完整结果见 q2_feature_distribution_comparison.csv。

## OOD/新颖度

- OOD企业数：50/302。
- OOD企业比例：0.165563。
- 使用的阈值：每项主特征的附件1训练分布0.01至0.99分位区间。
- OOD阈值在运行前由123家参考组确定，未使用302家目标分布调参。
- 缺失特征单独报告，不把缺失自动当作分布外。

| enterprise_id | ood_feature_count | ood_feature_fraction | missing_primary_feature_count | novelty_score_max_tail_distance | ood_flag | threshold_source | threshold_rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| E419 | 2 | 0.13333333333333333 | 0 | 20.261179941119305 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E381 | 2 | 0.13333333333333333 | 0 | 3.958201487007899 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E125 | 2 | 0.13333333333333333 | 0 | 2.3156982337637424 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E124 | 2 | 0.13333333333333333 | 0 | 1.7949889136225352 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E420 | 1 | 0.06666666666666667 | 0 | 1.5492509308233122 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E217 | 4 | 0.26666666666666666 | 0 | 1.1137530668967899 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E187 | 4 | 0.26666666666666666 | 0 | 1.1095356284865965 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E264 | 2 | 0.13333333333333333 | 0 | 0.9816944619527999 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E273 | 2 | 0.13333333333333333 | 0 | 0.6684161456261938 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E373 | 1 | 0.06666666666666667 | 0 | 0.49627877001889004 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E424 | 1 | 0.06666666666666667 | 0 | 0.3063092506719977 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E242 | 2 | 0.13333333333333333 | 0 | 0.18777964222432983 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E185 | 1 | 0.06666666666666667 | 0 | 0.1650201980896961 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E191 | 1 | 0.06666666666666667 | 0 | 0.1586872056632929 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E336 | 1 | 0.06666666666666667 | 0 | 0.1548619057027065 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E360 | 2 | 0.13333333333333333 | 0 | 0.14285714285714285 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E127 | 1 | 0.06666666666666667 | 0 | 0.1281210320044032 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E324 | 1 | 0.06666666666666667 | 0 | 0.08034914165565928 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E312 | 1 | 0.06666666666666667 | 0 | 0.0681425210044054 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |
| E193 | 1 | 0.06666666666666667 | 0 | 0.0653428288182572 | True | attachment1_123_only | any_primary_feature_outside_reference_quantile_interval |

完整302家OOD结果见 q2_ood_scores.csv；每项训练分位边界见 q2_ood_reference_bounds.csv。

## 对问题二后续建模的影响

- 分布比较只描述协变量差异，不能证明目标组的真实违约率变化。
- OOD标记可作为后续高不确定性降额或人工复核的输入，但阈值不能在看过风险模型结果后调整。
- 本阶段没有把两组数据合并后拟合缩尾、缺失填补或标准化参数。
