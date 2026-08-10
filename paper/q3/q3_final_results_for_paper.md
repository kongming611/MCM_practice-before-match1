# Q3 可写入论文的证据与段落要点

本文档是依据当前 CSV/JSON 产物整理的结果证据卡，不是终稿。数字均来自当前工作区的 Q3 全链运行；金额字段以 `10k CNY` 计，即万元，不能解释为真实利润或实际损失。

## 1. 结果范围与证据文件

- 样本：302家企业，ID/名称精确连接通过；行业映射为145家 `unknown`。
- 压力来源：[国家统计局 2020Q1 分行业 GDP 页面](https://www.stats.gov.cn/sj/zxfb/202302/t20230203_1900688.html)。该页面为2020-04-18发布的初步核算，含2020年3月；在Q2发票窗口之后，只作为回顾性外部压力标尺。
- 风险表：[q3_scenario_risk_scores.csv](../../data/processed/q3_scenario_risk_scores.csv)。
- 优化与对照：[q3_robust_credit_strategy.csv](../../data/processed/q3_robust_credit_strategy.csv)、[q3_portfolio_scenario_summary.csv](../../outputs/q3/tables/q3_portfolio_scenario_summary.csv)、[q3_fixed_q2_strategy_under_robust_risk.csv](../../outputs/q3/tables/q3_fixed_q2_strategy_under_robust_risk.csv)。
- 全链证据：[q3_full_validation.json](../../outputs/q3/reports/q3_full_validation.json)、[q3_run_manifest.json](../../outputs/q3/reports/q3_run_manifest.json)。

## 2. 组合结果：identity、robust 和 fixed-Q2 对照

下表直接对应 `q3_portfolio_scenario_summary.csv` 与 `q3_robust_optimization_validation.json`；`fixed-Q2-under-robust` 的两项结果来自固定Q2决策、利率和额度后，用 robust 风险重新计算的模型隐含经济量。

| 方案 | 名义预算（万元） | 选择企业数 | 预期实际放款（万元） | 模型隐含净收益（万元） | 模型隐含信用损失（万元） |
|---|---:|---:|---:|---:|---:|
| identity（Q2风险） | 10000 | 164 | 2122.868113 | 103.82419750 | 62.95931693 |
| robust（重新优化） | 10000 | 164 | 1919.463858 | 89.42684021 | 68.69132702 |
| severe（重新优化） | 10000 | 164 | 1919.463858 | 89.42684021 | 68.69132702 |
| fixed-Q2-under-robust | 10000 | 164 | 2122.868113 | 88.58970784 | 75.17742509 |

robust 与 severe 相同是单调风险覆盖下的结构结果，不能当作两份独立证据。相对于固定Q2策略，在同一 robust 风险候选经济量下重新优化的差值为：净收益 **+0.83713237万元**，模型隐含信用损失 **-6.48609807万元**。该差值说明在当前参数和约束下重新分配候选额度的模型结果，不代表实际经营改善。

identity 的净收益和信用损失分别为103.82419750和62.95931693万元；robust 中心结果分别为89.42684021和68.69132702万元。所有方案均满足名义预算严格等式 `10000` 个 `10k CNY` 单位（1亿元）；预期实际放款额不是预算约束的替代定义。

## 3. 行业资金迁移与集中度

结果来自 [q3_industry_exposure_comparison.csv](../../outputs/q3/tables/q3_industry_exposure_comparison.csv) 的 identity/robust 行，按行业代码复算名义额度：

| 行业 | identity（万元） | robust（万元） | 变化（万元） |
|---|---:|---:|---:|
| unknown | 5600 | 5450 | -150 |
| wholesale_retail | 650 | 700 | +50 |
| leasing_business_services | 300 | 350 | +50 |
| other_services | 650 | 700 | +50 |
| 其他行业合计 | 2800 | 2800 | 0 |

因此 unknown 行业迁出150万元，批零、租赁商务和其他服务各增加50万元。行业额度 HHI 从 **0.3568** 降至 **0.3419**；HHI 只是组合集中度摘要，不是行业额度上限。当前模型未设置新增行业上限，不能从这张表推断行业配额政策。

## 4. 压力覆盖与行业异质性段落要点

- 压力幅度按 `d_h=max(0,-g_h)`（百分点）和 `S_h=clip(d_h/40,0,1)` 计算；住宿餐饮 `g=-35.3`、`S=0.8825`，在 severe 下压力量裁剪至1，乘数为1.25。
- 正增长的金融业（+6.0）和信息软件IT（+13.2）保持 `S=0` 和乘数1；模型没有将宏观正增长转化为个体风险奖励。
- unknown 的145行使用已知行业最大 `S=0.8825`；这是一条保守政策规则，不能解读为145家企业都属于住宿餐饮，也不等于这些企业的真实行业冲击。
- 行业映射结果应与 [q3_industry_mapping_validation.json](../../outputs/q3/reports/q3_industry_mapping_validation.json) 联读；名称规则证据不足时保持 unknown，避免把产品词或“科技”等泛词当成行业事实。

## 5. 敏感性与稳定性

敏感性表 [q3_sensitivity_summary.csv](../../outputs/q3/tables/q3_sensitivity_summary.csv) 含13行：正式 robust 中心、11个单因素扰动和1个探索性组合。正式中心为净收益89.42684021万元、模型隐含信用损失68.69132702万元。

在本次测试范围内，各单因素净收益极差如下（仅表示登记网格内的确定性重算范围，不是统计置信区间）：

| 扰动轴 | 网格 | 净收益极差（万元） |
|---|---|---:|
| LGD | 0.30/0.70 | 55.0204 |
| funding cost rate | 0.02/0.04 | 38.3721 |
| rho | 0.10/0.50 | 21.3288 |
| c_pp | 20/60 | 5.1171 |
| lambda_severe | 1.125/1.875 | 2.1697 |
| unknown policy | p50_known_S（相对 max_known_S 中心） | 仅1个替代点，不构成极差 |

探索性组合 `exploratory_combined_adverse` 的净收益为31.7440万元、模型隐含信用损失为81.2656万元；它是边界探索，不是预测，也不进入正式选择频率统计。

策略稳定性来自 [q3_enterprise_strategy_stability.csv](../../outputs/q3/tables/q3_enterprise_strategy_stability.csv)：在排除探索性组合后，12个正式/单因素场景中，`robust_selected_core=162`、`selection_sensitive=8`、`consistently_not_selected=132`。标签按选择频率（≥0.80、介于0和0.80、等于0）生成，只是策略选择频率分析规则，不是企业真实稳健性、违约分类或风险分层。

## 6. 图和可引用证据

三张图均已输出 PDF/SVG/PNG/TIFF，图形 QA 为 `PASS`，来源表和图形契约记录在 [q3_figure_qa.json](../../outputs/q3/reports/q3_figure_qa.json)：

- [q3_industry_shock_heatmap.pdf](../../outputs/q3/figures/q3_industry_shock_heatmap.pdf)：行业压力强度与正增长中性对照；
- [q3_industry_allocation_shift.pdf](../../outputs/q3/figures/q3_industry_allocation_shift.pdf)：identity 到 robust 的行业额度迁移及 HHI；
- [q3_sensitivity_results.pdf](../../outputs/q3/figures/q3_sensitivity_results.pdf)：13个确定性场景的净收益、模型隐含信用损失和策略重合度。

本材料不列举“典型企业”故事，以避免事后挑选个案。若团队后续需要加入企业示例，必须先在 `q3_strategy_adjustments.csv` 中登记可复现的筛选规则（例如“决策改变且额度变化非零”），再同时报告筛选范围和全部符合者，不能只挑有利案例。

## 7. 写作边界

- “风险上升/下降”应写成模型覆盖风险变化，不写成疫情导致的真实违约率变化。
- “净收益/信用损失”应加“预期”或“模型隐含”限定，并注明单位万元。
- robust=severe 只能说明当前单调场景下最大值落在 severe；不能宣称独立稳健性验证。
- 结果段落应与假设/局限 [q3_assumptions_and_limitations.md](q3_assumptions_and_limitations.md) 和复现说明 [q3_reproducibility.md](q3_reproducibility.md) 一起使用。
- 提交前由团队依据 [Q3人工确认包](q3_manual_confirmation_packet.md) 完成参数口径、unknown策略、行业上限、外部资料使用和图表单位确认；自动 `PASS` 不替代签字。
